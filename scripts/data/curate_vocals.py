#!/usr/bin/env python3
"""Turn separated vocal stems into dataset/<track>/vocals/<song_id>/<song_id>__pNN.wav parts."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from _audio import read_csv, record_path, resolve_path, setup_console, sha256_file, slugify, write_csv

VOCALS_MARKER = "_(vocals)_"
NAMES_COLUMNS = ["source_stem", "song_id"]
CURATION_FILE = "_curation.json"
WINDOWS_MAX_PATH = 260  # long paths are off by default; +20 leaves room for the temp name and build_manifest
HELD_OUT_FOLDERS = {"reference", "cover_input"}  # ground truth and the song to convert: never training data
STRIP_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".opus", ".webm")

# Removed before picking the title. Applied after NFKC, so fullwidth forms are already ASCII.
BRACKET_BLOCKS = [r"【[^】]*】", r"\[[^\]]*\]", r"〔[^〕]*〕", r"《[^》]*》"]
DECORATIONS = [
    r"@\S+",
    r"클린\s*버전",
    r"무자막",
    r"무대\s*오디오\s*only",
    r"cover\s+by\.?\s*\S*",
    r"규티비",
    r"\bofficial\s+(?:m/?v|music\s+video|audio|video|lyric\s+video)\b",
    r"\bm/v\b",
    r"\bmv\b",
]
# " _ ", "__", " - ", "|" and the Hangul letter "ㅣ" (U+3163, NFKC turns it into U+1175) used as a bar.
SEPARATORS = r"\s+_\s+|_{2,}|\s*[|" + chr(0x3163) + chr(0x1175) + r"]\s*|\s+-\s+|\s+_$"
QUOTED = r"(?<!\w)[\"'‘“「『]([^\"'’”」』]+)[\"'’”」』](?!\w)"
SYMBOL_CATEGORIES = {"So", "Sk", "Cs", "Co", "Cn"}
INVISIBLE = {chr(0x200D), chr(0xFE0E), chr(0xFE0F), chr(0x20E3)}  # ZWJ, variation selectors, keycap
SEGMENT_TRIM = " _-.,!?~:;"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy separated *_(vocals)_* stems into vocals/<song_id>/<song_id>__pNN.wav, cutting out "
        "excluded time ranges (each removed range splits the song into parts; nothing is crossfaded). "
        "Writes <output>/../vocals_names.csv (editable source_stem -> song_id) and <output>/_curation.json.",
        epilog="Build the manifest from the result with: build_manifest.py --input <output> --song-level 0 ...",
    )
    parser.add_argument("--separated", required=True, help='Folder of separated stems; only files with "_(vocals)_".')
    parser.add_argument("--output", required=True, help="Curated vocals folder, e.g. dataset/private/vocals.")
    parser.add_argument(
        "--names",
        default=None,
        help="CSV with source_stem,song_id overriding the default names (e.g. an edited vocals_names.csv).",
    )
    parser.add_argument(
        "--names-out", default=None, help="Where to write the name table. Default: vocals_names.csv next to --output."
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="CSV of ranges to remove: song_id or source_stem, start_s, end_s, reason. Times are seconds or "
        "[h:]mm:ss[.f] on the stem's timeline; empty start = 0, empty end = end of file.",
    )
    parser.add_argument(
        "--artist",
        action="append",
        default=[],
        help="Singer alias (repeatable, case-insensitive), dropped from default song_ids, e.g. --artist NAME.",
    )
    parser.add_argument(
        "--infer-artist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also treat the --separated folder name and title segments repeated in >= 3 stems as aliases "
        "(default: on).",
    )
    parser.add_argument(
        "--mono", action=argparse.BooleanOptionalAction, default=True, help="Downmix to mono (default: on)."
    )
    parser.add_argument("--min-part-s", type=float, default=2.0, help="Drop parts shorter than this (s).")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and name table without writing.")
    return parser.parse_args()


# ------------------------------------------------------------------- naming

def loose_key(text: str) -> str:
    """Comparison key tolerant of Unicode form, case, spacing, and punctuation differences."""
    text = unicodedata.normalize("NFKC", text).casefold()
    for ext in STRIP_EXTENSIONS:
        if text.endswith(ext):
            text = text[: -len(ext)]
    return re.sub(r"[\W_]+", "", text)


def strip_symbols(text: str) -> str:
    return "".join(
        " " if unicodedata.category(ch) in SYMBOL_CATEGORIES else ch for ch in text if ch not in INVISIBLE
    )


def artist_pattern(artists: list[str]) -> re.Pattern | None:
    aliases = sorted({unicodedata.normalize("NFKC", a).strip() for a in artists if a.strip()}, key=len, reverse=True)
    if not aliases:
        return None
    alternation = "|".join(re.escape(a) for a in aliases)
    # "규현(Kyuhyun)의 ", "가수(Alias)이 부르는 ": the alias, its bracketed twin, a particle.
    return re.compile(
        rf"[^\s(]*\(?(?:{alternation})\)?(?:\s*\([^)]*\))?(?:의|이가|이|가|는|은)?\s*(?:부르는)?\s*", re.IGNORECASE
    )


def after_artist(text: str, artists: re.Pattern | None) -> str:
    """Text after the last artist mention ("규현의 X" -> "X"), or the text itself."""
    if artists is None:
        return text
    matches = list(artists.finditer(text))
    rest = text[matches[-1].end() :].strip(SEGMENT_TRIM) if matches else ""
    return rest or (artists.sub(" ", text).strip(SEGMENT_TRIM) if matches else text)


def clean_text(stem: str) -> str:
    text = strip_symbols(unicodedata.normalize("NFKC", stem))
    for pattern in BRACKET_BLOCKS + DECORATIONS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def split_segments(text: str) -> list[str]:
    segments = [s.strip(SEGMENT_TRIM) for s in re.split(SEPARATORS, text)]
    return [s for s in segments if s]


def infer_artists(stems: list[str], folder_name: str, min_count: int = 3) -> list[str]:
    """Aliases guessed from the data: the folder name, and segments repeated across >= min_count stems.

    "규현(Kyuhyun) - 숨", "규현(Kyuhyun) - 한숨", ... -> "규현(Kyuhyun)", "규현", "Kyuhyun".
    """
    counts: dict[str, int] = defaultdict(int)
    for stem in stems:
        for segment in set(split_segments(clean_text(stem))):
            counts[segment.casefold()] += 1
    generic = folder_name.casefold() in {"separated", "vocals", "stems", "output"}
    aliases = {folder_name} if re.fullmatch(r"[^\W\d_][\w ]*", folder_name) and not generic else set()
    for segment, count in counts.items():
        if count >= min_count:
            aliases.add(segment)
            pair = re.fullmatch(r"(.+?)\s*\((.+)\)", segment)
            if pair:
                aliases.update(part.strip() for part in pair.groups())
    return sorted(aliases)


def title_text(stem: str, artists: re.Pattern | None) -> str:
    text = clean_text(stem)
    quoted = [q.strip(SEGMENT_TRIM) for q in re.findall(QUOTED, text) if q.strip(SEGMENT_TRIM)]
    if quoted:
        # Korean cover titles quote "artist - title"; keep the title.
        pieces = [p.strip(SEGMENT_TRIM) for p in quoted[0].split("-")]
        return next((p for p in reversed(pieces) if p), quoted[0])
    segments = split_segments(text)
    if not segments:
        return stem
    plain = [s for s in segments if artists is None or not artists.search(s)]
    return plain[0] if plain else after_artist(segments[0], artists)


def default_song_id(stem: str, artists: re.Pattern | None) -> str:
    song_id = slugify(title_text(stem, artists))
    return song_id if song_id != "x" else slugify(stem)


def checked_song_id(song_id: str, source_stem: str) -> str:
    """build_manifest re-slugifies the folder name, so a song_id must already be slug-stable."""
    cleaned = slugify(song_id.strip())
    if cleaned != song_id.strip():
        print(f"warning: song_id {song_id!r} for {source_stem!r} is not filename-safe; using {cleaned!r}")
    return cleaned


# ------------------------------------------------------------------ inputs

@dataclass
class Exclusion:
    start_s: float
    end_s: float | None
    reason: str


@dataclass
class Source:
    path: Path
    stem: str
    song_id: str = ""
    exclusions: list[Exclusion] = field(default_factory=list)


def find_sources(separated: Path) -> list[Source]:
    sources = []
    for path in sorted(separated.iterdir()):
        name = path.name
        if path.is_file() and VOCALS_MARKER in name and path.suffix.lower() in (".wav", ".flac"):
            sources.append(Source(path=path, stem=name[: name.rfind(VOCALS_MARKER)]))
    return sources


def read_names(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if rows and not set(NAMES_COLUMNS) <= set(rows[0]):
        raise SystemExit(f"{record_path(path)} needs columns: {', '.join(NAMES_COLUMNS)}")
    return {loose_key(r["source_stem"]): r["song_id"] for r in rows if r.get("source_stem") and r.get("song_id")}


def assign_names(sources: list[Source], names: dict[str, str], artists: re.Pattern | None) -> None:
    used = set()
    for source in sources:
        key = loose_key(source.stem)
        if key in names:
            source.song_id = checked_song_id(names[key], source.stem)
            used.add(key)
        else:
            source.song_id = default_song_id(source.stem, artists)
    for key in sorted(set(names) - used):
        print(f"warning: names entry matches no stem and is ignored: {key}")
    by_song = defaultdict(list)
    for source in sources:
        by_song[source.song_id].append(source.stem)
    for song_id, stems in sorted(by_song.items()):
        if len(stems) > 1:
            print(f"note: {len(stems)} stems share song_id {song_id!r} and become parts of one song: {stems}")


def parse_time(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    seconds = 0.0
    for piece in value.split(":"):
        seconds = seconds * 60.0 + float(piece)
    return seconds


def read_exclusions(path: Path, sources: list[Source]) -> None:
    rows = read_csv(path)
    if rows and not ({"start_s", "end_s"} <= set(rows[0]) and {"song_id", "source_stem"} & set(rows[0])):
        raise SystemExit(f"{record_path(path)} needs columns: song_id or source_stem, start_s, end_s, reason")
    by_stem = defaultdict(list)
    by_song = defaultdict(list)
    for source in sources:
        by_stem[loose_key(source.stem)].append(source)
        by_song[source.song_id].append(source)

    problems = []
    for line, row in enumerate(rows, 2):
        song_id, stem = (row.get("song_id") or "").strip(), (row.get("source_stem") or "").strip()
        if (song_id or stem).startswith("#"):
            continue
        if stem:
            matched = by_stem.get(loose_key(stem), [])
        else:
            matched = by_song.get(song_id, [])
        if len(matched) != 1:
            label = f"source_stem {stem!r}" if stem else f"song_id {song_id!r}"
            hint = " (use source_stem)" if matched else ""
            problems.append(f"line {line}: {label} matches {len(matched)} stems{hint}")
            continue
        try:
            start, end = parse_time(row.get("start_s") or ""), parse_time(row.get("end_s") or "")
        except ValueError:
            problems.append(f"line {line}: bad time {row.get('start_s')!r}..{row.get('end_s')!r}")
            continue
        start = start or 0.0
        if end is not None and end <= start:
            problems.append(f"line {line}: end_s must be after start_s")
            continue
        matched[0].exclusions.append(Exclusion(start, end, (row.get("reason") or "").strip()))
    if problems:
        raise SystemExit("Exclusion file problems:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------- planning

def keep_ranges(frames: int, sr: int, exclusions: list[Exclusion]) -> list[tuple[int, int]]:
    """Half-open sample ranges left after removing the excluded ones."""
    removed = []
    for ex in exclusions:
        start = min(frames, max(0, int(round(ex.start_s * sr))))
        end = frames if ex.end_s is None else min(frames, max(0, int(round(ex.end_s * sr))))
        if end > start:
            removed.append((start, end))
    kept, pos = [], 0
    for start, end in sorted(removed):
        if start > pos:
            kept.append((pos, start))
        pos = max(pos, end)
    if pos < frames:
        kept.append((pos, frames))
    return kept


def plan_song(song_id: str, sources: list[Source], output: Path, min_part_s: float, mono: bool) -> dict:
    record = {"song_id": song_id, "sources": [], "parts": []}
    number = 0
    for source in sorted(sources, key=lambda s: s.stem):
        info = sf.info(str(source.path))
        sr, frames = int(info.samplerate), int(info.frames)
        for ex in source.exclusions:
            if ex.start_s * sr >= frames or (ex.end_s is not None and ex.end_s * sr > frames + sr):
                print(f"warning: {song_id}: exclusion {ex.start_s:.2f}-{ex.end_s} s is past the end ({frames / sr:.2f} s)")
        dropped = []
        for start, end in keep_ranges(frames, sr, source.exclusions):
            if (end - start) / sr < min_part_s:
                dropped.append({"start_sample": start, "end_sample": end, "duration_s": round((end - start) / sr, 3)})
                continue
            number += 1
            target = output / song_id / f"{song_id}__p{number:02d}.wav"
            if os.name == "nt" and len(str(target)) + 20 >= WINDOWS_MAX_PATH:
                raise SystemExit(f"Path too long for Windows ({len(str(target))}); shorten {song_id!r} or --output")
            record["parts"].append({
                "path": record_path(target),
                "source_stem": source.stem,
                "start_sample": start,
                "end_sample": end,
                "start_s": round(start / sr, 6),
                "end_s": round(end / sr, 6),
                "duration_s": round((end - start) / sr, 3),
            })
        record["sources"].append({
            "source_stem": source.stem,
            "path": record_path(source.path),
            "sha256": sha256_file(source.path),
            "sr": sr,
            "channels": int(info.channels),
            "frames": frames,
            "duration_s": round(frames / sr, 3),
            "mono": mono,
            "exclusions": [{"start_s": e.start_s, "end_s": e.end_s, "reason": e.reason} for e in source.exclusions],
            "dropped_parts": dropped,
        })
    return record


# ----------------------------------------------------------------- writing

def write_part(target: Path, source: Path, start: int, end: int, mono: bool) -> None:
    with sf.SoundFile(str(source)) as handle:
        handle.seek(start)
        data = handle.read(end - start, dtype="float32", always_2d=True)
        sr = handle.samplerate
    if mono:
        data = data.mean(axis=1)
    peak = float(np.abs(data).max()) if data.size else 0.0
    if peak > 1.0:
        print(f"warning: {target.name} peaks at {peak:.3f} (> 0 dBFS) and will clip in 24-bit PCM")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        sf.write(tmp_name, data, sr, format="WAV", subtype="PCM_24")
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def previous_parts(old: dict) -> dict[str, dict]:
    """Part path -> what produced it, from the last run's _curation.json."""
    sha = {}
    for song in old.get("songs", []):
        for src in song.get("sources", []):
            sha[(song["song_id"], src["source_stem"])] = (src.get("sha256"), src.get("mono"))
    parts = {}
    for song in old.get("songs", []):
        for part in song.get("parts", []):
            source_sha, mono = sha.get((song["song_id"], part["source_stem"]), (None, None))
            parts[part["path"]] = {**part, "source_sha256": source_sha, "mono": mono}
    return parts


def part_is_current(part: dict, source: dict, old: dict | None) -> bool:
    if old is None or not resolve_path(part["path"]).exists():
        return False
    same = all(old.get(k) == part[k] for k in ("source_stem", "start_sample", "end_sample"))
    return same and old["source_sha256"] == source["sha256"] and old["mono"] == source["mono"] and "bytes" in old and (
        resolve_path(part["path"]).stat().st_size == old["bytes"]
    )


def write_song(record: dict, old_parts: dict[str, dict]) -> tuple[int, int]:
    sources = {s["source_stem"]: s for s in record["sources"]}
    written = skipped = 0
    for part in record["parts"]:
        source = sources[part["source_stem"]]
        target = resolve_path(part["path"])
        if part_is_current(part, source, old_parts.get(part["path"])):
            skipped += 1
        else:
            write_part(target, resolve_path(source["path"]), part["start_sample"], part["end_sample"], source["mono"])
            written += 1
        part["bytes"] = target.stat().st_size
    return written, skipped


def remove_stale(output: Path, song_ids: set[str], wanted: set[Path]) -> int:
    """Delete <song_id>__pNN.wav files of managed songs that this run did not produce."""
    removed = 0
    for song_id in sorted(song_ids):
        folder = output / song_id
        if not folder.is_dir():
            continue
        pattern = re.compile(rf"{re.escape(song_id)}__p\d+\.wav")
        for path in folder.iterdir():
            if path.is_file() and pattern.fullmatch(path.name) and path.resolve() not in wanted:
                path.unlink()
                removed += 1
        if not any(folder.iterdir()):
            folder.rmdir()
    return removed


def write_names(path: Path, sources: list[Source], user_names: bool) -> None:
    rows = [{"source_stem": s.stem, "song_id": s.song_id} for s in sorted(sources, key=lambda s: s.stem)]
    if path.exists() and not user_names:
        old = {loose_key(r.get("source_stem", "")): r.get("song_id", "") for r in read_csv(path)}
        edited = [r for r in rows if old.get(loose_key(r["source_stem"]), r["song_id"]) != r["song_id"]]
        if edited:
            backup = path.with_name(path.name + ".bak")
            os.replace(path, backup)
            print(
                f"warning: {record_path(path)} had {len(edited)} name(s) different from the defaults; "
                f"saved it as {record_path(backup)}. Pass --names to use edited names."
            )
    write_csv(path, rows, NAMES_COLUMNS)


# ------------------------------------------------------------------ report

def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def print_table(records: list[dict]) -> None:
    rows = []
    for record in records:
        for src in record["sources"]:
            parts = [p for p in record["parts"] if p["source_stem"] == src["source_stem"]]
            kept = sum(p["duration_s"] for p in parts)
            rows.append([
                record["song_id"],
                f"{src['duration_s'] / 60:.2f}",
                f"{(src['duration_s'] - kept) / 60:.2f}",
                f"{kept / 60:.2f}",
                str(len(parts)),
                src["source_stem"],
            ])
    header = ["song_id", "min", "cut", "kept", "parts", "source_stem"]
    widths = [max(display_width(r[i]) for r in rows + [header]) for i in range(len(header) - 1)]
    for row in [header] + rows:
        print("  " + "  ".join(pad(cell, widths[i]) for i, cell in enumerate(row[:-1])) + "  " + row[-1])
    total = sum(r["duration_s"] for rec in records for r in rec["parts"]) / 60
    print(f"  {len(records)} song(s), {sum(len(r['parts']) for r in records)} part(s), {total:.1f} min kept")


def main() -> None:
    setup_console()
    args = parse_args()
    separated, output = resolve_path(args.separated), resolve_path(args.output)
    names_out = resolve_path(args.names_out) if args.names_out else output.parent / "vocals_names.csv"
    if not separated.is_dir():
        raise SystemExit(f"Separated folder does not exist: {separated}")
    if HELD_OUT_FOLDERS & {part.casefold() for part in separated.resolve().parts}:
        raise SystemExit(f"Refusing to curate {record_path(separated)}: reference/cover_input audio must never be trained on")
    if args.min_part_s < 0:
        raise SystemExit("--min-part-s must be >= 0.")

    sources = find_sources(separated)
    if not sources:
        raise SystemExit(f'No "{VOCALS_MARKER}" stems in: {separated}')
    names_path = resolve_path(args.names) if args.names else None
    names = read_names(names_path) if names_path else {}
    artists = args.artist
    if args.infer_artist:
        inferred = [a for a in infer_artists([s.stem for s in sources], separated.name) if a not in args.artist]
        if inferred:
            print(f"artist aliases inferred from the stems (dropped from default names): {inferred}")
        artists = args.artist + inferred
    assign_names(sources, names, artist_pattern(artists))
    exclude_path = resolve_path(args.exclude) if args.exclude else None
    if exclude_path:
        read_exclusions(exclude_path, sources)

    groups = defaultdict(list)
    for source in sources:
        groups[source.song_id].append(source)
    print(f"planning {len(sources)} stem(s) from {record_path(separated)}")
    records = [plan_song(song, groups[song], output, args.min_part_s, args.mono) for song in sorted(groups)]
    for record in records:
        if not record["parts"]:
            print(f"warning: {record['song_id']}: no part is >= {args.min_part_s} s; the song is left out")
    print_table(records)
    if args.dry_run:
        print("dry run: nothing written")
        return

    curation_path = output / CURATION_FILE
    old = json.loads(curation_path.read_text(encoding="utf-8")) if curation_path.exists() else {}
    old_parts = previous_parts(old)
    written = skipped = 0
    for record in records:
        w, s = write_song(record, old_parts)
        written, skipped = written + w, skipped + s
    wanted = {resolve_path(p["path"]).resolve() for r in records for p in r["parts"]}
    managed = set(groups) | {song["song_id"] for song in old.get("songs", [])}
    removed = remove_stale(output, managed, wanted)

    write_names(names_out, sources, names_path is not None)
    curation = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "separated": record_path(separated),
        "output": record_path(output),
        "names": {"path": record_path(names_path), "sha256": sha256_file(names_path)} if names_path else None,
        "exclude": {"path": record_path(exclude_path), "sha256": sha256_file(exclude_path)} if exclude_path else None,
        "names_out": record_path(names_out),
        "mono": args.mono,
        "min_part_s": args.min_part_s,
        "subtype": "PCM_24",
        "songs": records,
    }
    curation_path.parent.mkdir(parents=True, exist_ok=True)
    curation_path.write_text(json.dumps(curation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"done: {written} part(s) written, {skipped} unchanged, {removed} stale removed -> {record_path(output)}; "
        f"names: {record_path(names_out)}"
    )


if __name__ == "__main__":
    main()
