#!/usr/bin/env python3
"""Build the frozen cross-conversion source set from other singers' vocals."""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _audio import (
    CLIP_COLUMNS,
    HOP_S,
    FrameAnalysis,
    analyze_frames,
    annotated_lyrics,
    iter_audio_files,
    load_audio,
    median_f0,
    read_csv,
    record_path,
    resolve_path,
    setup_console,
    slugify,
    song_name,
    to_mono,
    trim_inactive,
    write_csv,
    write_wav,
)

STEP_S = 0.25
WINDOWS_PER_FILE = 2
TARGET_F0_METHOD = "dio"  # pooled median over many files; dio is ~20x faster than harvest and close enough
# Same song under different titles (song_id slugs). GTSinger: the sopranos' 눈, 코, 입 is the tenor's EYES, NOSE, LIPS.
SONG_ALIASES = {"눈_코_입": "EYES_NOSE_LIPS"}


@dataclass
class Candidate:
    singer: str
    song_id: str
    path: str
    start: int
    end: int
    voiced_frames: int
    median_f0: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick the most voiced windows per singer (spread over songs) and write source_eval clips."
    )
    parser.add_argument("--input", required=True, help="Folder of other singers' vocals, or a manifest CSV.")
    parser.add_argument("--track", required=True, choices=["private", "public"])
    parser.add_argument(
        "--singer-from",
        choices=["parent", "arg"],
        default="parent",
        help="parent: top-level folder under --input is the singer (manifest: its singer column). arg: use --singer.",
    )
    parser.add_argument("--singer", default=None, help="Singer name for --singer-from arg.")
    parser.add_argument("--only-singers", nargs="+", default=None, help="Keep only these singers.")
    parser.add_argument(
        "--song-level",
        type=int,
        default=None,
        help="Folder level (0-based, under the singer folder for 'parent') used as song_id. Default: file stem.",
    )
    parser.add_argument("--glob", default="**/*", help='Glob under --input. Default: "**/*".')
    parser.add_argument("--exclude", action="append", default=[], help="fnmatch pattern on the relative path to skip.")
    parser.add_argument("--clips-per-singer", type=int, default=4)
    parser.add_argument("--clip-len", type=float, default=10.0, help="Window length (s).")
    parser.add_argument("--min-clip-len", type=float, default=5.0, help="Shortest acceptable clip (s).")
    parser.add_argument("--min-voiced", type=float, default=0.6, help="Minimum voiced-frame ratio of a clip.")
    parser.add_argument("--max-files-per-singer", type=int, default=0, help="Seeded subsample before analysis (0 = all).")
    parser.add_argument("--target-manifest", default=None, help="Target singer manifest; sets transpose from median F0.")
    parser.add_argument("--target-f0-hz", type=float, default=None, help="Use this target median F0 instead.")
    parser.add_argument(
        "--allow-target-songs",
        action="store_true",
        help="Keep source songs whose song_id also appears in --target-manifest (skipped by default).",
    )
    parser.add_argument(
        "--exclude-songs",
        nargs="+",
        default=[],
        help="Extra source song_ids to skip, e.g. a target song under another title (built in: "
        + ", ".join(f"{k}={v}" for k, v in SONG_ALIASES.items()) + ").",
    )
    parser.add_argument("--output-dir", default=None, help="Default: dataset/<track>/source_eval.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--f0-method", choices=["harvest", "dio"], default="harvest")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="Replace an existing clips.csv (the set should stay frozen).")
    return parser.parse_args()


# ------------------------------------------------------------------- inputs

def rows_from_dir(input_dir: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    rows = []
    for path in iter_audio_files(input_dir, args.glob, args.exclude):
        relative = path.relative_to(input_dir)
        if args.singer_from == "parent":
            if len(relative.parts) < 2:
                continue
            singer, inner = relative.parts[0], Path(*relative.parts[1:])
        else:
            singer, inner = args.singer, relative
        rows.append({"path": record_path(path), "singer": singer, "song_id": slugify(song_name(inner, args.song_level))})
    return rows


def load_inputs(args: argparse.Namespace) -> list[dict[str, str]]:
    source = resolve_path(args.input)
    if source.is_file() and source.suffix.lower() == ".csv":
        rows = [{"path": r["path"], "singer": r["singer"], "song_id": r["song_id"]} for r in read_csv(source)]
        if args.singer_from == "arg":
            rows = [{**row, "singer": args.singer} for row in rows]
    elif source.is_dir():
        rows = rows_from_dir(source, args)
    else:
        raise SystemExit(f"--input is neither a folder nor a CSV: {source}")
    if args.only_singers:
        rows = [row for row in rows if row["singer"] in set(args.only_singers)]
    if not rows:
        raise SystemExit("No input files found.")
    return rows


def song_key(song_id: str) -> str:
    return SONG_ALIASES.get(song_id, song_id).casefold()


def drop_target_songs(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    """Skip songs the target singer also sings, so cross-conversion never replays a trained melody.

    Titles are compared after SONG_ALIASES and case folding; --exclude-songs covers other aliases.
    """
    excluded = {song_key(song) for song in args.exclude_songs}
    target_songs: list[str] = []
    if args.target_manifest and not args.allow_target_songs:
        target_songs = sorted({row["song_id"] for row in read_csv(resolve_path(args.target_manifest))})
        excluded |= {song_key(song) for song in target_songs}
    kept = [row for row in rows if song_key(row["song_id"]) not in excluded]
    dropped = sorted({row["song_id"] for row in rows} - {row["song_id"] for row in kept})
    if dropped:
        print(f"skipping {len(rows) - len(kept)} file(s) of {len(dropped)} song(s) shared with the target "
              f"or excluded: {dropped}")
    if target_songs:
        print(f"target songs: {target_songs}")
        print(f"kept source songs (check for the same song under another title): "
              f"{sorted({row['song_id'] for row in kept})}")
    return kept


def subsample(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    if limit <= 0:
        return rows
    by_singer: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_singer[row["singer"]].append(row)
    rng = random.Random(seed)
    kept = []
    for singer in sorted(by_singer):
        group = sorted(by_singer[singer], key=lambda r: r["path"])
        kept.extend(group if len(group) <= limit else rng.sample(group, limit))
    return kept


# ----------------------------------------------------------------- windows

def best_windows(analysis: FrameAnalysis, args: argparse.Namespace) -> list[tuple[int, int, int]]:
    """Up to WINDOWS_PER_FILE non-overlapping (start, end, voiced_frames) windows, most voiced first."""
    active = np.flatnonzero(analysis.active)
    if active.size == 0:
        return []
    first, last = int(active[0]), int(active[-1]) + 1
    length, min_length = analysis.frame(args.clip_len), analysis.frame(args.min_clip_len)
    step = analysis.frame(STEP_S)
    starts = [first] if last - first <= length else list(range(first, last - length + 1, step))
    voiced_cum = np.concatenate([[0], np.cumsum(analysis.voiced)])

    scored = []
    for start in starts:
        s, e = trim_inactive(analysis, start, min(start + length, last))
        if e - s < min_length:
            continue
        voiced = int(voiced_cum[e] - voiced_cum[s])
        if voiced / (e - s) >= args.min_voiced:
            scored.append((s, e, voiced))
    scored.sort(key=lambda item: (-item[2], item[0]))

    chosen: list[tuple[int, int, int]] = []
    for s, e, voiced in scored:
        if all(e <= cs or s >= ce for cs, ce, _ in chosen):
            chosen.append((s, e, voiced))
        if len(chosen) == WINDOWS_PER_FILE:
            break
    return chosen


def analyze_row(row: dict[str, str], args: argparse.Namespace) -> list[Candidate]:
    data, sr = load_audio(resolve_path(row["path"]))
    analysis = analyze_frames(to_mono(data), sr, args.f0_method)
    return [
        Candidate(row["singer"], row["song_id"], row["path"], s, e, voiced, median_f0(analysis.f0[s:e]))
        for s, e, voiced in best_windows(analysis, args)
    ]


def file_f0(path: str, f0_method: str) -> np.ndarray:
    data, sr = load_audio(resolve_path(path))
    f0 = analyze_frames(to_mono(data), sr, f0_method).f0
    return f0[f0 > 0]


def select(candidates: list[Candidate], per_singer: int, seed: int) -> list[Candidate]:
    """Round-robin over songs (best song first) so each singer's clips come from different songs."""
    by_singer: dict[str, dict[str, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for cand in candidates:
        by_singer[cand.singer][cand.song_id].append(cand)

    rng = random.Random(seed)
    picked = []
    for singer in sorted(by_singer):
        songs = by_singer[singer]
        for group in songs.values():
            group.sort(key=lambda c: (-c.voiced_frames, c.path, c.start))
        order = sorted(songs)
        rng.shuffle(order)
        order.sort(key=lambda song: -songs[song][0].voiced_frames)
        chosen: list[Candidate] = []
        for round_index in range(max(len(g) for g in songs.values())):
            for song in order:
                if len(chosen) < per_singer and round_index < len(songs[song]):
                    chosen.append(songs[song][round_index])
        if len(chosen) < per_singer:
            print(f"warning: only {len(chosen)} clip(s) available for {singer}")
        picked.extend(chosen)
    return picked


# ---------------------------------------------------------------- transpose

def target_median_f0(args: argparse.Namespace, pool: ProcessPoolExecutor) -> float | None:
    if args.target_f0_hz:
        return args.target_f0_hz
    if not args.target_manifest:
        return None
    rows = read_csv(resolve_path(args.target_manifest))
    train = [row for row in rows if row.get("split") == "train"]
    if not train:
        print("warning: target manifest has no split=train rows; using all target files")
        train = rows
    print(f"estimating target median F0 from {len(train)} file(s)")
    voiced = list(pool.map(file_f0, [r["path"] for r in train], [TARGET_F0_METHOD] * len(train), chunksize=4))
    pooled = np.concatenate(voiced) if voiced else np.array([])
    if pooled.size == 0:
        raise SystemExit("No voiced frames found in the target files.")
    return float(np.median(pooled))


def transpose_for(target_f0: float | None, clip_f0: float) -> int:
    if target_f0 is None or not math.isfinite(clip_f0) or clip_f0 <= 0:
        return 0
    return int(max(-12, min(12, round(12.0 * math.log2(target_f0 / clip_f0)))))


# ------------------------------------------------------------------- output

def write_clips(picked: list[Candidate], output_dir: Path, target_f0: float | None) -> list[dict[str, str]]:
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for stale in clips_dir.glob("*.wav"):
        stale.unlink()

    counters: dict[str, int] = defaultdict(int)
    rows = []
    for cand in sorted(picked, key=lambda c: (c.singer, c.song_id, c.path, c.start)):
        counters[cand.singer] += 1
        clip_id = f"{slugify(cand.singer)}_{counters[cand.singer]:02d}_{slugify(cand.song_id, 24)}"
        data, sr = load_audio(resolve_path(cand.path))
        start_s, end_s = cand.start * HOP_S, cand.end * HOP_S
        target = clips_dir / f"{clip_id}.wav"
        write_wav(target, to_mono(data)[int(round(start_s * sr)) : int(round(end_s * sr))], sr)
        transpose = transpose_for(target_f0, cand.median_f0)
        rows.append({
            "clip_id": clip_id,
            "path": record_path(target),
            "singer": cand.singer,
            "song_id": cand.song_id,
            "start_s": f"{start_s:.2f}",
            "end_s": f"{end_s:.2f}",
            "source_path": cand.path,
            "transpose": str(transpose),
            "lyrics": annotated_lyrics(resolve_path(cand.path), start_s, end_s),
        })
        voiced_ratio = cand.voiced_frames / max(cand.end - cand.start, 1)
        print(
            f"  {clip_id}: {end_s - start_s:.1f}s voiced {voiced_ratio:.0%} "
            f"f0 {cand.median_f0:.0f} Hz transpose {transpose:+d}"
        )
    return rows


def main() -> None:
    setup_console()
    args = parse_args()
    if args.singer_from == "arg" and not args.singer:
        raise SystemExit("--singer-from arg needs --singer.")
    if not 0 < args.min_clip_len <= args.clip_len:
        raise SystemExit("Expected 0 < --min-clip-len <= --clip-len.")

    output_dir = resolve_path(args.output_dir or f"dataset/{args.track}/source_eval")
    clips_csv = output_dir / "clips.csv"
    if clips_csv.exists() and not args.force:
        raise SystemExit(f"{record_path(clips_csv)} exists; the source set is frozen. Pass --force to rebuild it.")

    rows = subsample(drop_target_songs(load_inputs(args), args), args.max_files_per_singer, args.seed)
    singers = sorted({row["singer"] for row in rows})
    print(f"analyzing {len(rows)} file(s) from {len(singers)} singer(s): {singers}")

    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        candidates = [c for group in pool.map(analyze_row, rows, [args] * len(rows), chunksize=4) for c in group]
        picked = select(candidates, args.clips_per_singer, args.seed)
        target_f0 = target_median_f0(args, pool)

    if target_f0:
        print(f"target median F0: {target_f0:.1f} Hz")
    rows_out = write_clips(picked, output_dir, target_f0)
    write_csv(clips_csv, rows_out, CLIP_COLUMNS)
    minutes = sum(float(r["end_s"]) - float(r["start_s"]) for r in rows_out) / 60.0
    print(f"done: {len(rows_out)} clip(s), {minutes:.1f} min -> {record_path(clips_csv)}")


if __name__ == "__main__":
    main()
