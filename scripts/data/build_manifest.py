#!/usr/bin/env python3
"""Build or extend dataset/<track>/manifest.csv from a folder of vocal files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _audio import (
    MANIFEST_COLUMNS,
    audio_info,
    iter_audio_files,
    read_csv,
    record_path,
    resolve_path,
    setup_console,
    sha256_file,
    short_hash,
    slugify,
    song_name,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk audio files (wav/flac/mp3/m4a/ogg) and write manifest rows (one per file)."
    )
    parser.add_argument("--input", required=True, help="Folder with vocal files (repo-relative or absolute).")
    parser.add_argument("--track", required=True, choices=["private", "public"])
    parser.add_argument("--singer", required=True, help="Singer name written to every row.")
    parser.add_argument(
        "--song-level",
        type=int,
        default=None,
        help="Use the Nth path component under --input (0-based) as song_id. "
        "Default: file stem (one file = one song). GTSinger singer folder: 1.",
    )
    parser.add_argument("--glob", default="**/*", help='Glob under --input. Default: "**/*".')
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help='fnmatch pattern on the path relative to --input to skip (repeatable), e.g. "*/Paired_Speech_Group/*".',
    )
    parser.add_argument("--output", default=None, help="Manifest path. Default: dataset/<track>/manifest.csv.")
    parser.add_argument("--append", action="store_true", help="Merge into an existing manifest, skipping known paths.")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep files whose content (sha256) repeats an earlier row. Default: keep only the first copy.",
    )
    return parser.parse_args()


def describe_file(path: Path, input_dir: Path, singer: str, song_level: int | None) -> dict[str, str]:
    relative = path.relative_to(input_dir)
    repo_path = record_path(path)
    song_id = slugify(song_name(relative, song_level))
    duration, sr, channels = audio_info(path)
    return {
        "file_id": f"{song_id}_{short_hash(repo_path)}",
        "path": repo_path,
        "song_id": song_id,
        "singer": singer,
        "duration_s": f"{duration:.3f}",
        "sr": str(sr),
        "channels": str(channels),
        "sha256": sha256_file(path),
        "split": "",
    }


def merge_rows(existing: list[dict[str, str]], new: list[dict[str, str]], append: bool) -> list[dict[str, str]]:
    """Append mode keeps existing rows; rebuild mode keeps a file's split if its content is unchanged."""
    previous = {row["path"]: row for row in existing}
    if append:
        return existing + [row for row in new if row["path"] not in previous]
    for row in new:
        old = previous.get(row["path"])
        if old and old.get("sha256") == row["sha256"]:
            row["split"] = old.get("split", "")
    return new


def drop_duplicates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the first row per sha256 (existing rows come first, then by path)."""
    seen: dict[str, str] = {}
    kept = []
    for row in rows:
        if row["sha256"] in seen:
            continue
        seen[row["sha256"]] = row["path"]
        kept.append(row)
    dropped = len(rows) - len(kept)
    if dropped:
        print(f"skipped {dropped} file(s) with content identical to another row (pass --keep-duplicates to keep them)")
    return kept


def check_ids(rows: list[dict[str, str]]) -> None:
    ids = Counter(row["file_id"] for row in rows)
    clashes = [file_id for file_id, count in ids.items() if count > 1]
    if clashes:
        raise SystemExit(f"Duplicate file_id values: {clashes[:5]}")


def main() -> None:
    setup_console()
    args = parse_args()
    input_dir = resolve_path(args.input)
    output = resolve_path(args.output or f"dataset/{args.track}/manifest.csv")

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    files = iter_audio_files(input_dir, args.glob, args.exclude)
    if not files:
        raise SystemExit(f"No audio files found in: {input_dir}")

    print(f"scanning {len(files)} file(s) in {record_path(input_dir)}")
    new_rows = []
    for index, path in enumerate(files, 1):
        new_rows.append(describe_file(path, input_dir, args.singer, args.song_level))
        if index % 100 == 0:
            print(f"  {index}/{len(files)}")

    existing = read_csv(output) if output.exists() else []
    if existing and not args.append:
        print(f"rebuilding existing manifest (splits kept for unchanged files): {record_path(output)}")
    rows = merge_rows(existing, sorted(new_rows, key=lambda row: row["path"]), args.append)
    if not args.keep_duplicates:
        rows = drop_duplicates(rows)
    rows.sort(key=lambda row: row["path"])
    check_ids(rows)
    write_csv(output, rows, MANIFEST_COLUMNS)

    songs = {row["song_id"] for row in rows}
    minutes = sum(float(row["duration_s"]) for row in rows) / 60.0
    print(
        f"done: manifest has {len(rows)} file(s) ({len(rows) - len(existing):+d} vs before), "
        f"{len(songs)} song(s), {minutes:.1f} min -> {record_path(output)}"
    )


if __name__ == "__main__":
    main()
