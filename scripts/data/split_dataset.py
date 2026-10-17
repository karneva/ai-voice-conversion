#!/usr/bin/env python3
"""Song-level train/held-out split, plus optional train copy and held-out clip cutting."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from _audio import (
    APPLIO_EXTENSIONS,
    AUDIO_EXTENSIONS,
    CLIP_COLUMNS,
    HOP_S,
    MANIFEST_COLUMNS,
    FrameAnalysis,
    analyze_frames,
    annotated_lyrics,
    load_audio,
    read_csv,
    record_path,
    resolve_path,
    setup_console,
    to_mono,
    trim_inactive,
    write_csv,
    write_wav,
)

TRAIN_INDEX_COLUMNS = ["file_id", "path", "source_path", "song_id", "sha256"]
SPLIT_TRIES = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign split=train/heldout per song (never splitting a song) and write it back to the manifest."
    )
    parser.add_argument("--manifest", required=True, help="Manifest CSV (dataset/<track>/manifest.csv).")
    parser.add_argument("--heldout-fraction", type=float, default=0.12, help="Target held-out share of total duration.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-heldout-songs", type=int, default=2)
    parser.add_argument(
        "--resplit",
        action="store_true",
        help="Ignore existing split values. Default keeps them and sends new songs to train (held-out stays frozen).",
    )
    parser.add_argument("--materialize", action="store_true", help="Copy train files and cut held-out clips.")
    parser.add_argument("--clip-min", type=float, default=5.0, help="Preferred minimum clip length (s).")
    parser.add_argument("--clip-max", type=float, default=10.0, help="Maximum clip length (s).")
    parser.add_argument("--clip-floor", type=float, default=3.0, help="Drop clips shorter than this (s).")
    parser.add_argument("--min-voiced", type=float, default=0.6, help="Drop clips with fewer voiced frames than this.")
    parser.add_argument(
        "--max-clips", type=int, default=0, help="Keep at most N clips, spread evenly over held-out songs (0 = all)."
    )
    parser.add_argument("--f0-method", choices=["harvest", "dio"], default="harvest")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="Worker processes for clip cutting.")
    parser.add_argument("--train-dir", default=None, help="Default: train/ next to the manifest.")
    parser.add_argument("--heldout-dir", default=None, help="Default: heldout/ next to the manifest.")
    return parser.parse_args()


# ------------------------------------------------------------------ splitting

def song_durations(rows: list[dict[str, str]]) -> dict[str, float]:
    durations: dict[str, float] = defaultdict(float)
    for row in rows:
        durations[row["song_id"]] += float(row["duration_s"])
    return dict(durations)


def existing_splits(rows: list[dict[str, str]]) -> dict[str, str]:
    splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("split"):
            splits[row["song_id"]].add(row["split"])
    mixed = [song for song, values in splits.items() if len(values) > 1]
    if mixed:
        raise SystemExit(f"Songs with mixed split values (fix the manifest or pass --resplit): {mixed}")
    return {song: values.pop() for song, values in splits.items()}


def choose_heldout(durations: dict[str, float], fraction: float, min_songs: int, seed: int) -> set[str]:
    """Best of many seeded shuffles: the song prefix (>= min_songs) whose duration is closest to the target."""
    songs = sorted(durations)
    if len(songs) <= min_songs:
        raise SystemExit(f"Need more than {min_songs} songs to split, found {len(songs)}.")
    target = fraction * sum(durations.values())
    rng = random.Random(seed)
    best: list[str] = []
    best_error = float("inf")
    for _ in range(SPLIT_TRIES):
        order = songs[:]
        rng.shuffle(order)
        total = 0.0
        for count, song in enumerate(order[:-1], 1):
            total += durations[song]
            if count >= min_songs and abs(total - target) < best_error:
                best, best_error = order[:count], abs(total - target)
            if total >= target and count >= min_songs:
                break
    return set(best)


def assign_splits(rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    durations = song_durations(rows)
    previous = {} if args.resplit else existing_splits(rows)
    if previous:
        new_songs = sorted(set(durations) - set(previous))
        if new_songs:
            print(f"keeping existing splits; {len(new_songs)} new song(s) go to train: {new_songs}")
        splits = {song: previous.get(song, "train") for song in durations}
    else:
        heldout = choose_heldout(durations, args.heldout_fraction, args.min_heldout_songs, args.seed)
        splits = {song: "heldout" if song in heldout else "train" for song in durations}
    for row in rows:
        row["split"] = splits[row["song_id"]]


# ---------------------------------------------------------------- train copy

def materialize_train(rows: list[dict[str, str]], train_dir: Path) -> None:
    index_path = train_dir / "_index.csv"
    old_index = read_csv(index_path) if index_path.exists() else []
    train_dir.mkdir(parents=True, exist_ok=True)

    new_index = []
    for row in (r for r in rows if r["split"] == "train"):
        source = resolve_path(row["path"])
        suffix = source.suffix.lower() if source.suffix.lower() in APPLIO_EXTENSIONS else ".wav"
        target = train_dir / f"{row['file_id']}{suffix}"
        new_index.append({
            "file_id": row["file_id"],
            "path": record_path(target),
            "source_path": row["path"],
            "song_id": row["song_id"],
            "sha256": row["sha256"],
        })

    # Applio trains on every audio file in the folder, so anything not listed for train must go,
    # including files the old index never recorded (interrupted copy, files added by hand).
    wanted = {resolve_path(entry["path"]).name for entry in new_index}
    removed = 0
    for path in train_dir.iterdir():
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS and path.name not in wanted:
            path.unlink()
            removed += 1

    previous = {entry["path"]: entry["sha256"] for entry in old_index}
    copied = 0
    for entry in new_index:
        target = resolve_path(entry["path"])
        if target.exists() and previous.get(entry["path"]) == entry["sha256"]:
            continue
        source = resolve_path(entry["source_path"])
        if source.suffix.lower() in APPLIO_EXTENSIONS:
            shutil.copy2(source, target)
        else:
            data, sr = load_audio(source)
            write_wav(target, data, sr)
        copied += 1

    write_csv(index_path, new_index, TRAIN_INDEX_COLUMNS)
    print(f"train: {len(new_index)} file(s) in {record_path(train_dir)} ({copied} copied, {removed} stale removed)")


# -------------------------------------------------------------- held-out clips

def plan_clips(analysis: FrameAnalysis, args: argparse.Namespace) -> list[tuple[int, int]]:
    """Cut the active part of a file into consecutive clips, cutting at the quietest frame."""
    active = np.flatnonzero(analysis.active)
    if active.size == 0:
        return []
    f_min, f_max = analysis.frame(args.clip_min), analysis.frame(args.clip_max)
    f_floor = analysis.frame(args.clip_floor)
    pos, last = int(active[0]), int(active[-1]) + 1
    clips = []
    while pos < last:
        remaining = last - pos
        if remaining <= f_max:
            cut = last
        else:
            low = pos + f_min
            high = pos + (min(f_max, remaining - f_min) if remaining < f_max + f_min else f_max)
            high = max(high, low + 1)
            cut = low + int(np.argmin(analysis.rms_db[low:high]))
        start, end = trim_inactive(analysis, pos, cut)
        if end - start >= f_floor and analysis.voiced_fraction(start, end) >= args.min_voiced:
            clips.append((start, end))
        following = active[active >= cut]
        if following.size == 0:
            break
        pos = int(following[0])
    return clips


def plan_file(row: dict[str, str], args: argparse.Namespace) -> list[tuple[int, int]]:
    data, sr = load_audio(resolve_path(row["path"]))
    return plan_clips(analyze_frames(to_mono(data), sr, args.f0_method), args)


def cap_clips(planned: list, max_clips: int, seed: int) -> list:
    """Seeded round-robin over songs so a capped clip set still covers every held-out song."""
    if max_clips <= 0 or len(planned) <= max_clips:
        return planned
    by_song: dict[str, list] = defaultdict(list)
    for item in planned:
        by_song[item[0]["song_id"]].append(item)
    rng = random.Random(seed)
    for song in sorted(by_song):
        rng.shuffle(by_song[song])
    picked = []
    while len(picked) < max_clips:
        for song in sorted(by_song):
            if by_song[song] and len(picked) < max_clips:
                picked.append(by_song[song].pop())
    order = {id(item): index for index, item in enumerate(planned)}
    return sorted(picked, key=lambda item: order[id(item)])


def materialize_heldout(rows: list[dict[str, str]], heldout_dir: Path, args: argparse.Namespace) -> list[dict]:
    clips_dir = heldout_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for stale in clips_dir.glob("*.wav"):
        stale.unlink()

    heldout = [r for r in rows if r["split"] == "heldout"]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        plans = list(pool.map(plan_file, heldout, [args] * len(heldout)))

    planned = [(row, number, span) for row, plan in zip(heldout, plans) for number, span in enumerate(plan)]
    planned = cap_clips(planned, args.max_clips, args.seed)

    clip_rows = []
    cache: dict[str, tuple[np.ndarray, int]] = {}
    for row, number, (start, end) in planned:
        if row["path"] not in cache:
            data, sr = load_audio(resolve_path(row["path"]))
            cache.clear()
            cache[row["path"]] = (to_mono(data), sr)
        mono, sr = cache[row["path"]]
        start_s, end_s = start * HOP_S, end * HOP_S
        clip_id = f"{row['file_id']}_{number:02d}"
        target = clips_dir / f"{clip_id}.wav"
        write_wav(target, mono[int(round(start_s * sr)) : int(round(end_s * sr))], sr)
        clip_rows.append({
            "clip_id": clip_id,
            "path": record_path(target),
            "singer": row["singer"],
            "song_id": row["song_id"],
            "start_s": f"{start_s:.2f}",
            "end_s": f"{end_s:.2f}",
            "source_path": row["path"],
            "transpose": "0",
            "lyrics": annotated_lyrics(resolve_path(row["path"]), start_s, end_s),
        })
    write_csv(heldout_dir / "clips.csv", clip_rows, CLIP_COLUMNS)
    return clip_rows


def print_stats(rows: list[dict[str, str]], clip_rows: list[dict] | None) -> None:
    minutes = defaultdict(float)
    songs = defaultdict(set)
    for row in rows:
        minutes[row["split"]] += float(row["duration_s"]) / 60.0
        songs[row["split"]].add(row["song_id"])
    total = sum(minutes.values())
    for split in ("train", "heldout"):
        share = 100.0 * minutes[split] / total if total else 0.0
        print(f"{split}: {len(songs[split])} song(s), {minutes[split]:.1f} min ({share:.1f}%)")
    print(f"heldout songs: {sorted(songs['heldout'])}")
    if clip_rows is not None:
        clip_minutes = sum(float(c["end_s"]) - float(c["start_s"]) for c in clip_rows) / 60.0
        print(f"heldout clips: {len(clip_rows)} ({clip_minutes:.1f} min)")


def main() -> None:
    setup_console()
    args = parse_args()
    if not 0.0 < args.heldout_fraction < 1.0:
        raise SystemExit("--heldout-fraction must be between 0 and 1.")
    if not 0.0 < args.clip_floor <= args.clip_min <= args.clip_max:
        raise SystemExit("Expected 0 < --clip-floor <= --clip-min <= --clip-max.")

    manifest = resolve_path(args.manifest)
    rows = read_csv(manifest)
    if not rows:
        raise SystemExit(f"Manifest is empty: {manifest}")
    assign_splits(rows, args)
    write_csv(manifest, rows, MANIFEST_COLUMNS)
    print(f"split written to {record_path(manifest)}")

    clip_rows = None
    if args.materialize:
        materialize_train(rows, resolve_path(args.train_dir) if args.train_dir else manifest.with_name("train"))
        heldout_dir = resolve_path(args.heldout_dir) if args.heldout_dir else manifest.with_name("heldout")
        clip_rows = materialize_heldout(rows, heldout_dir, args)
        print(f"clips: {record_path(heldout_dir / 'clips.csv')}")
    print_stats(rows, clip_rows)


if __name__ == "__main__":
    main()
