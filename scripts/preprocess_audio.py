#!/usr/bin/env python3
"""Convert audio files to mono 40 kHz WAV segments for RVC training."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


DEFAULT_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert audio files to mono WAV and slice them into fixed-length segments."
    )
    parser.add_argument("--input", required=True, type=Path, help="Directory containing raw audio files.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for processed WAV segments.")
    parser.add_argument("--sample-rate", default=40000, type=int, help="Output sample rate. Default: 40000.")
    parser.add_argument(
        "--segment-seconds",
        default=8.0,
        type=float,
        help="Segment length in seconds. Default: 8.0.",
    )
    parser.add_argument(
        "--min-segment-seconds",
        default=3.0,
        type=float,
        help="Drop tail segments shorter than this value. Default: 3.0.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Input file extensions to include.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing segment files.")
    return parser.parse_args()


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required binary not found on PATH: {name}")


def normalized_extensions(extensions: list[str]) -> set[str]:
    return {(ext if ext.startswith(".") else f".{ext}").lower() for ext in extensions}


def iter_audio_files(input_dir: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def ffprobe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def segment_count(duration: float, segment_seconds: float, min_segment_seconds: float) -> int:
    full_segments = int(duration // segment_seconds)
    remainder = duration - (full_segments * segment_seconds)
    return full_segments + int(remainder >= min_segment_seconds)


def output_stem(input_dir: Path, audio_file: Path) -> str:
    relative = audio_file.relative_to(input_dir).with_suffix("")
    return "__".join(relative.parts)


def run_ffmpeg(
    audio_file: Path,
    output_pattern: Path,
    sample_rate: int,
    segment_seconds: float,
    overwrite: bool,
) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(audio_file),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-reset_timestamps",
        "1",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    require_binary("ffmpeg")
    require_binary("ffprobe")

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if args.segment_seconds <= 0:
        raise SystemExit("--segment-seconds must be greater than 0.")
    if args.min_segment_seconds < 0:
        raise SystemExit("--min-segment-seconds must be 0 or greater.")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = iter_audio_files(input_dir, normalized_extensions(args.extensions))
    if not files:
        raise SystemExit(f"No audio files found in: {input_dir}")

    total_written = 0
    for audio_file in files:
        duration = ffprobe_duration(audio_file)
        expected = segment_count(duration, args.segment_seconds, args.min_segment_seconds)
        if expected == 0:
            print(f"skip short file: {audio_file}")
            continue

        stem = output_stem(input_dir, audio_file)
        output_pattern = output_dir / f"{stem}_%03d.wav"

        if not args.overwrite and any(output_dir.glob(f"{stem}_*.wav")):
            print(f"skip existing segments: {audio_file}")
            continue

        run_ffmpeg(audio_file, output_pattern, args.sample_rate, args.segment_seconds, args.overwrite)

        generated = sorted(output_dir.glob(f"{stem}_*.wav"))
        for tail in generated[expected:]:
            tail.unlink()

        written = min(len(generated), expected)
        total_written += written
        print(f"{audio_file} -> {written} segment(s)")

    print(f"done: {total_written} segment(s) written to {output_dir}")


if __name__ == "__main__":
    main()
