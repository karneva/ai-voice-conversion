#!/usr/bin/env python3
"""Split songs into vocals and instrumental with a UVR Roformer model (audio-separator)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEPARATOR = ROOT / "separator" / "env" / "Scripts" / "audio-separator.exe"
MODEL_DIR = ROOT / "separator" / "models"
DEFAULT_MODEL = "vocals_mel_band_roformer.ckpt"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".wma"}
# libsndfile (used by audio-separator to read audio) opens these directly; others are decoded first.
NATIVE_EXTENSIONS = {".wav", ".flac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate vocals and instrumental from songs.")
    parser.add_argument("input", type=Path, help="Audio file or folder (not recursive).")
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "separated", help="Output folder.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Separation model. Default: {DEFAULT_MODEL}.")
    return parser.parse_args()


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def decode_to_wav(audio_file: Path, temp_dir: Path) -> Path:
    """Decode a compressed file to 24-bit PCM WAV, keeping its sample rate and channels."""
    target = temp_dir / f"{audio_file.stem}.wav"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(audio_file), "-map", "0:a:0", "-c:a", "pcm_s24le", str(target),
    ]
    subprocess.run(command, check=True)
    return target


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # emoji file names on a cp949 console
    args = parse_args()
    if not SEPARATOR.exists():
        print(f"audio-separator not found: {SEPARATOR}", file=sys.stderr)
        return 1
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH. Open a new terminal after installing ffmpeg.", file=sys.stderr)
        return 1

    files = collect_inputs(args.input)
    if not files:
        print(f"No audio files found in: {args.input}", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    # Decode next to the output (not %TEMP%) so private audio never leaves the repo; hidden name keeps scanners away.
    with tempfile.TemporaryDirectory(prefix=".decode_", dir=args.output, ignore_cleanup_errors=True) as temp:
        temp_dir = Path(temp)
        inputs = []
        for audio_file in files:
            if audio_file.suffix.lower() in NATIVE_EXTENSIONS:
                inputs.append(audio_file)
            else:
                print(f"decode: {audio_file.name}")
                inputs.append(decode_to_wav(audio_file, temp_dir))

        command = [
            str(SEPARATOR), *map(str, inputs),
            "-m", args.model, "--model_file_dir", str(MODEL_DIR),
            "--output_dir", str(args.output), "--output_format", "WAV",
        ]
        return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
