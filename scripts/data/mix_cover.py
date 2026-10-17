#!/usr/bin/env python3
"""Mix converted vocals with the separated instrumental into a listenable cover."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from _audio import db, load_audio, record_path, resample, resolve_path, setup_console

LIMITER_WINDOW_S = 0.005
OUTPUT_FORMATS = {
    ".wav": ("WAV", "PCM_24"),
    ".flac": ("FLAC", "PCM_24"),
    ".mp3": ("MP3", "MPEG_LAYER_III"),
    ".ogg": ("OGG", "VORBIS"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mix vocals over an instrumental, loudness-normalize, and peak-limit to a ceiling."
    )
    parser.add_argument("--vocals", required=True, help="Converted vocal file.")
    parser.add_argument("--instrumental", required=True, help="Instrumental, e.g. the *_(other)_*.wav from separation.")
    parser.add_argument("--output", required=True, help="Output .wav, .flac, .mp3 or .ogg.")
    parser.add_argument("--vocal-gain-db", type=float, default=0.0, help="Gain applied to the vocals before mixing.")
    parser.add_argument("--instrumental-gain-db", type=float, default=0.0, help="Gain applied to the instrumental.")
    parser.add_argument("--vocal-offset-ms", type=float, default=0.0, help="Delay (+) or advance (-) the vocals.")
    parser.add_argument("--match-lufs", type=float, default=-14.0, help="Target integrated loudness.")
    parser.add_argument("--no-normalize", action="store_true", help="Skip loudness normalization.")
    parser.add_argument("--ceiling-db", type=float, default=-1.0, help="Peak ceiling in dBFS.")
    return parser.parse_args()


def match_channels(samples: np.ndarray, channels: int) -> np.ndarray:
    if samples.shape[1] == channels:
        return samples
    mono = samples.mean(axis=1, keepdims=True)
    return np.repeat(mono, channels, axis=1)


def shift(samples: np.ndarray, offset: int) -> np.ndarray:
    if offset > 0:
        return np.concatenate([np.zeros((offset, samples.shape[1]), samples.dtype), samples])
    return samples[-offset:] if offset < 0 else samples


def pad_to(samples: np.ndarray, length: int) -> np.ndarray:
    return np.pad(samples, ((0, length - samples.shape[0]), (0, 0)))


def loudness(samples: np.ndarray, sr: int) -> float:
    import pyloudnorm

    try:
        return float(pyloudnorm.Meter(sr).integrated_loudness(samples))
    except ValueError:
        return float("-inf")


def limit(samples: np.ndarray, sr: int, ceiling: float) -> np.ndarray:
    """Look-ahead peak limiter: the smoothed gain never exceeds the gain each sample needs."""
    peak = np.abs(samples).max(axis=1)
    needed = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    if needed.min() >= 1.0:
        return samples
    window = max(1, int(LIMITER_WINDOW_S * sr))
    gain = uniform_filter1d(minimum_filter1d(needed, size=2 * window + 1), size=window)
    return np.clip(samples * np.minimum(gain, needed)[:, None], -ceiling, ceiling)


def write_output(path: Path, samples: np.ndarray, sr: int) -> None:
    fmt = OUTPUT_FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise SystemExit(f"Unsupported output extension: {path.suffix} (use {', '.join(OUTPUT_FORMATS)})")
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"compression_level": 0.0, "bitrate_mode": "CONSTANT"} if fmt[0] == "MP3" else {}
    sf.write(str(path), samples, sr, format=fmt[0], subtype=fmt[1], **extra)


def main() -> None:
    setup_console()
    args = parse_args()
    vocals, vocal_sr = load_audio(resolve_path(args.vocals))
    inst, sr = load_audio(resolve_path(args.instrumental))
    output = resolve_path(args.output)

    vocals = match_channels(resample(vocals, vocal_sr, sr), inst.shape[1])
    vocals = shift(vocals, int(round(args.vocal_offset_ms / 1000.0 * sr)))
    length = max(vocals.shape[0], inst.shape[0])
    if abs(vocals.shape[0] - inst.shape[0]) > sr:
        print(f"warning: lengths differ by {abs(vocals.shape[0] - inst.shape[0]) / sr:.1f}s; padding the shorter one")

    mix = pad_to(vocals, length) * 10 ** (args.vocal_gain_db / 20) + pad_to(inst, length) * 10 ** (
        args.instrumental_gain_db / 20
    )
    mix = mix.astype(np.float64)
    before = loudness(mix, sr)
    if not args.no_normalize and np.isfinite(before):
        mix *= 10 ** ((args.match_lufs - before) / 20)
    mix = limit(mix, sr, 10 ** (args.ceiling_db / 20))

    write_output(output, mix.astype(np.float32), sr)
    print(
        f"mixed {length / sr:.1f}s at {sr} Hz: loudness {before:.1f} -> {loudness(mix, sr):.1f} LUFS, "
        f"peak {float(db(np.abs(mix).max())):.2f} dBFS -> {record_path(output)}"
    )


if __name__ == "__main__":
    main()
