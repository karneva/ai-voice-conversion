#!/usr/bin/env python3
"""Flag clipping, low bandwidth, silence and likely instrument bleed in manifest files."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _audio import (
    HOP_S,
    FrameAnalysis,
    analyze_frames,
    db,
    fmt,
    load_audio,
    median_f0,
    read_csv,
    record_path,
    resolve_path,
    runs,
    setup_console,
    to_mono,
    write_csv,
)

REPORT_COLUMNS = [
    "file_id", "path", "song_id", "split", "duration_s", "sr", "channels",
    "peak_dbfs", "clip_ratio", "lufs", "silence_ratio", "voiced_ratio", "voiced_s",
    "median_f0_hz", "rolloff95_hz", "bandwidth_hz", "n_suspect", "suspect_s", "flags",
]
SEGMENT_COLUMNS = ["path", "start_s", "end_s", "reason"]

CLIP_LEVEL = 0.999
SILENCE_DB = -45.0
SUSPECT_RMS_DB = -35.0
BAND_HZ = 250.0
CLIFF_DB = 22.0
CUTOFF_SEARCH_HZ = 6000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-file quality report plus a list of suspect segments to listen to."
    )
    parser.add_argument("--manifest", required=True, help="Manifest CSV (dataset/<track>/manifest.csv).")
    parser.add_argument("--report", default=None, help="Default: qc_report.csv next to the manifest.")
    parser.add_argument("--segments", default=None, help="Default: qc_segments.csv next to the manifest.")
    parser.add_argument(
        "--f0-method", choices=["dio", "harvest"], default="dio", help="pyworld F0 method (harvest: slower, ~20x)."
    )
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="Worker processes.")
    parser.add_argument("--min-suspect-s", type=float, default=1.0, help="Shortest loud-unvoiced run to flag.")
    parser.add_argument("--max-clip-ratio", type=float, default=1e-4, help="Flag clipping above this sample ratio.")
    parser.add_argument(
        "--min-bandwidth-hz", type=float, default=17000.0, help="Flag low_bandwidth for a low-pass cutoff below this."
    )
    parser.add_argument("--quiet-lufs", type=float, default=-35.0, help="Flag quiet below this integrated LUFS.")
    parser.add_argument("--max-silence-ratio", type=float, default=0.5, help="Flag mostly_silent above this.")
    parser.add_argument(
        "--min-voiced-active", type=float, default=0.5, help="Flag low_voiced when voiced/non-silent frames is below this."
    )
    return parser.parse_args()


def integrated_lufs(data: np.ndarray, sr: int) -> float:
    import pyloudnorm

    try:
        return float(pyloudnorm.Meter(sr).integrated_loudness(data if data.shape[1] > 1 else data[:, 0]))
    except ValueError:
        return float("nan")


def lowpass_cutoff(power: np.ndarray, freqs: np.ndarray, nyquist: float) -> float:
    """Frequency of a brick-wall drop in the long-term spectrum (lossy codec or upsampling), else Nyquist.

    Natural vocal spectra roll off gently. A codec low-pass leaves everything above a short transition band
    at least CLIFF_DB below the level just under the cutoff.
    """
    edges = np.arange(0.0, nyquist + BAND_HZ, BAND_HZ)
    band_db = np.array([
        10.0 * np.log10(power[(freqs >= lo) & (freqs < hi)].mean() + 1e-20) if np.any((freqs >= lo) & (freqs < hi)) else -200.0
        for lo, hi in zip(edges[:-1], edges[1:])
    ])
    for i in range(int(CUTOFF_SEARCH_HZ / BAND_HZ), len(band_db) - 2):
        if band_db[i - 2 : i].mean() - band_db[i + 2 :].max() > CLIFF_DB:
            return float(edges[i])
    return float(nyquist)


def spectral_stats(mono: np.ndarray, sr: int, loud: np.ndarray) -> tuple[float, float]:
    """Median 95% spectral rolloff and the low-pass cutoff of the non-silent frames."""
    import librosa

    hop = int(round(sr * HOP_S))
    spectrum = np.abs(librosa.stft(mono, n_fft=2048, hop_length=hop, center=True))
    n = min(spectrum.shape[1], len(loud))
    frames = spectrum[:, :n][:, loud[:n]]
    if frames.shape[1] == 0:
        return float("nan"), float("nan")
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    rolloff = librosa.feature.spectral_rolloff(S=frames, sr=sr, n_fft=2048, roll_percent=0.95)[0]
    cutoff = lowpass_cutoff(np.mean(frames**2, axis=1), freqs, sr / 2)
    return float(np.median(rolloff)), cutoff


def suspect_segments(analysis: FrameAnalysis, min_frames: int) -> list[tuple[int, int]]:
    loud_unvoiced = (analysis.rms_db > SUSPECT_RMS_DB) & ~analysis.voiced
    return [(s, e) for s, e in runs(loud_unvoiced) if e - s >= min_frames]


def clipping_segments(mono_abs_max: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Clipped regions in seconds, merging hits closer than 0.5 s."""
    hits = np.flatnonzero(mono_abs_max >= CLIP_LEVEL)
    if hits.size == 0:
        return []
    gap = int(0.5 * sr)
    breaks = np.flatnonzero(np.diff(hits) > gap)
    starts = np.concatenate([[hits[0]], hits[breaks + 1]])
    ends = np.concatenate([hits[breaks], [hits[-1]]])
    return [(s / sr, (e + 1) / sr) for s, e in zip(starts, ends)]


def check_file(row: dict[str, str], args: argparse.Namespace) -> tuple[dict[str, str], list[dict[str, str]]]:
    path = resolve_path(row["path"])
    base = {key: row.get(key, "") for key in ("file_id", "path", "song_id", "split", "duration_s", "sr", "channels")}
    try:
        data, sr = load_audio(path)
    except Exception as error:  # noqa: BLE001 - report and continue with other files
        return {**base, "flags": f"decode_error:{type(error).__name__}"}, []

    mono = to_mono(data)
    abs_max = np.abs(data).max(axis=1)
    analysis = analyze_frames(mono, sr, args.f0_method)
    silent = analysis.rms_db < SILENCE_DB
    voiced = analysis.voiced
    non_silent = max(int((~silent).sum()), 1)
    rolloff, bandwidth = spectral_stats(mono, sr, ~silent)

    clip_ratio = float((abs_max >= CLIP_LEVEL).mean())
    lufs = integrated_lufs(data, sr)
    suspects = suspect_segments(analysis, int(round(args.min_suspect_s / HOP_S)))
    clips = clipping_segments(abs_max, sr) if clip_ratio > args.max_clip_ratio else []

    flags = []
    if clip_ratio > args.max_clip_ratio:
        flags.append("clipping")
    if np.isfinite(bandwidth) and bandwidth < min(args.min_bandwidth_hz, 0.9 * sr / 2):
        flags.append("low_bandwidth")
    if np.isfinite(lufs) and lufs < args.quiet_lufs:
        flags.append("quiet")
    if silent.mean() > args.max_silence_ratio:
        flags.append("mostly_silent")
    if voiced.sum() / non_silent < args.min_voiced_active:
        flags.append("low_voiced")
    if suspects:
        flags.append("suspect_bleed")

    report = {
        **base,
        "peak_dbfs": fmt(float(db(abs_max.max())), 2),
        "clip_ratio": f"{clip_ratio:.6f}",
        "lufs": fmt(lufs, 2),
        "silence_ratio": fmt(float(silent.mean())),
        "voiced_ratio": fmt(float(voiced.mean())),
        "voiced_s": fmt(float(voiced.sum() * HOP_S), 2),
        "median_f0_hz": fmt(median_f0(analysis.f0), 1),
        "rolloff95_hz": fmt(rolloff, 0),
        "bandwidth_hz": fmt(bandwidth, 0),
        "n_suspect": str(len(suspects)),
        "suspect_s": fmt(sum(e - s for s, e in suspects) * HOP_S, 2),
        "flags": ";".join(flags),
    }
    segments = [
        {"path": row["path"], "start_s": fmt(s * HOP_S, 2), "end_s": fmt(e * HOP_S, 2), "reason": "suspect_bleed"}
        for s, e in suspects
    ] + [{"path": row["path"], "start_s": fmt(s, 2), "end_s": fmt(e, 2), "reason": "clipping"} for s, e in clips]
    segments.sort(key=lambda seg: float(seg["start_s"]))
    return report, segments


def print_summary(reports: list[dict[str, str]], segments: list[dict[str, str]]) -> None:
    total_s = sum(float(r["duration_s"] or 0) for r in reports)
    voiced_s = sum(float(r.get("voiced_s") or 0) for r in reports)
    flag_counts: dict[str, int] = {}
    for report in reports:
        for flag in filter(None, report["flags"].split(";")):
            name = flag.split(":")[0]
            flag_counts[name] = flag_counts.get(name, 0) + 1
    flagged = sum(1 for r in reports if r["flags"])
    listen_s = sum(float(s["end_s"]) - float(s["start_s"]) for s in segments)
    print(f"files: {len(reports)} ({flagged} flagged), total {total_s / 60:.1f} min, voiced {voiced_s / 60:.1f} min")
    for name, count in sorted(flag_counts.items(), key=lambda item: -item[1]):
        print(f"  {name}: {count}")
    print(f"segments to listen to: {len(segments)} ({listen_s / 60:.1f} min)")


def main() -> None:
    setup_console()
    args = parse_args()
    manifest = resolve_path(args.manifest)
    report_path = resolve_path(args.report) if args.report else manifest.with_name("qc_report.csv")
    segments_path = resolve_path(args.segments) if args.segments else manifest.with_name("qc_segments.csv")
    rows = read_csv(manifest)
    if not rows:
        raise SystemExit(f"Manifest is empty: {manifest}")

    print(f"checking {len(rows)} file(s) with {args.jobs} worker(s)")
    reports: list[dict[str, str]] = []
    segments: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for index, (report, segs) in enumerate(pool.map(check_file, rows, [args] * len(rows), chunksize=4), 1):
            reports.append(report)
            segments.extend(segs)
            if index % 50 == 0:
                print(f"  {index}/{len(rows)}")

    write_csv(report_path, reports, REPORT_COLUMNS)
    write_csv(segments_path, segments, SEGMENT_COLUMNS)
    print_summary(reports, segments)
    print(f"report: {record_path(report_path)}")
    print(f"segments: {record_path(segments_path)}")


if __name__ == "__main__":
    main()
