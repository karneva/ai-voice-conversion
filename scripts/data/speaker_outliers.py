#!/usr/bin/env python3
"""Flag time ranges whose voice is unlikely to be the target singer (duet partners, guests, MC talk)."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from _audio import (
    ANALYSIS_SR,
    HOP_S,
    REPO_ROOT,
    analyze_frames,
    fmt,
    iter_audio_files,
    load_audio,
    read_csv,
    record_path,
    resample,
    resolve_path,
    setup_console,
    short_hash,
    slugify,
    to_mono,
    write_csv,
)

OUTLIER_COLUMNS = ["path", "start_s", "end_s", "min_cos", "mean_cos", "n_windows"]
SUMMARY_COLUMNS = [
    "path", "duration_min", "voiced_min", "n_windows", "median_cos", "p10_cos",
    "n_segments", "flagged_s", "flagged_pct",
]
WINDOW_COLUMNS = ["path", "start_s", "end_s", "voiced_frac", "cos", "score", "flagged"]

MAD_SCALE = 1.4826  # MAD -> standard deviation for normal data
FILE_MEDIAN_K = 2.0  # file-level hint: file median below median - k * MAD of all file medians ...
FILE_MEDIAN_MARGIN = 0.1  # ... and at least this far below their median
FILE_MIN_WINDOWS = 30  # files with fewer scored windows are too noisy for the hint
MODELS_DIR = REPO_ROOT / "results" / "cache" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed 3 s voiced windows with ECAPA, compare each to a target-singer centroid, and list the time "
            "ranges that score unusually low so only those need listening."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Folder of vocal files (see --glob).")
    source.add_argument("--manifest", help="Manifest CSV; every row's path is scanned.")
    parser.add_argument("--glob", default="*_(vocals)_*.wav", help="File pattern under --input (default: %(default)s).")
    parser.add_argument(
        "--reference", nargs="+", default=None,
        help="Known-pure target vocals (files or folders) for the centroid. Default: robust centroid of all inputs.",
    )
    parser.add_argument("--reference-glob", default="**/*", help="File pattern for --reference folders.")
    parser.add_argument("--output", required=True, help="Segment CSV, e.g. dataset/<track>/speaker_outliers.csv.")
    parser.add_argument("--summary", default=None, help="Per-file CSV. Default: <output stem>_summary.csv.")
    parser.add_argument("--windows", default=None, help="Optional per-window CSV (all scored windows).")
    parser.add_argument("--plot", action="store_true", help="Save a cosine-over-time PNG per file in <output stem>_plots/.")
    parser.add_argument("--window-s", type=float, default=3.0, help="Embedding window length.")
    parser.add_argument("--hop-s", type=float, default=1.0, help="Window hop.")
    parser.add_argument("--min-voiced", type=float, default=0.5, help="Skip windows with fewer voiced frames than this.")
    parser.add_argument("--f0-method", choices=["dio", "harvest"], default="dio", help="pyworld voicing detector.")
    parser.add_argument(
        "--smooth", type=int, default=3,
        help="Score = mean cosine of this many neighbouring voiced windows (odd; 1 = raw cosine).",
    )
    parser.add_argument("--k", type=float, default=3.5, help="Threshold = median - k * MAD (scaled) of all window scores.")
    parser.add_argument("--floor", type=float, default=None, help="Also flag every window scoring below this.")
    parser.add_argument("--min-consecutive", type=int, default=2, help="Shortest run of flagged windows to report.")
    parser.add_argument(
        "--bridge", type=int, default=1, help="Unvoiced (skipped) windows allowed inside a run of flagged windows."
    )
    parser.add_argument("--merge-gap-s", type=float, default=2.0, help="Merge segments separated by at most this.")
    parser.add_argument("--iterations", type=int, default=3, help="Robust-centroid trimming passes (input centroid).")
    parser.add_argument("--drop-frac", type=float, default=0.25, help="Fraction of lowest-cosine windows dropped per pass.")
    parser.add_argument("--batch-size", type=int, default=32, help="ECAPA batch size.")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available).")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="Decode/voicing worker processes.")
    return parser.parse_args()


# ------------------------------------------------------------------ windows

@dataclass
class Prepared:
    path: Path
    wav: np.ndarray
    starts: np.ndarray
    voiced_frac: np.ndarray


@dataclass
class FileScores:
    path: Path
    duration_s: float
    starts: np.ndarray
    voiced_frac: np.ndarray
    keep: np.ndarray
    emb: np.ndarray
    cos: np.ndarray = field(default_factory=lambda: np.zeros(0))
    score: np.ndarray = field(default_factory=lambda: np.zeros(0))
    segments: list[dict] = field(default_factory=list)
    flagged: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def kept_cos(self) -> np.ndarray:
        return self.cos[self.keep]

    @property
    def kept_score(self) -> np.ndarray:
        return self.score[self.keep]


def window_starts(duration_s: float, window_s: float, hop_s: float) -> np.ndarray:
    if duration_s < window_s:
        return np.zeros(0)
    count = int(np.floor((duration_s - window_s) / hop_s + 1e-9)) + 1
    return np.round(np.arange(count) * hop_s, 6)


def prepare(path: Path, window_s: float, hop_s: float, f0_method: str) -> Prepared:
    data, sr = load_audio(path)
    wav = resample(to_mono(data), sr, ANALYSIS_SR).astype(np.float32)
    voiced = analyze_frames(wav, ANALYSIS_SR, f0_method).voiced
    starts = window_starts(len(wav) / ANALYSIS_SR, window_s, hop_s)
    per_window = int(round(window_s / HOP_S))
    fractions = np.array([voiced[int(round(s / HOP_S)) : int(round(s / HOP_S)) + per_window].mean() for s in starts])
    return Prepared(path, wav, starts, fractions if len(starts) else np.zeros(0))


def prepare_all(paths: list[Path], args: argparse.Namespace) -> Iterator[Prepared]:
    if not paths:
        return
    if args.jobs <= 1 or len(paths) == 1:
        for path in paths:
            yield prepare(path, args.window_s, args.hop_s, args.f0_method)
        return
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(paths))) as pool:
        n = len(paths)
        yield from pool.map(prepare, paths, [args.window_s] * n, [args.hop_s] * n, [args.f0_method] * n)


def embed_file(prep: Prepared, encoder, args: argparse.Namespace) -> FileScores:
    keep = prep.voiced_frac >= args.min_voiced
    size = int(round(args.window_s * ANALYSIS_SR))
    offsets = [int(round(s * ANALYSIS_SR)) for s in prep.starts[keep]]
    chunks = []
    for i in range(0, len(offsets), args.batch_size):
        batch = np.stack([prep.wav[o : o + size] for o in offsets[i : i + args.batch_size]])
        chunks.append(encoder.embed_batch(batch))
    emb = np.concatenate(chunks) if chunks else np.zeros((0, 192))
    return FileScores(prep.path, len(prep.wav) / ANALYSIS_SR, prep.starts, prep.voiced_frac, keep, emb)


def embed_paths(paths: list[Path], encoder, args: argparse.Namespace, label: str) -> list[FileScores]:
    scores = []
    for i, prep in enumerate(prepare_all(paths, args), 1):
        result = embed_file(prep, encoder, args)
        print(
            f"[{label} {i}/{len(paths)}] {record_path(prep.path)}: {result.duration_s / 60:.1f} min, "
            f"{int(result.keep.sum())}/{len(result.starts)} voiced windows",
            flush=True,
        )
        scores.append(result)
    return scores


# ------------------------------------------------------------------ scoring

def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def robust_centroid(emb: np.ndarray, iterations: int, drop_frac: float) -> np.ndarray:
    """Mean embedding, re-estimated after dropping the lowest-cosine windows each pass."""
    center = unit(emb.mean(axis=0))
    for _ in range(iterations):
        cos = emb @ center
        center = unit(emb[cos >= np.quantile(cos, drop_frac)].mean(axis=0))
    return center


def voiced_groups(keep: np.ndarray, bridge: int) -> list[np.ndarray]:
    """Kept window indices split where more than `bridge` unvoiced windows separate them."""
    kept = np.flatnonzero(keep)
    if not len(kept):
        return []
    return np.split(kept, np.flatnonzero(np.diff(kept) > 1 + bridge) + 1)


def smoothed(cos: np.ndarray, keep: np.ndarray, size: int, bridge: int) -> np.ndarray:
    """Centred moving average of the cosine over neighbouring voiced windows (shrinks at group edges)."""
    out = np.full(len(cos), np.nan)
    half = max(size, 1) // 2
    for group in voiced_groups(keep, bridge):
        values = cos[group]
        sums = np.concatenate([[0.0], np.cumsum(values)])
        lo = np.maximum(np.arange(len(values)) - half, 0)
        hi = np.minimum(np.arange(len(values)) + half + 1, len(values))
        out[group] = (sums[hi] - sums[lo]) / (hi - lo)
    return out


def robust_threshold(cos: np.ndarray, k: float, floor: float | None) -> tuple[float, float, float]:
    median = float(np.median(cos))
    mad = MAD_SCALE * float(np.median(np.abs(cos - median)))
    threshold = median - k * mad
    if floor is not None:
        threshold = max(threshold, floor)
    return threshold, median, mad


def flagged_runs(score: FileScores, threshold: float, args: argparse.Namespace) -> list[list[int]]:
    """Runs of flagged windows (grid indices); unvoiced windows up to --bridge do not break a run."""
    kept = np.flatnonzero(score.keep)
    runs: list[list[int]] = []
    current: list[int] = []
    previous = -10**9
    for idx in kept:
        low = score.score[idx] < threshold
        if low and current and idx - previous <= 1 + args.bridge:
            current.append(int(idx))
        elif low:
            if current:
                runs.append(current)
            current = [int(idx)]
        elif current:
            runs.append(current)
            current = []
        previous = idx
    if current:
        runs.append(current)
    return [run for run in runs if len(run) >= args.min_consecutive]


def segments_for(score: FileScores, threshold: float, args: argparse.Namespace) -> list[dict]:
    """Each window stands for its central hop-long cell, so a run maps to [first centre - hop/2, last centre + hop/2]."""
    half = args.hop_s / 2
    centre = args.window_s / 2
    merged: list[tuple[float, float, list[int]]] = []
    for run in flagged_runs(score, threshold, args):
        start = max(0.0, score.starts[run[0]] + centre - half)
        end = min(score.duration_s, score.starts[run[-1]] + centre + half)
        if merged and start - merged[-1][1] <= args.merge_gap_s:
            merged[-1] = (merged[-1][0], end, merged[-1][2] + run)
        else:
            merged.append((start, end, run))
    segments = []
    for start, end, idx in merged:
        values = score.cos[idx]
        segments.append({
            "path": record_path(score.path),
            "start_s": round(float(start), 2),
            "end_s": round(float(end), 2),
            "min_cos": float(values.min()),
            "mean_cos": float(values.mean()),
            "n_windows": len(idx),
            "_idx": idx,
        })
    return segments


def summarize(score: FileScores, hop_s: float) -> dict:
    kept = score.kept_cos
    voiced_s = len(kept) * hop_s
    flagged_s = sum(seg["end_s"] - seg["start_s"] for seg in score.segments)
    return {
        "path": record_path(score.path),
        "duration_min": fmt(score.duration_s / 60, 2),
        "voiced_min": fmt(voiced_s / 60, 2),
        "n_windows": len(kept),
        "median_cos": fmt(float(np.median(kept)), 4) if len(kept) else "",
        "p10_cos": fmt(float(np.quantile(kept, 0.1)), 4) if len(kept) else "",
        "n_segments": len(score.segments),
        "flagged_s": fmt(flagged_s, 1),
        "flagged_pct": fmt(100.0 * flagged_s / voiced_s, 1) if voiced_s else "",
        "_flagged_s": flagged_s,
    }


# ------------------------------------------------------------------ output

def mmss(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def segment_row(seg: dict) -> dict:
    return {**seg, "start_s": fmt(seg["start_s"], 2), "end_s": fmt(seg["end_s"], 2),
            "min_cos": fmt(seg["min_cos"], 4), "mean_cos": fmt(seg["mean_cos"], 4)}


def window_rows(score: FileScores, window_s: float) -> list[dict]:
    rel = record_path(score.path)
    return [
        {
            "path": rel,
            "start_s": fmt(s, 2),
            "end_s": fmt(s + window_s, 2),
            "voiced_frac": fmt(v, 3),
            "cos": fmt(c, 4) if k else "",
            "score": fmt(m, 4) if k else "",
            "flagged": int(f) if k else "",
        }
        for s, v, c, m, k, f in zip(
            score.starts, score.voiced_frac, score.cos, score.score, score.keep, score.flagged
        )
    ]


def plot_file(score: FileScores, threshold: float, median: float, out_dir: Path, window_s: float) -> Path:
    import warnings

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    warnings.filterwarnings("ignore", message="Glyph .* missing")

    rel = record_path(score.path)
    width = float(np.clip(6 + score.duration_s / 60 * 0.6, 8, 24))
    fig, ax = plt.subplots(figsize=(width, 3.2))
    centres = score.starts + window_s / 2
    keep = score.keep
    for seg in score.segments:
        ax.axvspan(seg["start_s"], seg["end_s"], color="#d62728", alpha=0.15, lw=0)
    ax.plot(centres, np.where(keep, score.cos, np.nan), color="#9db4d6", lw=0.6, label="window cosine")
    ax.plot(centres, np.where(keep, score.score, np.nan), color="#4c72b0", lw=1.0, label="smoothed score")
    ax.scatter(centres[score.flagged], score.score[score.flagged], s=8, color="#d62728", zorder=3)
    ax.axhline(threshold, color="#d62728", ls="--", lw=0.8, label=f"threshold {threshold:.3f}")
    ax.axhline(median, color="#555555", ls=":", lw=0.8, label=f"median {median:.3f}")
    ax.set_xlim(0, score.duration_s)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: mmss(x)))
    ax.set_xlabel("time (mm:ss)")
    ax.set_ylabel("cosine to target")
    ax.set_ylim(min(-0.1, float(np.nanmin(np.where(keep, score.cos, np.nan), initial=0.0))), 1.0)
    ax.set_title(Path(rel).name, fontsize=8, fontfamily=["Malgun Gothic", "DejaVu Sans"])
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{slugify(score.path.stem, 40)}_{short_hash(rel)}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def print_report(scores: list[FileScores], summaries: list[dict], threshold: float) -> None:
    by_path = {record_path(s.path): s for s in scores}
    ordered = sorted(summaries, key=lambda row: (-row["_flagged_s"], row["path"]))
    print("\nFiles by flagged time (listen to these ranges):")
    for row in ordered:
        if not row["_flagged_s"]:
            continue
        print(f"\n  {row['flagged_s']:>6} s ({row['flagged_pct']}% of voiced)  median {row['median_cos']}  {row['path']}")
        for seg in sorted(by_path[row["path"]].segments, key=lambda s: s["start_s"]):
            print(
                f"      {mmss(seg['start_s'])}-{mmss(seg['end_s'])}  "
                f"min {seg['min_cos']:.3f}  mean {seg['mean_cos']:.3f}  ({seg['n_windows']} windows)"
            )
    clean = [row["path"] for row in ordered if not row["_flagged_s"]]
    if clean:
        print(f"\nNo segment below {threshold:.3f} in {len(clean)} file(s):")
        for path in clean:
            print(f"      {path}")
    low = low_median_files(summaries)
    if low:
        print("\nWhole-file check: median cosine far below the other files (mostly another voice, or a very different channel):")
        for row in low:
            print(f"      median {row['median_cos']}  {row['path']}")


def low_median_files(summaries: list[dict], k: float = FILE_MEDIAN_K) -> list[dict]:
    rows = [row for row in summaries if row["median_cos"] != "" and row["n_windows"] >= FILE_MIN_WINDOWS]
    if len(rows) < 4:
        return []
    medians = np.array([float(row["median_cos"]) for row in rows])
    threshold, center, _ = robust_threshold(medians, k, None)
    threshold = min(threshold, center - FILE_MEDIAN_MARGIN)
    return sorted((row for row, m in zip(rows, medians) if m < threshold), key=lambda row: float(row["median_cos"]))


# ------------------------------------------------------------------ main

def collect_inputs(args: argparse.Namespace) -> list[Path]:
    if args.manifest:
        manifest = resolve_path(args.manifest)
        paths = [resolve_path(row["path"]) for row in read_csv(manifest)]
    else:
        folder = resolve_path(args.input)
        if not folder.is_dir():
            raise SystemExit(f"--input is not a folder: {folder}")
        paths = iter_audio_files(folder, args.glob)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} input file(s) missing, e.g. {record_path(missing[0])}")
    if not paths:
        raise SystemExit("No input files found.")
    return sorted(dict.fromkeys(paths))


def collect_references(values: list[str], pattern: str) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = resolve_path(value)
        if path.is_dir():
            paths.extend(iter_audio_files(path, pattern))
        elif path.is_file():
            paths.append(path)
        else:
            raise SystemExit(f"--reference not found: {value}")
    if not paths:
        raise SystemExit("No reference audio found.")
    return sorted(dict.fromkeys(paths))


def load_encoder(device: str | None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval_lib import gpu, speaker

    device = gpu.pick_device(device)
    print(f"Loading ECAPA speaker encoder on {device}", flush=True)
    return speaker.SpeakerEncoder(MODELS_DIR, device)


def main() -> None:
    setup_console()
    args = parse_args()
    output = resolve_path(args.output)
    summary_path = resolve_path(args.summary) if args.summary else output.with_name(f"{output.stem}_summary.csv")
    inputs = collect_inputs(args)
    references = collect_references(args.reference, args.reference_glob) if args.reference else []

    encoder = load_encoder(args.device)
    try:
        scores = embed_paths(inputs, encoder, args, "input")
        known = {s.path: s for s in scores}
        ref_new = [p for p in references if p not in known]
        ref_scores = [known[p] for p in references if p in known] + embed_paths(ref_new, encoder, args, "reference")
    finally:
        encoder.close()

    all_emb = np.concatenate([s.emb for s in scores])
    if len(all_emb) < 2:
        raise SystemExit("Fewer than 2 voiced windows in the inputs; nothing to compare.")
    if references:
        ref_emb = np.concatenate([s.emb for s in ref_scores])
        if not len(ref_emb):
            raise SystemExit("No voiced windows in the reference files.")
        center = unit(ref_emb.mean(axis=0))
        print(f"Centroid: mean of {len(ref_emb)} reference windows from {len(references)} file(s)")
    else:
        center = robust_centroid(all_emb, args.iterations, args.drop_frac)
        print(f"Centroid: {len(all_emb)} input windows, {args.iterations} passes dropping the lowest {args.drop_frac:.0%}")

    for score in scores:
        score.cos = np.full(len(score.starts), np.nan)
        score.cos[score.keep] = score.emb @ center
        score.score = smoothed(score.cos, score.keep, args.smooth, args.bridge)
    threshold, median, mad = robust_threshold(np.concatenate([s.kept_score for s in scores]), args.k, args.floor)
    print(f"Threshold: score < {threshold:.4f} (median {median:.4f}, scaled MAD {mad:.4f}, k {args.k}, "
          f"smooth {args.smooth}"
          f"{f', floor {args.floor}' if args.floor is not None else ''})")

    for score in scores:
        score.segments = segments_for(score, threshold, args)
        score.flagged = np.zeros(len(score.starts), dtype=bool)
        for seg in score.segments:
            score.flagged[seg["_idx"]] = True

    segments = sorted((seg for s in scores for seg in s.segments), key=lambda g: (g["mean_cos"], g["path"], g["start_s"]))
    summaries = [summarize(s, args.hop_s) for s in scores]
    write_csv(output, [segment_row(seg) for seg in segments], OUTLIER_COLUMNS)
    write_csv(summary_path, sorted(summaries, key=lambda r: (-r["_flagged_s"], r["path"])), SUMMARY_COLUMNS)
    print(f"Wrote {record_path(output)} ({len(segments)} segment(s)) and {record_path(summary_path)}")
    if args.windows:
        windows_path = resolve_path(args.windows)
        write_csv(windows_path, [row for s in scores for row in window_rows(s, args.window_s)], WINDOW_COLUMNS)
        print(f"Wrote {record_path(windows_path)}")
    if args.plot:
        plot_dir = output.with_name(f"{output.stem}_plots")
        for score in scores:
            plot_file(score, threshold, median, plot_dir, args.window_s)
        print(f"Plots in {record_path(plot_dir)}")

    print_report(scores, summaries, threshold)


if __name__ == "__main__":
    main()
