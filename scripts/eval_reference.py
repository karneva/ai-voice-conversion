#!/usr/bin/env python3
"""Paired evaluation of converted vocals against the target singer's real cover of the same song.

The source is the original singer's separated vocals of a song; the reference is the target
singer's own (possibly partial, rearranged, re-timed) cover of that song in the same key, never
used for training. Converting the full source vocals keeps the source timeline, so one
source<->reference time map serves every checkpoint.

Subcommands:
  align    which parts of the source the reference sings, plus a frame-level time map:
           the reference is cut at long pauses and each segment is matched by subsequence DTW
           (CENS chroma + per-file standardized MFCC + vocal activity, 20 ms frames). Each
           candidate path is verified by absolute-pitch agreement (same key): a candidate is
           accepted when pitch_agreement - cost_ratio >= min_match_score, and the choice among
           accepted candidates prefers the song order
  convert  convert the FULL source vocals with run checkpoints (Applio core.py infer via
           make_cover.py helpers, transpose 0, the run's infer grid by default)
  score    on aligned regions voiced in both tracks: sim_ref (ECAPA cosine of paired 3 s
           windows), mcd_ref (band-limited DTW on mel-cepstra), f0_corr_ref (information only),
           each with the unconverted source as the baseline
  all      align + convert + score

Outputs: runs/<run_id>/reference/{alignment.json, <E>e/<combo>/converted.wav, metrics.csv} and an
upsert into results/reference.csv. The canonical alignment is
results/cache/reference_alignment_<song>.json, keyed by the input sha256s and the alignment config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import make_cover  # noqa: E402  (also puts scripts/data on sys.path)
from _audio import REPO_ROOT, record_path, resolve_path, setup_console, sha256_file, slugify  # noqa: E402
from eval_lib import audio, mcd, pitch, speaker  # noqa: E402
from eval_lib.gpu import StageMonitor, nvidia_smi_used_mib, pick_device  # noqa: E402
from evaluate import fmt, read_csv, write_csv  # noqa: E402
from run_experiment import APPLIO_LOGS, combo_tag  # noqa: E402

EVAL_VERSION = "ref-1.0.0"
ALIGN_VERSION = "ref-align/2"
SR = audio.SR
TRANSPOSE = 0  # the reference is in the source key
DEFAULT_SOURCE = "dataset/private/separated/cover_input"
DEFAULT_REFERENCE = "dataset/private/separated/reference"
RUNS_DIR = REPO_ROOT / "runs"
CACHE_DIR = REPO_ROOT / "results" / "cache"
RESULTS_CSV = REPO_ROOT / "results" / "reference.csv"
DEFAULT_GRID = [{"f0_method": "rmvpe", "index_rate": 0.3, "protect": 0.33, "volume_envelope": 1.0}]

REFERENCE_COLUMNS = [
    "run_id", "epoch", "combo", "n_windows", "aligned_s",
    "sim_ref", "sim_ref_source", "sim_ref_self", "mcd_ref", "mcd_ref_source", "f0_corr_ref",
    "eval_version",
]
UPSERT_KEY = ("run_id", "epoch", "combo")
METRICS_COLUMNS = [
    "run_id", "epoch", "step", "combo", "n_windows", "aligned_s", "voiced_both_s", "n_mcd_frames",
    "sim_ref", "sim_ref_source", "sim_ref_gain", "sim_ref_self", "sim_ref_centroid", "sim_ref_source_centroid",
    "mcd_ref", "mcd_ref_source", "mcd_ref_gain", "f0_corr_ref", "f0_corr_ref_source",
    "converted_path", "converted_sha256", "alignment_key", "eval_version",
]

# Subsequence DTW steps (ref frames, source frames): local tempo ratio within [0.5, 2]. The (2, 1)
# step covers two reference frames, so it counts twice and D[-1] / n is a mean per reference frame.
SIGMA = np.array([[1, 1], [1, 2], [2, 1]])
WEIGHTS_MUL = np.array([1.0, 1.0, 2.0])


@dataclass(frozen=True)
class AlignConfig:
    hop_s: float = 0.02
    active_top_db: float = 35.0
    active_floor_db: float = -55.0
    chroma_smooth: int = 9
    mfcc_smooth: int = 5
    n_mfcc: int = 13
    chroma_weight: float = 1.0
    mfcc_weight: float = 1.0
    activity_weight: float = 0.5
    min_pause_s: float = 1.0
    min_active_s: float = 1.0
    min_segment_s: float = 6.0
    max_merge_gap_s: float = 4.0
    max_segment_s: float = 30.0
    candidates: int = 4
    pitch_tolerance_cents: float = 100.0
    min_pitch_s: float = 1.0
    min_match_score: float = 0.12
    order_penalty: float = 0.05
    order_tolerance_s: float = 2.0
    anchor_step: int = 5

    @property
    def hop(self) -> int:
        return int(round(self.hop_s * SR))

    def frames(self, seconds: float) -> int:
        return int(round(seconds / self.hop_s))


@dataclass(frozen=True)
class ScoreConfig:
    window_s: float = 3.0
    window_hop_s: float = 1.0
    min_voiced: float = 0.5
    embed_batch: int = 16
    dtw_band_s: float = 0.3


# ---------------------------------------------------------------- helpers


def log(message: str) -> None:
    print(message, flush=True)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):d}:{seconds % 60:04.1f}"


def json_hash(data: Any) -> str:
    return hashlib.sha1(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find_vocals(value: str) -> Path:
    """A file, or a folder holding exactly one *_(vocals)_*.wav."""
    path = resolve_path(value)
    if path.is_file():
        return path
    found = sorted(path.glob(f"*{make_cover.VOCALS_MARK}*.wav")) if path.is_dir() else []
    if len(found) != 1:
        raise SystemExit(f"Expected one *{make_cover.VOCALS_MARK}*.wav in {value}, found {len(found)}")
    return found[0]


def song_name(source: Path, given: str | None) -> str:
    return slugify(given or source.name.split(make_cover.VOCALS_MARK)[0] or source.stem)


def reference_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id / "reference"


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- alignment features


@dataclass
class AlignFeatures:
    chroma: np.ndarray  # (12, T) CENS, unit norm
    mfcc: np.ndarray  # (n, T) standardized over active frames, smoothed, unit norm
    activity: np.ndarray  # (T,) soft vocal activity 0..1
    active: np.ndarray  # (T,) bool
    cents: np.ndarray  # (T,) F0 in cents re 55 Hz, NaN where unvoiced

    def __len__(self) -> int:
        return len(self.active)

    def pooled(self, factor: int) -> "AlignFeatures":
        """Average every `factor` frames (for the overview plot)."""
        n = len(self) // factor

        def pool(x: np.ndarray) -> np.ndarray:
            return x[..., : n * factor].reshape(*x.shape[:-1], n, factor).mean(axis=-1)

        return AlignFeatures(unit(pool(self.chroma)), unit(pool(self.mfcc)), pool(self.activity),
                             pool(self.active) > 0.5, self.cents[: n * factor : factor])


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=0, keepdims=True), 1e-9)


def pitch_cents(x: np.ndarray, hop_s: float) -> np.ndarray:
    """pyworld DIO + StoneMask F0 (CPU) on the alignment grid, 5-frame median filtered; NaN = unvoiced."""
    from _audio import HOP_S, estimate_f0
    from scipy.ndimage import median_filter

    f0 = estimate_f0(x, SR, "dio")[:: max(1, int(round(hop_s / HOP_S)))]
    cents = np.where(f0 > 0, 1200.0 * np.log2(np.maximum(f0, 1.0) / 55.0), -1.0)
    cents = median_filter(cents, size=5)
    return np.where(cents > 0, cents, np.nan)


def align_features(x: np.ndarray, cfg: AlignConfig) -> AlignFeatures:
    import librosa
    from scipy.ndimage import uniform_filter1d

    hop = cfg.hop
    chroma = librosa.feature.chroma_cens(y=x, sr=SR, hop_length=hop, win_len_smooth=cfg.chroma_smooth)
    mfcc = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=cfg.n_mfcc + 1, n_fft=1024, hop_length=hop, n_mels=40)[1:]
    db = audio.frame_rms_db(x, frame_length=2 * hop, hop_length=hop)
    cents = pitch_cents(x, cfg.hop_s)
    n = min(chroma.shape[1], mfcc.shape[1], len(db), len(cents))
    chroma, mfcc, db, cents = chroma[:, :n], mfcc[:, :n], db[:n], cents[:n]
    threshold = max(float(db.max()) - cfg.active_top_db, cfg.active_floor_db)
    active = db > threshold
    activity = 1.0 / (1.0 + np.exp(-(db - threshold) / 3.0))
    basis = mfcc[:, active] if active.sum() > 10 else mfcc
    mfcc = (mfcc - basis.mean(axis=1, keepdims=True)) / np.maximum(basis.std(axis=1, keepdims=True), 1e-6)
    mfcc = uniform_filter1d(mfcc, size=cfg.mfcc_smooth, axis=1)
    return AlignFeatures(unit(chroma).astype(np.float32), unit(mfcc).astype(np.float32),
                         activity.astype(np.float32), active, cents)


def cost_matrix(ref: AlignFeatures, src: AlignFeatures, rows: slice, cfg: AlignConfig) -> np.ndarray:
    """Frame cost (reference rows x all source frames), normalized to about 0..1.

    Content (chroma + MFCC cosine distance) counts where both frames are active; elsewhere it is
    a neutral 0.5, and an activity mismatch term is added.
    """
    ar, as_ = ref.activity[rows], src.activity
    content = cfg.chroma_weight * (1.0 - ref.chroma[:, rows].T @ src.chroma)
    content += cfg.mfcc_weight * 0.5 * (1.0 - ref.mfcc[:, rows].T @ src.mfcc)
    neutral = 0.5 * (cfg.chroma_weight + cfg.mfcc_weight)
    gate = np.outer(ar, as_)
    cost = gate * content
    cost += (1.0 - gate) * neutral
    cost += cfg.activity_weight * np.abs(ar[:, None] - as_[None, :])
    cost /= cfg.chroma_weight + cfg.mfcc_weight + cfg.activity_weight
    return cost.astype(np.float64)


# ---------------------------------------------------------------- segmentation


def active_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def segment_reference(feat: AlignFeatures, cfg: AlignConfig) -> list[tuple[int, int]]:
    """Reference frame spans [start, end) cut at pauses >= min_pause_s, short ones merged, long ones split."""
    spans: list[list[int]] = []
    for start, end in active_runs(feat.active):
        if spans and start - spans[-1][1] < cfg.frames(cfg.min_pause_s):
            spans[-1][1] = end
        else:
            spans.append([start, end])
    spans = [s for s in spans if feat.active[s[0]:s[1]].sum() >= cfg.frames(cfg.min_active_s)]
    spans = merge_short(spans, cfg)
    result: list[tuple[int, int]] = []
    for start, end in spans:
        result.extend(split_long(feat.active, start, end, cfg))
    return result


def merge_short(spans: list[list[int]], cfg: AlignConfig) -> list[list[int]]:
    spans = [list(s) for s in spans]
    while True:
        short = [i for i, (s, e) in enumerate(spans) if e - s < cfg.frames(cfg.min_segment_s)]
        best = None
        for i in short:
            for j in (i - 1, i + 1):
                if 0 <= j < len(spans):
                    gap = spans[max(i, j)][0] - spans[min(i, j)][1]
                    if gap <= cfg.frames(cfg.max_merge_gap_s) and (best is None or gap < best[0]):
                        best = (gap, min(i, j))
        if best is None:
            return spans
        k = best[1]
        spans[k:k + 2] = [[spans[k][0], spans[k + 1][1]]]


def split_long(active: np.ndarray, start: int, end: int, cfg: AlignConfig) -> list[tuple[int, int]]:
    """Split at the longest pause in the middle 60%, recursively, until spans fit max_segment_s."""
    if end - start <= cfg.frames(cfg.max_segment_s):
        return [(start, end)]
    lo, hi = start + int(0.2 * (end - start)), start + int(0.8 * (end - start))
    gaps = [(e - s, s, e) for s, e in active_runs(~active[lo:hi])]
    if gaps:
        _, s, e = max(gaps)
        cut_start, cut_end = lo + s, lo + e
    else:
        cut_start = cut_end = (start + end) // 2
    return split_long(active, start, cut_start, cfg) + split_long(active, cut_end, end, cfg)


# ---------------------------------------------------------------- matching


@dataclass
class Candidate:
    src_start: int
    src_end: int
    cost: float  # mean frame cost along the path
    ratio: float  # end cost / median end cost over all source positions (lower = more distinctive)
    path: np.ndarray  # (k, 2) absolute (reference frame, source frame), ascending
    agreement: float = 0.0  # fraction of jointly voiced path frames within pitch_tolerance_cents
    voiced_s: float = 0.0  # jointly voiced seconds along the path

    @property
    def score(self) -> float:
        return self.agreement - self.ratio


def pitch_agreement(path: np.ndarray, ref: AlignFeatures, src: AlignFeatures, cfg: AlignConfig) -> tuple[float, float]:
    """(agreement, voiced seconds); agreement is 0 below min_pitch_s of jointly voiced frames."""
    a, b = ref.cents[path[:, 0]], src.cents[path[:, 1]]
    both = ~np.isnan(a) & ~np.isnan(b)
    voiced_s = float(both.sum()) * cfg.hop_s
    if voiced_s < cfg.min_pitch_s:
        return 0.0, voiced_s
    return float(np.mean(np.abs(a[both] - b[both]) <= cfg.pitch_tolerance_cents)), voiced_s


def match_segment(ref: AlignFeatures, src: AlignFeatures, span: tuple[int, int], cfg: AlignConfig) -> list[Candidate]:
    import librosa

    r0, r1 = span
    n = r1 - r0
    cost = cost_matrix(ref, src, slice(r0, r1), cfg)
    acc, steps = librosa.sequence.dtw(C=cost, subseq=True, step_sizes_sigma=SIGMA, weights_mul=WEIGHTS_MUL,
                                      backtrack=False, return_steps=True)
    end_cost = acc[-1] / n
    finite = np.isfinite(end_cost)
    if not finite.any():
        return []
    null = float(np.median(end_cost[finite]))
    candidates: list[Candidate] = []
    for j in np.argsort(np.where(finite, end_cost, np.inf))[: 50 * cfg.candidates]:
        if len(candidates) >= cfg.candidates:
            break
        if any(abs(int(j) - (c.src_end - 1)) < n // 4 for c in candidates):
            continue
        wp = librosa.sequence.dtw_backtracking(steps, step_sizes_sigma=SIGMA, subseq=True, start=int(j))[::-1]
        s0, s1 = int(wp[0, 1]), int(wp[-1, 1]) + 1
        if any(min(s1, c.src_end) - max(s0, c.src_start) > 0.5 * min(s1 - s0, c.src_end - c.src_start)
               for c in candidates):
            continue
        path = np.column_stack([wp[:, 0] + r0, wp[:, 1]])
        agreement, voiced_s = pitch_agreement(path, ref, src, cfg)
        candidates.append(Candidate(s0, s1, float(cost[wp[:, 0], wp[:, 1]].mean()), float(end_cost[j] / null), path,
                                    agreement, voiced_s))
    return candidates


def choose_candidates(per_segment: list[list[Candidate]], cfg: AlignConfig) -> list[int | None]:
    """Viterbi over accepted candidates: -score + a penalty for jumping back in the source."""
    accepted = [[k for k, c in enumerate(cands) if c.score >= cfg.min_match_score] for cands in per_segment]
    order = [i for i, ks in enumerate(accepted) if ks]
    choice: list[int | None] = [None] * len(per_segment)
    if not order:
        return choice
    tol = cfg.frames(cfg.order_tolerance_s)
    score = {k: -per_segment[order[0]][k].score for k in accepted[order[0]]}
    back: list[dict[int, int]] = []
    for prev_i, i in zip(order, order[1:]):
        new_score, pointers = {}, {}
        for k in accepted[i]:
            cur = per_segment[i][k]

            def total(pk: int) -> float:
                prev = per_segment[prev_i][pk]
                jump = cfg.order_penalty if cur.src_start < prev.src_end - tol else 0.0
                return score[pk] + jump

            best = min(score, key=total)
            new_score[k], pointers[k] = total(best) - cur.score, best
        back.append(pointers)
        score = new_score
    k = min(score, key=score.get)
    for step, i in enumerate(reversed(order)):
        choice[i] = k
        if step < len(back):
            k = back[-1 - step][k]
    return choice


def path_anchors(path: np.ndarray, cfg: AlignConfig) -> list[list[float]]:
    """Monotonic reference->source map sampled every anchor_step frames, in seconds."""
    ref_idx, src_idx = path[:, 0], path[:, 1].astype(np.float64)
    first = int(ref_idx.min())
    counts = np.bincount(ref_idx - first)
    sums = np.bincount(ref_idx - first, weights=src_idx)
    seen = np.flatnonzero(counts) + first
    frames = np.arange(first, int(ref_idx.max()) + 1)
    mapped = np.maximum.accumulate(np.interp(frames, seen, sums[seen - first] / counts[seen - first]))
    keep = np.unique(np.r_[np.arange(0, len(frames), cfg.anchor_step), len(frames) - 1])
    return [[round(float(frames[k]) * cfg.hop_s, 3), round(float(mapped[k]) * cfg.hop_s, 3)] for k in keep]


def candidate_record(c: Candidate, cfg: AlignConfig) -> dict[str, Any]:
    return {"src_start_s": round(c.src_start * cfg.hop_s, 2), "src_end_s": round(c.src_end * cfg.hop_s, 2),
            "cost": round(c.cost, 4), "cost_ratio": round(c.ratio, 4), "pitch_agreement": round(c.agreement, 4),
            "pitch_voiced_s": round(c.voiced_s, 1), "match_score": round(c.score, 4)}


def run_alignment(source: Path, reference: Path, cfg: AlignConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """(alignment record, plot data)."""
    t0 = time.perf_counter()
    src_wav, ref_wav = audio.load_16k(source), audio.load_16k(reference)
    src, ref = align_features(src_wav, cfg), align_features(ref_wav, cfg)
    spans = segment_reference(ref, cfg)
    log(f"align: reference {mmss(len(ref_wav) / SR)} cut into {len(spans)} segment(s); "
        f"source {mmss(len(src_wav) / SR)}")
    per_segment = [match_segment(ref, src, span, cfg) for span in spans]
    choice = choose_candidates(per_segment, cfg)
    segments = []
    for index, (span, cands, k) in enumerate(zip(spans, per_segment, choice)):
        record: dict[str, Any] = {
            "index": index,
            "ref_start_s": round(span[0] * cfg.hop_s, 2),
            "ref_end_s": round(span[1] * cfg.hop_s, 2),
            "ref_active_s": round(float(ref.active[span[0]:span[1]].sum()) * cfg.hop_s, 2),
            "status": "matched" if k is not None else "rejected",
            "candidates": [candidate_record(c, cfg) for c in cands],
        }
        if k is not None:
            chosen = cands[k]
            record.update(candidate_record(chosen, cfg))
            record["candidate"] = k
            record["tempo_ratio"] = round((chosen.src_end - chosen.src_start) / (span[1] - span[0]), 3)
            record["map"] = path_anchors(chosen.path, cfg)
        else:
            best = max(cands, key=lambda c: c.score) if cands else None
            record["reason"] = (f"best match score {best.score:.3f} < {cfg.min_match_score}" if best else "no candidate")
        segments.append(record)
    matched = [s for s in segments if s["status"] == "matched"]
    summary = {
        "n_segments": len(segments),
        "n_matched": len(matched),
        "reference_s": round(len(ref_wav) / SR, 2),
        "source_s": round(len(src_wav) / SR, 2),
        "reference_active_s": round(float(ref.active.sum()) * cfg.hop_s, 2),
        "matched_ref_s": round(sum(s["ref_end_s"] - s["ref_start_s"] for s in matched), 2),
        "matched_ref_active_s": round(sum(s["ref_active_s"] for s in matched), 2),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    plot = {"ref": ref, "src": src, "spans": spans, "per_segment": per_segment, "choice": choice}
    return {"segments": segments, "summary": summary}, plot


def print_alignment(record: dict[str, Any]) -> None:
    log("  seg  reference          source             tempo  ratio  pitch  score  status")
    for s in record["segments"]:
        ref = f"{mmss(s['ref_start_s'])}-{mmss(s['ref_end_s'])}"
        if s["status"] == "matched":
            src = f"{mmss(s['src_start_s'])}-{mmss(s['src_end_s'])}"
            log(f"  {s['index']:>3}  {ref:<17}  {src:<17}  {s['tempo_ratio']:5.2f}  {s['cost_ratio']:.3f}  "
                f"{s['pitch_agreement']:.3f}  {s['match_score']:.3f}  matched")
        else:
            log(f"  {s['index']:>3}  {ref:<17}  {'-':<17}  {'':5}  {'':5}  {'':5}  {'':5}  rejected ({s['reason']})")
    m = record["summary"]
    log(f"  matched {m['n_matched']}/{m['n_segments']} segment(s): {m['matched_ref_active_s']:.1f}s of "
        f"{m['reference_active_s']:.1f}s active reference audio")


def plot_alignment(plot: dict[str, Any], cfg: AlignConfig, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["DejaVu Sans", "Malgun Gothic"]

    factor = 5
    ref, src = plot["ref"].pooled(factor), plot["src"].pooled(factor)
    cost = cost_matrix(ref, src, slice(0, len(ref)), cfg)
    step = cfg.hop_s * factor
    fig, ax = plt.subplots(figsize=(10, 10 * len(ref) / max(len(src), 1) + 1.2))
    ax.imshow(cost, origin="lower", aspect="auto", cmap="gray", vmin=np.percentile(cost, 1),
              vmax=np.percentile(cost, 50), extent=(0, len(src) * step, 0, len(ref) * step))
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
    for i, (span, cands, k) in enumerate(zip(plot["spans"], plot["per_segment"], plot["choice"])):
        ax.axhspan(span[0] * cfg.hop_s, span[1] * cfg.hop_s, color="#ffffff", alpha=0.06, linewidth=0)
        for j, c in enumerate(cands):
            xy = c.path * cfg.hop_s
            if j == k:
                ax.plot(xy[:, 1], xy[:, 0], color=colors[i % len(colors)], linewidth=2.2)
                ax.text(xy[-1, 1], xy[-1, 0], f" {i}: {c.score:.2f}", color=colors[i % len(colors)], fontsize=8,
                        va="center")
            else:
                ax.plot(xy[:, 1], xy[:, 0], color="#e34948", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_xlabel("source time (s)")
    ax.set_ylabel("reference time (s)")
    ax.set_title(f"{title}\ncost: dark = similar · chosen path per reference segment (solid, label = match score); "
                 "other candidates dashed red",
                 loc="left", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {record_path(path)}")


def alignment_inputs(source: Path, reference: Path) -> dict[str, Any]:
    return {name: {"path": record_path(p), "sha256": sha256_file(p)} for name, p in
            (("source", source), ("reference", reference))}


def load_or_align(source: Path, reference: Path, song: str, cfg: AlignConfig, *, force: bool = False,
                  plot_path: Path | None = None, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    inputs = alignment_inputs(source, reference)
    key = json_hash({"version": ALIGN_VERSION, "config": asdict(cfg),
                     "source": inputs["source"]["sha256"], "reference": inputs["reference"]["sha256"]})
    cache = cache_dir / f"reference_alignment_{song}.json"
    cached = read_json(cache)
    if not force and cached and cached.get("key") == key and plot_path is None:
        log(f"align: reuse {record_path(cache)} (key {key})")
        return cached
    record, plot = run_alignment(source, reference, cfg)
    record = {"version": ALIGN_VERSION, "key": key, "created_at": now_iso(), "song": song, "inputs": inputs,
              "config": asdict(cfg), **record}
    write_json(cache, record)
    log(f"wrote {record_path(cache)}")
    if plot_path is not None:
        plot_alignment(plot, cfg, plot_path, f"{song}: reference vs source alignment")
    return record


def copy_alignment(record: dict[str, Any], run_id: str) -> None:
    target = reference_dir(run_id) / "alignment.json"
    if (read_json(target) or {}).get("key") != record["key"]:
        write_json(target, record)
        log(f"wrote {record_path(target)}")


# ---------------------------------------------------------------- conversion


def run_config(run_id: str) -> dict[str, Any]:
    info = read_json(RUNS_DIR / run_id / "run.json") or {}
    return info.get("config") or read_json(REPO_ROOT / "configs" / f"{run_id}.json") or {}


def infer_entries(cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    grid = (cfg.get("infer") or {}).get("grid") or DEFAULT_GRID
    if args.index_rate is None and args.protect is None:
        return [dict(e) for e in grid]
    entry = dict(grid[0])
    if args.index_rate is not None:
        entry["index_rate"] = args.index_rate
    if args.protect is not None:
        entry["protect"] = args.protect
    return [entry]


def available_epochs(model_id: str) -> dict[int, Path]:
    found = {}
    for pth in (APPLIO_LOGS / model_id).glob(f"{model_id}_*e_*s.pth"):
        epoch = pth.name[len(model_id) + 1:].split("e_")[0]
        if epoch.isdigit():
            found[int(epoch)] = pth
    return dict(sorted(found.items()))


def select_epochs(spec: str, available: dict[int, Path]) -> list[int]:
    if spec == "all":
        return list(available)
    wanted = sorted({int(v.strip().rstrip("e")) for v in spec.split(",") if v.strip()})
    missing = [e for e in wanted if e not in available]
    if missing:
        raise SystemExit(f"Checkpoints not found for epoch(s) {missing}; available: {list(available)}")
    return wanted


def file_record(path: Path | None) -> dict[str, str] | None:
    return {"path": record_path(path), "sha256": sha256_file(path)} if path else None


def convert_one(run_id: str, epoch: int, model: make_cover.Model, entry: dict[str, Any], source: dict[str, str],
                split_audio: bool, force: bool) -> Path:
    combo = combo_tag(entry)
    out_dir = reference_dir(run_id) / f"{epoch}e" / combo
    target, record_file = out_dir / "converted.wav", out_dir / "converted.json"
    settings = {**entry, "index_rate": entry["index_rate"] if model.index else 0.0, "transpose": TRANSPOSE,
                "embedder": model.embedder_custom or model.embedder, "split_audio": split_audio}
    inputs = {"source": source, "pth": file_record(model.pth), "index": file_record(model.index)}
    old = read_json(record_file) or {}
    if not force and target.is_file() and old.get("inputs") == inputs and old.get("settings") == settings:
        log(f"convert: {epoch}e/{combo} up to date")
        return target
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    ns = SimpleNamespace(f0_method=entry["f0_method"], index_rate=entry["index_rate"], protect=entry["protect"],
                         volume_envelope=entry["volume_envelope"], split_audio=split_audio)
    t0 = time.perf_counter()
    converted, command = make_cover.convert(model, resolve_path(source["path"]), work, TRANSPOSE, ns)
    os.replace(converted, target)
    shutil.copyfile(work / "applio_infer.log", out_dir / "applio_infer.log")
    shutil.rmtree(work, ignore_errors=True)  # holds a copy of the private source vocals
    seconds = time.perf_counter() - t0
    write_json(record_file, {
        "created_at": now_iso(), "run_id": run_id, "epoch": epoch, "combo": combo, "settings": settings,
        "inputs": inputs, "output": {"path": record_path(target), "sha256": sha256_file(target)},
        "applio_command": make_cover.format_command(command), "seconds": round(seconds, 1),
    })
    log(f"convert: {epoch}e/{combo} -> {record_path(target)} ({seconds:.0f}s)")
    return target


def cmd_convert(args: argparse.Namespace) -> None:
    cfg = run_config(args.run)
    model_id = cfg.get("model_run_id", args.run)
    available = available_epochs(model_id)
    if not available:
        raise SystemExit(f"No {model_id}_<E>e_<S>s.pth in {record_path(APPLIO_LOGS / model_id)}")
    epochs = select_epochs(args.epochs, available)
    entries = infer_entries(cfg, args)
    source_path = find_vocals(args.source)
    source = file_record(source_path)
    used = nvidia_smi_used_mib()
    log(f"convert: {len(epochs)} epoch(s) x {len(entries)} combo(s) of {record_path(source_path)}"
        + (f"; GPU memory in use {used} MiB" if used is not None else ""))
    for epoch in epochs:
        model = make_cover.resolve_model(SimpleNamespace(run=model_id, epoch=epoch, pth=None, index=None, embedder=None))
        for entry in entries:
            convert_one(args.run, epoch, model, entry, source, args.split_audio, args.force)


# ---------------------------------------------------------------- scoring: tracks


@dataclass
class Track:
    path: Path
    sha256: str
    wav: np.ndarray
    f0: pitch.F0Track | None = None
    mcep: np.ndarray | None = None  # (coeffs, frames) at McdConfig hop
    energy_db: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return len(self.wav) / SR

    def voiced_at(self, times: np.ndarray) -> np.ndarray:
        idx = np.round(np.asarray(times) / F0_HOP_S).astype(int)
        ok = (idx >= 0) & (idx < len(self.f0.voiced))
        out = np.zeros(len(idx), dtype=bool)
        out[ok] = self.f0.voiced[idx[ok]]
        return out

    def f0_at(self, times: np.ndarray) -> np.ndarray:
        idx = np.clip(np.round(np.asarray(times) / F0_HOP_S).astype(int), 0, len(self.f0.f0) - 1)
        return self.f0.f0[idx]

    def voiced_fraction(self, start: float, end: float) -> float:
        a, b = max(0, int(round(start / F0_HOP_S))), min(len(self.f0.voiced), int(round(end / F0_HOP_S)))
        return float(self.f0.voiced[a:b].mean()) if b > a else 0.0

    def window(self, center: float, length_s: float) -> np.ndarray:
        n = int(round(length_s * SR))
        start = int(round((center - length_s / 2) * SR))
        piece = self.wav[max(0, start): max(0, start) + n]
        return np.pad(piece, (0, n - len(piece)))


F0_HOP_S = pitch.PitchConfig().hop_length / pitch.SAMPLE_RATE


def load_track(path: Path) -> Track:
    return Track(path, sha256_file(path), audio.load_16k(path))


def f0_cache_path(sha: str) -> Path:
    tag = json_hash(pitch.PitchConfig().as_dict())[:8]
    return CACHE_DIR / "reference_f0" / f"{sha[:24]}_{tag}.npz"


def fill_f0(tracks: list[Track], device: str) -> dict[str, Any] | None:
    missing = []
    for track in tracks:
        cached = f0_cache_path(track.sha256)
        if cached.exists():
            data = np.load(cached)
            track.f0 = pitch.F0Track(f0=data["f0"], voiced=data["voiced"])
        else:
            missing.append(track)
    if not missing:
        return None
    with StageMonitor("pitch", device) as mon:
        extractor = pitch.PitchExtractor(device)
        try:
            for track in missing:
                track.f0 = extractor.extract(track.wav)
                path = f0_cache_path(track.sha256)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, f0=track.f0.f0, voiced=track.f0.voiced)
                mon.tick()
        finally:
            extractor.close()
    return mon.stats


def fill_mcep(track: Track) -> None:
    if track.mcep is None:
        track.mcep, track.energy_db = mcd._spectra(track.wav, mcd.McdConfig())


# ---------------------------------------------------------------- scoring: regions and windows


@dataclass
class Region:
    index: int
    ref_start: float
    ref_end: float
    anchors: np.ndarray  # (k, 2) reference s, source s

    def to_src(self, t: np.ndarray | float) -> np.ndarray:
        return np.interp(t, self.anchors[:, 0], self.anchors[:, 1])

    @property
    def src_start(self) -> float:
        return float(self.anchors[0, 1])

    @property
    def src_end(self) -> float:
        return float(self.anchors[-1, 1])


def regions_from(alignment: dict[str, Any]) -> list[Region]:
    return [Region(s["index"], s["ref_start_s"], s["ref_end_s"], np.asarray(s["map"], dtype=np.float64))
            for s in alignment["segments"] if s["status"] == "matched"]


@dataclass
class Window:
    region: int
    ref_center: float
    src_center: float


def select_windows(regions: list[Region], ref: Track, src: Track, cfg: ScoreConfig) -> list[Window]:
    """Reference windows inside matched regions whose mapped source window is voiced too."""
    half = cfg.window_s / 2
    windows = []
    for region in regions:
        center = region.ref_start + half
        while center + half <= region.ref_end + 1e-6:
            sc = float(region.to_src(center))
            if (half <= sc <= src.duration - half
                    and ref.voiced_fraction(center - half, center + half) >= cfg.min_voiced
                    and src.voiced_fraction(sc - half, sc + half) >= cfg.min_voiced):
                windows.append(Window(region.index, round(center, 3), round(sc, 3)))
            center += cfg.window_hop_s
    return windows


def embed_windows(encoder: speaker.SpeakerEncoder, track: Track, centers: list[float], cfg: ScoreConfig) -> np.ndarray:
    chunks = []
    for i in range(0, len(centers), cfg.embed_batch):
        batch = np.stack([track.window(c, cfg.window_s) for c in centers[i:i + cfg.embed_batch]])
        chunks.append(encoder.embed_batch(batch))
    return np.concatenate(chunks) if chunks else np.zeros((0, 192))


def far_centroids(ref_embs: np.ndarray, centers: np.ndarray, min_gap: float) -> np.ndarray:
    """Per window: centroid of the reference windows that do not overlap it (leave-one-out)."""
    out = np.zeros_like(ref_embs)
    for i, c in enumerate(centers):
        others = ref_embs[np.abs(centers - c) >= min_gap]
        out[i] = speaker.normalize(others.mean(axis=0)) if len(others) else np.nan
    return out


def rowwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def similarity(out_embs: np.ndarray, ref_embs: np.ndarray, centroids: np.ndarray) -> dict[str, Any]:
    paired, to_centroid = rowwise_cos(out_embs, ref_embs), rowwise_cos(out_embs, centroids)
    return {"paired": paired, "centroid": to_centroid}


# ---------------------------------------------------------------- scoring: spectra and pitch


def mcep_grid() -> tuple[float, float]:
    """(hop, offset) in seconds: mcd.py frames are not centered, frame k covers k*hop .. k*hop + window."""
    mc = mcd.McdConfig()
    return mc.hop_ms / 1000.0, mc.win_ms / 2000.0


def region_pairs(region: Region, ref: Track, out: Track, cfg: ScoreConfig) -> tuple[np.ndarray, np.ndarray]:
    """DTW path (ref mcep frame, out mcep frame) inside a band around the alignment map."""
    import librosa
    from scipy.spatial.distance import cdist

    mc = mcd.McdConfig()
    hop, off = mcep_grid()
    empty = np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    i0 = max(0, int(math.ceil((region.ref_start - off) / hop)))
    i1 = min(int((region.ref_end - off) / hop), ref.mcep.shape[1] - 1)
    if i1 - i0 < 10:
        return empty
    expected = region.to_src(np.arange(i0, i1 + 1) * hop + off)
    j0 = max(0, int((expected.min() - cfg.dtw_band_s - off) / hop))
    j1 = min(out.mcep.shape[1] - 1, int(math.ceil((expected.max() + cfg.dtw_band_s - off) / hop)))
    if j1 - j0 < 10:
        return empty
    cost = cdist(ref.mcep[mc.s:mc.D, i0:i1 + 1].T, out.mcep[mc.s:mc.D, j0:j1 + 1].T)
    outside = np.abs((np.arange(j0, j1 + 1) * hop + off)[None, :] - expected[:, None]) > cfg.dtw_band_s
    cost[outside] = 1e4
    if cost.shape[0] <= cost.shape[1]:
        _, wp = librosa.sequence.dtw(C=cost, subseq=True)
    else:  # librosa 1.0 returns swapped path columns for a tall subseq matrix, so transpose ourselves
        _, wp = librosa.sequence.dtw(C=cost.T, subseq=True)
        wp = wp[:, ::-1]
    wp = wp[::-1]
    keep = ~outside[wp[:, 0], wp[:, 1]]
    return wp[keep, 0] + i0, wp[keep, 1] + j0


def spectral_scores(regions: list[Region], ref: Track, out: Track, cfg: ScoreConfig) -> dict[str, Any]:
    """MCD (project scale, c1..c15) and F0 correlation on DTW pairs voiced in both, per region and pooled."""
    from mel_cepstral_distance.computation import get_MCD_k

    mc = mcd.McdConfig()
    hop, off = mcep_grid()
    fill_mcep(ref)
    fill_mcep(out)
    loud = ref.energy_db > float(ref.energy_db.max()) - mc.top_db
    per_region, mcd_all, f0_ref_all, f0_out_all, ref_frames = [], [], [], [], 0
    for region in regions:
        pi, pj = region_pairs(region, ref, out, cfg)
        ti, tj = pi * hop + off, pj * hop + off
        mask = ref.voiced_at(ti) & out.voiced_at(tj) & loud[pi]
        pi, pj, ti, tj = pi[mask], pj[mask], ti[mask], tj[mask]
        values = get_MCD_k(ref.mcep[:, pi], out.mcep[:, pj], mc.s, mc.D) if len(pi) else np.zeros(0)
        f0_ref, f0_out = ref.f0_at(ti), out.f0_at(tj)
        mcd_all.append(values)
        f0_ref_all.append(f0_ref)
        f0_out_all.append(f0_out)
        ref_frames += len(np.unique(pi))
        corr = pearson(np.log2(f0_ref), np.log2(f0_out)) if len(pi) else None
        per_region.append({"index": region.index, "n_frames": int(len(pi)),
                           "mcd": round(float(values.mean()), 4) if len(values) else None,
                           "f0_corr": round(corr, 4) if corr is not None else None})
    values = np.concatenate(mcd_all) if mcd_all else np.zeros(0)
    f0_ref = np.concatenate(f0_ref_all) if f0_ref_all else np.zeros(0)
    f0_out = np.concatenate(f0_out_all) if f0_out_all else np.zeros(0)
    corr = pearson(np.log2(f0_ref), np.log2(f0_out)) if len(f0_ref) else None
    return {
        "mcd": float(values.mean()) if len(values) else None,
        "f0_corr": corr,
        "n_frames": int(len(values)),
        "voiced_both_s": round(ref_frames * hop, 2),
        "regions": per_region,
    }


# ---------------------------------------------------------------- scoring: orchestration


@dataclass
class Output:
    run_id: str
    epoch: int
    combo: str
    path: Path
    detail_path: Path
    track: Track | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def discover_outputs(run_id: str, epochs: str, combos: str) -> list[Output]:
    root = reference_dir(run_id)
    wanted_epochs = None if epochs == "all" else {int(v.strip().rstrip("e")) for v in epochs.split(",") if v.strip()}
    wanted_combos = None if combos == "all" else {c.strip() for c in combos.split(",") if c.strip()}
    outputs = []
    for wav in sorted(root.glob("*e/*/converted.wav")):
        epoch_dir, combo = wav.parent.parent.name, wav.parent.name
        if not epoch_dir[:-1].isdigit():
            continue
        epoch = int(epoch_dir[:-1])
        if (wanted_epochs is None or epoch in wanted_epochs) and (wanted_combos is None or combo in wanted_combos):
            outputs.append(Output(run_id, epoch, combo, wav, wav.parent / "scores.json"))
    if wanted_epochs:
        missing = wanted_epochs - {o.epoch for o in outputs}
        if missing:
            raise SystemExit(f"No converted.wav for epoch(s) {sorted(missing)} under {record_path(root)}; run 'convert'")
    return sorted(outputs, key=lambda o: (o.epoch, o.combo))


def check_provenance(output: Output, source_sha: str) -> None:
    record = read_json(output.path.parent / "converted.json") or {}
    recorded = ((record.get("inputs") or {}).get("source") or {}).get("sha256")
    if recorded is None:
        log(f"warning: {record_path(output.path)} has no converted.json; cannot check its source")
    elif recorded != source_sha:
        raise SystemExit(f"{record_path(output.path)} was converted from a different source; rerun 'convert'")


def score_key(alignment: dict[str, Any], cfg: ScoreConfig) -> str:
    return json_hash({"alignment": alignment["key"], "score": asdict(cfg), "eval_version": EVAL_VERSION,
                      "mcd": mcd.McdConfig().as_dict(), "pitch": pitch.PitchConfig().as_dict()})


def step_of(run_id: str, epoch: int) -> str:
    model_id = run_config(run_id).get("model_run_id", run_id)
    pth = available_epochs(model_id).get(epoch)
    return pth.stem.rsplit("_", 1)[-1].rstrip("s") if pth else ""


def score_pairs(source: Path, reference: Path, alignment: dict[str, Any], outputs: list[Output],
                device: str, cfg: ScoreConfig) -> dict[str, Any]:
    """Score `outputs` (tracks loaded) and the unconverted source against the reference. Returns the baseline."""
    regions = regions_from(alignment)
    if not regions:
        raise SystemExit("The alignment has no matched segment; nothing to score")
    ref, src = load_track(reference), load_track(source)
    stats = [s for s in [fill_f0([ref, src] + [o.track for o in outputs], device)] if s]
    windows = select_windows(regions, ref, src, cfg)
    if len(windows) < 2:
        raise SystemExit(f"Only {len(windows)} voiced window(s) in the aligned regions")
    ref_centers = [w.ref_center for w in windows]
    src_centers = [w.src_center for w in windows]
    with StageMonitor("speaker", device) as mon:
        encoder = speaker.SpeakerEncoder(CACHE_DIR / "models", device)
        try:
            ref_embs = embed_windows(encoder, ref, ref_centers, cfg)
            src_embs = embed_windows(encoder, src, src_centers, cfg)
            out_embs = [embed_windows(encoder, o.track, src_centers, cfg) for o in outputs]
            mon.tick(len(windows) * (2 + len(outputs)))
        finally:
            encoder.close()
    stats.append(mon.stats)
    centroids = far_centroids(ref_embs, np.asarray(ref_centers), cfg.window_s)
    self_cos = rowwise_cos(ref_embs, centroids)
    regions_s = sum(r.ref_end - r.ref_start for r in regions)

    def detail(track: Track, embs: np.ndarray) -> dict[str, Any]:
        sim = similarity(embs, ref_embs, centroids)
        spectral = spectral_scores(regions, ref, track, cfg)
        return {
            "sim_ref": float(np.mean(sim["paired"])),
            "sim_ref_centroid": float(np.nanmean(sim["centroid"])),
            "mcd_ref": spectral["mcd"],
            "f0_corr_ref": spectral["f0_corr"],
            "n_mcd_frames": spectral["n_frames"],
            "voiced_both_s": spectral["voiced_both_s"],
            "regions": spectral["regions"],
            "window_cos": [round(float(v), 4) for v in sim["paired"]],
        }

    with StageMonitor("mcd", "cpu") as mon:
        baseline = detail(src, src_embs)
        mon.tick()
        for output, embs in zip(outputs, out_embs):
            output.detail = detail(output.track, embs)
            mon.tick()
    stats.append(mon.stats)
    baseline.update({
        "n_windows": len(windows),
        "aligned_s": round(regions_s, 2),
        "sim_ref_self": float(np.nanmean(self_cos)),
        "windows": [asdict(w) for w in windows],
        "stages": stats,
    })
    return baseline


def metrics_row(output: Output, baseline: dict[str, Any], key: str) -> dict[str, Any]:
    d = output.detail
    gain = lambda a, b: None if a is None or b is None else a - b  # noqa: E731
    return {
        "run_id": output.run_id, "epoch": output.epoch, "step": step_of(output.run_id, output.epoch),
        "combo": output.combo, "n_windows": baseline["n_windows"], "aligned_s": baseline["aligned_s"],
        "voiced_both_s": d["voiced_both_s"], "n_mcd_frames": d["n_mcd_frames"],
        "sim_ref": d["sim_ref"], "sim_ref_source": baseline["sim_ref"],
        "sim_ref_gain": gain(d["sim_ref"], baseline["sim_ref"]),
        "sim_ref_self": baseline["sim_ref_self"], "sim_ref_centroid": d["sim_ref_centroid"],
        "sim_ref_source_centroid": baseline["sim_ref_centroid"],
        "mcd_ref": d["mcd_ref"], "mcd_ref_source": baseline["mcd_ref"],
        "mcd_ref_gain": gain(baseline["mcd_ref"], d["mcd_ref"]),
        "f0_corr_ref": d["f0_corr_ref"], "f0_corr_ref_source": baseline["f0_corr_ref"],
        "converted_path": record_path(output.path), "converted_sha256": output.track.sha256 if output.track else "",
        "alignment_key": key, "eval_version": EVAL_VERSION,
    }


def upsert_reference(rows: list[dict[str, Any]], path: Path = RESULTS_CSV) -> None:
    existing = read_csv(path) if path.exists() else []

    def key(row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(fmt(row.get(c)) for c in UPSERT_KEY)

    merged = {key(r): r for r in existing}
    for row in rows:
        merged[key(row)] = {c: fmt(row.get(c)) for c in REFERENCE_COLUMNS}
    ordered = sorted(merged.values(), key=lambda r: (r["run_id"], int(r["epoch"] or 0), r["combo"]))
    write_csv(path, REFERENCE_COLUMNS, ordered)
    log(f"upserted {len(rows)} row(s) into {record_path(path)}")


def print_table(rows: list[dict[str, Any]]) -> None:
    def f(v: Any, spec: str) -> str:
        return "-" if v is None or v == "" else format(float(v), spec)

    log(f"{'epoch':>5} {'sim_ref':>8} {'(Tim)':>7} {'gain':>7} {'self':>6} {'mcd_ref':>8} {'(Tim)':>7} "
        f"{'gain':>6} {'f0_corr':>8}")
    for r in rows:
        log(f"{r['epoch']:>5} {f(r['sim_ref'], '.4f'):>8} {f(r['sim_ref_source'], '.4f'):>7} "
            f"{f(r['sim_ref_gain'], '+.4f'):>7} {f(r['sim_ref_self'], '.3f'):>6} {f(r['mcd_ref'], '.3f'):>8} "
            f"{f(r['mcd_ref_source'], '.3f'):>7} {f(r['mcd_ref_gain'], '+.3f'):>6} {f(r['f0_corr_ref'], '.3f'):>8}")


def cmd_score(args: argparse.Namespace, alignment: dict[str, Any] | None = None) -> None:
    source, reference = find_vocals(args.source), find_vocals(args.reference)
    cfg = ScoreConfig()
    if alignment is None:
        alignment = load_or_align(source, reference, song_name(source, args.song), AlignConfig())
    copy_alignment(alignment, args.run)
    outputs = discover_outputs(args.run, args.epochs, args.combos)
    if not outputs:
        raise SystemExit(f"No converted outputs under {record_path(reference_dir(args.run))}; run 'convert' first")
    source_sha = sha256_file(source)
    for o in outputs:
        check_provenance(o, source_sha)
        o.track = load_track(o.path)
    key = score_key(alignment, cfg)
    todo = []
    for o in outputs:
        old = read_json(o.detail_path) or {}
        if not args.force and old.get("score_key") == key and old.get("converted_sha256") == o.track.sha256:
            o.detail = old["detail"]
        else:
            todo.append(o)
    device = pick_device(args.device)
    log(f"score: {len(outputs)} output(s), {len(todo)} to score, {len(outputs) - len(todo)} reused; device {device}")
    baseline = score_pairs(source, reference, alignment, todo, device, cfg)
    for o in todo:
        write_json(o.detail_path, {"score_key": key, "converted_sha256": o.track.sha256, "created_at": now_iso(),
                                   "detail": o.detail})
    rows = [metrics_row(o, baseline, alignment["key"]) for o in outputs]
    run_dir = reference_dir(args.run)
    write_csv(run_dir / "metrics.csv", METRICS_COLUMNS, rows)
    log(f"wrote {record_path(run_dir / 'metrics.csv')}")
    write_json(run_dir / "baseline.json", {"created_at": now_iso(), "eval_version": EVAL_VERSION,
                                           "score_config": asdict(cfg), "alignment_key": alignment["key"],
                                           "source": record_path(source), "reference": record_path(reference),
                                           **baseline})
    upsert_reference(rows)
    print_table(rows)


# ---------------------------------------------------------------- CLI


def cmd_align(args: argparse.Namespace) -> dict[str, Any]:
    source, reference = find_vocals(args.source), find_vocals(args.reference)
    song = song_name(source, args.song)
    plot_path = None
    if args.plot:
        plot_path = (reference_dir(args.run) if args.run else CACHE_DIR) / f"alignment_{song}.png"
    record = load_or_align(source, reference, song, AlignConfig(), force=args.force, plot_path=plot_path)
    print_alignment(record)
    if args.run:
        copy_alignment(record, args.run)
    return record


def add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Source vocals (file, or a folder with one *_(vocals)_*.wav).")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="Target singer's cover vocals (file, or a folder with one *_(vocals)_*.wav).")
    parser.add_argument("--song", help="Song id for the alignment cache name. Default: slug of the source stem.")


def add_selection(parser: argparse.ArgumentParser, convert: bool) -> None:
    parser.add_argument("--run", required=True, help="Run id (runs/<run>, Applio/logs/<model_run_id>).")
    parser.add_argument("--epochs", default="all", help="Comma list like 20,100 or 'all'.")
    if convert:
        parser.add_argument("--index-rate", type=float, help="Override the run's infer grid (single combo).")
        parser.add_argument("--protect", type=float, help="Override the run's infer grid (single combo).")
        parser.add_argument("--split-audio", action="store_true",
                            help="Applio --split-audio (less VRAM; hard cuts at -60 dB silences).")
    else:
        parser.add_argument("--combos", default="all", help="Comma list of combo tags or 'all'.")
        parser.add_argument("--device", help="torch device for ECAPA/CREPE. Default: cuda if available.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("align", help="Align the reference cover to the source vocals (CPU).")
    add_inputs(p)
    p.add_argument("--run", help="Also copy the alignment (and plot) to runs/<run>/reference/.")
    p.add_argument("--plot", action="store_true", help="Save an alignment plot (paths over the cost matrix).")
    p.add_argument("--force", action="store_true", help="Recompute even if the cached alignment matches.")

    p = sub.add_parser("convert", help="Convert the full source vocals with run checkpoints (Applio, GPU).")
    add_inputs(p)
    add_selection(p, convert=True)
    p.add_argument("--force", action="store_true", help="Reconvert even if converted.json matches.")

    p = sub.add_parser("score", help="Score converted outputs against the reference.")
    add_inputs(p)
    add_selection(p, convert=False)
    p.add_argument("--force", action="store_true", help="Rescore even if scores.json matches.")

    p = sub.add_parser("all", help="align + convert + score.")
    add_inputs(p)
    add_selection(p, convert=True)
    p.add_argument("--combos", default="all", help=argparse.SUPPRESS)
    p.add_argument("--device", help="torch device for ECAPA/CREPE. Default: cuda if available.")
    p.add_argument("--plot", action="store_true", help="Save the alignment plot.")
    p.add_argument("--force", action="store_true", help="Redo every step.")
    args = parser.parse_args()
    for name in ("index_rate", "protect"):
        value = getattr(args, name, None)
        if value is not None and not 0 <= value <= (0.5 if name == "protect" else 1):
            parser.error(f"--{name.replace('_', '-')} out of range")
    return args


def main() -> int:
    setup_console()
    args = parse_args()
    if args.command == "align":
        cmd_align(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "score":
        cmd_score(args)
    else:
        alignment = cmd_align(args)
        cmd_convert(args)
        args.combos = "all" if args.index_rate is None and args.protect is None else combo_tag(
            infer_entries(run_config(args.run), args)[0])
        cmd_score(args, alignment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
