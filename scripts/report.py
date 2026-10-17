#!/usr/bin/env python3
"""Build portfolio figures and Markdown tables from results/runs.csv and Applio TensorBoard logs."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import t as t_dist


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMBO = "f0-rmvpe_ir0.30_p0.33_ve1.00"
SOURCE_RUN_ID = "_source"  # evaluate.py source-only rows: the unconverted input scored with the same metrics
LOSS_TAGS = ("loss_avg_50/g/total", "loss_avg_50/g/mel", "loss_avg_50/d/adv")
LOSS_LABELS = {
    "loss_avg_50/g/total": "Generator total loss",
    "loss_avg_50/g/mel": "Mel loss (x45)",
    "loss_avg_50/d/adv": "Discriminator loss",
}
CHECKPOINT_RE = re.compile(r"_(\d+)e_(\d+)s\.pth$")

# Text and surface tokens (light mode only: figures are static images).
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e6e5e1"
SURFACE = "#ffffff"
BASELINE_GRAY = "#a3a29c"
LOSS_COLOR = "#52514e"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
ORDINAL_BLUES = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")
PROTECT_MARKERS = ("o", "s", "^", "D", "v", "P")


@dataclass(frozen=True)
class Metric:
    label: str
    unit: str
    higher_is_better: bool
    fmt: str
    color: str

    @property
    def hint(self) -> str:
        return "↑ better" if self.higher_is_better else "↓ better"

    @property
    def decimals(self) -> int:
        match = re.search(r"\.(\d+)", self.fmt)
        return int(match.group(1)) if match else 0


# Colors follow the evaluation axis: similarity blue, intelligibility orange, pitch aqua,
# naturalness violet, spectral distortion magenta, source leakage red.
METRICS: dict[str, Metric] = {
    "sim_target_norm": Metric("Target similarity (normalized)", "0 = others, 1 = target", True, ".3f", SLOTS[0]),
    "sim_target": Metric("Target similarity", "ECAPA cosine", True, ".3f", SLOTS[0]),
    "sim_source": Metric("Source similarity", "ECAPA cosine", False, ".3f", SLOTS[7]),
    "dcer": Metric("ΔCER (converted − source)", "CER difference", False, "+.3f", SLOTS[1]),
    "cer": Metric("CER (converted)", "character error rate", False, ".3f", SLOTS[1]),
    "f0_rmse_cents": Metric("F0 RMSE", "cents", False, ".1f", SLOTS[2]),
    "gpe50": Metric("Gross pitch error (>50 c)", "fraction of voiced frames", False, ".3f", SLOTS[2]),
    "octave_err": Metric("Octave error (>600 c)", "fraction of voiced frames", False, ".3f", SLOTS[2]),
    "vuv_err": Metric("V/UV error", "fraction of frames", False, ".3f", SLOTS[2]),
    "fpc": Metric("F0 correlation", "Pearson r", True, ".3f", SLOTS[2]),
    "singmos": Metric("SingMOS (predicted)", "score", True, ".2f", SLOTS[6]),
    "utmos": Metric("UTMOS (predicted)", "score", True, ".2f", SLOTS[6]),
    "mcd": Metric("MCD", "MCD (not dB; project scale)", False, ".2f", SLOTS[4]),
}
TABLE_METRICS = (
    "sim_target_norm", "sim_target", "sim_source", "dcer", "f0_rmse_cents",
    "octave_err", "vuv_err", "gpe50", "fpc", "singmos", "utmos", "mcd",
)
CHART_METRICS = ("sim_target_norm", "sim_source", "dcer", "f0_rmse_cents", "octave_err", "singmos", "mcd")
EPOCH_CURVE_METRICS = ("sim_target_norm", "dcer", "f0_rmse_cents", "singmos")
NUMERIC_COLUMNS = (
    "epoch", "step", "index_rate", "protect", "volume_envelope", "n_clips", "n_pitch_clips",
    "sim_target", "sim_target_norm", "sim_source", "f0_rmse_cents", "gpe50", "octave_err",
    "vuv_err", "fpc", "cer", "cer_source", "dcer", "utmos", "utmos_source", "singmos",
    "singmos_source", "mcd",
)
# Pitch means skip clips with no jointly voiced frames, so a row covering fewer clips is not comparable.
PITCH_METRICS = ("f0_rmse_cents", "gpe50", "octave_err", "fpc")
# runs.csv stores these as corpus ratios (total edits / total reference characters), not clip means.
CORPUS_METRICS = ("cer", "dcer")
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0


# ---------------------------------------------------------------- CLI


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs-csv", default=Path("results/runs.csv"), type=Path,
                        help="Results table. Default: results/runs.csv.")
    parser.add_argument("--out-dir", default=Path("results"), type=Path,
                        help="Writes <out-dir>/figures and <out-dir>/tables. Default: results.")
    parser.add_argument("--runs-dir", default=Path("runs"), type=Path,
                        help="Run records (run.json, metrics/). Default: runs.")
    parser.add_argument("--applio-logs", default=Path("Applio/logs"), type=Path,
                        help="Applio logs root with <run_id>/eval event files. Default: Applio/logs.")
    parser.add_argument("--calibration-dir", default=Path("results"), type=Path,
                        help="Directory with calibration_<track>.json. Default: results.")
    parser.add_argument("--formats", default="png,svg", help="Figure formats. Default: png,svg.")
    parser.add_argument("--dpi", default=200, type=int, help="PNG resolution. Default: 200.")


def add_set(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--set", default="source_eval", choices=("source_eval", "heldout"),
                        help="Evaluation set. Default: source_eval.")


def add_combo(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--combo", default=None,
                        help=f"Inference setting tag, e.g. {BASELINE_COMBO}. "
                             "Default: the only combo, else the baseline combo, else the one with most epochs.")


def add_select_set(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--select-set", default="heldout", choices=("source_eval", "heldout"),
                        help="Set on which the best epoch is chosen; its row on --set is reported. "
                             "Equal to --set = select and report on one set (optimistic). Default: heldout.")


def add_epoch_choice(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epoch", default="best", help="best, last, or an epoch number. Default: best.")
    parser.add_argument("--metric-for-best", default="sim_target_norm", choices=sorted(METRICS),
                        help="Metric that defines the best epoch. Default: sim_target_norm.")
    add_select_set(parser)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make figures (PNG/SVG) and Markdown tables from results/runs.csv.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("epoch-curve", help="E1: eval metrics vs epoch aligned with smoothed training losses.")
    p.add_argument("--run", required=True, help="Run ID.")
    add_set(p)
    add_combo(p)
    p.add_argument("--metrics", default=",".join(EPOCH_CURVE_METRICS),
                   help=f"Comma-separated metrics. Default: {','.join(EPOCH_CURVE_METRICS)}.")
    p.add_argument("--primary", default="sim_target_norm", choices=sorted(METRICS),
                   help="Metric whose best epoch is drawn across all panels. Default: sim_target_norm.")
    p.add_argument("--tb-dir", type=Path, default=None,
                   help="TensorBoard event directory. Default: <applio-logs>/<run>/eval.")
    p.add_argument("--smooth", default=0.9, type=float, help="EMA weight for loss curves (0 = off). Default: 0.9.")
    p.add_argument("--no-loss", action="store_true", help="Skip the training-loss panels.")
    add_common(p)

    p = sub.add_parser("pareto", help="E2: index_rate x protect trade-off between dCER and target similarity.")
    p.add_argument("--run", required=True, help="Run ID.")
    p.add_argument("--epoch", required=True, type=int, help="Checkpoint epoch.")
    add_set(p)
    p.add_argument("--f0-method", default=None, help="Filter f0 method. Default: the most common one.")
    p.add_argument("--volume-envelope", default=None, type=float, help="Filter volume envelope. Default: the most common one.")
    p.add_argument("--x-metric", default="dcer", choices=sorted(METRICS), help="Default: dcer.")
    p.add_argument("--y-metric", default="sim_target_norm", choices=sorted(METRICS), help="Default: sim_target_norm.")
    add_common(p)

    p = sub.add_parser("compare", help="Ablation table and bar chart against a baseline run.")
    p.add_argument("--baseline", required=True, help="Baseline run ID.")
    p.add_argument("--runs", required=True, help="Comma-separated run IDs to compare.")
    p.add_argument("--name", default=None, help="Output name. Default: compare_<baseline>_<set>.")
    add_set(p)
    add_combo(p)
    add_epoch_choice(p)
    add_common(p)

    p = sub.add_parser("data-curve", help="E5: quality vs dataset minutes.")
    p.add_argument("--runs", required=True, action="append",
                   help="Comma-separated run IDs forming one series. Repeat for more series.")
    p.add_argument("--series-names", default=None, help="Comma-separated legend names, one per --runs.")
    p.add_argument("--minutes-from", default="auto", choices=("auto", "run.json", "model_info"),
                   help="Where to read dataset length. Default: auto (run.json, then model_info.json).")
    p.add_argument("--minutes", default=None, help="Explicit overrides: run_id=minutes,run_id=minutes.")
    p.add_argument("--metrics", default="sim_target_norm,dcer,f0_rmse_cents", help="Comma-separated metrics.")
    p.add_argument("--name", default=None, help="Output name. Default: e5_data_curve_<set>.")
    add_set(p)
    add_combo(p)
    add_epoch_choice(p)
    add_common(p)

    p = sub.add_parser("summary", help="One Markdown table of all runs at their best epoch.")
    add_set(p)
    add_combo(p)
    p.add_argument("--metric-for-best", default="sim_target_norm", choices=sorted(METRICS),
                   help="Metric that defines the best epoch. Default: sim_target_norm.")
    add_select_set(p)
    add_common(p)
    return parser.parse_args()


# ---------------------------------------------------------------- data access


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def combo_tag(f0_method: object, index_rate: float, protect: float, volume_envelope: float) -> str:
    return f"f0-{f0_method}_ir{index_rate:.2f}_p{protect:.2f}_ve{volume_envelope:.2f}"


def load_runs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Results table not found: {path}")
    df = pd.read_csv(path)
    missing = {"run_id", "set", "epoch"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else np.nan
    for column in ("run_id", "set", "track", "f0_method", "cer_ref"):
        df[column] = df[column].astype("string") if column in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    source_rows = df[df["run_id"] == SOURCE_RUN_ID].copy()
    df = df[df["run_id"] != SOURCE_RUN_ID].dropna(subset=["epoch"]).copy()
    df.attrs["source_rows"] = source_rows
    df["epoch"] = df["epoch"].astype(int)
    df["combo"] = [
        combo_tag(m, ir, p, ve) if not any(pd.isna(v) for v in (m, ir, p, ve)) else "unknown"
        for m, ir, p, ve in zip(df["f0_method"], df["index_rate"], df["protect"], df["volume_envelope"])
    ]
    return df


def select_rows(df: pd.DataFrame, run_id: str, eval_set: str) -> pd.DataFrame:
    rows = df[(df["run_id"] == run_id) & (df["set"] == eval_set)]
    if rows.empty:
        known = ", ".join(sorted(df.loc[df["set"] == eval_set, "run_id"].dropna().unique()))
        raise SystemExit(f"No rows for run '{run_id}' on set '{eval_set}'. Runs on that set: {known or 'none'}")
    return rows


def pick_combo(rows: pd.DataFrame, combo: str | None, run_id: str) -> str:
    combos = rows["combo"].value_counts()
    if combo:
        if combo not in combos.index:
            raise SystemExit(f"Combo '{combo}' not found for run '{run_id}'. Available: {', '.join(combos.index)}")
        return combo
    if len(combos) == 1:
        return str(combos.index[0])
    if BASELINE_COMBO in combos.index:
        return BASELINE_COMBO
    chosen = str(combos.index[0])
    print(f"note: run {run_id} has {len(combos)} combos; using {chosen} (most rows). Pass --combo to choose.")
    return chosen


def combo_rows(df: pd.DataFrame, run_id: str, eval_set: str, combo: str | None) -> tuple[pd.DataFrame, str]:
    rows = select_rows(df, run_id, eval_set)
    chosen = pick_combo(rows, combo, run_id)
    rows = rows[rows["combo"] == chosen].sort_values("epoch")
    return rows.drop_duplicates("epoch", keep="last"), chosen


def has_values(rows: pd.DataFrame, metric: str) -> bool:
    return metric in rows.columns and rows[metric].notna().any()


def full_pitch_coverage(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Drop rows whose pitch metric skipped clips (silent/unvoiced outputs would otherwise look better)."""
    if metric not in PITCH_METRICS or "n_pitch_clips" not in rows or rows["n_pitch_clips"].isna().all():
        return rows
    complete = rows[rows["n_pitch_clips"].isna() | (rows["n_pitch_clips"] >= rows["n_clips"])]
    if complete.empty:
        print(f"warning: every row has pitch metrics on fewer clips than n_clips; {metric} best ignores coverage.")
        return rows
    if len(complete) < len(rows):
        skipped = sorted(set(rows["epoch"]) - set(complete["epoch"]))
        print(f"note: {metric}: epochs {skipped} skipped for best-epoch selection (pitch metrics missing on some clips).")
    return complete


def best_row(rows: pd.DataFrame, metric: str) -> pd.Series | None:
    valid = full_pitch_coverage(rows.dropna(subset=[metric]), metric)
    if valid.empty:
        return None
    index = valid[metric].idxmax() if METRICS[metric].higher_is_better else valid[metric].idxmin()
    return valid.loc[index]


def choose_epoch_row(rows: pd.DataFrame, how: str, metric: str, run_id: str) -> pd.Series:
    if how == "last":
        return rows.loc[rows["epoch"].idxmax()]
    if how == "best":
        row = best_row(rows, metric)
        if row is None:
            print(f"warning: run {run_id} has no {metric} values; falling back to last epoch.")
            return rows.loc[rows["epoch"].idxmax()]
        return row
    try:
        epoch = int(how)
    except ValueError:
        raise SystemExit(f"--epoch must be best, last, or an integer, got: {how}")
    match = rows[rows["epoch"] == epoch]
    if match.empty:
        raise SystemExit(f"Run {run_id} has no epoch {epoch}. Available: {sorted(rows['epoch'].tolist())}")
    return match.iloc[-1]


def pick_epoch_row(df: pd.DataFrame, rows: pd.DataFrame, how: str, metric: str, select_set: str, eval_set: str,
                   run_id: str, combo: str) -> tuple[pd.Series, str]:
    """Choose the epoch on select_set (for 'best') and return its row on eval_set plus the set it was chosen on."""
    if how != "best" or select_set == eval_set:
        return choose_epoch_row(rows, how, metric, run_id), eval_set
    selection = df[(df["run_id"] == run_id) & (df["set"] == select_set) & (df["combo"] == combo)]
    selection = selection[selection["epoch"].isin(rows["epoch"])].drop_duplicates("epoch", keep="last")
    best = best_row(selection, metric) if not selection.empty else None
    if best is None:
        print(f"warning: run {run_id} has no {metric} on {select_set} at the epochs evaluated on {eval_set}; "
              f"selecting on {eval_set} itself.")
        return choose_epoch_row(rows, how, metric, run_id), eval_set
    return rows[rows["epoch"] == int(best["epoch"])].iloc[-1], select_set


def selection_note(metric: str, eval_set: str, selected_on: list[str]) -> str:
    others = sorted(set(selected_on) - {eval_set})
    if not others:
        return (f"each run's best epoch by `{metric}` on this set ({eval_set}). This selects the checkpoint on the "
                "evaluation set itself, so the scores are optimistic; pass --select-set with the other set for "
                "unbiased numbers")
    note = f"each run's best epoch by `{metric}` on {', '.join(others)}, reported on {eval_set}"
    if eval_set in selected_on:
        note += f" (runs without rows on the selection set were selected on {eval_set}, so theirs are optimistic)"
    return note


def best_epoch_text(metric: str, eval_set: str, selected_on: list[str]) -> str:
    others = sorted(set(selected_on) - {eval_set})
    return f"best epoch by {metric}" + (f" on {', '.join(others)}" if others else " on this set")


def cer_reference_note(rows: pd.DataFrame | list[pd.Series]) -> list[str]:
    frame = pd.DataFrame(rows) if isinstance(rows, list) else rows
    if "cer_ref" not in frame:
        return []
    kinds = set(frame["cer_ref"].dropna().astype(str)) - {""}
    if not kinds & {"asr", "mixed"}:
        return []
    return ["- CER reference: some clips have no lyrics, so their reference is Whisper's transcript of the source. "
            "For those clips cer_source is 0 and ΔCER is CER against the source ASR, not against the lyrics."]


Bounds = tuple[float | None, float | None]


def clip_metrics(runs_dir: Path, run_id: str, eval_set: str, epoch: int, combo: str,
                 bounds: Bounds = (None, None)) -> pd.DataFrame | None:
    """Per-clip metrics written by evaluate.py, with the run-level derived columns added."""
    path = runs_dir / run_id / "metrics" / eval_set / f"{epoch}e" / f"{combo}.csv"
    if not path.exists():
        return None
    try:
        clips = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        print(f"warning: cannot read {path}: {error}")
        return None
    if "dcer" not in clips and {"cer", "cer_source"} <= set(clips.columns):
        clips["dcer"] = pd.to_numeric(clips["cer"], errors="coerce") - pd.to_numeric(clips["cer_source"], errors="coerce")
    ceiling, floor = bounds
    if "sim_target_norm" not in clips and "sim_target" in clips and ceiling is not None and floor is not None \
            and ceiling > floor:
        clips["sim_target_norm"] = (pd.to_numeric(clips["sim_target"], errors="coerce") - floor) / (ceiling - floor)
    return clips


def ci95(values: pd.Series) -> float:
    """Half-width of the t-based 95% CI of the mean (clip counts are small)."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return math.nan
    return float(t_dist.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))


def bootstrap_ratio_ci(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """Half-width of the 95% percentile bootstrap CI of sum(numerator) / sum(denominator), resampling clips."""
    if len(numerator) < 2:
        return math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    index = rng.integers(0, len(numerator), size=(BOOTSTRAP_SAMPLES, len(numerator)))
    ratios = numerator[index].sum(axis=1) / denominator[index].sum(axis=1)
    low, high = np.percentile(ratios, [2.5, 97.5])
    return float((high - low) / 2)


def corpus_parts(clips: pd.DataFrame, metric: str) -> pd.DataFrame | None:
    """Per-clip character edits (metric x reference length) and reference lengths, indexed by clip_id."""
    if metric not in clips or "ref_chars" not in clips or "clip_id" not in clips:
        return None
    chars = pd.to_numeric(clips["ref_chars"], errors="coerce")
    edits = pd.to_numeric(clips[metric], errors="coerce") * chars
    parts = pd.DataFrame({"edits": edits.to_numpy(), "chars": chars.to_numpy()}, index=clips["clip_id"])
    return parts[(parts["chars"] > 0) & parts["edits"].notna()]


def metric_ci(clips: pd.DataFrame | None, metric: str) -> float:
    """95% CI half-width over clips, with the same estimator as the runs.csv value (corpus ratio for CER)."""
    if clips is None or metric not in clips:
        return math.nan
    if metric not in CORPUS_METRICS:
        return ci95(clips[metric])
    parts = corpus_parts(clips, metric)
    if parts is None:
        return math.nan
    return bootstrap_ratio_ci(parts["edits"].to_numpy(dtype=float), parts["chars"].to_numpy(dtype=float))


def paired_delta_ci(base: pd.DataFrame | None, other: pd.DataFrame | None, metric: str) -> float:
    if base is None or other is None or metric not in base or metric not in other:
        return math.nan
    if "clip_id" not in base or "clip_id" not in other:
        return math.nan
    if metric in CORPUS_METRICS:
        base_parts, other_parts = corpus_parts(base, metric), corpus_parts(other, metric)
        if base_parts is None or other_parts is None:
            return math.nan
        merged = other_parts.join(base_parts, how="inner", rsuffix="_base")
        return bootstrap_ratio_ci((merged["edits"] - merged["edits_base"]).to_numpy(dtype=float),
                                  merged["chars_base"].to_numpy(dtype=float))
    merged = other[["clip_id", metric]].merge(base[["clip_id", metric]], on="clip_id", suffixes=("", "_base"))
    return ci95(pd.to_numeric(merged[metric], errors="coerce") - pd.to_numeric(merged[f"{metric}_base"], errors="coerce"))


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"warning: cannot read {path}: {error}")
        return None


def find_number(data: object, keys: tuple[str, ...]) -> float | None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys and isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        for value in data.values():
            found = find_number(value, keys)
            if found is not None:
                return found
    return None


def steps_per_epoch(df: pd.DataFrame, run_id: str, applio_logs: Path, runs_dir: Path) -> float | None:
    rows = df[(df["run_id"] == run_id) & (df["step"] > 0) & (df["epoch"] > 0)]
    if not rows.empty:
        ratios = (rows["step"] / rows["epoch"]).round(6).unique()
        if len(ratios) > 1:
            print(f"warning: run {run_id} has inconsistent step/epoch ratios {ratios[:5]}; using the median.")
        return float(np.median(rows["step"] / rows["epoch"]))
    ratios = []
    for pth in (applio_logs / run_id).glob(f"{run_id}_*e_*s.pth"):
        match = CHECKPOINT_RE.search(pth.name)
        if match and int(match.group(1)) > 0:
            ratios.append(int(match.group(2)) / int(match.group(1)))
    if ratios:
        return float(np.median(ratios))
    return find_number(read_json(runs_dir / run_id / "run.json"), ("steps_per_epoch",))


def read_losses(tb_dir: Path, tags: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    if not tb_dir.exists():
        print(f"warning: TensorBoard directory not found: {tb_dir}")
        return {}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    guidance = {"scalars": 0, "images": 1, "audio": 1, "histograms": 1, "tensors": 1, "compressedHistograms": 1}
    accumulator = EventAccumulator(str(tb_dir), size_guidance=guidance)
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    losses = {}
    for tag in tags:
        if tag not in available:
            print(f"warning: tag {tag} not found in {tb_dir}")
            continue
        events = accumulator.Scalars(tag)
        frame = pd.DataFrame({"step": [e.step for e in events], "value": [e.value for e in events]})
        losses[tag] = frame.drop_duplicates("step", keep="last").sort_values("step").reset_index(drop=True)
    return losses


def ema(values: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0 or len(values) == 0:
        return values
    smoothed = np.empty_like(values, dtype=float)
    last, debias = 0.0, 0.0
    for i, value in enumerate(values):
        last = last * weight + (1 - weight) * value
        debias = debias * weight + (1 - weight)
        smoothed[i] = last / debias
    return smoothed


def calibration_bounds(calibration_dir: Path, track: object) -> Bounds:
    """(ceiling, floor) of raw target similarity from evaluate.py calibrate."""
    if track is None or pd.isna(track) or not str(track):
        return None, None
    data = read_json(calibration_dir / f"calibration_{track}.json")
    if not isinstance(data, dict):
        return None, None
    ceiling, floor = data.get("sim_ceiling"), data.get("sim_floor")
    if not isinstance(ceiling, (int, float)):
        ceiling = find_number(data, ("upper", "upper_bound"))
    if not isinstance(floor, (int, float)):
        floor = find_number(data, ("lower", "lower_bound"))
    return (float(ceiling) if ceiling is not None else None, float(floor) if floor is not None else None)


def run_track(rows: pd.DataFrame) -> str | None:
    tracks = rows["track"].dropna()
    return str(tracks.iloc[0]) if not tracks.empty else None


def dataset_minutes(run_id: str, source: str, runs_dir: Path, applio_logs: Path) -> float | None:
    if source in ("auto", "run.json"):
        data = read_json(runs_dir / run_id / "run.json")
        minutes = find_number(data, ("dataset_minutes", "train_minutes", "minutes"))
        if minutes is not None:
            return minutes
        seconds = find_number(data, ("dataset_seconds", "train_seconds", "total_seconds", "dataset_duration_s"))
        if seconds is not None:
            return seconds / 60
    if source in ("auto", "model_info"):
        for path in (runs_dir / run_id / "applio" / "model_info.json", applio_logs / run_id / "model_info.json"):
            seconds = find_number(read_json(path), ("total_seconds",))
            if seconds is not None:
                return seconds / 60
    return None


# ---------------------------------------------------------------- figure helpers


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Malgun Gothic"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "svg.fonttype": "path",
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
    })


def panel_title(ax: plt.Axes, metric: str, compact: bool = False) -> None:
    """Wide panels put the direction hint on the right; narrow panels append its arrow to the title."""
    info = METRICS[metric]
    if compact:
        ax.set_title(f"{info.label} {info.hint[0]}", loc="left", fontsize=8.5)
    else:
        ax.set_title(f"{info.label}  ", loc="left")
        ax.set_title(info.hint, loc="right", fontsize=8, color=INK_MUTED, fontweight="normal")
    ax.set_ylabel(info.unit)


def figure_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.01, 0.995, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.995 - 0.33 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=8.5, color=INK_SECONDARY)


def save_figure(fig: plt.Figure, out_dir: Path, name: str, formats: str, dpi: int) -> list[Path]:
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in [f.strip().lower() for f in formats.split(",") if f.strip()]:
        path = figure_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
        written.append(path)
    plt.close(fig)
    for path in written:
        print(f"wrote {display_path(path)}")
    return written


def fmt_value(value: object, fmt: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return "–"
    return format(float(value), fmt)


def markdown_table(header: list[str], rows: list[list[str]], align: list[str]) -> str:
    marks = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks[a] for a in align) + " |"]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def command_line() -> str:
    def relative(arg: str) -> str:
        path = Path(arg)
        return display_path(path) if path.is_absolute() else arg

    return "evaluation\\env\\python.exe scripts/report.py " + " ".join(shlex.quote(relative(a)) for a in sys.argv[1:])


def write_table(out_dir: Path, name: str, text: str) -> Path:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / f"{name}.md"
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"wrote {display_path(path)}")
    return path


def present_metrics(frame: pd.DataFrame, metrics: tuple[str, ...] | list[str]) -> list[str]:
    return [m for m in metrics if has_values(frame, m)]


def parse_metric_list(text: str) -> list[str]:
    metrics = [m.strip() for m in text.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise SystemExit(f"Unknown metrics: {unknown}. Known: {sorted(METRICS)}")
    return metrics


# ---------------------------------------------------------------- epoch-curve


def epoch_curve(args: argparse.Namespace, df: pd.DataFrame, out_dir: Path) -> None:
    rows, combo = combo_rows(df, args.run, args.set, args.combo)
    metrics = present_metrics(rows, parse_metric_list(args.metrics))
    if not metrics:
        raise SystemExit(f"None of the requested metrics has values for run {args.run}.")
    runs_dir, applio_logs = repo_path(args.runs_dir), repo_path(args.applio_logs)

    losses: dict[str, pd.DataFrame] = {}
    spe = None
    if not args.no_loss:
        spe = steps_per_epoch(df, args.run, applio_logs, runs_dir)
        if spe is None:
            print("warning: steps per epoch unknown (no step column, checkpoints, or run.json); skipping losses.")
        else:
            tb_dir = repo_path(args.tb_dir) if args.tb_dir else applio_logs / args.run / "eval"
            losses = read_losses(tb_dir, LOSS_TAGS)

    bands = {m: [] for m in metrics}
    bounds = calibration_bounds(repo_path(args.calibration_dir), run_track(rows))
    for epoch in rows["epoch"]:
        clips = clip_metrics(runs_dir, args.run, args.set, int(epoch), combo, bounds)
        for m in metrics:
            bands[m].append(metric_ci(clips, m))

    n_panels = len(metrics) + len(losses)
    fig, axes = plt.subplots(n_panels, 1, sharex=True, figsize=(7.6, 0.4 + 1.55 * n_panels),
                             gridspec_kw={"hspace": 0.55})
    axes = np.atleast_1d(axes)
    epochs = rows["epoch"].to_numpy()
    primary = best_row(rows, args.primary) if has_values(rows, args.primary) else None
    best_lines = []

    for ax, metric in zip(axes, metrics):
        info = METRICS[metric]
        values = rows[metric].to_numpy(dtype=float)
        band = np.array(bands[metric], dtype=float)
        if np.isfinite(band).any():
            ax.fill_between(epochs, values - band, values + band, color=info.color, alpha=0.15, linewidth=0)
        ax.plot(epochs, values, color=info.color, marker="o", markersize=3.5, markeredgecolor=SURFACE,
                markeredgewidth=0.8)
        best = best_row(rows, metric)
        if best is not None:
            ax.axvline(best["epoch"], color=info.color, linestyle=":", linewidth=1.2)
            ax.plot(best["epoch"], best[metric], marker="o", markersize=8, markerfacecolor="none",
                    markeredgecolor=info.color, markeredgewidth=1.5)
            ax.annotate(f"best {format(best[metric], info.fmt)} @ {int(best['epoch'])}",
                        xy=(best["epoch"], best[metric]), xytext=(7, 7 if info.higher_is_better else -7),
                        textcoords="offset points", fontsize=7.5, color=INK_SECONDARY,
                        va="bottom" if info.higher_is_better else "top", zorder=5,
                        bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85})
            ax.margins(y=0.22)
            low, high = ax.get_ylim()
            pad = 0.15 * (high - low)
            if info.higher_is_better:
                ax.set_ylim(low, high + pad)
            else:
                ax.set_ylim(low - pad, high)
            best_lines.append(f"{metric}={int(best['epoch'])}")
        panel_title(ax, metric)

    for ax, (tag, frame) in zip(axes[len(metrics):], losses.items()):
        x = frame["step"].to_numpy() / spe
        y = frame["value"].to_numpy(dtype=float)
        ax.plot(x, y, color=LOSS_COLOR, alpha=0.18, linewidth=0.8)
        ax.plot(x, ema(y, args.smooth), color=LOSS_COLOR, linewidth=1.8)
        ax.set_title(f"{LOSS_LABELS.get(tag, tag)}  ", loc="left")
        ax.set_title(f"train · {tag}", loc="right", fontsize=7.5, color=INK_MUTED, fontweight="normal")
        ax.set_ylabel("loss (50-step avg)")

    if primary is not None:
        for ax in axes:
            ax.axvline(primary["epoch"], color=INK, linestyle="--", linewidth=1.0, alpha=0.7, zorder=0)

    max_epoch = max([epochs.max()] + [f["step"].max() / spe for f in losses.values()]) if losses else epochs.max()
    axes[-1].set_xlim(0, max_epoch * 1.02)
    axes[-1].set_xlabel("epoch")
    n_clips = rows["n_clips"].dropna()
    clip_text = f", n={int(n_clips.median())} clips" if not n_clips.empty else ""
    band_text = " · band = 95% CI over clips" if any(np.isfinite(b).any() for b in map(np.array, bands.values())) else ""
    loss_text = f" · losses EMA {args.smooth:g}, {spe:.0f} steps/epoch" if losses else ""
    primary_text = (f"dashed = best {args.primary} (epoch {int(primary['epoch'])}), " if primary is not None else "")
    figure_title(fig, f"{args.run}: eval quality vs training loss by epoch",
                 f"set {args.set}{clip_text} · {combo}{band_text}{loss_text}\n"
                 f"{primary_text}dotted = best epoch per metric · gray thin = raw loss_avg_50")
    fig.subplots_adjust(top=1 - 1.15 / fig.get_figheight())

    name = f"e1_epoch_curve_{args.run}_{args.set}" + ("" if combo == BASELINE_COMBO else f"_{combo}")
    save_figure(fig, out_dir, name, args.formats, args.dpi)
    write_epoch_table(args, rows, metrics, combo, losses, spe, out_dir, name)


def write_epoch_table(args: argparse.Namespace, rows: pd.DataFrame, metrics: list[str], combo: str,
                      losses: dict[str, pd.DataFrame], spe: float | None, out_dir: Path, name: str) -> None:
    header = ["epoch", "step", "n_clips"] + [f"{m} {METRICS[m].hint[0]}" for m in metrics]
    loss_at_epoch = {}
    for tag, frame in losses.items():
        epoch_of = frame["step"] / spe
        loss_at_epoch[tag] = {int(e): float(ema(frame["value"].to_numpy(dtype=float), args.smooth)[
            np.searchsorted(epoch_of.to_numpy(), e, side="right") - 1]) for e in rows["epoch"]
            if (epoch_of <= e).any()}
        header.append(f"{tag.removeprefix('loss_avg_50/')} (EMA)")
    bests = {m: best_row(rows, m) for m in metrics}
    table_rows = []
    for _, row in rows.iterrows():
        cells = [str(int(row["epoch"])), fmt_value(row["step"], ".0f"), fmt_value(row["n_clips"], ".0f")]
        for m in metrics:
            text = fmt_value(row[m], METRICS[m].fmt)
            best = bests[m]
            cells.append(f"**{text}**" if best is not None and int(best["epoch"]) == int(row["epoch"]) else text)
        for tag in losses:
            cells.append(fmt_value(loss_at_epoch[tag].get(int(row["epoch"]), math.nan), ".2f"))
        table_rows.append(cells)
    text = [
        f"# {args.run}: metrics by epoch ({args.set})", "",
        markdown_table(header, table_rows, ["r"] * len(header)), "",
        f"- Inference setting: `{combo}`. Bold = best epoch per metric (↑ higher / ↓ lower is better).",
        "- Best epochs are selected on this evaluation set, so their scores are optimistic.",
        *cer_reference_note(rows),
        f"- Reproduce: `{command_line()}`",
    ]
    write_table(out_dir, name, "\n".join(text))


# ---------------------------------------------------------------- pareto


def pareto_front(points: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    x_sign = 1 if not METRICS[x].higher_is_better else -1
    y_sign = 1 if METRICS[y].higher_is_better else -1
    ordered = points.assign(_x=points[x] * x_sign, _y=points[y] * y_sign).sort_values(["_x", "_y"], ascending=[True, False])
    front, best_y = [], -math.inf
    for index, row in ordered.iterrows():
        if row["_y"] > best_y:
            front.append(index)
            best_y = row["_y"]
    return points.loc[front]


def most_common(values: pd.Series) -> object:
    counts = values.dropna().value_counts()
    return counts.index[0] if not counts.empty else None


def place_labels(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, labels: list[str], colors: list[str]) -> None:
    """Greedy point labelling: first candidate offset that avoids earlier labels, markers and the axes edge."""
    offsets = ((6, 3, "left", "bottom"), (6, -3, "left", "top"), (-6, 3, "right", "bottom"),
               (-6, -3, "right", "top"), (0, 8, "center", "bottom"), (0, -8, "center", "top"))
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    pad = 5 * fig.dpi / 72
    markers = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]
    taken = []

    def collides(box) -> bool:
        if box.x0 < axes_box.x0 or box.x1 > axes_box.x1 or box.y0 < axes_box.y0 or box.y1 > axes_box.y1:
            return True
        if any(box.overlaps(other) for other in taken):
            return True
        return any(box.x0 - pad < mx < box.x1 + pad and box.y0 - pad < my < box.y1 + pad for mx, my in markers)

    for x, y, label, color in sorted(zip(xs, ys, labels, colors), key=lambda item: -item[1]):
        text = None
        for dx, dy, ha, va in offsets:
            if text is not None:
                text.remove()
            text = ax.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                               fontsize=7, color=color, zorder=4)
            box = text.get_window_extent(renderer).expanded(1.04, 1.1)
            if not collides(box):
                break
        else:
            dx, dy, ha, va = offsets[0]
            text.remove()
            text = ax.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                               fontsize=7, color=color, zorder=4)
            box = text.get_window_extent(renderer)
        taken.append(box)


def pareto(args: argparse.Namespace, df: pd.DataFrame, out_dir: Path) -> None:
    rows = select_rows(df, args.run, args.set)
    rows = rows[rows["epoch"] == args.epoch]
    if rows.empty:
        raise SystemExit(f"Run {args.run} has no rows at epoch {args.epoch} on set {args.set}.")
    f0 = args.f0_method or most_common(rows["f0_method"])
    ve = args.volume_envelope if args.volume_envelope is not None else most_common(rows["volume_envelope"])
    rows = rows[(rows["f0_method"] == f0) & np.isclose(rows["volume_envelope"], ve)]
    rows = rows.dropna(subset=[args.x_metric, args.y_metric, "index_rate", "protect"])
    rows = rows.drop_duplicates(["index_rate", "protect"], keep="last")
    if len(rows) < 2:
        raise SystemExit(f"Need at least two index_rate/protect combos with {args.x_metric} and {args.y_metric}.")

    x, y = args.x_metric, args.y_metric
    rates = sorted(rows["index_rate"].unique())
    protects = sorted(rows["protect"].unique())
    if len(protects) > len(PROTECT_MARKERS):
        raise SystemExit(f"At most {len(PROTECT_MARKERS)} protect values are supported, got {len(protects)}.")
    ramp = np.linspace(0, len(ORDINAL_BLUES) - 1, len(rates)).round().astype(int) if len(rates) > 1 else [2]
    rate_color = {r: ORDINAL_BLUES[i] for r, i in zip(rates, ramp)}
    protect_marker = dict(zip(protects, PROTECT_MARKERS))
    front = pareto_front(rows, x, y)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.grid(True, axis="both")
    fx, fy = front.sort_values(x)[x].to_numpy(), front.sort_values(x)[y].to_numpy()
    ax.plot(fx, fy, color=INK_MUTED, linewidth=1.2, linestyle="-", zorder=1, drawstyle="default")
    for _, row in rows.iterrows():
        on_front = row.name in front.index
        ax.scatter(row[x], row[y], s=70 if on_front else 46, marker=protect_marker[row["protect"]],
                   color=rate_color[row["index_rate"]], edgecolor=INK if on_front else SURFACE,
                   linewidth=1.2 if on_front else 1.0, zorder=3)
    baseline = rows[np.isclose(rows["index_rate"], 0.30) & np.isclose(rows["protect"], 0.33)]
    if not baseline.empty:
        b = baseline.iloc[0]
        ax.scatter(b[x], b[y], s=230, marker="o", facecolor="none", edgecolor=INK, linestyle="--", linewidth=1.0, zorder=2)
        ax.annotate("Baseline-0", xy=(b[x], b[y]), xytext=(-8, -14), textcoords="offset points",
                    fontsize=7.5, color=INK, ha="right")

    x_lo, x_hi = rows[x].min(), rows[x].max()
    span = (x_hi - x_lo) or 1.0
    ax.set_xlim(x_lo - 0.06 * span, x_hi + 0.18 * span)  # room for labels right of the last points
    ax.margins(y=0.08)
    ax.set_xlabel(f"{METRICS[x].label} — {METRICS[x].hint}")
    ax.set_ylabel(f"{METRICS[y].label} — {METRICS[y].hint}")
    rate_handles = [Line2D([], [], marker="o", linestyle="", color=rate_color[r], markersize=7, label=f"{r:.2f}") for r in rates]
    protect_handles = [Line2D([], [], marker=protect_marker[p], linestyle="", color=INK_MUTED, markersize=7,
                              label=f"{p:.2f}" + (" (off)" if p >= 0.5 else "")) for p in protects]
    front_handle = [Line2D([], [], color=INK_MUTED, linewidth=1.2, label="Pareto front"),
                    Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK,
                           markersize=7, label="on front")]
    first = ax.legend(handles=rate_handles, title="index_rate", loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      alignment="left", title_fontsize=8)
    ax.add_artist(first)
    second = ax.legend(handles=protect_handles, title="protect", loc="upper left", bbox_to_anchor=(1.01, 0.58),
                       alignment="left", title_fontsize=8)
    ax.add_artist(second)
    ax.legend(handles=front_handle, loc="upper left", bbox_to_anchor=(1.01, 0.22))
    right = METRICS[x].higher_is_better
    up = METRICS[y].higher_is_better
    arrow = {(False, True): "↖", (True, True): "↗", (False, False): "↙", (True, False): "↘"}[(right, up)]
    ax.text(0.99 if right else 0.01, 0.99 if up else 0.01, f"better {arrow}", transform=ax.transAxes,
            ha="right" if right else "left", va="top" if up else "bottom", fontsize=8, color=INK_MUTED)
    n_clips = rows["n_clips"].dropna()
    clip_text = f", n={int(n_clips.median())} clips" if not n_clips.empty else ""
    figure_title(fig, f"{args.run} @ epoch {args.epoch}: index_rate × protect trade-off",
                 f"set {args.set}{clip_text} · f0 {f0} · volume_envelope {ve:.2f} · labels = index_rate/protect")
    fig.subplots_adjust(top=1 - 0.75 / fig.get_figheight())
    place_labels(ax, rows[x].to_numpy(dtype=float), rows[y].to_numpy(dtype=float),
                 [f"{r['index_rate']:.2f}/{r['protect']:.2f}" for _, r in rows.iterrows()],
                 [INK_SECONDARY if i in front.index else INK_MUTED for i in rows.index])

    name = f"e2_pareto_{args.run}_{args.epoch}e_{args.set}"
    save_figure(fig, out_dir, name, args.formats, args.dpi)

    table = rows.sort_values([y, x], ascending=[not METRICS[y].higher_is_better, METRICS[x].higher_is_better])
    extra = present_metrics(table, [m for m in ("sim_source", "f0_rmse_cents", "singmos") if m not in (x, y)])
    header = ["index_rate", "protect", "front", f"{x} {METRICS[x].hint[0]}", f"{y} {METRICS[y].hint[0]}"]
    header += [f"{m} {METRICS[m].hint[0]}" for m in extra] + ["n_clips"]
    body = [[f"{r['index_rate']:.2f}", f"{r['protect']:.2f}", "●" if r.name in front.index else "",
             fmt_value(r[x], METRICS[x].fmt), fmt_value(r[y], METRICS[y].fmt)]
            + [fmt_value(r[m], METRICS[m].fmt) for m in extra] + [fmt_value(r["n_clips"], ".0f")]
            for _, r in table.iterrows()]
    text = [
        f"# {args.run} @ epoch {args.epoch}: index_rate × protect ({args.set})", "",
        markdown_table(header, body, ["r", "r", "c"] + ["r"] * (len(header) - 3)), "",
        f"- f0 `{f0}`, volume_envelope {ve:.2f}. ● = on the Pareto front of {x} and {y}.",
        "- protect 0.50 disables consonant protection in Applio (rvc/infer/pipeline.py).",
        *cer_reference_note(rows),
        f"- Reproduce: `{command_line()}`",
    ]
    write_table(out_dir, name, "\n".join(text))


# ---------------------------------------------------------------- compare / summary


def reference_rows(df: pd.DataFrame, rows: list[pd.Series], metrics: list[str], calibration_dir: Path,
                   track: str | None, eval_set: str) -> list[list[str]]:
    """Ceiling/floor from calibration plus the unconverted input scored by evaluate.py source-only."""
    refs = []
    ceiling, floor = calibration_bounds(calibration_dir, track)
    if ceiling is not None:
        refs.append(("Ref: target held-out (ceiling, LOO)", {"sim_target": ceiling, "sim_target_norm": 1.0}))
    if floor is not None:
        refs.append(("Ref: other singers (floor)", {"sim_target": floor, "sim_target_norm": 0.0}))
    label = "Ref: unconverted target vocals" if eval_set == "heldout" else "Ref: unconverted source vocals"
    source = df.attrs.get("source_rows", pd.DataFrame())
    if not source.empty:
        source = source[(source["set"] == eval_set) & ((source["track"] == track) if track else True)]
    if not source.empty:
        refs.append((label, {m: float(source[m].mean()) for m in metrics if source[m].notna().any()}))
    else:
        frame = pd.DataFrame(rows)
        values = {m: float(frame[f"{m}_source"].mean()) for m in ("utmos", "singmos")
                  if f"{m}_source" in frame and frame[f"{m}_source"].notna().any()}
        if values or ("cer_source" in frame and frame["cer_source"].notna().any()):
            refs.append((label, {**values, "dcer": 0.0}))
    return [[name] + [fmt_value(v.get(m, math.nan), METRICS[m].fmt) for m in metrics] for name, v in refs]


def delta_cell(value: float, base: float, ci: float, metric: str) -> str:
    info = METRICS[metric]
    if pd.isna(value):
        return "–"
    text = format(value, info.fmt)
    if pd.isna(base):
        return text
    delta = value - base
    digits = info.decimals
    if round(abs(delta), digits) == 0:
        return f"{text} (=)"
    better = (delta > 0) == info.higher_is_better
    ci_text = f" ±{ci:.{digits}f}" if not pd.isna(ci) else ""
    return f"{text} ({delta:+.{digits}f}{ci_text} {'▲' if better else '▼'})"


def gather_compare(args: argparse.Namespace, df: pd.DataFrame,
                   run_ids: list[str]) -> tuple[list[pd.Series], list[str], list[str]]:
    selected, combos, selected_on = [], [], []
    for run_id in run_ids:
        combo = args.combo
        rows = select_rows(df, run_id, args.set)
        if combo and combo not in set(rows["combo"]):
            raise SystemExit(f"Run {run_id} has no rows for combo {combo}.")
        rows, chosen = combo_rows(df, run_id, args.set, combo)
        row, chosen_on = pick_epoch_row(df, rows, args.epoch, args.metric_for_best, args.select_set, args.set,
                                        run_id, chosen)
        selected.append(row)
        combos.append(chosen)
        selected_on.append(chosen_on)
    return selected, combos, selected_on


def compare(args: argparse.Namespace, df: pd.DataFrame, out_dir: Path) -> None:
    others = [r.strip() for r in args.runs.split(",") if r.strip() and r.strip() != args.baseline]
    run_ids = [args.baseline] + others
    selected, combos, selected_on = gather_compare(args, df, run_ids)
    frame = pd.DataFrame(selected)
    metrics = present_metrics(frame, TABLE_METRICS)
    runs_dir, calibration_dir = repo_path(args.runs_dir), repo_path(args.calibration_dir)
    base = selected[0]
    track = None if pd.isna(base["track"]) else str(base["track"])
    clips = [clip_metrics(runs_dir, r, args.set, int(row["epoch"]), c, calibration_bounds(calibration_dir, row["track"]))
             for r, row, c in zip(run_ids, selected, combos)]

    header = ["run", "epoch", "n_clips"] + [f"{m} {METRICS[m].hint[0]}" for m in metrics]
    body = []
    for i, (run_id, row) in enumerate(zip(run_ids, selected)):
        cells = [f"**{run_id}** (baseline)" if i == 0 else run_id, str(int(row["epoch"])), fmt_value(row["n_clips"], ".0f")]
        for m in metrics:
            if i == 0:
                ci = metric_ci(clips[0], m)
                text = fmt_value(row[m], METRICS[m].fmt)
                cells.append(text + (f" ±{ci:.{METRICS[m].decimals}f}" if not pd.isna(ci) else ""))
            else:
                cells.append(delta_cell(row[m], base[m], paired_delta_ci(clips[0], clips[i], m), m))
        body.append(cells)
    refs = reference_rows(df, selected, metrics, calibration_dir, track, args.set)
    body += [[r[0], "", ""] + r[1:] for r in refs]

    epoch_note = {
        "best": selection_note(args.metric_for_best, args.set, selected_on),
        "last": "each run's last evaluated epoch",
    }.get(args.epoch, f"epoch {args.epoch} for every run")
    combo_note = ", ".join(sorted(set(combos)))
    text = [
        f"# Ablation vs {args.baseline} ({args.set})", "",
        markdown_table(header, body, ["l", "r", "r"] + ["r"] * len(metrics)), "",
        f"- Epoch: {epoch_note}.",
        f"- Inference setting: `{combo_note}`.",
        "- Cells show value (delta vs baseline). ▲ = better, ▼ = worse, (=) = unchanged at the shown precision, "
        "judged by each metric's direction "
        "(↑ higher / ↓ lower is better). ± = 95% CI over clips (paired by clip for deltas) when per-clip metrics "
        "exist: t-based for clip means, clip bootstrap for the corpus CER/ΔCER.",
        "- Reference rows: target bounds from the calibration file; source vocals = the unconverted input "
        "(ΔCER is 0 by definition, MOS columns are source scores).",
        *cer_reference_note(selected),
        f"- Reproduce: `{command_line()}`",
    ]
    name = args.name or f"compare_{args.baseline}_{args.set}"
    write_table(out_dir, name, "\n".join(text))
    compare_chart(args, run_ids, selected, clips, name, out_dir, selected_on)


def compare_chart(args: argparse.Namespace, run_ids: list[str], selected: list[pd.Series],
                  clips: list[pd.DataFrame | None], name: str, out_dir: Path, selected_on: list[str]) -> None:
    frame = pd.DataFrame(selected)
    metrics = present_metrics(frame, CHART_METRICS)
    if not metrics:
        print("warning: no chartable metrics; skipping the bar chart.")
        return
    ncols = min(3, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    width = max(7.6, 1.1 + ncols * max(2.75, 0.42 * len(run_ids) + 0.9))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 0.6 + 2.5 * nrows), squeeze=False,
                             gridspec_kw={"hspace": 0.95, "wspace": 0.38})
    positions = np.arange(len(run_ids))
    for ax, metric in zip(axes.flat, metrics):
        info = METRICS[metric]
        values = frame[metric].to_numpy(dtype=float)
        errors = np.array([metric_ci(c, metric) for c in clips])
        colors = [BASELINE_GRAY] + [info.color] * (len(run_ids) - 1)
        valid = np.isfinite(values)
        ax.bar(positions[valid], values[valid], width=0.68, color=[c for c, ok in zip(colors, valid) if ok],
               edgecolor=SURFACE, linewidth=2)
        has_err = valid & np.isfinite(errors)
        if has_err.any():
            ax.errorbar(positions[has_err], values[has_err], yerr=errors[has_err], fmt="none",
                        ecolor=INK_SECONDARY, elinewidth=1, capsize=2.5)
        if valid[0]:
            ax.axhline(values[0], color=INK_SECONDARY, linestyle="--", linewidth=0.9)
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        for x, v, e in zip(positions, values, np.nan_to_num(errors)):
            if math.isnan(v):
                ax.annotate("n/a", xy=(x, 0), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=6.5, color=INK_MUTED)
                continue
            tip = v + e if v >= 0 else v - e
            ax.annotate(format(v, info.fmt), xy=(x, tip), xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5, color=INK_SECONDARY)
        ax.set_xticks(positions, run_ids, rotation=35, ha="right", rotation_mode="anchor", fontsize=7.5)
        ax.margins(y=0.18)
        if not np.any(np.abs(values[valid]) + np.nan_to_num(errors[valid]) > 0):
            ax.set_ylim(0, 1)  # all zero: avoid a degenerate, offset-labelled axis
        panel_title(ax, metric, compact=True)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    epoch_text = {"best": best_epoch_text(args.metric_for_best, args.set, selected_on),
                  "last": "last epoch"}.get(args.epoch, f"epoch {args.epoch}")
    figure_title(fig, f"Ablation vs {run_ids[0]}", f"set {args.set} · {epoch_text} · ↑/↓ = direction of better · "
                 "gray = baseline, dashed = baseline value"
                 + (" · whiskers = 95% CI over clips" if any(c is not None for c in clips) else ""))
    fig.subplots_adjust(top=1 - 0.8 / fig.get_figheight())
    save_figure(fig, out_dir, name, args.formats, args.dpi)


def summary(args: argparse.Namespace, df: pd.DataFrame, out_dir: Path) -> None:
    on_set = df[df["set"] == args.set]
    if on_set.empty:
        raise SystemExit(f"No rows for set {args.set}.")
    selected, combos, selected_on = [], [], []
    for run_id in sorted(on_set["run_id"].dropna().unique()):
        rows = on_set[on_set["run_id"] == run_id]
        if args.combo and args.combo in set(rows["combo"]):
            chosen = args.combo
        elif args.combo:
            print(f"note: run {run_id} has no {args.combo} rows; using its own default combo.")
            chosen = pick_combo(rows, None, run_id)
        else:
            chosen = pick_combo(rows, None, run_id)
        rows = rows[rows["combo"] == chosen].drop_duplicates("epoch", keep="last")
        row, chosen_on = pick_epoch_row(df, rows, "best", args.metric_for_best, args.select_set, args.set,
                                        run_id, chosen)
        selected.append(row)
        combos.append(chosen)
        selected_on.append(chosen_on)
    frame = pd.DataFrame(selected)
    metrics = present_metrics(frame, TABLE_METRICS)
    show_combo = len(set(combos)) > 1
    header = ["run", "track", "best epoch", "step", "epochs evaluated"] + (["combo"] if show_combo else []) + ["n_clips"]
    header += [f"{m} {METRICS[m].hint[0]}" for m in metrics]
    body = []
    for row, combo in zip(selected, combos):
        evaluated = on_set[(on_set["run_id"] == row["run_id"]) & (on_set["combo"] == combo)]["epoch"].nunique()
        cells = [str(row["run_id"]), "" if pd.isna(row["track"]) else str(row["track"]), str(int(row["epoch"])),
                 fmt_value(row["step"], ".0f"), str(evaluated)]
        cells += [f"`{combo}`"] if show_combo else []
        cells += [fmt_value(row["n_clips"], ".0f")] + [fmt_value(row[m], METRICS[m].fmt) for m in metrics]
        body.append(cells)
    for track in sorted(frame["track"].dropna().unique()):
        track_rows = [r for r in selected if r["track"] == track]
        for ref in reference_rows(df, track_rows, metrics, repo_path(args.calibration_dir), str(track), args.set):
            body.append([f"{ref[0]}", str(track), "", "", ""] + ([""] if show_combo else []) + [""] + ref[1:])
    n_fixed = 6 if show_combo else 5
    text = [
        f"# All runs at their best epoch ({args.set})", "",
        markdown_table(header, body, ["l", "l"] + ["r"] * (n_fixed - 2) + ["r"] * (len(header) - n_fixed)), "",
        f"- Best epoch: {selection_note(args.metric_for_best, args.set, selected_on)}.",
        f"- Inference setting: `{', '.join(sorted(set(combos)))}`" + (" (per-run, see combo column)." if show_combo else "."),
        "- ↑ higher is better, ↓ lower is better.",
        *cer_reference_note(selected),
        f"- Reproduce: `{command_line()}`",
    ]
    write_table(out_dir, f"summary_{args.set}", "\n".join(text))


# ---------------------------------------------------------------- data-curve


def parse_minutes(text: str | None) -> dict[str, float]:
    overrides = {}
    for item in (text or "").split(","):
        if not item.strip():
            continue
        run_id, _, value = item.partition("=")
        try:
            overrides[run_id.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"--minutes expects run_id=minutes, got: {item}")
    return overrides


def data_curve(args: argparse.Namespace, df: pd.DataFrame, out_dir: Path) -> None:
    series_runs = [[r.strip() for r in group.split(",") if r.strip()] for group in args.runs]
    names = [n.strip() for n in args.series_names.split(",")] if args.series_names else []
    if names and len(names) != len(series_runs):
        raise SystemExit("--series-names needs one name per --runs.")
    names = names or [f"series {i + 1}" for i in range(len(series_runs))]
    overrides = parse_minutes(args.minutes)
    runs_dir, applio_logs = repo_path(args.runs_dir), repo_path(args.applio_logs)

    records, selected_on = [], []
    for name, run_ids in zip(names, series_runs):
        for run_id in run_ids:
            minutes = overrides.get(run_id) or dataset_minutes(run_id, args.minutes_from, runs_dir, applio_logs)
            if minutes is None:
                raise SystemExit(f"Dataset minutes unknown for {run_id}; pass --minutes {run_id}=<min>.")
            rows, chosen = combo_rows(df, run_id, args.set, args.combo)
            row, chosen_on = pick_epoch_row(df, rows, args.epoch, args.metric_for_best, args.select_set, args.set,
                                            run_id, chosen)
            selected_on.append(chosen_on)
            records.append({"series": name, "run_id": run_id, "minutes": minutes, **row.drop(labels=["run_id"]).to_dict()})
    frame = pd.DataFrame(records)
    metrics = present_metrics(frame, parse_metric_list(args.metrics))
    if not metrics:
        raise SystemExit("None of the requested metrics has values.")

    fig, axes = plt.subplots(1, len(metrics), figsize=(1.0 + 2.9 * len(metrics), 3.3), squeeze=False,
                             gridspec_kw={"wspace": 0.4})
    ticks = sorted(frame["minutes"].unique())
    for ax, metric in zip(axes.flat, metrics):
        info = METRICS[metric]
        for i, (name, group) in enumerate(frame.groupby("series", sort=False)):
            group = group.sort_values("minutes")
            color = info.color if len(names) == 1 else SLOTS[i % len(SLOTS)]
            ax.plot(group["minutes"], group[metric], color=color, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=name)
            if len(names) <= 4 and len(names) > 1:
                last = group.dropna(subset=[metric]).tail(1)
                if not last.empty:
                    ax.annotate(name, xy=(last["minutes"].iloc[0], last[metric].iloc[0]), xytext=(5, 0),
                                textcoords="offset points", fontsize=7, color=INK_SECONDARY, va="center")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("training data (minutes, log scale)")
        ax.margins(x=0.12, y=0.15)
        panel_title(ax, metric, compact=True)
    if len(names) > 1:
        axes.flat[0].legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncols=len(names))
    epoch_text = {"best": best_epoch_text(args.metric_for_best, args.set, selected_on),
                  "last": "last epoch"}.get(args.epoch, f"epoch {args.epoch}")
    figure_title(fig, "Quality vs amount of training data",
                 f"set {args.set} · {epoch_text} · ↑/↓ = direction of better")
    fig.subplots_adjust(top=1 - 0.8 / fig.get_figheight())
    name = args.name or f"e5_data_curve_{args.set}"
    save_figure(fig, out_dir, name, args.formats, args.dpi)

    header = ["series", "run", "minutes", "epoch", "n_clips"] + [f"{m} {METRICS[m].hint[0]}" for m in metrics]
    body = [[r["series"], r["run_id"], f"{r['minutes']:.1f}", str(int(r["epoch"])), fmt_value(r["n_clips"], ".0f")]
            + [fmt_value(r[m], METRICS[m].fmt) for m in metrics] for _, r in frame.iterrows()]
    text = [
        f"# Quality vs training data ({args.set})", "",
        markdown_table(header, body, ["l", "l", "r", "r", "r"] + ["r"] * len(metrics)), "",
        f"- Epoch: {selection_note(args.metric_for_best, args.set, selected_on) if args.epoch == 'best' else epoch_text}.",
        *cer_reference_note(frame),
        f"- Minutes from {args.minutes_from}" + (" with --minutes overrides." if overrides else "."),
        f"- Reproduce: `{command_line()}`",
    ]
    write_table(out_dir, name, "\n".join(text))


# ---------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    setup_style()
    df = load_runs(repo_path(args.runs_csv))
    out_dir = repo_path(args.out_dir)
    handlers = {
        "epoch-curve": epoch_curve,
        "pareto": pareto,
        "compare": compare,
        "data-curve": data_curve,
        "summary": summary,
    }
    handlers[args.command](args, df, out_dir)


if __name__ == "__main__":
    main()
