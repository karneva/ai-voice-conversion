#!/usr/bin/env python3
"""Paired E4-no-masked minus E1-base statistics (CPU only, read-only on existing results).

Reference (gold = Kyuhyun's real partial cover of 사랑합니다, 82 overlapping 3 s windows, 8 regions):
  * sim_ref: per-window ECAPA cosines from scores.json (same 82 windows in both runs).
  * mcd_ref: frame-level MCD recomputed from converted.wav with scripts/eval_reference.py's own
    functions (region_pairs + get_MCD_k + the same voicing/loudness mask); the per-region means are
    checked against the stored scores.json values. Pooled MCD = mean over all kept frames.
  * Block bootstrap (B=10000): resampling units are 3 s blocks on the reference time axis
    (= 3 consecutive 1 s-hop windows; MCD frames by the same 3 s span), and, as a coarser check,
    the 8 regions. The same resample is used for sim and MCD and across epochs, so pooled
    (epochs >= 200) estimates carry the cross-epoch correlation.
  * Equal optimizer steps: E4 epoch e vs the E1 checkpoint with the nearest step count.
General sets (source_eval, heldout): paired by clip_id, clip-level bootstrap, metrics aggregated
the same way as scripts/evaluate.py (means; corpus CER), pooled over epochs 200..500 (every 20).

Writes stats.json and stats.md next to this file.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import eval_reference as er  # noqa: E402
from mel_cepstral_distance.computation import get_MCD_k  # noqa: E402

RUNS = ("E1-base", "E4-no-masked")
COMBO = "f0-rmvpe_ir0.30_p0.33_ve1.00"
REF_EPOCHS = (20, 100, 200, 300, 400, 500)
POOL_EPOCHS = (200, 300, 400, 500)
GEN_EPOCHS = tuple(range(200, 501, 20))
B = 10000
BLOCK_S = 3.0
SEED = 20261018


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v: str | None) -> float:
    try:
        return float(v) if v not in (None, "") else math.nan
    except ValueError:
        return math.nan


def steps_per_epoch() -> dict[str, int]:
    out = {}
    for row in read_csv(REPO / "results" / "runs.csv"):
        if row["run_id"] in RUNS and row["step"] and row["epoch"]:
            out[row["run_id"]] = round(int(row["step"]) / int(row["epoch"]))
    return out


def ci(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def p_two_sided(samples: np.ndarray) -> float:
    """Bootstrap two-sided p for H0: delta = 0 (share of resamples on the other side, doubled)."""
    lo, hi = float(np.mean(samples <= 0)), float(np.mean(samples >= 0))
    return float(min(1.0, 2 * min(lo, hi)))


def summarize(point: float, samples: np.ndarray) -> dict[str, float | list[float]]:
    return {"delta": float(point), "ci95": ci(samples), "p_boot": p_two_sided(samples),
            "se_boot": float(np.std(samples, ddof=1))}


# ---------------------------------------------------------------- reference: data


def frame_mcd(regions, ref, out, cfg) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per region: (ref frame time s, MCD per kept DTW pair). Mirrors er.spectral_scores."""
    mc = er.mcd.McdConfig()
    hop, off = er.mcep_grid()
    er.fill_mcep(ref)
    er.fill_mcep(out)
    loud = ref.energy_db > float(ref.energy_db.max()) - mc.top_db
    res = []
    for region in regions:
        pi, pj = er.region_pairs(region, ref, out, cfg)
        ti, tj = pi * hop + off, pj * hop + off
        mask = ref.voiced_at(ti) & out.voiced_at(tj) & loud[pi]
        pi, pj, ti = pi[mask], pj[mask], ti[mask]
        values = get_MCD_k(ref.mcep[:, pi], out.mcep[:, pj], mc.s, mc.D) if len(pi) else np.zeros(0)
        res.append((ti, np.asarray(values, dtype=np.float64)))
    return res


def load_tracks_with_f0(paths: list[Path]):
    tracks = [er.load_track(p) for p in paths]
    for t in tracks:
        cached = er.f0_cache_path(t.sha256)
        if not cached.exists():
            raise SystemExit(f"no cached F0 for {er.record_path(t.path)}; refusing to run RMVPE here")
        data = np.load(cached)
        t.f0 = er.pitch.F0Track(f0=data["f0"], voiced=data["voiced"])
    return tracks


def reference_data() -> dict:
    baseline = json.loads((REPO / "runs/E1-base/reference/baseline.json").read_text(encoding="utf-8"))
    alignment = json.loads((REPO / "runs/E1-base/reference/alignment.json").read_text(encoding="utf-8"))
    al4 = json.loads((REPO / "runs/E4-no-masked/reference/alignment.json").read_text(encoding="utf-8"))
    assert alignment["key"] == al4["key"] == baseline["alignment_key"]
    regions = er.regions_from(alignment)
    cfg = er.ScoreConfig(**baseline["score_config"])
    windows = baseline["windows"]
    region_by_idx = {r.index: r for r in regions}

    # block keys: (region, k) with k = floor((t - region start) / 3 s); windows by their span start
    def block_of(region_idx: int, t_start: float) -> tuple[int, int]:
        r = region_by_idx[region_idx]
        return region_idx, int(max(0.0, t_start - r.ref_start) // BLOCK_S)

    win_blocks = [block_of(w["region"], w["ref_center"] - cfg.window_s / 2) for w in windows]
    win_regions = [w["region"] for w in windows]

    ref_path = REPO / baseline["reference"]
    paths = [ref_path]
    for run in RUNS:
        for e in REF_EPOCHS:
            paths.append(REPO / f"runs/{run}/reference/{e}e/{COMBO}/converted.wav")
    tracks = load_tracks_with_f0(paths)
    ref = tracks[0]
    data = {"window_blocks": win_blocks, "window_regions": win_regions, "runs": {}}
    check = []
    i = 1
    for run in RUNS:
        data["runs"][run] = {}
        for e in REF_EPOCHS:
            out = tracks[i]
            i += 1
            scores = json.loads((out.path.parent / "scores.json").read_text(encoding="utf-8"))["detail"]
            assert len(scores["window_cos"]) == len(windows)
            per_region = frame_mcd(regions, ref, out, cfg)
            frames_t, frames_v, frames_reg = [], [], []
            for region, (t, v), stored in zip(regions, per_region, scores["regions"]):
                assert stored["index"] == region.index
                if stored["mcd"] is not None:
                    check.append(abs(float(v.mean()) - stored["mcd"]))
                    assert stored["n_frames"] == len(v), (run, e, region.index)
                frames_t.append(t)
                frames_v.append(v)
                frames_reg.append(np.full(len(v), region.index))
            t = np.concatenate(frames_t)
            v = np.concatenate(frames_v)
            reg = np.concatenate(frames_reg)
            assert abs(float(v.mean()) - scores["mcd_ref"]) < 1e-6, (run, e, float(v.mean()), scores["mcd_ref"])
            fblocks = [block_of(int(rg), float(tt)) for rg, tt in zip(reg, t)]
            data["runs"][run][e] = {
                "window_cos": np.asarray(scores["window_cos"], dtype=np.float64),
                "sim_ref_stored": scores["sim_ref"],
                "mcd_ref_stored": scores["mcd_ref"],
                "frame_mcd": v, "frame_block": fblocks, "frame_region": reg,
                "region_mcd": {r["index"]: r["mcd"] for r in scores["regions"]},
            }
            out.mcep = None  # keep memory small
            out.wav = None
            out.energy_db = None
    data["max_region_mcd_abs_diff"] = float(max(check))
    return data


# ---------------------------------------------------------------- reference: bootstrap


def unit_sums(keys: list, values: np.ndarray, units: list) -> tuple[np.ndarray, np.ndarray]:
    idx = {u: j for j, u in enumerate(units)}
    s = np.zeros(len(units))
    n = np.zeros(len(units))
    for k, v in zip(keys, values):
        j = idx[k]
        s[j] += v
        n[j] += 1
    return s, n


def reference_stats(data: dict, spe: dict[str, int]) -> dict:
    rng = np.random.default_rng(SEED)
    results = {}
    for unit_name in ("block3s", "region"):
        if unit_name == "block3s":
            win_keys = data["window_blocks"]
            units = sorted(set(win_keys) | {b for run in RUNS for e in REF_EPOCHS
                                            for b in data["runs"][run][e]["frame_block"]})
        else:
            win_keys = data["window_regions"]
            units = sorted(set(win_keys))
        U = len(units)
        counts = rng.multinomial(U, np.full(U, 1.0 / U), size=B).astype(np.float64)  # (B, U)
        ones = np.ones(U)

        sums = {}
        for run in RUNS:
            for e in REF_EPOCHS:
                d = data["runs"][run][e]
                ws, wn = unit_sums(win_keys, d["window_cos"], units)
                fkeys = d["frame_block"] if unit_name == "block3s" else [int(r) for r in d["frame_region"]]
                ms, mn = unit_sums(fkeys, d["frame_mcd"], units)
                sums[(run, e)] = (ws, wn, ms, mn)

        def metric(run: str, e: int, which: str, w: np.ndarray) -> np.ndarray | float:
            ws, wn, ms, mn = sums[(run, e)]
            s, n = (ws, wn) if which == "sim" else (ms, mn)
            return (w @ s) / (w @ n)

        def pair(e4: int, e1: int, which: str) -> tuple[float, np.ndarray]:
            pt = metric("E4-no-masked", e4, which, ones) - metric("E1-base", e1, which, ones)
            bs = metric("E4-no-masked", e4, which, counts) - metric("E1-base", e1, which, counts)
            return float(pt), bs

        res = {"n_units": U, "same_epoch": {}, "equal_steps": {}}
        pooled = {"sim": [], "mcd": []}
        for e in REF_EPOCHS:
            row = {}
            for which, name in (("sim", "d_sim_ref"), ("mcd", "d_mcd_ref")):
                pt, bs = pair(e, e, which)
                row[name] = summarize(pt, bs)
                if e in POOL_EPOCHS:
                    pooled[which].append((pt, bs))
            res["same_epoch"][e] = row
        res["same_epoch_pooled_ge200"] = {
            name: summarize(np.mean([p for p, _ in pooled[w]]), np.mean([b for _, b in pooled[w]], axis=0))
            for w, name in (("sim", "d_sim_ref"), ("mcd", "d_mcd_ref"))}

        # equal steps: each E4 checkpoint vs the E1 checkpoint with the nearest step count
        pooled = {"sim": [], "mcd": []}
        for e4 in REF_EPOCHS:
            s4 = e4 * spe["E4-no-masked"]
            e1 = min(REF_EPOCHS, key=lambda x: abs(x * spe["E1-base"] - s4))
            row = {"e1_epoch": e1, "e4_step": s4, "e1_step": e1 * spe["E1-base"],
                   "step_ratio_e4_over_e1": s4 / (e1 * spe["E1-base"])}
            for which, name in (("sim", "d_sim_ref"), ("mcd", "d_mcd_ref")):
                pt, bs = pair(e4, e1, which)
                row[name] = summarize(pt, bs)
                if e4 in POOL_EPOCHS:
                    pooled[which].append((pt, bs))
            res["equal_steps"][e4] = row
        res["equal_steps_pooled_e4_ge200"] = {
            name: summarize(np.mean([p for p, _ in pooled[w]]), np.mean([b for _, b in pooled[w]], axis=0))
            for w, name in (("sim", "d_sim_ref"), ("mcd", "d_mcd_ref"))}
        results[unit_name] = res

    # descriptive: per-region sign consistency and per-window win rate (same epoch)
    desc = {}
    regions = sorted(data["runs"]["E1-base"][20]["region_mcd"])
    for e in REF_EPOCHS:
        a, b = data["runs"]["E4-no-masked"][e], data["runs"]["E1-base"][e]
        dreg = {r: a["region_mcd"][r] - b["region_mcd"][r] for r in regions}
        dwin = a["window_cos"] - b["window_cos"]
        desc[e] = {
            "e1": {"sim_ref": b["sim_ref_stored"], "mcd_ref": b["mcd_ref_stored"]},
            "e4": {"sim_ref": a["sim_ref_stored"], "mcd_ref": a["mcd_ref_stored"]},
            "regions_e4_lower_mcd": int(sum(v < 0 for v in dreg.values())),
            "region_d_mcd": {str(r): round(v, 4) for r, v in dreg.items()},
            "windows_e4_higher_cos": int(np.sum(dwin > 0)),
            "n_windows": int(len(dwin)),
        }
    results["descriptive"] = desc
    return results


# ---------------------------------------------------------------- general sets


GEN_METRICS = {
    "source_eval": ("sim_target", "dcer", "f0_rmse_cents", "singmos"),
    "heldout": ("sim_target", "dcer", "f0_rmse_cents", "singmos", "mcd"),
}


def load_clips(run: str, set_name: str, epoch: int) -> dict[str, dict[str, float]]:
    rows = read_csv(REPO / f"runs/{run}/metrics/{set_name}/{epoch}e/{COMBO}.csv")
    out = {}
    for r in rows:
        chars = fnum(r["ref_chars"])
        out[r["clip_id"]] = {
            "sim_target": fnum(r["sim_target"]), "f0_rmse_cents": fnum(r["f0_rmse_cents"]),
            "singmos": fnum(r["singmos"]), "mcd": fnum(r["mcd"]),
            "edits": fnum(r["cer"]) * chars, "edits_src": fnum(r["cer_source"]) * chars, "chars": chars,
        }
    return out


def gen_aggregate(arr: dict[str, np.ndarray], metric: str, w: np.ndarray) -> np.ndarray:
    """arr[...] shape (clips,), w shape (B, clips) or (clips,). Mirrors evaluate.aggregate."""
    if metric == "dcer":
        chars = np.nan_to_num(arr["chars"])
        cer = (w @ np.nan_to_num(arr["edits"])) / (w @ chars)
        src = (w @ np.nan_to_num(arr["edits_src"])) / (w @ chars)
        return cer - src
    v = arr[metric]
    ok = ~np.isnan(v)
    return (w @ np.where(ok, v, 0.0)) / (w @ ok.astype(np.float64))


def general_stats(spe: dict[str, int]) -> dict:
    rng = np.random.default_rng(SEED + 1)
    res = {}
    for set_name, metrics in GEN_METRICS.items():
        cache = {(run, e): load_clips(run, set_name, e) for run in RUNS for e in range(20, 501, 20)}
        clips = sorted(set.intersection(*[set(c) for c in cache.values()]))
        n = len(clips)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
        ones = np.ones(n)

        def arrays(run: str, e: int) -> dict[str, np.ndarray]:
            c = cache[(run, e)]
            keys = c[clips[0]].keys()
            return {k: np.array([c[cl][k] for cl in clips]) for k in keys}

        def pooled_delta(pairs: list[tuple[int, int]], sel: np.ndarray | None = None) -> dict:
            out = {}
            w_pt = ones if sel is None else sel.astype(np.float64)
            if sel is None:
                w_bs = counts
            else:  # resample clips within the selection only
                k = int(sel.sum())
                w_bs = np.zeros((B, n))
                w_bs[:, sel] = rng.multinomial(k, np.full(k, 1.0 / k), size=B)
            for m in metrics:
                pts, bss, e1v, e4v = [], [], [], []
                for e4, e1 in pairs:
                    a4, a1 = arrays("E4-no-masked", e4), arrays("E1-base", e1)
                    p4, p1 = gen_aggregate(a4, m, w_pt), gen_aggregate(a1, m, w_pt)
                    pts.append(p4 - p1)
                    e4v.append(p4)
                    e1v.append(p1)
                    bss.append(gen_aggregate(a4, m, w_bs) - gen_aggregate(a1, m, w_bs))
                s = summarize(np.mean(pts), np.mean(bss, axis=0))
                s.update({"e1_mean": float(np.mean(e1v)), "e4_mean": float(np.mean(e4v))})
                out[m] = s
            return out

        same = [(e, e) for e in GEN_EPOCHS]
        equal = []
        for e4 in GEN_EPOCHS:
            s4 = e4 * spe["E4-no-masked"]
            e1 = min(range(20, 501, 20), key=lambda x: abs(x * spe["E1-base"] - s4))
            equal.append((e4, e1))
        per_epoch = {e: pooled_delta([(e, e)]) for e in POOL_EPOCHS}
        res[set_name] = {
            "n_clips": n,
            "pooled_same_epoch_200_500": pooled_delta(same),
            "pooled_equal_steps_e4_200_500": pooled_delta(equal),
            "equal_steps_pairs": equal,
            "per_epoch": per_epoch,
        }
        songs = sorted({c.rsplit("_", 2)[0] for c in clips})
        if set_name == "heldout" and len(songs) > 1:
            res[set_name]["per_song_same_epoch_200_500"] = {
                song: {"n_clips": int(sum(c.rsplit("_", 2)[0] == song for c in clips)),
                       **pooled_delta(same, np.array([c.rsplit("_", 2)[0] == song for c in clips]))}
                for song in songs}
    return res


# ---------------------------------------------------------------- output


def fmt_ci(s: dict, digits: int, signed: bool = True) -> str:
    sp = f"+.{digits}f" if signed else f".{digits}f"
    lo, hi = s["ci95"]
    star = "*" if (lo > 0 or hi < 0) else ""
    return f"{s['delta']:{sp}} [{lo:{sp}}, {hi:{sp}}]{star}"


def write_md(ref: dict, gen: dict, spe: dict, check: float) -> str:
    L = []
    L.append("# E4-no-masked vs E1-base: paired statistics")
    L.append("")
    L.append("Generated by `runs/E4-no-masked/analysis/stats.py` (CPU, B = 10000 bootstrap resamples, "
             "percentile 95% CIs). Delta = E4 - E1. `*` = CI excludes 0. "
             "Lower MCD is better (closer to the real singer); higher sim is better.")
    L.append("")
    L.append(f"Steps per epoch: E1 {spe['E1-base']}, E4 {spe['E4-no-masked']} "
             f"(ratio {spe['E4-no-masked'] / spe['E1-base']:.3f}). Frame-level MCD recomputed with "
             f"eval_reference.py functions; max |region mean - stored| = {check:.1e} dB, pooled values "
             "match scores.json.")
    L.append("")
    L.append("## Reference (gold = real cover, 82 windows / 8 regions, index rate 0.3)")
    L.append("")
    L.append("Primary: block bootstrap over 3 s blocks of the reference timeline "
             f"({ref['block3s']['n_units']} blocks; 3 consecutive 1 s-hop windows each, MCD frames by "
             f"the same span). Check: region bootstrap ({ref['region']['n_units']} regions, coarse).")
    L.append("")
    L.append("### Same epoch")
    L.append("")
    L.append("| epoch | E1 MCD | E4 MCD | dMCD [95% CI, 3 s blocks] | dMCD [CI, regions] | regions E4 better | "
             "E1 sim | E4 sim | dsim [CI, 3 s blocks] | dsim [CI, regions] | windows E4 higher |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in REF_EPOCHS:
        b, r, d = ref["block3s"]["same_epoch"][e], ref["region"]["same_epoch"][e], ref["descriptive"][e]
        L.append(f"| {e} | {d['e1']['mcd_ref']:.3f} | {d['e4']['mcd_ref']:.3f} | {fmt_ci(b['d_mcd_ref'], 3)} | "
                 f"{fmt_ci(r['d_mcd_ref'], 3)} | {d['regions_e4_lower_mcd']}/8 | {d['e1']['sim_ref']:.4f} | "
                 f"{d['e4']['sim_ref']:.4f} | {fmt_ci(b['d_sim_ref'], 4)} | {fmt_ci(r['d_sim_ref'], 4)} | "
                 f"{d['windows_e4_higher_cos']}/{d['n_windows']} |")
    b, r = ref["block3s"]["same_epoch_pooled_ge200"], ref["region"]["same_epoch_pooled_ge200"]
    L.append(f"| **pooled 200-500** | | | **{fmt_ci(b['d_mcd_ref'], 3)}** | {fmt_ci(r['d_mcd_ref'], 3)} | | | | "
             f"**{fmt_ci(b['d_sim_ref'], 4)}** | {fmt_ci(r['d_sim_ref'], 4)} | |")
    L.append("")
    L.append("### Equal optimizer steps (E4 epoch vs nearest-step E1 checkpoint among 20/100/200/300/400/500)")
    L.append("")
    L.append("| E4 epoch (step) | E1 epoch (step) | step ratio | dMCD [CI, 3 s blocks] | dMCD [CI, regions] | "
             "dsim [CI, 3 s blocks] | dsim [CI, regions] |")
    L.append("|---|---|---|---|---|---|---|")
    for e in REF_EPOCHS:
        b, r = ref["block3s"]["equal_steps"][e], ref["region"]["equal_steps"][e]
        L.append(f"| {e} ({b['e4_step']}) | {b['e1_epoch']} ({b['e1_step']}) | {b['step_ratio_e4_over_e1']:.2f} | "
                 f"{fmt_ci(b['d_mcd_ref'], 3)} | {fmt_ci(r['d_mcd_ref'], 3)} | {fmt_ci(b['d_sim_ref'], 4)} | "
                 f"{fmt_ci(r['d_sim_ref'], 4)} |")
    b, r = ref["block3s"]["equal_steps_pooled_e4_ge200"], ref["region"]["equal_steps_pooled_e4_ge200"]
    L.append(f"| **pooled E4 200-500** | | | **{fmt_ci(b['d_mcd_ref'], 3)}** | {fmt_ci(r['d_mcd_ref'], 3)} | "
             f"**{fmt_ci(b['d_sim_ref'], 4)}** | {fmt_ci(r['d_sim_ref'], 4)} |")
    L.append("")
    L.append("Note: E1 400/500 (62800/78500 steps) have no E4 counterpart (E4 ends at 47500 steps); "
             "E4 500 vs E1 300 is the closest match (47500 vs 47100 steps).")
    L.append("")
    L.append("## General eval sets (paired by clip_id, clip bootstrap, mean of per-epoch deltas)")
    L.append("")
    names = {"sim_target": ("dsim_target", 4), "dcer": ("dCER (= dCER-vs-source)", 4),
             "f0_rmse_cents": ("dF0 RMSE (cents)", 2), "singmos": ("dSingMOS", 3), "mcd": ("dMCD (dB)", 3)}
    for set_name, g in gen.items():
        L.append(f"### {set_name} ({g['n_clips']} clips)")
        L.append("")
        L.append("| metric | E1 (mean 200-500) | E4 (mean 200-500) | delta same epoch, pooled 200-500 [CI] | "
                 "delta equal steps, pooled E4 200-500 [CI] | " + " | ".join(f"delta @{e}" for e in POOL_EPOCHS) + " |")
        L.append("|---|---|---|---|---|" + "---|" * len(POOL_EPOCHS))
        for m in GEN_METRICS[set_name]:
            label, dg = names[m]
            s, q = g["pooled_same_epoch_200_500"][m], g["pooled_equal_steps_e4_200_500"][m]
            per = " | ".join(fmt_ci(g["per_epoch"][e][m], dg) for e in POOL_EPOCHS)
            L.append(f"| {label} | {s['e1_mean']:.{dg}f} | {s['e4_mean']:.{dg}f} | {fmt_ci(s, dg)} | "
                     f"{fmt_ci(q, dg)} | {per} |")
        L.append("")
        if "per_song_same_epoch_200_500" in g and set_name == "heldout":
            L.append("Per song (same epoch, pooled 200-500, clip bootstrap within song):")
            L.append("")
            L.append("| song | clips | " + " | ".join(names[m][0] for m in GEN_METRICS[set_name]) + " |")
            L.append("|---|---|" + "---|" * len(GEN_METRICS[set_name]))
            for song, sg in g["per_song_same_epoch_200_500"].items():
                L.append(f"| {song} | {sg['n_clips']} | "
                         + " | ".join(fmt_ci(sg[m], names[m][1]) for m in GEN_METRICS[set_name]) + " |")
            L.append("")
            L.append("Held-out clips come from only 2 songs, so the clip bootstrap treats correlated clips as "
                     "independent; the per-song split shows whether a pooled effect is driven by one song.")
            L.append("")
    L.append("Equal-step pairs for the general sets: E4 epoch e vs E1 epoch nearest to e x "
             f"{spe['E4-no-masked']}/{spe['E1-base']} (checkpoints every 20 epochs).")
    L.append("")
    L.append("Caveats: one song / one gold reference (82 windows from about 109 s), so the reference result "
             "is about this song; overlapping windows and adjacent blocks are still mildly dependent; "
             "8-region CIs are coarse (few clusters). Many deltas are tested, no multiplicity correction; "
             "read single `*` on the general sets with that in mind. E4 trains on 60% of the steps per epoch.")
    L.append("")
    return "\n".join(L)


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main() -> int:
    spe = steps_per_epoch()
    print("steps/epoch", spe, flush=True)
    data = reference_data()
    print("region MCD recompute max abs diff", data["max_region_mcd_abs_diff"], flush=True)
    ref = reference_stats(data, spe)
    gen = general_stats(spe)
    out = {"config": {"B": B, "block_s": BLOCK_S, "seed": SEED, "combo": COMBO, "ref_epochs": REF_EPOCHS,
                      "pool_epochs_reference": POOL_EPOCHS, "general_epochs": GEN_EPOCHS,
                      "steps_per_epoch": spe, "delta": "E4-no-masked minus E1-base",
                      "region_mcd_recompute_max_abs_diff": data["max_region_mcd_abs_diff"]},
           "reference": ref, "general": gen}
    (HERE / "stats.json").write_text(json.dumps(to_jsonable(out), indent=1, ensure_ascii=False), encoding="utf-8")
    md = write_md(ref, gen, spe, data["max_region_mcd_abs_diff"])
    (HERE / "stats.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
