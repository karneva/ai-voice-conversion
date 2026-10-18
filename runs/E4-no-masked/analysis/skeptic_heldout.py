#!/usr/bin/env python3
"""Skeptic check: is E4's +0.19 dB held-out MCD regression stationary colour (tilt/EQ) or fine timbre?

Same frame pairing and loudness mask as scripts/eval_lib/mcd.mel_cepstral_distance; variants:
c1..c15 (stored), c4..c15, c1..c3, per-clip cepstral mean normalisation; plus out-minus-target
2-8 kHz energy share. Clip bootstrap and per-song split, pooled over epochs 200..500 (every 20).
Writes skeptic_heldout.json next to this file.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from eval_lib import audio, mcd  # noqa: E402

COMBO = "f0-rmvpe_ir0.30_p0.33_ve1.00"
EPOCHS = tuple(range(200, 501, 20))
RUNS = ("E1-base", "E4-no-masked")
cfg = mcd.McdConfig()
cache: dict = {}


def spectra(path: Path):
    if path not in cache:
        x = audio.load_16k(path)
        c, edb = mcd._spectra(x, cfg)
        import librosa
        S = np.abs(librosa.stft(x / (np.abs(x).max() + 1e-12), n_fft=512, hop_length=128, win_length=512,
                                center=False)) ** 2
        cache[path] = (c, edb, S)
    return cache[path]


def clip_metrics(ref_path: Path, out_path: Path):
    rc, rdb, rS = spectra(ref_path)
    oc, _, oS = spectra(out_path)
    n = min(rc.shape[1], oc.shape[1])
    m = rdb[:n] > rdb[:n].max() - cfg.top_db
    X, Y = rc[1:16, :n][:, m], oc[1:16, :n][:, m]
    d = X - Y
    dc = (X - X.mean(1, keepdims=True)) - (Y - Y.mean(1, keepdims=True))
    fr = np.fft.rfftfreq(512, 1 / 16000)
    k = min(rS.shape[1], oS.shape[1], n)
    mm = m[:k]

    def share(S):
        return 10 * np.log10(S[(fr >= 2000) & (fr < 8000)][:, :k][:, mm].sum() / S[:, :k][:, mm].sum())

    return {"std": float(np.linalg.norm(d, axis=0).mean()), "hi": float(np.linalg.norm(d[3:], axis=0).mean()),
            "lo": float(np.linalg.norm(d[:3], axis=0).mean()), "cmn": float(np.linalg.norm(dc, axis=0).mean()),
            "d_share_2_8k": float(share(oS) - share(rS))}


rows = {}
for run in RUNS:
    for e in EPOCHS:
        p = REPO / f"runs/{run}/metrics/heldout/{e}e/{COMBO}.csv"
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                met = clip_metrics(REPO / r["reference_path"], REPO / r["output_path"])
                assert abs(met["std"] - float(r["mcd"])) < 1e-3, (run, e, r["clip_id"], met["std"], r["mcd"])
                rows[(run, e, r["clip_id"])] = met
    print(run, "done", flush=True)

clips = sorted({k[2] for k in rows})
song = np.array(["_".join(c.split("_")[:-2]) for c in clips])
rng = np.random.default_rng(11)
out = {"n_clips": len(clips), "songs": sorted(set(song))}
for var in ("std", "hi", "lo", "cmn", "d_share_2_8k"):
    D = np.array([[rows[("E4-no-masked", e, c)][var] - rows[("E1-base", e, c)][var] for e in EPOCHS] for c in clips])
    per_clip = D.mean(1)
    res = {"E1_mean": round(float(np.mean([rows[("E1-base", e, c)][var] for e in EPOCHS for c in clips])), 4),
           "E4_mean": round(float(np.mean([rows[("E4-no-masked", e, c)][var] for e in EPOCHS for c in clips])), 4),
           "d": round(float(per_clip.mean()), 4)}
    for name, sel in [("all", np.ones(len(clips), bool))] + [(s, song == s) for s in sorted(set(song))]:
        v = per_clip[sel]
        bs = v[rng.integers(0, len(v), size=(5000, len(v)))].mean(1)
        res[name] = {"d": round(float(v.mean()), 4),
                     "ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    res["by_epoch"] = {e: round(float(D[:, i].mean()), 3) for i, e in enumerate(EPOCHS)}
    out[var] = res
(HERE / "skeptic_heldout.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
