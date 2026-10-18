#!/usr/bin/env python3
"""Skeptic checks on the E4-no-masked vs E1-base claims (CPU only, read-only on existing results).

1. MCD on the gold reference: is the E4 gain timbre, or tilt / level / bandwidth?
   Same DTW pairs and mask as eval_reference.py; variants: c1..c15 (stored), c4..c15 (tilt removed),
   c1..c3 (tilt only), per-region cepstral mean normalisation (removes any stationary EQ).
   Bootstrap unit: 6 s blocks (theirs: 3 s) + leave-one-region-out jackknife range.
2. Level / brightness / bandwidth of the converted outputs (LUFS, 2-8 kHz share, centroid, >8 kHz share).
3. Speaker similarity with a different encoder: WavLM-base-plus-sv (offline HF cache), same 82 windows;
   paired cos to Kyuhyun, cos to leave-out Kyuhyun centroid, and cos to Tim (source leakage).
4. H1-H2 with a different F0 tracker (RMVPE cache) and plain FFT peak picking, all 6 epochs,
   step-matched pairs, plus a low-band control (A3-C#4).
Writes skeptic.json next to this file.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import eval_reference as er  # noqa: E402

RUNS = ("E1-base", "E4-no-masked")
COMBO = "f0-rmvpe_ir0.30_p0.33_ve1.00"
EPOCHS = (20, 100, 200, 300, 400, 500)
POOL = (200, 300, 400, 500)
STEP_PAIRS = [(200, 100), (300, 200), (400, 200), (500, 300)]  # (E4 epoch, nearest-step E1 epoch)
B = 4000
BLOCK_S = 6.0
SEED = 7


def load(path: Path) -> er.Track:
    t = er.load_track(path)
    data = np.load(er.f0_cache_path(t.sha256))
    t.f0 = er.pitch.F0Track(f0=data["f0"], voiced=data["voiced"])
    return t


def ci(x):
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]


baseline = json.loads((REPO / "runs/E1-base/reference/baseline.json").read_text(encoding="utf-8"))
alignment = json.loads((REPO / "runs/E1-base/reference/alignment.json").read_text(encoding="utf-8"))
regions = er.regions_from(alignment)
rby = {r.index: r for r in regions}
cfg = er.ScoreConfig(**baseline["score_config"])
windows = baseline["windows"]
ref = load(REPO / baseline["reference"])
src = load(REPO / baseline["source"])
outs = {(run, e): load(REPO / f"runs/{run}/reference/{e}e/{COMBO}/converted.wav") for run in RUNS for e in EPOCHS}
tracks = {"src": src, **outs}
OUT: dict = {}


def blk(region_idx, t):
    return (region_idx, int(max(0.0, t - rby[region_idx].ref_start) // BLOCK_S))


# ------------------------------------------------------------------ 1. MCD variants
mc = er.mcd.McdConfig()
hop, off = er.mcep_grid()
er.fill_mcep(ref)
loud = ref.energy_db > float(ref.energy_db.max()) - mc.top_db
frames = {}
for key, out in tracks.items():
    er.fill_mcep(out)
    rows = {"std": [], "hi": [], "lo": [], "cmn": [], "dc1": [], "blk": [], "reg": []}
    for region in regions:
        pi, pj = er.region_pairs(region, ref, out, cfg)
        ti, tj = pi * hop + off, pj * hop + off
        m = ref.voiced_at(ti) & out.voiced_at(tj) & loud[pi]
        pi, pj, ti = pi[m], pj[m], ti[m]
        if not len(pi):
            continue
        X, Y = ref.mcep[1:16, pi], out.mcep[1:16, pj]
        d = X - Y
        rows["std"].append(np.linalg.norm(d, axis=0))
        rows["hi"].append(np.linalg.norm(d[3:], axis=0))
        rows["lo"].append(np.linalg.norm(d[:3], axis=0))
        dc = (X - X.mean(1, keepdims=True)) - (Y - Y.mean(1, keepdims=True))
        rows["cmn"].append(np.linalg.norm(dc, axis=0))
        rows["dc1"].append(Y[0] - X[0])
        rows["blk"] += [blk(region.index, float(t)) for t in ti]
        rows["reg"].append(np.full(len(pi), region.index))
    frames[key] = {k: (np.concatenate(v) if k != "blk" else v) for k, v in rows.items()}

units = sorted({b for f in frames.values() for b in f["blk"]})
uidx = {u: i for i, u in enumerate(units)}
U = len(units)
rng = np.random.default_rng(SEED)
W = rng.multinomial(U, np.full(U, 1 / U), size=B).astype(float)


def usum(key, var):
    f = frames[key]
    s, n = np.zeros(U), np.zeros(U)
    for b, v in zip(f["blk"], f[var]):
        s[uidx[b]] += v
        n[uidx[b]] += 1
    return s, n


def jack(key4, key1, var):
    vals = []
    for r in rby:
        a = frames[key4]
        b = frames[key1]
        vals.append(a[var][a["reg"] != r].mean() - b[var][b["reg"] != r].mean())
    return [round(float(min(vals)), 4), round(float(max(vals)), 4)]


mcd_res = {"n_blocks6s": U, "tim_source": {v: round(float(frames["src"][v].mean()), 4)
                                            for v in ("std", "hi", "lo", "cmn", "dc1")}}
for var in ("std", "hi", "lo", "cmn", "dc1"):
    S = {k: usum(k, var) for k in outs}
    res = {"abs": {f"{r[:2]}_{e}": round(float(frames[(r, e)][var].mean()), 4) for r, e in outs}}

    def pair(e4, e1):
        s4, n4 = S[("E4-no-masked", e4)]
        s1, n1 = S[("E1-base", e1)]
        pt = s4.sum() / n4.sum() - s1.sum() / n1.sum()
        bs = (W @ s4) / (W @ n4) - (W @ s1) / (W @ n1)
        return pt, bs

    for e in EPOCHS:
        pt, bs = pair(e, e)
        res[f"same_{e}"] = {"d": round(float(pt), 4), "ci": ci(bs),
                            "loo_region": jack(("E4-no-masked", e), ("E1-base", e), var)}
    for name, pairs in (("same_pooled200_500", [(e, e) for e in POOL]), ("steps_pooled", STEP_PAIRS)):
        got = [pair(a, b) for a, b in pairs]
        res[name] = {"d": round(float(np.mean([g[0] for g in got])), 4),
                     "ci": ci(np.mean([g[1] for g in got], axis=0))}
    for a, b in STEP_PAIRS:
        pt, bs = pair(a, b)
        res[f"steps_{a}v{b}"] = {"d": round(float(pt), 4), "ci": ci(bs)}
    mcd_res[var] = res
OUT["mcd"] = mcd_res
print("MCD done", flush=True)

# ------------------------------------------------------------------ 2. level / brightness / bandwidth
import pyloudnorm as pyln  # noqa: E402
import soundfile as sf  # noqa: E402


def aligned_mask_src(n_frames, hop_s):
    t = np.arange(n_frames) * hop_s
    m = np.zeros(n_frames, bool)
    for r in regions:
        m |= (t >= r.src_start) & (t <= r.src_end)
    return m


def spec_stats(path: Path, f0track, aligned: bool, region_src=None):
    x, sr = sf.read(str(path), always_2d=True)
    x = x.mean(1)
    lufs = pyln.Meter(sr).integrated_loudness(x)
    n_fft, hp = 2048, int(sr * 0.01)
    import librosa
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hp)) ** 2
    fr = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    e = S.sum(0)
    edb = 10 * np.log10(e + 1e-12)
    vt = np.round(np.arange(S.shape[1]) * hp / sr / er.F0_HOP_S).astype(int)
    v = np.zeros(S.shape[1], bool)
    ok = vt < len(f0track.voiced)
    v[ok] = f0track.voiced[vt[ok]]
    m = v & (edb > edb.max() - 35)
    if aligned:
        m &= aligned_mask_src(S.shape[1], hp / sr) if region_src is None else region_src(S.shape[1], hp / sr)
    tot = S[:, m].sum(0)

    def share(lo, hi):
        return float(np.median(10 * np.log10(S[(fr >= lo) & (fr < hi)][:, m].sum(0) / tot + 1e-12)))

    cen = float(np.median((fr[:, None] * S[:, m]).sum(0) / tot))
    return {"lufs": round(float(lufs), 2), "rms_dbfs_voiced": round(float(np.median(edb[m])), 2),
            "share_2_8k_db": round(share(2000, 8000), 2), "share_4_8k_db": round(share(4000, 8000), 2),
            "share_gt8k_db": round(share(8000, sr / 2), 2), "centroid_hz": round(cen, 1), "sr": sr}


lvl = {}
for key, t in tracks.items():
    name = "tim" if key == "src" else f"{key[0][:2]}_{key[1]}"
    lvl[name] = spec_stats(t.path, t.f0, True)
lvl["cover_full"] = spec_stats(ref.path, ref.f0, False)
OUT["level_brightness_aligned_regions"] = lvl
print("level done", flush=True)

# ------------------------------------------------------------------ 3. WavLM-SV similarity
import torch  # noqa: E402
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector  # noqa: E402

snap = Path.home() / ".cache/huggingface/hub/models--microsoft--wavlm-base-plus-sv/snapshots"
mdir = next(p for p in snap.iterdir() if (p / "config.json").exists())
fe = Wav2Vec2FeatureExtractor.from_pretrained(str(mdir), local_files_only=True)
model = WavLMForXVector.from_pretrained(str(mdir), local_files_only=True).eval()
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))


def embed(track, centers):
    out = []
    for i in range(0, len(centers), 16):
        batch = [track.window(c, cfg.window_s) for c in centers[i:i + 16]]
        inp = fe(batch, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = model(**inp).embeddings
        out.append(torch.nn.functional.normalize(emb, dim=-1).numpy())
    return np.concatenate(out)


rc = [w["ref_center"] for w in windows]
sc = [w["src_center"] for w in windows]
E_ref = embed(ref, rc)
E_src = embed(src, sc)
rca = np.asarray(rc)
cent = np.stack([E_ref[np.abs(rca - c) >= cfg.window_s].mean(0) for c in rca])
cent /= np.linalg.norm(cent, axis=1, keepdims=True)
wblk = [blk(w["region"], w["ref_center"] - cfg.window_s / 2) for w in windows]
wu = sorted(set(wblk))
wi = np.array([wu.index(b) for b in wblk])
Uw = len(wu)
Ww = rng.multinomial(Uw, np.full(Uw, 1 / Uw), size=B).astype(float)
onehot = np.zeros((len(windows), Uw))
onehot[np.arange(len(windows)), wi] = 1
cnt = onehot.sum(0)
sv = {"tim": {"paired": round(float((E_src * E_ref).sum(1).mean()), 4),
              "centroid": round(float((E_src * cent).sum(1).mean()), 4)},
      "self_ref_centroid": round(float((E_ref * cent).sum(1).mean()), 4)}
cos = {}
for key, t in outs.items():
    E = embed(t, sc)
    cos[key] = {"paired": (E * E_ref).sum(1), "centroid": (E * cent).sum(1), "to_tim": (E * E_src).sum(1)}
    sv[f"{key[0][:2]}_{key[1]}"] = {k: round(float(v.mean()), 4) for k, v in cos[key].items()}
ecapa = {key: np.asarray(json.loads((outs[key].path.parent / "scores.json").read_text(encoding="utf-8"))
                         ["detail"]["window_cos"]) for key in outs}


def wpair(e4, e1, getter):
    a, b = getter(("E4-no-masked", e4)), getter(("E1-base", e1))
    s = (a - b) @ onehot
    return float((a - b).mean()), (Ww @ s) / (Ww @ cnt)


for kind in ("paired", "centroid", "to_tim", "ecapa_paired"):
    get = (lambda k: ecapa[k]) if kind == "ecapa_paired" else (lambda k, kind=kind: cos[k][kind])
    res = {}
    for e in EPOCHS:
        pt, bs = wpair(e, e, get)
        res[f"same_{e}"] = {"d": round(pt, 4), "ci": ci(bs)}
    for name, pairs in (("same_pooled200_500", [(e, e) for e in POOL]), ("steps_pooled", STEP_PAIRS)):
        got = [wpair(a, b, get) for a, b in pairs]
        res[name] = {"d": round(float(np.mean([g[0] for g in got])), 4), "ci": ci(np.mean([g[1] for g in got], 0))}
    sv[f"delta_{kind}"] = res
# rank agreement between encoders across the 12 outputs
keys = list(outs)
a = np.array([cos[k]["paired"].mean() for k in keys])
b = np.array([ecapa[k].mean() for k in keys])
from scipy.stats import spearmanr  # noqa: E402
sv["spearman_wavlm_vs_ecapa_12_outputs"] = round(float(spearmanr(a, b)[0]), 3)
sv["window_corr_wavlm_vs_ecapa_pooled"] = round(float(np.corrcoef(
    np.concatenate([cos[k]["paired"] for k in keys]), np.concatenate([ecapa[k] for k in keys]))[0, 1]), 3)
OUT["wavlm_sv"] = sv
print("wavlm done", flush=True)

# ------------------------------------------------------------------ 4. H1-H2 (RMVPE F0 + FFT peaks)
import librosa  # noqa: E402

BANDS = {"A3-C#4": (57, 61.5), "D4-E4": (61.5, 64.5), "F4-G4": (64.5, 68)}
NFFT, WIN = 4096, 800  # 50 ms hann at 16 kHz, zero-padded


def h1h2(track):
    x = track.wav
    hp = 160
    S = np.abs(librosa.stft(x, n_fft=NFFT, hop_length=hp, win_length=WIN, center=True))
    fr = librosa.fft_frequencies(sr=16000, n_fft=NFFT)
    n = min(S.shape[1], len(track.f0.f0))
    f0 = track.f0.f0[:n]
    v = track.f0.voiced[:n] & (f0 >= 150)
    edb = 20 * np.log10(np.linalg.norm(S[:, :n], axis=0) + 1e-9)
    thr = np.percentile(edb[v], 30) if v.any() else 0
    v &= edb > thr
    out = np.full(n, np.nan)
    for k in np.where(v)[0]:
        f = f0[k]

        def pk(h):
            sel = (fr >= h * f * 0.94) & (fr <= h * f * 1.06)
            return 20 * np.log10(S[sel, k].max() + 1e-9)
        out[k] = pk(1) - pk(2)
    midi = np.full(n, np.nan)
    midi[v] = 69 + 12 * np.log2(f0[v] / 440)
    return out, midi


H = {k: h1h2(t) for k, t in tracks.items()}
Href = h1h2(ref)
nT = len(H["src"][0])


def fit(a):
    return np.pad(a, (0, max(0, nT - len(a))), constant_values=np.nan)[:nT]


H = {k: (fit(a), fit(b)) for k, (a, b) in H.items()}
tt = np.arange(nT) * 0.01
REG = {"whole": np.ones(nT, bool), "02:29-03:23": (tt >= 149) & (tt < 203),
       "verse1_76.9-88.5": (tt >= 76.9) & (tt < 88.5), "aligned": aligned_mask_src(nT, 0.01)}
tim_midi = H["src"][1]
hres = {"tim_median": {}, "cover_median_whole": {}}
for bn, (lo, hi) in BANDS.items():
    mb = (Href[1] >= lo) & (Href[1] < hi)
    hres["cover_median_whole"][bn] = round(float(np.nanmedian(Href[0][mb])), 2) if mb.any() else None
    for rn, rm in REG.items():
        m = rm & (tim_midi >= lo) & (tim_midi < hi)
        hres["tim_median"][f"{rn}|{bn}"] = round(float(np.nanmedian(H["src"][0][m])), 2) if m.sum() > 20 else None
# cover median on aligned ref regions
tr = np.arange(len(Href[0])) * 0.01
ref_al = np.zeros(len(tr), bool)
for r in regions:
    ref_al |= (tr >= r.ref_start) & (tr <= r.ref_end)
hres["cover_median_aligned"] = {bn: round(float(np.nanmedian(Href[0][ref_al & (Href[1] >= lo) & (Href[1] < hi)])), 2)
                                for bn, (lo, hi) in BANDS.items()}
hres["abs_median_by_tim_band"] = {}
HB = 2.0  # bootstrap block (s)
for key in outs:
    name = f"{key[0][:2]}_{key[1]}"
    for rn, rm in REG.items():
        for bn, (lo, hi) in BANDS.items():
            m = rm & (tim_midi >= lo) & (tim_midi < hi) & np.isfinite(H[key][0][:nT])
            hres["abs_median_by_tim_band"][f"{name}|{rn}|{bn}"] = round(float(np.nanmedian(H[key][0][:nT][m])), 2) \
                if m.sum() > 20 else None


def hpair(e4, e1, rn, bn):
    lo, hi = BANDS[bn]
    a, b = H[("E4-no-masked", e4)][0][:nT], H[("E1-base", e1)][0][:nT]
    m = REG[rn] & (tim_midi >= lo) & (tim_midi < hi) & np.isfinite(a) & np.isfinite(b)
    d = (a - b)[m]
    if len(d) < 30:
        return None
    blocks = (tt[m] // HB).astype(int)
    ub, inv = np.unique(blocks, return_inverse=True)
    s = np.bincount(inv, weights=d)
    c = np.bincount(inv).astype(float)
    Wb = rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=B).astype(float)
    bs = (Wb @ s) / (Wb @ c)
    return {"d_mean": round(float(d.mean()), 2), "d_median": round(float(np.median(d)), 2), "ci_mean": ci(bs),
            "n_frames": int(len(d)), "n_blocks": int(len(ub))}


pairs = {}
for rn in REG:
    for bn in BANDS:
        for e in EPOCHS:
            pairs[f"{rn}|{bn}|same_{e}"] = hpair(e, e, rn, bn)
        for a4, a1 in STEP_PAIRS:
            pairs[f"{rn}|{bn}|steps_{a4}v{a1}"] = hpair(a4, a1, rn, bn)
hres["paired_E4_minus_E1"] = pairs
OUT["h1h2_rmvpe_fft"] = hres

(HERE / "skeptic.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", HERE / "skeptic.json")
