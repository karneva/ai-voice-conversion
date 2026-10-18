"""Second-opinion spot checks (CPU): limiter leak in recovery, source-brightness leak, 복면가왕 floor content,
crepe-periodicity cross-check of the voicing claims in A and D."""
import os, sys, glob, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np, soundfile as sf, librosa
from scipy.stats import spearmanr
R = r"C:\DEV\ai-voice-conversion"; D = os.path.join(R, r"runs\E1-base\diagnosis"); C = os.path.join(D, "_cache")
sys.path.insert(0, D); sys.path.insert(0, os.path.join(R, "scripts", "data"))
import mix_cover
from common import LIVE
out = {}

# 1) limiter gain reduction during the reconstructed mix (does vocal-free -40 dB residual cover loud passages?)
inst, sr = sf.read(glob.glob(R + r"\dataset\private\separated\cover_input\*(other)*.wav")[0], dtype="float64")
meta = json.load(open(R + r"\dataset\private\covers\사랑합니다__E1-base_500e_t+0.json", encoding="utf8"))
est, _ = sf.read(os.path.join(C, "conv_500e_est.wav"), dtype="float64")
g = 10 ** (meta["mix"]["vocal_level_match_db"] / 20)
n = min(len(est), len(inst))
mix = (g * est[:n, None] + inst[:n]) * 0.80535
peak = np.abs(mix).max(axis=1); ceil = 10 ** (-1.0 / 20)
need = np.minimum(1, ceil / np.maximum(peak, 1e-12))
gr_db = 20 * np.log10(need)
t = np.arange(n) / sr
inC = (t >= 149) & (t <= 203)
out["limiter"] = {"frac_samples_reducing": float((need < 1).mean()), "frac_in_C": float((need[inC] < 1).mean()),
                  "min_gain_db": float(gr_db.min())}
mix_mp3, _ = sf.read(R + r"\dataset\private\covers\사랑합니다__E1-base_500e_t+0.mp3", dtype="float64")
# leak test: in 0.5 s windows inside C, regress recovered vocal on instrumental (should be ~0)
im = inst[:n].mean(1); b = []
for s in np.arange(149, 203, 0.5):
    a_, z_ = int(s * sr), int((s + 0.5) * sr)
    x, y = im[a_:z_], est[a_:z_]
    b.append(float((x * y).sum() / ((x * x).sum() + 1e-12)))
out["limiter"]["inst_leak_coef_C_median"] = float(np.median(b)); out["limiter"]["inst_leak_coef_C_p95abs"] = float(np.percentile(np.abs(b), 95))

# 2) does converted HF tilt track the SOURCE stem's HF tilt frame-by-frame (pitch partialled)?
def F(name):
    z = np.load(os.path.join(C, f"feat_{name}.npz")); return {k: z[k] for k in z.files}
S, V, Rf = F("src"), F("c500"), F("ref")
T = min(len(S["f0"]), len(V["f0"]))
m = 69 + 12 * np.log2(np.where(S["f0"][:T] > 0, S["f0"][:T], 1) / 440)
sel = (S["f0"][:T] > 0) & (V["f0"][:T] > 0) & (m >= 61.5) & (S["rms_db"][:T] > np.percentile(S["rms_db"][S["f0"] > 0], 30))
def partial(a, b, c):
    ra = a - np.polyval(np.polyfit(c, a, 1), c); rb = b - np.polyval(np.polyfit(c, b, 1), c)
    return float(spearmanr(ra, rb).correlation)
out["src_leak"] = {k: {"rho_conv_vs_src_pitch_partialled": partial(V[k][:T][sel], S[k][:T][sel], m[sel])}
                   for k in ("hf_4_8k_db", "alpha_db", "h1h2")}
# low-pitch control
sel_lo = (S["f0"][:T] > 0) & (V["f0"][:T] > 0) & (m < 61.5) & (m > 50) & (S["rms_db"][:T] > np.percentile(S["rms_db"][S["f0"] > 0], 30))
for k in ("hf_4_8k_db", "alpha_db", "h1h2"):
    out["src_leak"][k]["rho_low_notes"] = partial(V[k][:T][sel_lo], S[k][:T][sel_lo], m[sel_lo])

# 3) 복면가왕 loud unvoiced frames: what are they? flatness and proximity to high notes
res = {}
for p in sorted(glob.glob(R + r"\dataset\private\train\*.wav")):
    stem = os.path.basename(p)[:-4]
    grp = "mbk" if any(stem.startswith(x) for x in LIVE) else ("kv" if stem.startswith("킬링보이스") else "studio")
    z = F("train__" + stem)
    f0, rms = z["f0"], z["rms_db"]
    if (f0 > 0).sum() < 100: continue
    vmed = np.median(rms[f0 > 0])
    loud_uv = (f0 == 0) & (rms > vmed - 6)
    mm = 69 + 12 * np.log2(np.where(f0 > 0, f0, 1) / 440)
    hi = (f0 > 0) & (mm >= 64.5)
    # within 1 s after a high note
    hi_after = np.convolve(hi.astype(float), np.r_[np.zeros(100), np.ones(100)], mode="same") > 0
    r = res.setdefault(grp, {"loud_uv_s": 0.0, "voiced_s": 0.0, "flat": [], "near_hi": 0.0, "dur": [], "hf": []})
    r["loud_uv_s"] += loud_uv.sum() * 0.01; r["voiced_s"] += (f0 > 0).sum() * 0.01
    r["flat"] += list(z["flat_1_8k"][loud_uv]); r["hf"] += list(z["hf_6_16k_db"][loud_uv]); r["near_hi"] += (loud_uv & hi_after).sum() * 0.01
    # run lengths
    d = np.diff(np.r_[0, loud_uv.astype(int), 0]); s0 = np.flatnonzero(d == 1); e0 = np.flatnonzero(d == -1)
    r["dur"] += list((e0 - s0) * 0.01)
for grp, r in res.items():
    out.setdefault("loud_unvoiced", {})[grp] = {
        "loud_uv_per_voiced_min_s": r["loud_uv_s"] / (r["voiced_s"] / 60), "median_flat_1_8k": float(np.median(r["flat"])) if r["flat"] else None,
        "median_hf_6_16k": float(np.median(r["hf"])) if r["hf"] else None,
        "share_within_1s_after_high_note": r["near_hi"] / max(r["loud_uv_s"], 1e-9),
        "runs_longer_than_300ms": int(np.sum(np.array(r["dur"]) > 0.3)), "longest_run_s": float(max(r["dur"])) if r["dur"] else 0}

# 4) crepe periodicity cross-check of the voicing claims (A: 70.8-71.3 s; D: unvoiced-loud frames)
P = {k: F(k) for k in ("src", "srcmp3", "c500", "c300")}
def win(name, a, b, key):
    z = P[name]; i = (z["t"] >= a) & (z["t"] < b); return float(np.mean(z[key][i]))
out["A_window_70.8_71.3"] = {k: {"crepe_per_mean": win(k, 70.8, 71.3, "crepe_per"), "frac_per>0.5": float(np.mean(P[k]["crepe_per"][(P[k]["t"] >= 70.8) & (P[k]["t"] < 71.3)] > 0.5))} for k in P}
Ts = min(len(P[k]["t"]) for k in P)
src = P["src"]; act = src["rms_db"][:Ts] > np.nanmax(src["rms_db"]) - 30
uvl = act & (src["crepe_per"][:Ts] < 0.3) & (src["f0"][:Ts] == 0)
out["D_unvoiced_loud_src_frames"] = {"n": int(uvl.sum())}
for k in ("srcmp3", "c300", "c500"):
    out["D_unvoiced_loud_src_frames"][k + "_crepe_per>0.5"] = float(np.mean(P[k]["crepe_per"][:Ts][uvl] > 0.5))
    out["D_unvoiced_loud_src_frames"][k + "_harvest_voiced"] = float(np.mean(P[k]["f0"][:Ts][uvl] > 0))
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(os.path.join(D, "skeptic_misc_stats.json"), "w", encoding="utf8"), indent=1, ensure_ascii=False)
