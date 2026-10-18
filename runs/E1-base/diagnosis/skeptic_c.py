"""Second-opinion check for item C with features NOT used by the first report.

Direct harmonic-peak measures from a 4096-pt FFT (not CheapTrick / D4C / flatness):
  h1h2_pk : H1-H2 from actual spectral peaks
  hnr48   : harmonic-peak vs inter-harmonic-valley level in 4-8 kHz (higher = more periodic / buzzy, lower = noisier)
  hrel48  : mean harmonic-peak level 4-8 kHz relative to the strongest of H1..H3 (harmonic tilt)
  floor   : level of non-vocal frames relative to the voiced median (crowd/bleed residue)
Pitch-binned so groups are compared at matched pitch. CPU only.
"""
import os, sys, glob, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np, librosa
R = r"C:\DEV\ai-voice-conversion"; D = os.path.join(R, r"runs\E1-base\diagnosis"); C = os.path.join(D, "_cache")
sys.path.insert(0, D)
from common import LIVE

SR = 44100; NF = 4096
FR = np.fft.rfftfreq(NF, 1 / SR)
BINS = {"A3-C#4": (57, 61.5), "D4-E4": (61.5, 64.5), "F4-G4": (64.5, 68)}


def group(stem):
    if any(stem.startswith(p) for p in LIVE): return "mbk"          # 복면가왕
    if stem.startswith("킬링보이스"): return "kv"
    if any(stem.startswith(s) for s in ("너였다면", "사건의", "희재")): return "gyutv"
    return "snow_mv"


def measure(y, f0, stride=1):
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=NF, hop_length=441, center=True))
    T = min(S.shape[1], len(f0)); S = S[:, :T]; f0 = f0[:T]
    P = 20 * np.log10(S + 1e-9)
    tot = 10 * np.log10((S ** 2).sum(0) + 1e-12)
    out = {k: np.full(T, np.nan) for k in ("h1h2_pk", "hnr48", "hrel48")}
    df = FR[1]
    for t in np.flatnonzero(f0 > 0)[::stride]:
        f = f0[t]
        if f < 150: continue
        col = P[:, t]
        def pk(fq, w=0.04):
            a, b = int(fq * (1 - w) / df), int(np.ceil(fq * (1 + w) / df)) + 1
            return col[a:b].max()
        def vl(fq, w=0.12):
            a, b = int(fq * (1 - w) / df), int(np.ceil(fq * (1 + w) / df)) + 1
            return np.percentile(col[a:b], 20)
        H = [pk(k * f) for k in (1, 2, 3)]
        out["h1h2_pk"][t] = H[0] - H[1]
        ks = np.arange(int(np.ceil(4000 / f)), int(8000 / f) + 1)
        if len(ks) < 3: continue
        hp = np.array([pk(k * f) for k in ks]); vv = np.array([vl((k + 0.5) * f) for k in ks])
        out["hnr48"][t] = 10 * np.log10(np.mean(10 ** (hp / 10))) - 10 * np.log10(np.mean(10 ** (vv / 10)))
        out["hrel48"][t] = 10 * np.log10(np.mean(10 ** (hp / 10))) - max(H)
    out["tot"] = tot
    return out


def load_f0(name):
    z = np.load(os.path.join(C, f"feat_{name}.npz")); return z["f0"], z["rms_db"]


def summarize(M, f0, rms, mask_extra=None):
    m = 69 + 12 * np.log2(np.where(f0 > 0, f0, 1) / 440.0)
    loud = rms > np.nanpercentile(rms[f0 > 0], 30) if (f0 > 0).any() else rms > -1e9
    res = {}
    for b, (lo, hi) in BINS.items():
        sel = (f0 > 0) & (m >= lo) & (m < hi) & loud
        if mask_extra is not None: sel &= mask_extra
        r = {"n": int(np.isfinite(M["hnr48"][sel]).sum())}
        for k in ("h1h2_pk", "hnr48", "hrel48"):
            v = M[k][sel]; v = v[np.isfinite(v)]
            r[k] = float(np.median(v)) if len(v) > 20 else None
        res[b] = r
    # noise floor: unvoiced frames within 45 dB of max, relative to voiced median level
    act = rms > np.nanmax(rms) - 45
    uv = (f0 == 0) & act
    res["floor_rel_db"] = float(np.median(rms[uv]) - np.median(rms[f0 > 0])) if uv.sum() > 50 else None
    res["unvoiced_active_frac"] = float(uv.sum() / max(1, act.sum()))
    return res


def main():
    out = {}
    song = {"src": os.path.join(C, "src_vocals.wav"), "srcmp3": os.path.join(C, "src_mp3_est.wav"),
            "c300": os.path.join(C, "conv_300e_est.wav"), "c500": os.path.join(C, "conv_500e_est.wav"),
            "ref": glob.glob(R + r"\dataset\private\separated\reference\*(vocals)*.wav")[0]}
    perframe = {}
    for name, p in song.items():
        y, _ = librosa.load(p, sr=SR, mono=True)
        f0, rms = load_f0(name if name != "srcmp3" else "srcmp3")
        M = measure(y, f0)
        perframe[name] = (M, f0, rms)
        out[name] = summarize(M, f0, rms)
        print(name, json.dumps(out[name]), flush=True)
    # training groups
    G = {}
    for p in sorted(glob.glob(R + r"\dataset\private\train\*.wav")):
        stem = os.path.basename(p)[:-4]
        f0, rms = load_f0("train__" + stem)
        y, _ = librosa.load(p, sr=SR, mono=True)
        M = measure(y, f0, stride=2)
        G.setdefault(group(stem), []).append((M, f0, rms))
    for g, lst in G.items():
        Mc = {k: np.concatenate([x[0][k] for x in lst]) for k in ("h1h2_pk", "hnr48", "hrel48")}
        f0c = np.concatenate([x[1][:len(x[0]["hnr48"])] for x in lst])
        # per-file rms normalised to its own voiced median so 'loud' gate is fair
        rmsc = np.concatenate([x[2][:len(x[0]["hnr48"])] - np.median(x[2][x[1] > 0]) for x in lst])
        out["train_" + g] = summarize(Mc, f0c, rmsc)
        out["train_" + g]["floor_rel_db_per_file"] = [summarize(x[0], x[1][:len(x[0]['hnr48'])], x[2][:len(x[0]['hnr48'])])["floor_rel_db"] for x in lst]
        print(g, json.dumps(out["train_" + g]), flush=True)

    # source-artifact test: frame-level, high notes, conv500 excess vs source covariates
    Ms, f0s, rmss = perframe["src"]; Mc5, f0c5, _ = perframe["c500"]
    T = min(len(f0s), len(f0c5))
    inst, _ = librosa.load(os.path.join(C, "inst_mono.wav"), sr=SR, mono=True)
    Si = np.abs(librosa.stft(inst, n_fft=2048, hop_length=441))[:, :T]
    fr2 = librosa.fft_frequencies(sr=SR, n_fft=2048)
    inst48 = 10 * np.log10((Si[(fr2 >= 4000) & (fr2 < 8000)] ** 2).sum(0) + 1e-12)
    st = np.load(os.path.join(C, "stereo.npz"))
    # eM/eS: mid/side energies at 0.1 s; side/mid ratio as a reverb/width proxy
    sm = st["eS"] - st["eM"]  # already dB
    sm10 = np.repeat(sm, 10)[:T]; sm10 = np.pad(sm10, (0, T - len(sm10)), constant_values=np.nan)
    m = 69 + 12 * np.log2(np.where(f0s[:T] > 0, f0s[:T], 1) / 440)
    sel = (f0s[:T] > 0) & (f0c5[:T] > 0) & (m >= 61.5) & (rmss[:T] > np.percentile(rmss[f0s > 0], 30))
    from scipy.stats import spearmanr
    reg = {}
    for k in ("hnr48", "hrel48", "h1h2_pk"):
        d = Mc5[k][:T] - Ms[k][:T]
        ok = sel & np.isfinite(d) & np.isfinite(sm10)
        cov = {"midi": m[ok], "src_" + k: Ms[k][:T][ok], "inst48_rel_src": inst48[ok] - Ms["tot"][:T][ok],
               "side_minus_mid_db": sm10[ok], "src_level": rmss[:T][ok]}
        reg[k] = {"n": int(ok.sum()), "median_delta": float(np.median(d[ok])),
                  "spearman": {c: round(float(spearmanr(d[ok], v).correlation), 3) for c, v in cov.items()}}
        X = np.column_stack([np.ones(ok.sum())] + [(v - v.mean()) / (v.std() + 1e-9) for v in cov.values()])
        beta, *_ = np.linalg.lstsq(X, d[ok], rcond=None)
        pred = X @ beta; r2 = 1 - ((d[ok] - pred) ** 2).sum() / ((d[ok] - d[ok].mean()) ** 2).sum()
        reg[k]["std_beta"] = {c: round(float(b), 3) for c, b in zip(cov, beta[1:])}; reg[k]["r2"] = round(float(r2), 3)
        # inside vs outside C at matched pitch
        tt = np.arange(T) * 0.01
        inC = (tt >= 149) & (tt <= 203)
        for nm, msk in (("inC", ok & inC), ("outC", ok & ~inC)):
            reg[k][nm] = float(np.median(d[msk])) if msk.sum() > 20 else None
    out["source_artifact_regression"] = reg
    print(json.dumps(reg, indent=1), flush=True)
    json.dump(out, open(os.path.join(D, "skeptic_C_stats.json"), "w", encoding="utf8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
