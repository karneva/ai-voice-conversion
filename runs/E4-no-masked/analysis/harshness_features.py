"""Frame features for the E1-vs-E4 high-note harshness comparison. CPU only.

Re-uses the E1 diagnosis feature code unchanged (imported read-only from runs/E1-base/diagnosis):
  features.compute     -> harvest F0, D4C aperiodicity bands, envelope H1-H2, 4-8 kHz / 6-16 kHz rel. energy, alpha, crepe periodicity
  skeptic_c.measure    -> peak H1-H2, harmonic-to-valley ratio 4-8 kHz (hnr48), harmonic tilt 4-8 kHz vs H1..H3 (hrel48)
plus one extra band (2-8 kHz energy relative to 50-16k total), computed exactly like features.hf_4_8k_db.

Inputs
  runs/<run>/reference/<E>e/f0-rmvpe_ir0.30_p0.33_ve1.00/converted.wav   (index-rate 0.3, 40 kHz; lag-corrected to the source)
  dataset/private/covers/사랑합니다__E4-no-masked_<E>e_t+0.mp3          (index-rate 0.75 cover; vocals recovered like recover.py)
  runs/E1-base/diagnosis/_cache/{src_vocals,conv_300e_est,conv_500e_est}.wav + reference vocals (feature npz re-used)
Outputs (git-ignored): runs/E4-no-masked/analysis/_cache/{feat_*.npz, extra_*.npz, rec_*.wav}
"""
import os, sys, glob, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
import numpy as np
from multiprocessing import Pool

R = r"C:\DEV\ai-voice-conversion"
DIAG = os.path.join(R, r"runs\E1-base\diagnosis")
DC = os.path.join(DIAG, "_cache")
A = os.path.join(R, r"runs\E4-no-masked\analysis")
C = os.path.join(A, "_cache")
COMBO = "f0-rmvpe_ir0.30_p0.33_ve1.00"
SR = 44100
RUNS = {"E1": "E1-base", "E4": "E4-no-masked"}
EPOCHS = (300, 400, 500)


def stft_extra(y):
    """2-8 kHz energy rel. total, same STFT / bands as features.compute (n_fft 2048, hop 441)."""
    import librosa
    D = np.abs(librosa.stft(y.astype(np.float32), n_fft=2048, hop_length=441, center=True))
    fr = librosa.fft_frequencies(sr=SR, n_fft=2048)
    P = D ** 2
    def bandE(lo, hi):
        m = (fr >= lo) & (fr < hi)
        return P[m].sum(axis=0) + 1e-12
    return 10 * np.log10(bandE(2000, 8000) / bandE(50, 16000))


def estimate_lag(src, y, max_ms=60):
    """Lag (samples, y relative to src) from the cross-correlation of 1 kHz amplitude envelopes."""
    def env(x):
        k = SR // 1000
        n = len(x) // k
        return np.abs(x[: n * k]).reshape(n, k).mean(axis=1)
    a, b = env(src), env(y)
    n = min(len(a), len(b)); a = a[:n] - a[:n].mean(); b = b[:n] - b[:n].mean()
    best, bl = -np.inf, 0
    for l in range(-max_ms, max_ms + 1):
        if l >= 0:
            c = (a[: n - l] * b[l:]).sum()
        else:
            c = (a[-l:] * b[: n + l]).sum()
        if c > best:
            best, bl = c, l
    return bl * (SR // 1000)


def shift(y, lag, n):
    """Return y advanced by lag samples (lag>0: y is late), length n."""
    if lag > 0:
        y = y[lag:]
    elif lag < 0:
        y = np.pad(y, (-lag, 0))
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def recover_cover(mp3, meta_path, out_wav):
    """recover.py method: voc_est = (mix/a - inst)/g, a fitted on vocal-free frames, lag by xcorr."""
    import soundfile as sf
    inst, sr = sf.read(glob.glob(R + r"\dataset\private\separated\cover_input\*(other)*.wav")[0], dtype="float64")
    voc_src, _ = sf.read(glob.glob(R + r"\dataset\private\separated\cover_input\*(vocals)*.wav")[0], dtype="float64")
    meta = json.load(open(meta_path, encoding="utf8"))
    mix, msr = sf.read(mp3, dtype="float64")
    assert msr == sr
    g = 10 ** (meta["mix"]["vocal_level_match_db"] / 20)
    m = mix[:, 0]; x = inst[:, 0]
    L = 2 ** 22
    Af = np.fft.rfft(m[:L] if len(m) >= L else np.pad(m, (0, L - len(m))))
    Bf = np.fft.rfft(x[:L] if len(x) >= L else np.pad(x, (0, L - len(x))))
    cc = np.fft.irfft(Af * np.conj(Bf)); cc = np.concatenate([cc[-5000:], cc[:5001]])
    lag = int(np.argmax(cc)) - 5000
    if lag >= 0:
        mm = mix[lag:]; ii = inst[: len(mm)]
    else:
        mm = np.pad(mix, ((-lag, 0), (0, 0))); ii = inst
    n = min(len(mm), len(ii)); mm, ii = mm[:n], ii[:n]
    vs = voc_src[:n] if len(voc_src) >= n else np.pad(voc_src, ((0, n - len(voc_src)), (0, 0)))
    hop = 4410; k = n // hop
    ve = np.sqrt((vs[: k * hop] ** 2).mean(axis=1).reshape(k, hop).mean(axis=1))
    mask = np.repeat(ve < np.percentile(ve, 15), hop)
    mq = mm[: k * hop][mask]; iq = ii[: k * hop][mask]
    a = float((mq * iq).sum() / (iq * iq).sum())
    res = mq - a * iq
    est = ((mm / a - ii) / g).mean(axis=1)
    sf.write(out_wav, est.astype(np.float32), sr)
    return dict(lag=lag, a_fit=a, g=g, quiet_residual_db=float(10 * np.log10((res ** 2).mean() / (mq ** 2).mean())))


def job(args):
    name, path, kind = args
    sys.path.insert(0, DIAG)
    import librosa, soundfile as sf
    import features, skeptic_c
    info = {"name": name, "path": os.path.relpath(path, R), "kind": kind}
    fpath = os.path.join(C, f"feat_{name}.npz")
    if kind == "cached":  # feature npz already computed by the E1 diagnosis
        f = dict(np.load(os.path.join(DC, f"feat_{name}.npz")))
        y, _ = librosa.load(path, sr=SR, mono=True)
    else:
        if kind == "recover":
            rec = os.path.join(C, f"rec_{name}.wav")
            if not os.path.exists(rec):
                info["recover"] = recover_cover(path, path[:-4] + ".json", rec)
            path = rec
        y = features.load_mono(path, SR)
        src = features.load_mono(os.path.join(DC, "src_vocals.wav"), SR)
        lag = estimate_lag(src, y)
        info["lag_samples"] = int(lag)
        y = shift(y, lag, len(src))
        if os.path.exists(fpath):
            f = dict(np.load(fpath))
        else:
            f = features.compute(y, with_salience=False, with_crepe=True)
            np.savez_compressed(fpath, **f)
    epath = os.path.join(C, f"extra_{name}.npz")
    if not os.path.exists(epath):
        M = skeptic_c.measure(y, f["f0"])
        hf28 = stft_extra(y)
        T = len(f["f0"])
        hf28 = hf28[:T] if len(hf28) >= T else np.pad(hf28, (0, T - len(hf28)), constant_values=np.nan)
        np.savez_compressed(epath, h1h2_pk=M["h1h2_pk"], hnr48=M["hnr48"], hrel48=M["hrel48"], hf_2_8k_db=hf28)
    return info


if __name__ == "__main__":
    ref = glob.glob(R + r"\dataset\private\separated\reference\*(vocals)*.wav")[0]
    jobs = [("src", os.path.join(DC, "src_vocals.wav"), "cached"),
            ("ref", ref, "cached"),
            ("c300", os.path.join(DC, "conv_300e_est.wav"), "cached"),   # E1 300e, index-rate 0.75 (recovered)
            ("c500", os.path.join(DC, "conv_500e_est.wav"), "cached")]   # E1 500e, index-rate 0.75 (recovered)
    for tag, run in RUNS.items():
        for ep in EPOCHS:
            jobs.append((f"{tag}_{ep}", os.path.join(R, "runs", run, "reference", f"{ep}e", COMBO, "converted.wav"), "conv"))
    for mp3 in sorted(glob.glob(R + r"\dataset\private\covers\사랑합니다__E4-no-masked_*e_t+0.mp3")):
        ep = os.path.basename(mp3).split("_")[-2].rstrip("e")
        if os.path.exists(mp3[:-4] + ".json"):
            jobs.append((f"E4_{ep}_075", mp3, "recover"))
    infos = []
    with Pool(min(len(jobs), 10)) as pool:
        for info in pool.imap_unordered(job, jobs):
            print("done", json.dumps(info, ensure_ascii=False), flush=True)
            infos.append(info)
    json.dump(sorted(infos, key=lambda d: d["name"]), open(os.path.join(C, "jobs.json"), "w", encoding="utf8"),
              ensure_ascii=False, indent=1)
