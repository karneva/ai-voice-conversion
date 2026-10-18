"""Frame features (10 ms hop) for the cover diagnosis. CPU only.

f0 (pyworld harvest @16k), crepe-tiny periodicity, RMS, band energies, D4C aperiodicity bands (@32k),
H1-H2, spectral flatness 1-8 kHz, spectral flux, harmonic salience of a second pitch.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import librosa
import pyworld as pw

HOP_S = 0.01
FP_MS = 10.0


def load_mono(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float64)


def crepe_periodicity(y16):
    import torch, torchcrepe
    torch.set_num_threads(4)
    x = torch.tensor(y16, dtype=torch.float32)[None]
    f0, per = torchcrepe.predict(x, 16000, 160, 50.0, 1100.0, "tiny", batch_size=2048,
                                 device="cpu", return_periodicity=True, pad=True)
    return f0[0].numpy(), per[0].numpy()


def second_pitch_salience(S, freqs, f0, fmin=80.0, fmax=1000.0):
    """Harmonic-summation salience of the strongest pitch that is NOT a harmonic/subharmonic of f0.
    S: magnitude (bins x frames) at the f0 frame rate. Returns (s1, s2, f2)."""
    cands = fmin * 2 ** (np.arange(0, np.log2(fmax / fmin) * 1200, 20) / 1200)
    H = 8
    w = 0.8 ** np.arange(H)
    df = freqs[1] - freqs[0]
    T = S.shape[1]
    s1 = np.zeros(T); s2 = np.zeros(T); f2 = np.zeros(T)
    logS = np.log1p(100 * S / (S.max(axis=0, keepdims=True) + 1e-9))
    for t in range(T):
        col = logS[:, t]
        def sal(c):
            idx = np.round(np.outer(c, np.arange(1, H + 1)) / df).astype(int)
            idx = np.clip(idx, 0, len(col) - 1)
            return (col[idx] * w).sum(axis=1)
        if f0[t] <= 0:
            s = sal(cands); s1[t] = s.max(); continue
        # remove main harmonics (+-40 cents)
        c2 = col.copy()
        for h in range(1, 30):
            fh = f0[t] * h
            if fh > freqs[-1]:
                break
            lo, hi = fh * 2 ** (-40 / 1200), fh * 2 ** (40 / 1200)
            c2[int(lo / df): int(np.ceil(hi / df)) + 1] = 0
        s1[t] = sal(np.array([f0[t]]))[0]
        col_backup = col; col = c2
        s = sal(cands)
        col = col_backup
        # exclude candidates near f0 * k or f0 / k
        r = np.log2(cands / f0[t]) * 1200
        bad = np.zeros_like(cands, bool)
        for k in (1, 2, 3, 4):
            bad |= np.abs(r - 1200 * np.log2(k)) < 60
            bad |= np.abs(r + 1200 * np.log2(k)) < 60
        s[bad] = 0
        j = int(np.argmax(s)); s2[t] = s[j]; f2[t] = cands[j]
    return s1, s2, f2


def compute(y44, sr=44100, with_salience=True, with_crepe=True):
    y16 = librosa.resample(y44, orig_sr=sr, target_sr=16000)
    y32 = librosa.resample(y44, orig_sr=sr, target_sr=32000)
    f0, t = pw.harvest(y16, 16000, f0_floor=70.0, f0_ceil=1100.0, frame_period=FP_MS)
    sp32 = None
    ap32 = pw.d4c(y32, f0, t, 32000)  # (T, 1025) full aperiodicity, 0..1
    sp32 = pw.cheaptrick(y32, f0, t, 32000)
    T = len(f0)
    fr32 = np.linspace(0, 16000, ap32.shape[1])

    def band_mean(a, lo, hi):
        m = (fr32 >= lo) & (fr32 < hi)
        return a[:, m].mean(axis=1)
    ap_db = 20 * np.log10(np.clip(ap32, 1e-4, 1))
    feats = dict(t=t, f0=f0)
    feats["ap_0_4k"] = band_mean(ap_db, 300, 4000)    # lower = more harmonic
    feats["ap_4_8k"] = band_mean(ap_db, 4000, 8000)
    feats["ap_8_12k"] = band_mean(ap_db, 8000, 12000)
    # H1-H2 from cheaptrick envelope (smoothed, so this is an envelope-based H1*-H2* proxy)
    def env_at(fq):
        idx = np.clip(np.round(fq / (16000 / (sp32.shape[1] - 1))).astype(int), 0, sp32.shape[1] - 1)
        return 10 * np.log10(sp32[np.arange(T), idx] + 1e-20)
    feats["h1h2"] = np.where(f0 > 0, env_at(f0) - env_at(2 * f0), np.nan)

    # STFT @44.1k aligned to 10 ms frames
    hop = 441
    D = np.abs(librosa.stft(y44.astype(np.float32), n_fft=2048, hop_length=hop, center=True))
    D = D[:, :T] if D.shape[1] >= T else np.pad(D, ((0, 0), (0, T - D.shape[1])))
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    P = D ** 2

    def bandE(lo, hi):
        m = (fr >= lo) & (fr < hi)
        return P[m].sum(axis=0) + 1e-12
    tot = bandE(50, 16000)
    feats["rms_db"] = 10 * np.log10(tot / 1024)
    feats["hf_6_16k_db"] = 10 * np.log10(bandE(6000, 16000) / tot)
    feats["hf_4_8k_db"] = 10 * np.log10(bandE(4000, 8000) / tot)
    feats["alpha_db"] = 10 * np.log10(bandE(1000, 5000) / bandE(50, 1000))  # spectral tilt (alpha ratio)
    m = (fr >= 1000) & (fr < 8000)
    Pm = P[m] + 1e-12
    feats["flat_1_8k"] = np.exp(np.mean(np.log(Pm), axis=0)) / np.mean(Pm, axis=0)
    m2 = (fr >= 4000) & (fr < 12000)
    Pm2 = P[m2] + 1e-12
    feats["flat_4_12k"] = np.exp(np.mean(np.log(Pm2), axis=0)) / np.mean(Pm2, axis=0)
    L = np.log(D + 1e-6)
    flux = np.r_[0, np.maximum(np.diff(L, axis=1), 0).sum(axis=0)]
    feats["flux"] = flux
    mh = fr >= 2000
    feats["flux_hf"] = np.r_[0, np.maximum(np.diff(L[mh], axis=1), 0).sum(axis=0)]
    if with_crepe:
        cf0, per = crepe_periodicity(y16.astype(np.float32))
        n = min(T, len(per))
        feats["crepe_f0"] = np.pad(cf0[:n], (0, T - n))
        feats["crepe_per"] = np.pad(per[:n], (0, T - n))
    if with_salience:
        # 16k STFT, 4096 window for pitch resolution
        D16 = np.abs(librosa.stft(y16.astype(np.float32), n_fft=4096, hop_length=160, center=True))
        D16 = D16[:, :T] if D16.shape[1] >= T else np.pad(D16, ((0, 0), (0, T - D16.shape[1])))
        f16 = librosa.fft_frequencies(sr=16000, n_fft=4096)
        keep = f16 <= 5000
        s1, s2, f2 = second_pitch_salience(D16[keep], f16[keep], f0)
        feats["sal1"], feats["sal2"], feats["f2"] = s1, s2, f2
    return feats
