"""Recover converted vocals from the final cover mp3: mp3 = limit(a*(g*voc + inst)).
voc_est = (mp3/a - inst)/g, with a fitted by least squares on vocal-free frames and lag searched."""
import os, sys, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np, soundfile as sf
R = r"C:\DEV\ai-voice-conversion"
C = os.path.join(R, r"runs\E1-base\diagnosis\_cache")
inst, sr = sf.read(glob.glob(R + r"\dataset\private\separated\cover_input\*(other)*.wav")[0], dtype="float64")
voc_src, sr2 = sf.read(glob.glob(R + r"\dataset\private\separated\cover_input\*(vocals)*.wav")[0], dtype="float64")
print("inst", inst.shape, sr, "src", voc_src.shape, sr2)
out = {}
for ep in (300, 500):
    meta = json.load(open(R + rf"\dataset\private\covers\사랑합니다__E1-base_{ep}e_t+0.json", encoding="utf8"))
    mix, msr = sf.read(R + rf"\dataset\private\covers\사랑합니다__E1-base_{ep}e_t+0.mp3", dtype="float64")
    assert msr == sr
    g = 10 ** (meta["mix"]["vocal_level_match_db"] / 20)
    a0 = 10 ** ((-14.0 - meta["mix"]["mix_loudness_before_lufs"]) / 20)
    n = min(len(mix), len(inst))
    # lag search on mono via cross-corr of a 30 s chunk
    seg = slice(int(5 * sr), int(35 * sr))
    best = None
    for lag in range(-3000, 3001):
        pass
    m = mix[:, 0]; x = inst[:, 0]
    L = 2 ** 22
    A = np.fft.rfft(m[:L] if len(m) >= L else np.pad(m, (0, L - len(m))))
    B = np.fft.rfft(x[:L] if len(x) >= L else np.pad(x, (0, L - len(x))))
    cc = np.fft.irfft(A * np.conj(B))
    cc = np.concatenate([cc[-5000:], cc[:5001]])
    lag = int(np.argmax(cc)) - 5000
    print(ep, "lag samples", lag)
    if lag >= 0:
        mm = mix[lag:]; ii = inst[: len(mm)]
    else:
        mm = np.pad(mix, ((-lag, 0), (0, 0))); ii = inst
    n = min(len(mm), len(ii)); mm, ii = mm[:n], ii[:n]
    # fit a on frames where source vocals are quiet
    vs = voc_src[:n] if len(voc_src) >= n else np.pad(voc_src, ((0, n - len(voc_src)), (0, 0)))
    hop = 4410
    k = n // hop
    ve = np.sqrt((vs[: k * hop] ** 2).mean(axis=1).reshape(k, hop).mean(axis=1))
    quiet = ve < np.percentile(ve, 15)
    mask = np.repeat(quiet, hop)
    mq = mm[: k * hop][mask]; iq = ii[: k * hop][mask]
    a = float((mq * iq).sum() / (iq * iq).sum())
    res = mq - a * iq
    print(ep, f"a0={a0:.4f} a_fit={a:.4f} g={g:.3f}  residual on quiet frames {10*np.log10((res**2).mean()/(mq**2).mean()):.1f} dB rel mix")
    est = (mm / a - ii) / g
    est = est.mean(axis=1)
    sf.write(os.path.join(C, f"conv_{ep}e_est.wav"), est.astype(np.float32), sr)
    out[ep] = dict(lag=lag, a0=a0, a_fit=a, g=g, quiet_residual_db=float(10*np.log10((res**2).mean()/(mq**2).mean())))
sf.write(os.path.join(C, "src_vocals.wav"), voc_src.mean(axis=1).astype(np.float32), sr2)
sf.write(os.path.join(C, "inst_mono.wav"), inst.mean(axis=1).astype(np.float32), sr)
json.dump(out, open(os.path.join(C, "..", "recover_stats.json"), "w"), indent=1)
