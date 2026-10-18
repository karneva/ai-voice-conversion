"""Control for the recovery path: push Tim's SOURCE vocals through the same mix -> limiter -> MP3 -> subtract chain,
so any bias of the mp3-recovered estimate (e.g. MP3 removing noise between HF harmonics) shows up as src_mp3 - src."""
import os, sys, glob, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np, soundfile as sf
R = r"C:\DEV\ai-voice-conversion"; D = os.path.join(R, r"runs\E1-base\diagnosis"); C = os.path.join(D, "_cache")
sys.path.insert(0, os.path.join(R, "scripts", "data")); sys.path.insert(0, D)
import mix_cover, features
inst, sr = sf.read(glob.glob(R + r"\dataset\private\separated\cover_input\*(other)*.wav")[0], dtype="float64")
voc, _ = sf.read(glob.glob(R + r"\dataset\private\separated\cover_input\*(vocals)*.wav")[0], dtype="float64")
g = 1.0
mix = voc * g + inst
before = mix_cover.loudness(mix, sr); a = 10 ** ((-14.0 - before) / 20); mix *= a
mix = mix_cover.limit(mix, sr, 10 ** (-1.3 / 20))
p = os.path.join(C, "src_through_mp3.mp3")
sf.write(p, mix.astype(np.float32), sr, format="MP3", subtype="MPEG_LAYER_III", compression_level=0.0, bitrate_mode="CONSTANT")
m, _ = sf.read(p, dtype="float64")
n = min(len(m), len(inst)); est = ((m[:n] / a - inst[:n]) / g).mean(axis=1)
sf.write(os.path.join(C, "src_mp3_est.wav"), est.astype(np.float32), sr)
f = features.compute(est, with_salience=False, with_crepe=True)
np.savez_compressed(os.path.join(C, "feat_srcmp3.npz"), **f)
print("ok", before)
