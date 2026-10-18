"""Characterise the loud harvest-unvoiced runs in 복면가왕 stems (CPU)."""
import os, sys, glob, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np, librosa
R = r"C:\DEV\ai-voice-conversion"; D = os.path.join(R, r"runs\E1-base\diagnosis"); C = os.path.join(D, "_cache")
sys.path.insert(0, D)
from common import LIVE
import features
rows = []
for p in sorted(glob.glob(R + r"\dataset\private\train\*.wav")):
    stem = os.path.basename(p)[:-4]
    if not any(stem.startswith(x) for x in LIVE): continue
    z = np.load(os.path.join(C, f"feat_train__{stem}.npz"))
    f0, rms = z["f0"], z["rms_db"]
    if (f0 > 0).sum() < 100: continue
    vmed = np.median(rms[f0 > 0]); lu = (f0 == 0) & (rms > vmed - 6)
    d = np.diff(np.r_[0, lu.astype(int), 0]); s0 = np.flatnonzero(d == 1); e0 = np.flatnonzero(d == -1)
    keep = [(a, b) for a, b in zip(s0, e0) if b - a >= 30]
    if not keep: continue
    y, _ = librosa.load(p, sr=16000, mono=True)
    cf0, per = features.crepe_periodicity(y.astype(np.float32))
    for a, b in keep:
        pr = per[a:b]; ff = cf0[a:b]
        prev_f0 = f0[max(0, a - 30):a]; prev_f0 = prev_f0[prev_f0 > 0]
        rows.append(dict(file=stem, start=round(a * 0.01, 2), dur=round((b - a) * 0.01, 2), rel_db=round(float(np.median(rms[a:b]) - vmed), 1),
                         crepe_per=round(float(np.median(pr)), 2), crepe_f0=round(float(np.median(ff)), 0),
                         prev_note_hz=round(float(np.median(prev_f0)), 0) if len(prev_f0) else None,
                         flat=round(float(np.median(z["flat_1_8k"][a:b])), 3)))
rows.sort(key=lambda r: -r["dur"])
print(len(rows), "runs >=0.3 s"); [print(r) for r in rows[:20]]
per = np.array([r["crepe_per"] for r in rows]); print("frac runs crepe_per>0.5:", float((per > 0.5).mean()))
same = [r for r in rows if r["prev_note_hz"] and r["crepe_f0"] and abs(1200*np.log2(r["crepe_f0"]/r["prev_note_hz"])) < 100]
print("runs whose crepe pitch = preceding sung note (+-100c):", len(same), "of", sum(1 for r in rows if r["prev_note_hz"]))
json.dump(rows, open(os.path.join(D, "skeptic_mbk_tails.json"), "w", encoding="utf8"), ensure_ascii=False, indent=0)
