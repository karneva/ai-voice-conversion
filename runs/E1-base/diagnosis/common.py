import os, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

R = r"C:\DEV\ai-voice-conversion"
D = os.path.join(R, r"runs\E1-base\diagnosis")
C = os.path.join(D, "_cache")
_f = r"C:\Windows\Fonts\malgun.ttf"
if os.path.exists(_f):
    font_manager.fontManager.addfont(_f)
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

COL = {"src": "#4c78a8", "c300": "#f58518", "c500": "#e45756", "ref": "#54a24b", "inst": "#888888"}
LABEL = {"src": "Tim source stem", "c300": "300e converted", "c500": "500e converted", "ref": "Kyuhyun real cover"}

LIVE = ("바람기억", "숨_", "에너제틱", "오르막길", "한숨", "혜야")


def feat(name):
    z = np.load(os.path.join(C, f"feat_{name}.npz"))
    return {k: z[k] for k in z.files}


def train_group(stem):
    if any(stem.startswith(p) for p in LIVE):
        return "복면가왕 live (6 songs)"
    if stem.startswith("킬링보이스"):
        return "킬링보이스 live medley"
    return "studio/home (MV, 규티비)"


def midi(f):
    f = np.asarray(f, float)
    out = np.full(f.shape, np.nan)
    m = f > 0
    out[m] = 69 + 12 * np.log2(f[m] / 440.0)
    return out


def note_name(m):
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    m = int(round(m))
    return f"{names[m % 12]}{m // 12 - 1}"


def voiced_mask(F, rel_db=35.0):
    rms = F["rms_db"]
    act = rms > np.nanmax(rms) - rel_db
    return (F["f0"] > 0) & act
