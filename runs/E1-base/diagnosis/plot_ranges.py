"""Spectrogram + F0 overlays: source / 300e / 500e / instrumental per listener range, plus metric strip."""
import sys
from common import *
import librosa, librosa.display, soundfile as sf

RANGES = {
    "A_0109": (66.0, 74.0, 69.0, 71.0, "A) 01:09-01:11  'something mixed in / ad-lib'"),
    "B_0147": (104.0, 112.0, 107.0, 110.0, "B) 01:47-01:50  'slightly weird'"),
    "D_0324": (201.0, 212.0, 204.0, 210.0, "D) 03:24-03:30  '그댈 사랑합니다' slurred"),
}
if len(sys.argv) > 1:
    a, b = float(sys.argv[2]), float(sys.argv[3])
    RANGES = {sys.argv[1]: (a, b, a, b, sys.argv[1])}

audio = {k: librosa.load(os.path.join(C, f), sr=44100, mono=True)[0] for k, f in
         [("src", "src_vocals.wav"), ("c300", "conv_300e_est.wav"), ("c500", "conv_500e_est.wav"), ("inst", "inst_mono.wav")]}
F = {k: feat(k) for k in ("src", "c300", "c500")}
st = np.load(os.path.join(C, "stereo.npz"))

for key, (a, b, ha, hb, title) in RANGES.items():
    fig, axes = plt.subplots(6, 1, figsize=(13, 17), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 3, 3, 2.2, 1.6, 1.6]))
    i0, i1 = int(a * 44100), int(b * 44100)
    ref_max = None
    for ax, k in zip(axes[:4], ["src", "c300", "c500", "inst"]):
        y = audio[k][i0:i1]
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=4096, hop_length=256)), ref=1.0)
        if ref_max is None:
            ref_max = S.max()
        librosa.display.specshow(S, sr=44100, hop_length=256, x_axis="time", y_axis="log", ax=ax,
                                 vmin=ref_max - 80, vmax=ref_max, cmap="magma", x_coords=None)
        ax.set_ylim(70, 12000)
        ax.set_ylabel("Hz")
        tt = np.arange(S.shape[1]) * 256 / 44100
        if k in F:
            f = F[k]; m = (f["t"] >= a) & (f["t"] < b)
            f0 = np.where(f["f0"][m] > 0, f["f0"][m], np.nan)
            ax.plot(f["t"][m] - a, f0, color="cyan", lw=1.4, label=f"{k} F0 (harvest)")
            if k != "src":
                fs = F["src"]; ms = (fs["t"] >= a) & (fs["t"] < b)
                ax.plot(fs["t"][ms] - a, np.where(fs["f0"][ms] > 0, fs["f0"][ms], np.nan), color="white",
                        lw=0.9, ls="--", label="source F0")
            ax.legend(loc="upper right", fontsize=8, facecolor="black", labelcolor="white")
        ax.set_title({"src": "Tim source vocal stem (input to RVC)", "c300": "300e converted (recovered from mp3)",
                      "c500": "500e converted (recovered from mp3)", "inst": "Tim instrumental stem"}[k], fontsize=10, loc="left")
        ax.axvspan(ha - a, hb - a, color="lime", alpha=0.0, ec="lime", lw=1.5, ls="--")
        ax.set_xlabel("")
    # metric strip 1: levels
    ax = axes[4]
    tt = np.arange(len(st["eM"])) * 0.1
    m = (tt >= a) & (tt < b)
    ax.plot(tt[m] - a, st["eM"][m], color=COL["src"], label="source mid (L+R) 200-4k dB")
    ax.plot(tt[m] - a, st["eS"][m], color="#72b7b2", label="source side (L-R) dB")
    ax.plot(tt[m] - a, st["eI"][m], color=COL["inst"], ls=":", label="instrumental 200-4k dB")
    ax.set_ylim(-70, -5); ax.legend(fontsize=8, ncol=3, loc="lower left"); ax.set_ylabel("dB")
    ax.axvspan(ha - a, hb - a, color="lime", alpha=0.12)
    # metric strip 2: second-pitch salience and periodicity
    ax = axes[5]
    for k in ("src", "c500"):
        f = F[k]; mm = (f["t"] >= a) & (f["t"] < b)
        r = f["sal2"][mm] / np.maximum(f["sal1"][mm], 1e-6)
        r = np.convolve(np.nan_to_num(r), np.ones(5) / 5, mode="same")
        ax.plot(f["t"][mm] - a, r, color=COL[k], label=f"{k}: 2nd-pitch salience / main")
        ax.plot(f["t"][mm] - a, f["crepe_per"][mm], color=COL[k], ls=":", lw=1, label=f"{k}: crepe periodicity")
    ax.set_ylim(0, 1.2); ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.axvspan(ha - a, hb - a, color="lime", alpha=0.12)
    ax.set_xlabel(f"seconds from {int(a // 60)}:{a % 60:04.1f}  (green = listener range {int(ha//60)}:{ha%60:02.0f}-{int(hb//60)}:{hb%60:02.0f})")
    for ax in axes[:4]:
        ax.axvline(ha - a, color="lime", ls="--", lw=1.2); ax.axvline(hb - a, color="lime", ls="--", lw=1.2)
    fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    out = os.path.join(D, f"range_{key}.png")
    fig.savefig(out); plt.close(fig)
    print(out)
