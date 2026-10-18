"""D) consonant/transient sharpness and E) expressive features: Kyuhyun real cover vs conversion vs Tim source."""
import json
from common import *

F = {k: feat(k) for k in ("src", "c300", "c500", "ref")}
out = {}

# ------------------------------------------------------------------ F0 tracking errors (all ranges)
def f0_err(k, a, b):
    s, c = F["src"], F[k]
    m = (s["t"] >= a) & (s["t"] < b) & (s["f0"] > 0) & (c["f0"] > 0)
    d = 1200 * np.log2(c["f0"][m] / s["f0"][m])
    vs = (s["t"] >= a) & (s["t"] < b) & voiced_mask(s)
    return dict(n=int(m.sum()), median_abs_cents=float(np.median(np.abs(d))) if m.any() else None,
                frac_over_300c=float(np.mean(np.abs(d) > 300)) if m.any() else None,
                frac_octave=float(np.mean(np.abs(np.abs(d) - 1200) < 150)) if m.any() else None,
                voiced_lost=float(np.mean(c["f0"][vs] <= 0)) if vs.any() else None,
                crepe_per_src=float(np.median(s["crepe_per"][vs])) if vs.any() else None,
                crepe_per_conv=float(np.median(c["crepe_per"][vs])) if vs.any() else None)
RANGES = {"A": (69, 71), "B": (107, 110), "C": (149, 203), "D": (204, 210), "whole": (0, 229)}
out["f0_tracking"] = {r: {k: f0_err(k, *ab) for k in ("c300", "c500")} for r, ab in RANGES.items()}

# second pitch salience in ranges vs song
def sal_ratio(k, a, b):
    f = F[k]; m = (f["t"] >= a) & (f["t"] < b) & (f["rms_db"] > np.nanmax(f["rms_db"]) - 35)
    r = f["sal2"][m] / np.maximum(f["sal1"][m], 1e-6)
    return float(np.median(r)) if m.any() else None
out["second_pitch_ratio"] = {r: {k: sal_ratio(k, *ab) for k in ("src", "c500")} for r, ab in RANGES.items()}

# ------------------------------------------------------------------ D: onsets / unvoiced consonants
def onsets(f, thr_rise=12.0):
    """Frames where level rises >= thr_rise dB within 30 ms from a quieter state (consonant / note attacks)."""
    r = f["rms_db"]; act = r > np.nanmax(r) - 35
    idx = []
    for i in range(3, len(r) - 3):
        if act[i] and r[i] - r[i - 3] >= thr_rise and r[i] >= r[i - 1] and (not idx or i - idx[-1] > 8):
            idx.append(i)
    return np.array(idx)

s = F["src"]
on = onsets(s)
def onset_stats(k, sel):
    c = F[k]; res = []
    for i in on[sel]:
        w = slice(max(0, i - 3), i + 4)
        ps, pc = s["flux_hf"][w].max(), c["flux_hf"][w].max()
        res.append(pc / max(ps, 1e-6))
    return float(np.median(res)) if res else None, len(res)
tt = s["t"][on]
out["onset_hf_flux_ratio_conv_over_src"] = {
    "whole": {k: onset_stats(k, np.ones(len(on), bool)) for k in ("c300", "c500")},
    "D_204_210": {k: onset_stats(k, (tt >= 204) & (tt < 210)) for k in ("c300", "c500")},
}
# unvoiced consonant HF energy (ㅅ/ㅎ/ㄷ release etc.): frames where source is unvoiced but active
def unvoiced_hf(k, a, b):
    c = F[k]
    m = (s["t"] >= a) & (s["t"] < b) & (s["f0"] <= 0) & (s["rms_db"] > np.nanmax(s["rms_db"]) - 30)
    # absolute HF level in dB: hf ratio + rms
    hs = s["hf_4_8k_db"][m] + s["rms_db"][m]; hc = c["hf_4_8k_db"][m] + c["rms_db"][m]
    return dict(n=int(m.sum()), median_delta_db=float(np.median(hc - hs)) if m.any() else None,
                conv_voiced_frac=float(np.mean(c["f0"][m] > 0)) if m.any() else None)
out["unvoiced_consonant_hf"] = {r: {k: unvoiced_hf(k, *ab) for k in ("c300", "c500")} for r, ab in RANGES.items()}

# D figure: level + HF flux, 203.5-210.5
a, b = 203.5, 210.5
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for k in ("src", "c300", "c500"):
    f = F[k]; m = (f["t"] >= a) & (f["t"] < b)
    axes[0].plot(f["t"][m], f["rms_db"][m], color=COL[k], label=LABEL[k], lw=1.3)
    axes[1].plot(f["t"][m], f["flux_hf"][m], color=COL[k], label=LABEL[k], lw=1.1)
    axes[2].plot(f["t"][m], f["hf_4_8k_db"][m] + f["rms_db"][m], color=COL[k], label=LABEL[k], lw=1.1)
for i in on[(tt >= a) & (tt < b)]:
    for ax in axes:
        ax.axvline(s["t"][i], color="k", lw=0.6, ls=":")
axes[0].set_ylabel("level dB"); axes[1].set_ylabel("spectral flux >2 kHz"); axes[2].set_ylabel("4-8 kHz level dB")
axes[0].legend(fontsize=8); axes[2].set_xlabel("song time (s); dotted = onsets detected in source")
axes[0].set_title("D) 03:23.5-03:30.5 '그댈 사랑합니다': attack/consonant energy, source vs converted", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(D, "D_transients_0324.png")); plt.close(fig)

# ------------------------------------------------------------------ E: expressive features on aligned phrases
al = json.load(open(os.path.join(R, r"runs\E1-base\reference\alignment.json"), encoding="utf8"))
segs = [(g["src_start_s"], g["src_end_s"], g["ref_start_s"], g["ref_end_s"]) for g in al["segments"] if g.get("status") == "matched"][:7]

def vibrato(f0, t, mask):
    """Held-note vibrato: rate (Hz) and extent (+- cents) from detrended cents on voiced runs >= 0.5 s."""
    m = midi(f0) * 100
    rates, exts = [], []
    i = 0
    while i < len(m):
        if not mask[i] or np.isnan(m[i]):
            i += 1; continue
        j = i
        while j + 1 < len(m) and mask[j + 1] and not np.isnan(m[j + 1]) and abs(m[j + 1] - m[j]) < 80:
            j += 1
        if j - i + 1 >= 50:
            x = m[i:j + 1]
            k = 25; trend = np.convolve(np.pad(x, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
            d = x - trend
            if np.std(x - np.median(x)) < 120:  # a held note, not a run
                spec = np.abs(np.fft.rfft(d * np.hanning(len(d)), 1024)); fr = np.fft.rfftfreq(1024, 0.01)
                band = (fr >= 3.5) & (fr <= 9)
                rates.append(fr[band][np.argmax(spec[band])]); exts.append(np.std(d) * np.sqrt(2))
        i = j + 1
    return rates, exts

def attack_times(f, a, b):
    r = f["rms_db"]; t = f["t"]; m = (t >= a) & (t < b)
    idx = np.where(m)[0]; out_ = []
    rr = np.convolve(r, np.ones(3) / 3, "same")
    floor = np.nanmax(r) - 40
    i = idx[0]
    while i < idx[-1]:
        if rr[i] < floor and rr[i + 1] >= floor:
            w = rr[i:i + 40]; pk = int(np.argmax(w)); pv = w[pk]
            if pv - floor > 15:
                a10 = np.argmax(w >= pv - 20); a90 = np.argmax(w >= pv - 3)
                out_.append((a90 - a10) * 10.0)
            i += 20
        i += 1
    return out_

def feats_on(k, a, b):
    f = F[k]; m = (f["t"] >= a) & (f["t"] < b)
    v = m & voiced_mask(f) & (f["crepe_per"] > 0.5)
    act = m & (f["rms_db"] > np.nanmax(f["rms_db"]) - 40)
    rates, exts = vibrato(f["f0"], f["t"], v)
    return dict(ap_0_4k=np.nanmedian(f["ap_0_4k"][v]), h1h2=np.nanmedian(f["h1h2"][v]), alpha=np.nanmedian(f["alpha_db"][v]),
                hf_6_16k=np.nanmedian(f["hf_6_16k_db"][v]),
                vib_rate=np.median(rates) if rates else np.nan, vib_ext=np.median(exts) if exts else np.nan, n_vib=len(rates),
                attack_ms=np.median(attack_times(f, a, b)) if attack_times(f, a, b) else np.nan,
                dyn_range=np.percentile(f["rms_db"][act], 95) - np.percentile(f["rms_db"][act], 10),
                f0_median_midi=np.nanmedian(midi(f["f0"][v])))
rows = []
for (sa, sb, ra, rb) in segs:
    row = {"src_range": (sa, sb), "ref_range": (ra, rb)}
    for k in ("src", "c300", "c500"):
        row[k] = feats_on(k, sa, sb)
    row["ref"] = feats_on("ref", ra, rb)
    rows.append(row)
keys = ["ap_0_4k", "h1h2", "alpha", "hf_6_16k", "vib_rate", "vib_ext", "attack_ms", "dyn_range", "f0_median_midi"]
summary = {}
for kk in keys:
    vals = {k: np.array([r[k][kk] for r in rows], float) for k in ("src", "c300", "c500", "ref")}
    summary[kk] = {k: float(np.nanmedian(v)) for k, v in vals.items()}
    # per-segment: is conversion closer to source or to Kyuhyun?
    d_src = np.nanmedian(np.abs(vals["c500"] - vals["src"])); d_ref = np.nanmedian(np.abs(vals["c500"] - vals["ref"]))
    ok = np.isfinite(vals["c500"]) & np.isfinite(vals["src"])
    corr = float(np.corrcoef(vals["c500"][ok], vals["src"][ok])[0, 1]) if ok.sum() > 2 else None
    summary[kk].update(c500_dist_to_src=float(d_src), c500_dist_to_ref=float(d_ref), corr_c500_src_across_segments=corr)
out["E_expressive"] = summary
out["E_segments"] = [{"src": r["src_range"], "ref": r["ref_range"], **{k: {kk: round(float(r[k][kk]), 2) for kk in keys} for k in ("src", "c500", "ref")}} for r in rows]

# E figure
labels = {"ap_0_4k": "aperiodicity 0.3-4k dB\n(higher = breathier)", "h1h2": "H1-H2 dB (env)\n(higher = breathier)",
          "alpha": "alpha ratio dB\n(lower = softer tilt)", "hf_6_16k": "HF 6-16k rel. dB",
          "vib_rate": "vibrato rate Hz", "vib_ext": "vibrato extent +-cents", "attack_ms": "attack time ms\n(higher = softer onset)",
          "dyn_range": "dynamic range dB\n(p95-p10)"}
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
for ax, kk in zip(axes.flat, labels):
    for j, k in enumerate(("src", "c300", "c500", "ref")):
        v = np.array([r[k][kk] for r in rows], float)
        ax.scatter(np.full(len(v), j) + np.linspace(-0.15, 0.15, len(v)), v, color=COL[k], s=18, alpha=0.7)
        ax.hlines(np.nanmedian(v), j - 0.3, j + 0.3, color=COL[k], lw=3)
    for r in rows:  # connect src -> c500 -> per segment
        ax.plot([0, 2], [r["src"][kk], r["c500"][kk]], color="#bbbbbb", lw=0.6, zorder=0)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Tim src", "300e", "500e", "Kyuhyun"], fontsize=8)
    ax.set_title(labels[kk], fontsize=9, loc="left")
fig.suptitle("E) Expressive features on 7 aligned phrases (bar = median; grey line links source and 500e of the same phrase)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(os.path.join(D, "E_expressive_ref_vs_conv.png")); plt.close(fig)

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 3)
    if isinstance(o, np.integer): return int(o)
    return o
json.dump(clean(out), open(os.path.join(D, "DE_stats.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(json.dumps(clean(out), ensure_ascii=False, indent=1))
