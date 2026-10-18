"""E1 vs E4 high-note harshness: per pitch band measures, regions, paired E4-E1 differences, index-rate check, figure.

Run harshness_features.py first. CPU only. Writes harshness.json and harshness_E1_vs_E4.png next to this file.
Measure conventions are copied from the E1 diagnosis:
  * plot_harsh.py convention ("frame" measures: h1h2 env, hf_4_8k_db, hf_2_8k_db, alpha_db, ap_4_8k, ap_0_4k, flat_4_12k):
    conversion/source frames = source voiced (harvest F0>0, within 35 dB of max) & source crepe periodicity > 0.5
    & the signal's own level within 35 dB of its max; binned by SOURCE pitch. Real cover: its own voiced mask, its own pitch.
  * skeptic_c.py convention ("peak" measures: h1h2_pk, hnr48, hrel48): each signal's own harvest F0 (>=150 Hz) and pitch,
    loud gate = level above the 30th percentile of its own voiced frames.
  * consonant voicing: skeptic_misc.py (source loud & crepe<0.3 & harvest unvoiced -> converted crepe>0.5 / harvest voiced)
    and analyze_ed.py unvoiced_consonant_hf (source unvoiced & within 30 dB of max -> converted voiced fraction, 4-8 kHz level delta).
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

R = r"C:\DEV\ai-voice-conversion"
DIAG = os.path.join(R, r"runs\E1-base\diagnosis")
A = os.path.join(R, r"runs\E4-no-masked\analysis")
C = os.path.join(A, "_cache")
sys.path.insert(0, DIAG)
from common import plt, midi, voiced_mask  # noqa: E402  (also sets the Korean font)

BINS = {"A3-C#4": (57, 61.5), "D4-E4": (61.5, 64.5), "F4-G4": (64.5, 68)}
EPOCHS = (300, 400, 500)
FRAME_KEYS = ["h1h2", "hf_4_8k_db", "hf_2_8k_db", "alpha_db", "ap_4_8k", "ap_0_4k", "flat_4_12k", "hf_6_16k_db"]
PEAK_KEYS = ["h1h2_pk", "hnr48", "hrel48"]
MEAS = FRAME_KEYS + PEAK_KEYS
MIN_N = 20
rng = np.random.default_rng(0)


def load(name):
    d = os.path.join(C, f"feat_{name}.npz")
    if not os.path.exists(d):
        d = os.path.join(DIAG, "_cache", f"feat_{name}.npz")
    f = dict(np.load(d))
    e = dict(np.load(os.path.join(C, f"extra_{name}.npz")))
    T = len(f["f0"])
    for k, v in e.items():
        f[k] = v[:T] if len(v) >= T else np.pad(v, (0, T - len(v)), constant_values=np.nan)
    return f


jobs = json.load(open(os.path.join(C, "jobs.json"), encoding="utf8"))
NAMES = [j["name"] for j in jobs]
F = {n: load(n) for n in NAMES}
T = min(len(F[n]["f0"]) for n in NAMES if n != "ref")
for n in NAMES:
    if n != "ref":
        F[n] = {k: v[:T] for k, v in F[n].items()}
S = F["src"]
t = np.arange(T) * 0.01
src_m = midi(S["f0"])
ms = voiced_mask(S) & (S["crepe_per"] > 0.5)
Rf = F["ref"]
tr = np.arange(len(Rf["f0"])) * 0.01
ref_m = midi(Rf["f0"])
mr = voiced_mask(Rf) & (Rf["crepe_per"] > 0.5)

# ------------------------------------------------------------------ regions (source time; reference time for the real cover)
al = json.load(open(os.path.join(R, r"runs\E4-no-masked\reference\alignment.json"), encoding="utf8"))
segs = [s for s in al["segments"] if s["status"] == "matched"]


def src_to_ref(seg, ts):
    mp = np.array(seg["map"])  # [ref_s, src_s]
    return float(np.interp(ts, mp[:, 1], mp[:, 0]))


s6 = segs[6]
C_END = 203.0
REG_SRC = {"whole": [(0, 229)], "aligned": [(s["src_start_s"], s["src_end_s"]) for s in segs],
           "C_0229_0323": [(149, C_END)], "C_aligned_0301_0323": [(s6["src_start_s"], C_END)],
           "verse1_0117_0128": [(segs[4]["src_start_s"], segs[4]["src_end_s"])]}
REG_REF = {"whole": [(s["ref_start_s"], s["ref_end_s"]) for s in segs], "aligned": [(s["ref_start_s"], s["ref_end_s"]) for s in segs],
           "C_0229_0323": [(s6["ref_start_s"], src_to_ref(s6, C_END))], "C_aligned_0301_0323": [(s6["ref_start_s"], src_to_ref(s6, C_END))],
           "verse1_0117_0128": [(segs[4]["ref_start_s"], segs[4]["ref_end_s"])]}
REG_NOTE = {"whole": "conversions/Tim: whole song; real cover: its 8 aligned regions (only part that matches the song)",
            "aligned": "all four signals restricted to the 8 aligned regions (src 20.6-107.1 s + 181.5-209.1 s)",
            "C_0229_0323": "listener range 02:29-03:23; the real cover only covers 03:01.5-03:23 of it (aligned region 6)",
            "C_aligned_0301_0323": "part of 02:29-03:23 that the real cover sings (src 181.5-203.0 s)",
            "verse1_0117_0128": "verse-1 high notes ~01:19 (aligned region 4, src 76.9-88.5 s; held E4/F#4 at 79.7, 81.1, 82.3, 85.0 s)"}


def in_regions(tt, regs):
    m = np.zeros(len(tt), bool)
    for a, b in regs:
        m |= (tt >= a) & (tt < b)
    return m


def frame_sel(name, region):
    """plot_harsh convention. Returns (mask, pitch) for the frame measures."""
    if name == "ref":
        return mr & in_regions(tr, REG_REF[region]), ref_m
    f = F[name]
    return ms & (f["rms_db"] > np.nanmax(f["rms_db"]) - 35) & in_regions(t, REG_SRC[region]), src_m


def peak_sel(name, region):
    """skeptic_c convention: own F0, loud = above 30th pct of own voiced level."""
    f = F[name]
    f0, rms = f["f0"], f["rms_db"]
    m = midi(np.where(f0 > 0, f0, np.nan))
    loud = rms > np.nanpercentile(rms[f0 > 0], 30)
    tt = tr if name == "ref" else t
    regs = REG_REF[region] if name == "ref" else REG_SRC[region]
    return (f0 > 0) & loud & in_regions(tt, regs), m


def band_values(name, region, key):
    sel, x = (peak_sel if key in PEAK_KEYS else frame_sel)(name, region)
    out = {}
    for b, (lo, hi) in BINS.items():
        v = F[name][key][sel & (x >= lo) & (x < hi)]
        v = v[np.isfinite(v)]
        out[b] = v
    return out


def med(v):
    return float(np.median(v)) if len(v) > MIN_N else None


SIGNALS = ["src", "ref"] + [f"E1_{e}" for e in EPOCHS] + [f"E4_{e}" for e in EPOCHS] + \
          [n for n in ("c300", "c500") if n in F] + sorted(n for n in NAMES if n.endswith("_075"))
LAB = {"src": "Tim source", "ref": "Kyuhyun real cover", "c300": "E1 300e @ir0.75", "c500": "E1 500e @ir0.75"}
for n in NAMES:
    if n.endswith("_075"):
        LAB[n] = f"E4 {n.split('_')[1]}e @ir0.75"
    elif n.startswith("E"):
        LAB[n] = f"{n.split('_')[0]} {n.split('_')[1]}e @ir0.3"

out = {"settings": dict(bands_midi=BINS, min_frames=MIN_N, regions_src=REG_SRC, regions_ref=REG_REF, region_notes=REG_NOTE,
                        signals=LAB, jobs=jobs)}

# ------------------------------------------------------------------ 1) medians per signal / region / band / measure
tab = {}
for region in REG_SRC:
    tab[region] = {}
    for n in SIGNALS:
        tab[region][n] = {}
        for key in MEAS:
            bv = band_values(n, region, key)
            tab[region][n][key] = {b: med(v) for b, v in bv.items()}
            tab[region][n][key]["n"] = {b: int(len(v)) for b, v in bv.items()}
out["medians"] = tab

# ------------------------------------------------------------------ 2) replication check against the E1 diagnosis numbers
rep = {}
sk = json.load(open(os.path.join(DIAG, "skeptic_C_stats.json"), encoding="utf8"))
cs = json.load(open(os.path.join(DIAG, "C_stats.json"), encoding="utf8"))
for n in ("src", "c300", "c500", "ref"):
    for key in PEAK_KEYS:
        for b in BINS:
            # skeptic_c used the whole reference file, so recompute ref over its full length here
            f = F[n]; f0, rms = f["f0"], f["rms_db"]
            m = midi(np.where(f0 > 0, f0, np.nan)); loud = rms > np.nanpercentile(rms[f0 > 0], 30)
            lo, hi = BINS[b]; v = f[key][(f0 > 0) & loud & (m >= lo) & (m < hi)]; v = v[np.isfinite(v)]
            rep[f"skeptic_c/{n}/{b}/{key}"] = dict(mine=round(float(np.median(v)), 3), diagnosis=round(sk[n][b][key], 3))
for n in ("src", "c300", "c500"):
    for key in ("hf_6_16k_db", "ap_4_8k"):
        f = F[n]; m = ms & (f["rms_db"] > np.nanmax(f["rms_db"]) - 35)
        rep[f"plot_harsh/{n}/high_median/{key}"] = dict(mine=round(float(np.nanmedian(f[key][m & (src_m >= 62)])), 3),
                                                        diagnosis=round(cs["vs_pitch"][key][n]["high_median"], 3))
out["replication_check"] = rep
out["replication_max_abs_diff"] = max(abs(v["mine"] - v["diagnosis"]) for v in rep.values())


# ------------------------------------------------------------------ 3) paired E4 - E1 at identical source frames (same input, same epoch)
def paired(a, b, region, key):
    """Median of b - a over frames both signals keep, per band (source pitch), with a 1-s block bootstrap 95% CI."""
    res = {}
    if key in PEAK_KEYS:
        sa, _ = peak_sel(a, region); sb, _ = peak_sel(b, region)
        base = sa & sb & ms
    else:
        sa, _ = frame_sel(a, region); sb, _ = frame_sel(b, region)
        base = sa & sb
    d_all = F[b][key] - F[a][key]
    for bn, (lo, hi) in BINS.items():
        sel = base & (src_m >= lo) & (src_m < hi) & np.isfinite(d_all)
        idx = np.flatnonzero(sel)
        if len(idx) <= MIN_N:
            res[bn] = None; continue
        d = d_all[idx]; blk = idx // 100
        ub = np.unique(blk); groups = [d[blk == u] for u in ub]
        boots = []
        for _ in range(1000):
            pick = rng.integers(0, len(groups), len(groups))
            boots.append(np.median(np.concatenate([groups[i] for i in pick])))
        res[bn] = dict(median=round(float(np.median(d)), 3), ci95=[round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
                       n=int(len(d)), blocks=int(len(ub)))
    return res


pd = {}
for region in ("whole", "aligned", "C_0229_0323", "verse1_0117_0128"):
    pd[region] = {}
    for key in MEAS:
        pd[region][key] = {str(e): paired(f"E1_{e}", f"E4_{e}", region, key) for e in EPOCHS}
out["paired_E4_minus_E1_ir0.3"] = pd

# index-rate check: same model, 0.75 minus 0.3; and E4 vs E1 at 0.75
ir = {}
pairs = [("E1_300", "c300"), ("E1_500", "c500")] + [(f"E4_{n.split('_')[1]}", n) for n in NAMES if n.endswith("_075")]
for region in ("whole", "aligned", "C_0229_0323"):
    ir[region] = {}
    for key in MEAS:
        ir[region][key] = {f"{b}-{a}": paired(a, b, region, key) for a, b in pairs if a in F and b in F}
        for n in NAMES:
            if n.endswith("_075"):
                for c in ("c300", "c500"):
                    ir[region][key][f"{n}-{c}"] = paired(c, n, region, key)
out["index_rate_check"] = ir


# ------------------------------------------------------------------ 4) consonant voicing side check
def consonant(n, region):
    c = F[n]; reg = in_regions(t, REG_SRC[region])
    act30 = S["rms_db"] > np.nanmax(S["rms_db"]) - 30
    uvl = reg & act30 & (S["crepe_per"] < 0.3) & (S["f0"] == 0)
    uv = reg & act30 & (S["f0"] <= 0)
    hs = S["hf_4_8k_db"][uv] + S["rms_db"][uv]; hc = c["hf_4_8k_db"][uv] + c["rms_db"][uv]
    return dict(n_loud_unvoiced=int(uvl.sum()), crepe_voiced_frac=round(float(np.mean(c["crepe_per"][uvl] > 0.5)), 3) if uvl.any() else None,
                harvest_voiced_frac=round(float(np.mean(c["f0"][uvl] > 0)), 3) if uvl.any() else None,
                n_unvoiced_active=int(uv.sum()), unvoiced_active_conv_voiced_frac=round(float(np.mean(c["f0"][uv] > 0)), 3) if uv.any() else None,
                unvoiced_hf48_delta_db=round(float(np.median(hc - hs)), 3) if uv.any() else None)


out["consonant_voicing"] = {region: {n: consonant(n, region) for n in SIGNALS if n not in ("ref",)}
                            for region in ("whole", "C_0229_0323", "verse1_0117_0128")}


# ------------------------------------------------------------------ 5) toward the real cover? (aligned regions, high bands)
def run_val(region, run, key, b):
    v = [tab[region][f"{run}_{e}"][key][b] for e in EPOCHS]
    v = [x for x in v if x is not None]
    return (float(np.median(v)), float(min(v)), float(max(v))) if v else (None, None, None)


toward = {}
for region in ("aligned", "C_aligned_0301_0323", "verse1_0117_0128"):
    toward[region] = {}
    for key in MEAS:
        for b in BINS:
            r = tab[region]["ref"][key][b]; s = tab[region]["src"][key][b]
            e1 = run_val(region, "E1", key, b); e4 = run_val(region, "E4", key, b)
            if r is None or e1[0] is None or e4[0] is None:
                continue
            g1, g4 = e1[0] - r, e4[0] - r
            toward[region].setdefault(key, {})[b] = dict(
                ref=round(r, 3), src=round(s, 3) if s is not None else None,
                E1_median_of_epochs=round(e1[0], 3), E1_epoch_range=[round(e1[1], 3), round(e1[2], 3)],
                E4_median_of_epochs=round(e4[0], 3), E4_epoch_range=[round(e4[1], 3), round(e4[2], 3)],
                E1_gap_to_ref=round(g1, 3), E4_gap_to_ref=round(g4, 3),
                share_of_E1_gap_closed=round((abs(g1) - abs(g4)) / abs(g1), 3) if abs(g1) >= 0.5 else None,  # ratio meaningless when E1 already within 0.5 dB
                E4_closer=bool(abs(g4) < abs(g1)))
out["toward_real_cover"] = toward


# ------------------------------------------------------------------ 6) figure: bands x measure, aligned regions
PANELS = [("h1h2_pk", "H1-H2, spectral peaks (dB)", "higher = less pressed / breathier"),
          ("h1h2", "H1-H2, CheapTrick envelope (dB)", "higher = less pressed"),
          ("hrel48", "harmonic tilt: 4-8 kHz harmonics vs H1..H3 (dB)", "higher = brighter"),
          ("hnr48", "harmonic-to-valley ratio 4-8 kHz (dB)", "higher = buzzier / more periodic"),
          ("hf_4_8k_db", "4-8 kHz energy rel. total (dB)", "higher = brighter"),
          ("hf_2_8k_db", "2-8 kHz energy rel. total (dB)", "higher = brighter"),
          ("ap_4_8k", "D4C aperiodicity 4-8 kHz (dB)", "lower = more harmonic"),
          ("ap_0_4k", "D4C aperiodicity 0.3-4 kHz (dB)", "lower = more harmonic")]
STY = {"src": dict(color="#8a8984", marker="s", label="Tim source (input)", off=-0.21, ls="--"),
       "E1": dict(color="#2a78d6", marker="o", label="E1-base (with 6 TV stages)", off=-0.07, ls="-"),
       "E4": dict(color="#eb6834", marker="D", label="E4-no-masked (stages removed)", off=0.07, ls="-"),
       "ref": dict(color="#008300", marker="^", label="Kyuhyun real cover", off=0.21, ls="-")}
region = "aligned"
fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
xb = np.arange(len(BINS))
for ax, (key, title, hint) in zip(axes.flat, PANELS):
    for s, st in STY.items():
        if s in ("E1", "E4"):
            vals = [run_val(region, s, key, b) for b in BINS]
            y = np.array([v[0] if v[0] is not None else np.nan for v in vals])
            lo = np.array([v[1] if v[1] is not None else np.nan for v in vals]); hi = np.array([v[2] if v[2] is not None else np.nan for v in vals])
            ax.vlines(xb + st["off"], lo, hi, color=st["color"], lw=2.2, alpha=0.55)
        else:
            y = np.array([tab[region][s][key][b] if tab[region][s][key][b] is not None else np.nan for b in BINS])
        ax.plot(xb + st["off"], y, color=st["color"], ls=st["ls"], lw=1.6, marker=st["marker"], ms=8,
                mec="white", mew=1.2, label=st["label"], zorder=3)
    ax.set_xticks(xb); ax.set_xticklabels(list(BINS), fontsize=9)
    ax.set_xlim(-0.5, len(BINS) - 0.5)
    ax.set_title(title, fontsize=10, loc="left")
    ax.text(0.01, 0.985, hint, transform=ax.transAxes, fontsize=8, color="#52514e", va="top")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.margins(y=0.15)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("High-note phonation per pitch band, 8 aligned regions (109 s): E1 vs E4 at index-rate 0.3\n"
             "E1/E4 marker = median of 300/400/500e, vertical bar = range over those epochs; real cover binned by its own pitch",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(A, "harshness_E1_vs_E4.png")); plt.close(fig)


# ------------------------------------------------------------------ 7) headline: paired E4-E1 range over epochs, CI sign counts
head = {}
for region, d in pd.items():
    head[region] = {}
    for key in MEAS:
        for b in ("D4-E4", "F4-G4"):
            v = [d[key][str(e)][b] for e in EPOCHS if d[key][str(e)][b] is not None]
            if not v:
                continue
            head[region].setdefault(key, {})[b] = dict(
                E4_minus_E1_range=[min(x["median"] for x in v), max(x["median"] for x in v)],
                epochs_ci_above_0=sum(x["ci95"][0] > 0 for x in v), epochs_ci_below_0=sum(x["ci95"][1] < 0 for x in v))
out["headline_paired_E4_minus_E1"] = head
out["verdict"] = ("Partly yes. Removing the TV stages made the high notes clearly less pressed (H1-H2 +1.8 to +6.9 dB at D4-G4, every epoch, "
                  "about 30-80% of the E1-to-real-cover gap closed) and, in 02:29-03:23 and in 2-8 kHz everywhere, less bright "
                  "(down to the real cover's level at D4-E4 in the aligned part of that section). Outside that section 4-8 kHz brightness "
                  "and harmonic tilt did not move toward the cover, H1-H2 is still 5-10 dB below the cover, and 0.3-4 kHz aperiodicity at "
                  "F4-G4 went further from it. Consonant voicing is unchanged. See harshness.md.")


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 3)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


json.dump(clean(out), open(os.path.join(A, "harshness.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("replication max abs diff", out["replication_max_abs_diff"])
