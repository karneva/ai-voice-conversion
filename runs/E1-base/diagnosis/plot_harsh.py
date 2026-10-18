"""C) harshness vs pitch (source / 300e / 500e / Kyuhyun real cover) and training pitch coverage by source type."""
import json
from common import *

out = {}
F = {k: feat(k) for k in ("src", "c300", "c500", "ref")}
ms = voiced_mask(F["src"]) & (F["src"]["crepe_per"] > 0.5)
src_m = midi(F["src"]["f0"])

# ---- held notes in the cover (source F0 stable >= 0.25 s)
def held_notes(f0, t, mask, min_s=0.25, tol=0.6):
    m = midi(f0); notes = []
    i = 0; n = len(m)
    while i < n:
        if not mask[i] or np.isnan(m[i]):
            i += 1; continue
        j = i
        while j + 1 < n and mask[j + 1] and not np.isnan(m[j + 1]) and abs(m[j + 1] - np.median(m[i:j + 1])) < tol:
            j += 1
        if (j - i + 1) * 0.01 >= min_s:
            notes.append((t[i], t[j], float(np.median(m[i:j + 1]))))
        i = j + 1
    return notes

notes = held_notes(F["src"]["f0"], F["src"]["t"], ms)
notes_sorted = sorted(notes, key=lambda x: -x[2])
out["cover_top_held_notes"] = [dict(start=round(a, 2), end=round(b, 2), midi=round(m, 2), note=note_name(m)) for a, b, m in notes_sorted[:15]]
top = np.percentile([n[2] for n in notes], 90)
high_thr = float(np.floor(top))  # MIDI threshold for "the song's top pitches"
out["high_threshold_midi"] = high_thr
out["high_threshold_note"] = note_name(high_thr)
print("high threshold", high_thr, note_name(high_thr))

# ---- frame metrics vs source pitch (frames aligned: conversions keep timing)
metrics = [("hf_6_16k_db", "HF energy 6-16 kHz rel. total (dB)"),
           ("ap_4_8k", "D4C aperiodicity 4-8 kHz (dB, higher = noisier)"),
           ("flat_4_12k", "spectral flatness 4-12 kHz"),
           ("ap_0_4k", "D4C aperiodicity 0.3-4 kHz (dB, ~ -HNR)")]
bins = np.arange(np.floor(np.nanmin(src_m[ms])), np.ceil(np.nanmax(src_m[ms])) + 1)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
rng = np.random.default_rng(0)
out["vs_pitch"] = {}
for ax, (key, lab) in zip(axes.flat, metrics):
    rows = {}
    for k in ("src", "c300", "c500"):
        f = F[k]
        m = ms & (f["rms_db"] > np.nanmax(f["rms_db"]) - 35)
        x = src_m[m]; y = f[key][m]
        sel = rng.random(len(x)) < 0.25
        ax.scatter(x[sel] + rng.normal(0, 0.12, sel.sum()), y[sel], s=2, alpha=0.15, color=COL[k])
        med = [np.nanmedian(y[(x >= b) & (x < b + 1)]) if ((x >= b) & (x < b + 1)).sum() > 30 else np.nan for b in bins]
        ax.plot(bins + 0.5, med, color=COL[k], lw=2.2, marker="o", ms=3, label=LABEL[k])
        rows[k] = med
        lo = x < high_thr - 7; hi = x >= high_thr - 2
        out["vs_pitch"].setdefault(key, {})[k] = dict(low_median=float(np.nanmedian(y[lo])), high_median=float(np.nanmedian(y[hi])),
                                                       slope_per_semitone=float(np.polyfit(x[np.isfinite(y)], y[np.isfinite(y)], 1)[0]))
    # Kyuhyun's own cover, binned by its own pitch
    f = F["ref"]; m = voiced_mask(f) & (f["crepe_per"] > 0.5); x = midi(f["f0"])[m]; y = f[key][m]
    med = [np.nanmedian(y[(x >= b) & (x < b + 1)]) if ((x >= b) & (x < b + 1)).sum() > 30 else np.nan for b in bins]
    ax.plot(bins + 0.5, med, color=COL["ref"], lw=2, ls="--", marker="s", ms=3, label=LABEL["ref"] + " (own pitch)")
    out["vs_pitch"][key]["ref"] = dict(low_median=float(np.nanmedian(y[x < high_thr - 7])), high_median=float(np.nanmedian(y[x >= high_thr - 2])))
    ax.axvline(high_thr - 2, color="k", ls=":", lw=1)
    ax.set_xlabel("pitch (MIDI, from source F0)"); ax.set_ylabel(lab); ax.set_title(lab, fontsize=10, loc="left")
    tick = np.arange(bins[0], bins[-1] + 1, 2)
    ax.set_xticks(tick); ax.set_xticklabels([f"{int(t)}\n{note_name(t)}" for t in tick], fontsize=8)
    if key == "flat_4_12k":
        ax.set_ylim(0, 0.6)
    ax.legend(fontsize=8)
fig.suptitle("C) Harshness proxies vs pitch: converted (300e/500e) vs Tim source vs Kyuhyun real cover  (median per semitone)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(os.path.join(D, "C_harshness_vs_pitch.png")); plt.close(fig)

# ---- delta conv - src per pitch, and inside 02:29-03:23 vs outside
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, key in zip(axes, ["hf_6_16k_db", "ap_4_8k"]):
    for k in ("c300", "c500"):
        f = F[k]
        m = ms & np.isfinite(f[key]) & np.isfinite(F["src"][key])
        d = f[key][m] - F["src"][key][m]; x = src_m[m]
        med = [np.nanmedian(d[(x >= b) & (x < b + 1)]) if ((x >= b) & (x < b + 1)).sum() > 30 else np.nan for b in bins]
        ax.plot(bins + 0.5, med, color=COL[k], lw=2, marker="o", ms=3, label=f"{k} - source")
        tt = F["src"]["t"][m]
        inC = (tt >= 149) & (tt <= 203)
        out.setdefault("delta", {}).setdefault(key, {})[k] = dict(
            inC_high=float(np.nanmedian(d[inC & (x >= high_thr - 2)])), inC_low=float(np.nanmedian(d[inC & (x < high_thr - 7)])),
            outside_high=float(np.nanmedian(d[~inC & (x >= high_thr - 2)])), all=float(np.nanmedian(d)))
    ax.axhline(0, color="k", lw=0.8); ax.axvline(high_thr - 2, color="k", ls=":", lw=1)
    ax.set_xlabel("pitch (MIDI)"); ax.set_title(f"converted minus source: {key}", loc="left", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(D, "C_delta_conv_minus_source.png")); plt.close(fig)

# ---- training coverage
import glob as _g
groups = {}
for p in sorted(_g.glob(os.path.join(C, "feat_train__*.npz"))):
    stem = os.path.basename(p)[len("feat_train__"):-4]
    f = feat("train__" + stem)
    g = train_group(stem)
    m = voiced_mask(f, 40)
    d = groups.setdefault(g, dict(midi=[], hf=[], ap48=[], flat=[], unv_hi=0, frames=0, total_s=0, files=[]))
    d["midi"].append(midi(f["f0"])[m]); d["hf"].append(f["hf_6_16k_db"][m]); d["ap48"].append(f["ap_4_8k"][m]); d["flat"].append(f["flat_4_12k"][m])
    act = f["rms_db"] > np.nanmax(f["rms_db"]) - 30
    d["unv_hi"] += int((act & (f["f0"] <= 0)).sum()); d["frames"] += int(act.sum())
    d["total_s"] += len(f["f0"]) * 0.01; d["files"].append(stem)
for d in groups.values():
    for k in ("midi", "hf", "ap48", "flat"):
        d[k] = np.concatenate(d[k])
order = ["복면가왕 live (6 songs)", "킬링보이스 live medley", "studio/home (MV, 규티비)"]
gcol = {order[0]: "#e45756", order[1]: "#f58518", order[2]: "#4c78a8"}
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
hb = np.arange(40, 84)
cover_x = src_m[ms]
ax = axes[0]
bottom = np.zeros(len(hb) - 1)
for g in order:
    h, _ = np.histogram(groups[g]["midi"], bins=hb); h = h * 0.01
    ax.bar(hb[:-1] + 0.5, h, bottom=bottom, width=0.9, color=gcol[g], label=f"{g}: {groups[g]['total_s']/60:.1f} min")
    bottom += h
hc, _ = np.histogram(cover_x, bins=hb)
ax2 = ax.twinx(); ax2.step(hb[:-1] + 0.5, hc * 0.01, where="mid", color="k", lw=1.5, label="cover source (Tim) voiced s")
ax2.set_ylabel("cover source voiced seconds")
ax.axvline(high_thr - 2, color="k", ls=":"); ax.set_xlabel("MIDI pitch"); ax.set_ylabel("training voiced seconds")
tick = np.arange(40, 84, 4); ax.set_xticks(tick); ax.set_xticklabels([f"{t}\n{note_name(t)}" for t in tick], fontsize=8)
ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right")
ax.set_title("Training voiced time per semitone by source type", loc="left", fontsize=10)
out["train"] = {}
tot_hi = sum(((groups[g]["midi"] >= high_thr - 2).sum()) for g in order) * 0.01
for g in order:
    x = groups[g]["midi"]
    out["train"][g] = dict(total_min=round(groups[g]["total_s"] / 60, 2), voiced_min=round(len(x) * 0.01 / 60, 2),
                           s_at_or_above_high=round(float((x >= high_thr - 2).sum() * 0.01), 1),
                           share_of_high=round(float((x >= high_thr - 2).sum() * 0.01 / max(tot_hi, 1e-9)), 3),
                           s_at_or_above_top=round(float((x >= high_thr).sum() * 0.01), 1),
                           unvoiced_active_frac=round(d["unv_hi"] / max(d["frames"], 1), 3) if False else round(groups[g]["unv_hi"] / max(groups[g]["frames"], 1), 3))
out["cover_high_s"] = round(float((cover_x >= high_thr - 2).sum() * 0.01), 1)
# shares
ax = axes[1]
names = order
vals = [out["train"][g]["s_at_or_above_high"] for g in names]
ax.bar(range(3), vals, color=[gcol[g] for g in names])
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.0f}s\n({100*out['train'][names[i]]['share_of_high']:.0f}%)", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
ax.set_ylabel(f"voiced seconds >= {note_name(high_thr-2)} (MIDI {high_thr-2:.0f})")
ax.set_title(f"Where the high-note training data comes from", loc="left", fontsize=10)
# noise proxy vs pitch by group
ax = axes[2]
for g in order:
    x = groups[g]["midi"]; y = groups[g]["flat"]
    med = [np.nanmedian(y[(x >= b) & (x < b + 1)]) if ((x >= b) & (x < b + 1)).sum() > 50 else np.nan for b in hb[:-1]]
    ax.plot(hb[:-1] + 0.5, med, color=gcol[g], lw=2, marker="o", ms=3, label=g)
    lo = x < high_thr - 7; hi = x >= high_thr - 2
    out["train"][g].update(flat_low=float(np.nanmedian(y[lo])), flat_high=float(np.nanmedian(y[hi])),
                           ap48_low=float(np.nanmedian(groups[g]["ap48"][lo])), ap48_high=float(np.nanmedian(groups[g]["ap48"][hi])),
                           hf_low=float(np.nanmedian(groups[g]["hf"][lo])), hf_high=float(np.nanmedian(groups[g]["hf"][hi])))
ax.axvline(high_thr - 2, color="k", ls=":")
ax.set_xlabel("MIDI pitch"); ax.set_ylabel("spectral flatness 4-12 kHz (voiced frames)")
ax.set_title("Noise proxy in voiced training frames vs pitch", loc="left", fontsize=10); ax.legend(fontsize=8)
ax.set_xticks(tick); ax.set_xticklabels([f"{t}\n{note_name(t)}" for t in tick], fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(D, "C_training_pitch_coverage.png")); plt.close(fig)
json.dump(out, open(os.path.join(D, "C_stats.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
