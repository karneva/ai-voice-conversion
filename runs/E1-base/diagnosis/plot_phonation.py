"""C/E link: high-note phonation (H1-H2, alpha ratio) vs pitch for each training source type, the conversion and Kyuhyun's real cover."""
from common import *
import glob
G = {}
for p in sorted(glob.glob(os.path.join(C, "feat_train__*.npz"))):
    stem = os.path.basename(p)[12:-4]; f = feat("train__" + stem); g = train_group(stem)
    if g.startswith("studio"):
        g = "규티비 home covers (train)" if any(stem.startswith(s) for s in ("너였다면", "사건의", "희재")) else "첫눈처럼 MV studio (train)"
    elif g.startswith("복면"): g = "복면가왕 live (train)"
    else: g = "킬링보이스 live (train)"
    m = voiced_mask(f, 40)
    d = G.setdefault(g, dict(x=[], h1h2=[], alpha=[]))
    d["x"].append(midi(f["f0"])[m]); d["h1h2"].append(f["h1h2"][m]); d["alpha"].append(f["alpha_db"][m])
F = {k: feat(k) for k in ("src", "c300", "c500", "ref")}
ms = voiced_mask(F["src"]) & (F["src"]["crepe_per"] > 0.5)
for k in ("src", "c300", "c500"):
    G[LABEL[k]] = dict(x=[midi(F["src"]["f0"])[ms]], h1h2=[F[k]["h1h2"][ms]], alpha=[F[k]["alpha_db"][ms]])
mr = voiced_mask(F["ref"]) & (F["ref"]["crepe_per"] > 0.5)
G[LABEL["ref"]] = dict(x=[midi(F["ref"]["f0"])[mr]], h1h2=[F["ref"]["h1h2"][mr]], alpha=[F["ref"]["alpha_db"][mr]])
style = {"복면가왕 live (train)": ("#e45756", ":", 1.6), "킬링보이스 live (train)": ("#f58518", ":", 1.6),
         "규티비 home covers (train)": ("#4c78a8", ":", 1.6), "첫눈처럼 MV studio (train)": ("#9d755d", ":", 1.2),
         LABEL["src"]: ("#72b7b2", "-", 2.0), LABEL["c300"]: ("#b279a2", "-", 2.0), LABEL["c500"]: ("#d62728", "-", 2.8),
         LABEL["ref"]: ("#2ca02c", "--", 2.8)}
bins = np.arange(46, 70)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
for ax, key, lab in zip(axes, ["h1h2", "alpha"], ["H1-H2 (dB, envelope)  higher = breathier/softer, lower = pressed",
                                                  "alpha ratio 1-5k vs <1k (dB)  higher = brighter/brassier"]):
    for g, d in G.items():
        x = np.concatenate(d["x"]); y = np.concatenate(d[key])
        med = [np.nanmedian(y[(x >= b) & (x < b + 1)]) if ((x >= b) & (x < b + 1)).sum() > 40 else np.nan for b in bins]
        c, ls, lw = style[g]
        ax.plot(bins + 0.5, med, color=c, ls=ls, lw=lw, marker="o" if ls != ":" else None, ms=3, label=g)
    ax.axvline(62, color="k", lw=0.8, ls="-.")
    ax.text(62.2, ax.get_ylim()[1] if False else 0.98, "D4", transform=ax.get_xaxis_transform(), fontsize=8, va="top")
    tick = np.arange(46, 70, 2); ax.set_xticks(tick); ax.set_xticklabels([f"{t}\n{note_name(t)}" for t in tick], fontsize=8)
    ax.set_xlabel("pitch (MIDI)"); ax.set_title(lab, fontsize=10, loc="left")
axes[0].legend(fontsize=8, ncol=2, loc="lower left")
fig.suptitle("High-note phonation: the conversion follows the live-stage belting profile, Kyuhyun's real (규티비) cover follows the home-cover profile",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(D, "CE_high_note_phonation_by_source.png")); plt.close(fig)
print("ok")
