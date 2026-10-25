"""Figure: E3 evidence-quality sweep (mean AUROC ± 1 s.e. over 20 runs) per dataset; dashed = best label-blind baseline."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import glob, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

EV = {"rel": 1, "sign": 2}
LEVELS = [1, 1.5, 2, 3, 5, 10, 20]
COL = {"rel": "#2a78d6", "sign": "#eb6834"}
NAME = {"rel": "relative change", "sign": "sign change"}
DS = [("NF-CICIDS2018-v3", "NF-CICIDS2018-v3"), ("NF-ToN-IoT-v3", "NF-ToN-IoT-v3"), ("NF-UNSW-NB15-v3", "NF-UNSW-NB15-v3")]
def auc(y, s):
    a = [roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].mean() < 1]
    return float(np.mean(a)) if a else np.nan
res = {}
for f in sorted(glob.glob(f"{ROOT}/results/e3/e3_*.npz")):
    m = json.load(open(f.replace(".npz", ".json"))); z3 = np.load(f)
    z2 = np.load(f"{ROOT}/results/e2/e2_{m['dataset']}_{m['seed']}.npz")
    phi = np.abs(z2["phi_n"]); sdf = z2["phi_n"].std(0) + 1e-12
    base = [z2["S"], z2["bootSD"], phi, z2["bootSD"] / (phi + 0.05 * sdf)]
    for e, k in EV.items():
        y = z2[f"ev_{e}"]
        res.setdefault((m["dataset"], e, "base"), []).append(np.nanmax([auc(y, s) for s in base]))
        for l in LEVELS:
            res.setdefault((m["dataset"], e, l), []).append(auc(y, z3[f"P_r0.9_L{l}"][k]))
json.dump({"|".join(map(str, k)): v for k, v in res.items()}, open(f"{ROOT}/figures/fig_e3_data.json", "w"))

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, (ds, title) in zip(axes, DS):
    for e in EV:
        v = [np.array(res[(ds, e, l)]) for l in LEVELS]
        mu = np.array([np.nanmean(x) for x in v]); se = np.array([np.nanstd(x) / np.sqrt(np.sum(~np.isnan(x))) for x in v])
        ax.plot(LEVELS, mu, color=COL[e], lw=2, marker="o", ms=4, label=f"Z, {NAME[e]}")
        ax.fill_between(LEVELS, mu - se, mu + se, color=COL[e], alpha=0.15, lw=0)
        b = np.nanmean(res[(ds, e, "base")])
        ax.axhline(b, color=COL[e], lw=1.2, ls="--", label=f"best baseline, {NAME[e]}")
    ax.set_xscale("log"); ax.set_xticks(LEVELS); ax.set_xticklabels(["1", "1.5", "2", "3", "5", "10", "20"])
    ax.minorticks_off(); ax.set_title(title, fontsize=9, color="#0b0b0b"); ax.set_xlabel("evidence lift L")
    ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_ylim(0.55, 1.01)
axes[0].set_ylabel("AUROC")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(f"{ROOT}/figures/fig_e3.pdf"); fig.savefig(f"{ROOT}/figures/fig_e3.png", dpi=200)
print("saved")
