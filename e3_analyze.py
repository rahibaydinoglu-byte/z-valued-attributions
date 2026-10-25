"""E3 analysis as pre-registered (XAI-Cyber_E3_protocol.md)."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import glob, json
import numpy as np
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score
EV = {"abs": 0, "rel": 1, "sign": 2}
BASE = ["S", "bootSD", "|phi|", "bootSD/|phi|"]
LEVELS = [1, 1.5, 2, 3, 5, 10, 20]
def auc(y, s):
    a = [roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].mean() < 1]
    return float(np.mean(a)) if a else np.nan
rows = []
for f in sorted(glob.glob(f"{ROOT}/results/e3/e3_*.npz")):
    m = json.load(open(f.replace(".npz", ".json"))); z3 = np.load(f)
    z2 = np.load(f"{ROOT}/results/e2/e2_{m['dataset']}_{m['seed']}.npz")
    phi = np.abs(z2["phi_n"]); sdf = z2["phi_n"].std(0) + 1e-12
    base = {"S": z2["S"], "bootSD": z2["bootSD"], "|phi|": phi, "bootSD/|phi|": z2["bootSD"] / (phi + 0.05 * sdf)}
    r = {"ds": m["dataset"], "seed": m["seed"], "noise": m["train_noise"], "caps": {(e["recall"], e["lift"]): e["capped"] for e in m["evidence"]}}
    for e, k in EV.items():
        y = z2[f"ev_{e}"]
        r[(e, "best")] = np.nanmax([auc(y, s) for s in base.values()])
        for key in z3.files:
            rr, ll = key[3:].split("_L"); r[(e, float(rr), float(ll))] = auc(y, z3[key][k])
    rows.append(r)
DS = sorted({r["ds"] for r in rows})
print("runs", len(rows), {d: sum(r["ds"] == d for r in rows) for d in DS})
summary = {}
for d in DS:
    R = [r for r in rows if r["ds"] == d]
    print(f"\n=== {d} (train noise {np.mean([r['noise'] for r in R]):.3f}; capped levels: {sorted({l for r in R for (rc, l), c in r['caps'].items() if c and rc == 0.9})})")
    for e in ["rel", "sign", "abs"]:
        bb = np.array([r[(e, "best")] for r in R])
        means = [np.nanmean([r[(e, 0.9, float(l))] for r in R]) for l in LEVELS]
        print(f"  [{e}] best baseline {np.nanmean(bb):.3f} | " + "  ".join(f"L{l}={m_:.3f}" for l, m_ in zip(LEVELS, means)) +
              f" | r=0.5: L3={np.nanmean([r[(e, 0.5, 3.0)] for r in R]):.3f} L10={np.nanmean([r[(e, 0.5, 10.0)] for r in R]):.3f}")
        if e in ("rel", "sign"):
            rho_s = spearmanr(LEVELS, means).correlation
            top, low = np.array([r[(e, 0.9, 20.0)] for r in R]), np.array([r[(e, 0.9, 1.0)] for r in R])
            ok = ~np.isnan(top) & ~np.isnan(low); p_top = wilcoxon(top[ok], low[ok], alternative="greater").pvalue
            # fixed-sequence from highest lift downward
            Lstar, seq = "none", []
            for l in reversed(LEVELS):
                z = np.array([r[(e, 0.9, float(l))] for r in R]); ok = ~np.isnan(z) & ~np.isnan(bb)
                p = wilcoxon(z[ok], bb[ok], alternative="greater").pvalue; seq.append((l, round(np.mean(z[ok] - bb[ok]), 3), p))
                if p < 0.05: Lstar = l
                else: break
            summary[(d, e)] = Lstar
            print(f"      H9 Spearman(L, mean AUROC) = {rho_s:.3f}; L20 > L1 p = {p_top:.1e}")
            print(f"      fixed sequence (L, mean diff vs best baseline, p): {[(l, dd, f'{p:.1e}') for l, dd, p in seq]} -> L* = {Lstar}")
print("\nL* summary:", summary)
