import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import json, glob, numpy as np
from scipy.stats import wilcoxon
R = [json.load(open(f"{ROOT}/results/e8/e8_{s}.json")) for s in range(141, 151)]
print(f"n={len(R)} | TSK AUC {np.mean([r['tsk_auc'] for r in R]):.3f}  GB AUC {np.mean([r['gb_auc'] for r in R]):.3f} | sec/run {np.mean([r['seconds'] for r in R]):.0f}")
ps = {}
for e in ("rel", "sign"):
    z = np.array([r[f"{e}|Z"] for r in R]); b = np.array([r[f"{e}|best"] for r in R]); fx = np.array([r[f"{e}|bootSD/|phi|"] for r in R])
    ps[e] = wilcoxon(z - b, alternative="greater").pvalue
    print(f"[{e}] rate {np.mean([r[f'{e}|rate'] for r in R]):.3f} meanP {np.mean([r[f'{e}|meanP'] for r in R]):.3f} brier {np.mean([r[f'{e}|brier'] for r in R]):.4f} | "
          f"Z {z.mean():.3f} best {b.mean():.3f} diff {np.mean(z-b):+.3f} wins {(z>b).sum()}/10 p {ps[e]:.1e} | fixed {fx.mean():.3f} diff {np.mean(z-fx):+.3f} wins {(z>fx).sum()} p {wilcoxon(z-fx, alternative='greater').pvalue:.1e}")
    for k in ("S", "bootSD", "|phi|", "bootSD/|phi|"): print(f"     {k}: {np.mean([r[f'{e}|{k}'] for r in R]):.3f}")
p = sorted(ps.values()); print("Holm:", {k: min(1, v * (2 if v == p[0] else 1)) for k, v in ps.items()})
