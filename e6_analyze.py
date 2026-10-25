"""Gaussian vs lattice point value (Sec. 7.3, 'Is the exact law needed?'): run- and subsample-level summary of results/e6."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import json, glob, numpy as np
from scipy.stats import wilcoxon
R = {}
for f in glob.glob(f"{ROOT}/results/e6/e6_*.json"):
    s = int(f.split("_")[-2]); R.setdefault(s, []).append(json.load(open(f)))
for e in ("rel", "sign"):
    for n in ("ED", "MARKOV", "GAUSS", "LATTICE"):
        print(e, n, "AUROC %.3f" % np.mean([r[f"{e}|{n}|auc"] for rr in R.values() for r in rr]))
    for m in ("auc", "brier", "ece"):
        g = np.array([np.mean([r[f"{e}|GAUSS|{m}"] for r in R[s]]) for s in sorted(R)])
        l = np.array([np.mean([r[f"{e}|LATTICE|{m}"] for r in R[s]]) for s in sorted(R)])
        d = g - l; better = (d > 0) if m == "auc" else (d < 0)
        print(f"  [{e}] {m}: Gaussian {g.mean():.4f} lattice {l.mean():.4f} | Gaussian better in {better.sum()}/20 | two-sided Wilcoxon p {wilcoxon(d).pvalue:.1e}")
