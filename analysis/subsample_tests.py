"""Subsample-level primary tests for E1 and E1c (n = 20; feature sets averaged), fixed comparator, and feature set B alone."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import glob, numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
def auc(y, s):
    a = [roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].mean() < 1]; return np.mean(a) if a else np.nan
for study, dirn, zk in (("E1", "e1", "P_PROV"), ("E1c", "e1c", "P_BLK-FS")):
    for e, k in (("rel", 1), ("sign", 2)):
        per = {}
        for f in sorted(glob.glob(f"{ROOT}/results/{dirn}/e1_*_[AB].npz")):
            z = np.load(f); s = int(f.split("_")[-2]); fs = f[-5]; y = z[f"ev_{e}"]; sdf = z["phi_n"].std(0) + 1e-12
            b = [auc(y, z["S"]), auc(y, z["bootSD"]), auc(y, np.abs(z["phi_n"])), auc(y, z["bootSD"] / (np.abs(z["phi_n"]) + 0.05 * sdf))]
            per.setdefault(s, {})[fs] = (auc(y, z[zk][k]), max(b), b[3])
        seeds = sorted(per)
        if study == "E1": seeds = [s for s in seeds if 100 <= s <= 119]
        D = np.array([[np.mean([per[s][f][i] for f in per[s]]) for i in range(3)] for s in seeds])
        d, d2 = D[:, 0] - D[:, 1], D[:, 0] - D[:, 2]
        DB = np.array([per[s]["B"] for s in seeds]); dB = DB[:, 0] - DB[:, 1]
        print(f"{study} {e}: Z {D[:,0].mean():.3f} best {D[:,1].mean():.3f} diff {d.mean():+.3f} ({(d>0).sum()}/20, p {wilcoxon(d, alternative='greater').pvalue:.1e}) | "
              f"fixed {D[:,2].mean():.3f} diff {d2.mean():+.3f} (p {wilcoxon(d2, alternative='greater').pvalue:.1e}) | FS-B diff {dB.mean():+.3f} ({(dB>0).sum()}/20, p {wilcoxon(dB, alternative='greater').pvalue:.1e})")
