"""E2 revision analyses (Sec. 7.4): Z vs fixed comparator per dataset/event; NF-UNSW-NB15-v3 per-run diagnostics."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import json, glob, numpy as np
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score
def auc(y, s):
    a = [roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].mean() < 1]; return np.mean(a) if a else np.nan
for ds in ("NF-CICIDS2018-v3", "NF-ToN-IoT-v3", "NF-UNSW-NB15-v3"):
    for e, k in (("rel", 1), ("sign", 2)):
        Z, F = [], []
        for f in sorted(glob.glob(f"{ROOT}/results/e2/e2_{ds}_*.npz")):
            z = np.load(f); y = z[f"ev_{e}"]; sdf = z["phi_n"].std(0) + 1e-12
            if y.sum() == 0: continue
            fx = auc(y, z["bootSD"] / (np.abs(z["phi_n"]) + 0.05 * sdf)); zz = auc(y, z["P_BLK-FS"][k])
            if not (np.isnan(fx) or np.isnan(zz)): Z.append(zz); F.append(fx)
        Z, F = np.array(Z), np.array(F); d = Z - F
        print(f"{ds} {e}: n {len(d)} Z {Z.mean():.3f} fixed {F.mean():.3f} diff {d.mean():+.3f} wins {(d>0).sum()} two-sided p {wilcoxon(d).pvalue:.2g}")
rows = []
for f in sorted(glob.glob(f"{ROOT}/results/e2/e2_NF-UNSW-NB15-v3_*.npz")):
    z = np.load(f); m = json.load(open(f.replace(".npz", ".json"))); y = z["ev_rel"]; sdf = z["phi_n"].std(0) + 1e-12
    if y.sum() == 0: continue
    b = [auc(y, z["S"]), auc(y, z["bootSD"]), auc(y, np.abs(z["phi_n"])), auc(y, z["bootSD"] / (np.abs(z["phi_n"]) + 0.05 * sdf))]
    rows.append((m["tsk_auc_clean"], auc(y, z["P_BLK-FS"][1]), np.nanmax(b), *b))
r = np.array(rows, float); ok = ~np.isnan(r[:, 1]) & ~np.isnan(r[:, 2])
print("UNSW rel: mean per-run best %.3f; S %.3f bootSD %.3f |phi| %.3f ratio %.3f" % (np.nanmean(r[:, 2]), *np.nanmean(r[:, 3:], 0)))
print("UNSW rel: Spearman(TSK clean AUROC, Z - best) %.2f p %.2g" % spearmanr(r[ok, 0], r[ok, 1] - r[ok, 2]))
