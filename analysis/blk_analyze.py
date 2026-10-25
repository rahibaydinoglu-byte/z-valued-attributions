import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import glob, sys, numpy as np
from sklearn.metrics import roc_auc_score
d = sys.argv[1]
EV = {"abs": 0, "rel": 1, "sign": 2}
def auc(y, s):
    a = [roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].mean() < 1]
    return np.mean(a) if a else np.nan
fs = sorted(glob.glob(f"{d}/blk_*.npz")); print("runs", len(fs))
modes = [k[2:] for k in np.load(fs[0]).files if k.startswith("P_")]
for e, k in EV.items():
    print(f"[{e}]")
    for m in modes:
        A, MP, RT, BR = [], [], [], []
        for f in fs:
            z = np.load(f); y = z[f"ev_{e}"]; p = z[f"P_{m}"][k]
            A.append(auc(y, p)); MP.append(p.mean()); RT.append(y.mean()); BR.append(((p - y) ** 2).mean())
        print(f"   {m:8s} AUROC {np.nanmean(A):.3f}  meanP {np.mean(MP):.3f} vs rate {np.mean(RT):.3f}  |gap| {np.mean(np.abs(np.array(MP)-np.array(RT))):.3f}  Brier {np.mean(BR):.4f}")
