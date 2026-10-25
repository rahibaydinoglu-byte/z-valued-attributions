"""E1c confirmation analysis (Amendment 1 of XAI-Cyber_E1_protocol.md)."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import glob, json
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from sklearn.metrics import roc_auc_score
EV = {"abs": 0, "rel": 1, "sign": 2}
BASE = ["S", "bootSD", "|phi|", "bootSD/|phi|"]
def auc(y, s):
    a = [roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].mean() < 1]
    return float(np.mean(a)) if a else np.nan
R = []
for f in sorted(glob.glob(f"{ROOT}/results/e1c/e1_*.npz")):
    z = np.load(f); phi = np.abs(z["phi_n"]); sdf = z["phi_n"].std(0) + 1e-12
    base = {"S": z["S"], "bootSD": z["bootSD"], "|phi|": phi, "bootSD/|phi|": z["bootSD"] / (phi + 0.05 * sdf)}
    r = {"auc": {}, "brier": {}, "gap": {}, "meta": json.load(open(f.replace(".npz", ".json")))}
    for e, k in EV.items():
        y = z[f"ev_{e}"]
        for b, s in base.items(): r["auc"][(e, b)] = auc(y, s)
        for m in ["CL", "PROV", "PROV-AUD", "CONST", "ORACLE", "BLK-FS", "BLK-FMH"]:
            p = z["P_" + m][k]
            r["auc"][(e, m)] = auc(y, p); r["brier"][(e, m)] = float(((p - y) ** 2).mean()); r["gap"][(e, m)] = (float(p.mean()), float(y.mean()))
    R.append(r)
print("runs", len(R), "| seeds", sorted({r['meta']['seed'] for r in R})[0], "..", sorted({r['meta']['seed'] for r in R})[-1])
print(f"TSK AUC {np.mean([r['meta']['tsk_auc_clean'] for r in R]):.3f} | GB {np.mean([r['meta']['gb_auc_clean'] for r in R]):.3f} | lost max {max(r['meta']['lattice_lost_max'] for r in R):.1e}")
A = lambda e, m: np.array([r["auc"][(e, m)] for r in R]); Bv = lambda e, m: np.array([r["brier"][(e, m)] for r in R])
p5, p6 = {}, {}
for e in ["rel", "sign", "abs"]:
    bb = np.array([np.nanmax([r["auc"][(e, b)] for b in BASE]) for r in R])
    print(f"\n[{e}] best baseline/run AUROC {bb.mean():.3f}")
    for m in ["PROV", "BLK-FS", "BLK-FMH", "PROV-AUD", "CONST", "CL", "ORACLE"]:
        g = np.array([r["gap"][(e, m)] for r in R])
        print(f"   {m:9s} AUROC {np.nanmean(A(e, m)):.3f} ± {np.nanstd(A(e, m)):.3f} | Brier {Bv(e, m).mean():.4f} | meanP {g[:,0].mean():.3f} vs rate {g[:,1].mean():.3f} (|gap| {np.abs(g[:,0]-g[:,1]).mean():.3f})")
    p5[e] = wilcoxon(Bv(e, "BLK-FS"), Bv(e, "PROV"), alternative="less").pvalue
    p6[e] = wilcoxon(A(e, "BLK-FS"), bb, alternative="greater").pvalue
    print(f"   H5 Brier BLK-FS < PROV: diff {np.mean(Bv(e,'BLK-FS')-Bv(e,'PROV')):+.4f}, wins {int((Bv(e,'BLK-FS')<Bv(e,'PROV')).sum())}/{len(R)}, p = {p5[e]:.2e}")
    print(f"   H6 AUROC BLK-FS > best baseline: diff {np.nanmean(A(e,'BLK-FS')-bb):+.3f}, wins {int((A(e,'BLK-FS')>bb).sum())}/{len(R)}, p = {p6[e]:.2e}")
    print(f"   H7 AUROC BLK-FS vs PROV: diff {np.nanmean(A(e,'BLK-FS')-A(e,'PROV')):+.3f}, two-sided p = {wilcoxon(A(e,'BLK-FS'), A(e,'PROV')).pvalue:.2e}")
    print(f"   H1-rep PROV > best baseline: diff {np.nanmean(A(e,'PROV')-bb):+.3f}, p = {wilcoxon(A(e,'PROV'), bb, alternative='greater').pvalue:.2e}; "
          f"H2-rep PROV > CONST: diff {np.nanmean(A(e,'PROV')-A(e,'CONST')):+.3f}, p = {wilcoxon(A(e,'PROV'), A(e,'CONST'), alternative='greater').pvalue:.2e}")
    meths = BASE + ["CL", "PROV", "PROV-AUD", "CONST", "BLK-FS", "BLK-FMH"]
    M = np.column_stack([A(e, m) for m in meths]); M = M[~np.isnan(M).any(1)]
    ranks = np.mean([rankdata(-row) for row in M], 0); k, n = M.shape[1], M.shape[0]
    cd = 3.164 * np.sqrt(k * (k + 1) / (6 * n))     # q_0.05, k = 10 (Demsar 2006)
    print(f"   Friedman p = {friedmanchisquare(*M.T).pvalue:.1e}; CD = {cd:.2f}; ranks: " + ", ".join(f"{m}={x:.2f}" for m, x in sorted(zip(meths, ranks), key=lambda t: t[1])))
def holm(d):
    ps = sorted(d.items(), key=lambda t: t[1]); return {e: min(1, p * (len(ps) - i)) for i, (e, p) in enumerate(ps)}
print("\nHolm H5 (rel, sign):", holm({e: p5[e] for e in ["rel", "sign"]}))
print("Holm H6 (rel, sign):", holm({e: p6[e] for e in ["rel", "sign"]}))
