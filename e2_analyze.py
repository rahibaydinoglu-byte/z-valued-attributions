"""E2 analysis exactly as pre-registered (XAI-Cyber_E2_protocol.md)."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import glob, json
import numpy as np
from scipy.stats import wilcoxon, spearmanr, friedmanchisquare, rankdata
from sklearn.metrics import roc_auc_score
EV = {"abs": 0, "rel": 1, "sign": 2}
BASE = ["S", "bootSD", "|phi|", "bootSD/|phi|"]
ZM = ["CL", "PROV", "PROV+ii", "BLK-FS", "CONST", "ORACLE"]
def auc(y, s):
    a = [roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].mean() < 1]
    return float(np.mean(a)) if a else np.nan
def holm(d):
    ps = sorted(d.items(), key=lambda t: t[1]); out, run = {}, 0
    for i, (k, p) in enumerate(ps):
        run = max(run, min(1, p * (len(ps) - i))); out[k] = run
    return out
R = []
for f in sorted(glob.glob(f"{ROOT}/results/e2/e2_*.npz")):
    z = np.load(f); m = json.load(open(f.replace(".npz", ".json")))
    phi = np.abs(z["phi_n"]); sdf = z["phi_n"].std(0) + 1e-12
    base = {"S": z["S"], "bootSD": z["bootSD"], "|phi|": phi, "bootSD/|phi|": z["bootSD"] / (phi + 0.05 * sdf)}
    r = {"meta": m, "auc": {}, "brier": {}, "gap": {}, "strata": {}}
    for e, k in EV.items():
        y = z[f"ev_{e}"]
        for b, s in base.items(): r["auc"][(e, b)] = auc(y, s)
        for mm in ZM:
            p = z["P_" + mm][k]; r["auc"][(e, mm)] = auc(y, p)
            r["brier"][(e, mm)] = float(((p - y) ** 2).mean()); r["gap"][(e, mm)] = (float(p.mean()), float(y.mean()))
        r["best"] = r.get("best", {}); r["best"][e] = np.nanmax([r["auc"][(e, b)] for b in BASE])
        for st in ("near", "far"):
            msk = z["strata"] == st
            r["strata"][(e, st)] = (float(y[msk].mean()), auc(y[msk], z["P_BLK-FS"][k][msk]),
                                    np.nanmax([auc(y[msk], s[msk]) for s in base.values()]))
    R.append(r)
DS = sorted({r["meta"]["dataset"] for r in R})
print("runs", len(R), {d: sum(r["meta"]["dataset"] == d for r in R) for d in DS})
for d in DS:
    rr = [r for r in R if r["meta"]["dataset"] == d]
    mt = lambda k: np.mean([r["meta"][k] for r in rr])
    print(f"\n=== {d}: noise {mt('noise'):.3f} (M1a {np.mean([r['meta']['mech']['M1a'] for r in rr]):.3f}, M1b {np.mean([r['meta']['mech']['M1b'] for r in rr]):.3f}, M2 {np.mean([r['meta']['mech']['M2'] for r in rr]):.4f}) | "
          f"prior rho-AUC {mt('prior_rho_auc_pool'):.3f} | TSK {mt('tsk_auc_clean'):.3f} vs GB {mt('gb_auc_clean'):.3f} | lost max {max(r['meta']['lattice_lost_max'] for r in rr):.1e}")
    for e in ["rel", "sign", "abs"]:
        bb = np.array([r["best"][e] for r in rr])
        line = "  ".join(f"{mm}={np.nanmean([r['auc'][(e, mm)] for r in rr]):.3f}" for mm in BASE + ZM)
        print(f"  [{e}] best-base {np.nanmean(bb):.3f} | {line}")
H1, H2 = {}, {}
for d in DS:
    rr = [r for r in R if r["meta"]["dataset"] == d]
    for e in ["rel", "sign"]:
        z_ = np.array([r["auc"][(e, "BLK-FS")] for r in rr]); bb = np.array([r["best"][e] for r in rr]); c = np.array([r["auc"][(e, "CONST")] for r in rr])
        ok = ~np.isnan(z_) & ~np.isnan(bb)
        H1[(d, e)] = (wilcoxon(z_[ok], bb[ok], alternative="greater").pvalue, np.mean(z_[ok] - bb[ok]), int((z_[ok] > bb[ok]).sum()), int(ok.sum()))
        ok2 = ~np.isnan(z_) & ~np.isnan(c)
        H2[(d, e)] = (wilcoxon(z_[ok2], c[ok2], alternative="greater").pvalue, np.mean(z_[ok2] - c[ok2]))
h1 = holm({k: v[0] for k, v in H1.items()}); h2 = holm({k: v[0] for k, v in H2.items()})
print("\nH1-E2  Z[BLK-FS] > best baseline (per dataset, Holm over 6):")
for k, v in H1.items(): print(f"   {k[0]:18s} {k[1]:4s}: diff {v[1]:+.3f}  wins {v[2]}/{v[3]}  p {v[0]:.2e}  Holm {h1[k]:.2e}")
print("H2-E2  Z[BLK-FS] > Z[CONST] (Holm over 6):")
for k, v in H2.items(): print(f"   {k[0]:18s} {k[1]:4s}: diff {v[1]:+.3f}  Holm p {h2[k]:.2e}")
H8 = {}
for e in ["rel", "sign"]:
    g = np.array([r["auc"][(e, "BLK-FS")] - r["best"][e] for r in R]); pa = np.array([r["meta"]["prior_rho_auc_pool"] for r in R])
    ok = ~np.isnan(g); rho_s, p2 = spearmanr(pa[ok], g[ok]); H8[e] = (rho_s, p2 / 2 if rho_s > 0 else 1 - p2 / 2)
h8 = holm({e: v[1] for e, v in H8.items()})
print("H8 dose-response Spearman(prior rho-AUC, gain):", {e: (round(v[0], 3), f"Holm p {h8[e]:.2e}") for e, v in H8.items()})
H5 = {}
for e in ["rel", "sign"]:
    a = np.array([r["brier"][(e, "BLK-FS")] for r in R]); b = np.array([r["brier"][(e, "PROV")] for r in R])
    H5[e] = (wilcoxon(a, b, alternative="less").pvalue, np.mean(a - b), int((a < b).sum()))
h5 = holm({e: v[0] for e, v in H5.items()})
print("H5-E2 Brier BLK-FS < PROV:", {e: (round(v[1], 4), f"{v[2]}/{len(R)}", f"Holm p {h5[e]:.2e}") for e, v in H5.items()})
for e in ["rel", "sign"]:
    for mm in ["PROV", "BLK-FS", "ORACLE"]:
        g = np.array([r["gap"][(e, mm)] for r in R]); print(f"   calib [{e}] {mm:7s} meanP {g[:,0].mean():.3f} vs rate {g[:,1].mean():.3f}")
for e in ["rel", "sign", "abs"]:
    meths = BASE + ["CL", "PROV", "PROV+ii", "BLK-FS", "CONST"]
    M = np.array([[r["auc"][(e, m)] for m in meths] for r in R]); M = M[~np.isnan(M).any(1)]
    k, n = M.shape[1], M.shape[0]; ranks = np.mean([rankdata(-row) for row in M], 0)
    cd = 3.102 * np.sqrt(k * (k + 1) / (6 * n))   # q_0.05, k = 9 (Demsar 2006)
    print(f"Friedman [{e}] n={n} p={friedmanchisquare(*M.T).pvalue:.1e} CD={cd:.2f}: " + ", ".join(f"{m}={x:.2f}" for m, x in sorted(zip(meths, ranks), key=lambda t: t[1])))
for d in DS:
    rr = [r for r in R if r["meta"]["dataset"] == d]
    for e in ["rel", "sign"]:
        s = " | ".join(f"{st}: rate {np.nanmean([r['strata'][(e, st)][0] for r in rr]):.3f}, Z {np.nanmean([r['strata'][(e, st)][1] for r in rr]):.3f} vs base {np.nanmean([r['strata'][(e, st)][2] for r in rr]):.3f}" for st in ("near", "far"))
        print(f"  strata {d[:16]:16s} [{e}] {s}")
