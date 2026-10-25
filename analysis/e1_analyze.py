"""E1 analysis exactly as pre-registered (XAI-Cyber_E1_protocol.md)."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import glob, json
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from sklearn.metrics import roc_auc_score

EV = {"abs": 0, "rel": 1, "sign": 2}
BASE = ["S", "bootSD", "|phi|", "bootSD/|phi|"]
ZM = ["CL", "PROV", "PROV-AUD", "CONST", "ORACLE"]

def auc_mean(y, s):
    a = [roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].mean() < 1]
    return float(np.mean(a)) if a else np.nan

runs = []
for f in sorted(glob.glob(f"{ROOT}/results/e1/e1_*.npz")):
    z = np.load(f); meta = json.load(open(f.replace(".npz", ".json")))
    phi = np.abs(z["phi_n"]); sdf = z["phi_n"].std(0) + 1e-12
    base = {"S": z["S"], "bootSD": z["bootSD"], "|phi|": phi, "bootSD/|phi|": z["bootSD"] / (phi + 0.05 * sdf)}
    r = {"meta": meta, "auc": {}, "cal": {}, "strata": {}}
    modes = [k[2:] for k in z.files if k.startswith("P_")]
    for e, k in EV.items():
        y = z[f"ev_{e}"]
        for b, s in base.items(): r["auc"][(e, b)] = auc_mean(y, s)
        for m in modes:
            r["auc"][(e, "Z[" + m + "]")] = auc_mean(y, z["P_" + m][k])
            r["cal"][(e, m)] = (float(z["P_" + m][k].mean()), float(y.mean()), float(((z["P_" + m][k] - y) ** 2).mean()))
        for st in np.unique(z["strata"]):
            msk = z["strata"] == st
            r["strata"][(e, st, "rate")] = float(y[msk].mean())
            r["strata"][(e, st, "Z[PROV]")] = auc_mean(y[msk], z["P_PROV"][k][msk])
            r["strata"][(e, st, "best_base")] = np.nanmax([auc_mean(y[msk], s[msk]) for s in base.values()])
    runs.append(r)
print(f"runs: {len(runs)}  (seeds {sorted(set(r['meta']['seed'] for r in runs))[:3]}..., FS {sorted(set(r['meta']['fs'] for r in runs))})")
print(f"TSK AUC(clean) {np.mean([r['meta']['tsk_auc_clean'] for r in runs]):.3f} ± {np.std([r['meta']['tsk_auc_clean'] for r in runs]):.3f} | "
      f"HistGB {np.mean([r['meta']['gb_auc_clean'] for r in runs]):.3f} ± {np.std([r['meta']['gb_auc_clean'] for r in runs]):.3f}")
for k in ["CL", "PROV", "PROV-AUD"]:
    print(f"rho-AUC {k}: {np.mean([r['meta']['rho_auc'][k] for r in runs]):.3f}")
print(f"max lattice lost mass: {max(r['meta']['lattice_lost_max'] for r in runs):.1e}")

def arr(e, m): return np.array([r["auc"][(e, m)] for r in runs])
pvals = {}
for e in ["rel", "sign", "abs"]:
    bb = np.array([np.nanmax([r["auc"][(e, b)] for b in BASE]) for r in runs])
    print(f"\n[{e}]  best-baseline/run {np.nanmean(bb):.3f}")
    for m in BASE + ["Z[" + x + "]" for x in ZM]:
        v = arr(e, m); print(f"   {m:14s} {np.nanmean(v):.3f} ± {np.nanstd(v):.3f}")
    zp = arr(e, "Z[PROV]")
    p = wilcoxon(zp, bb, alternative="greater").pvalue; pvals[e] = p
    print(f"   H1  Z[PROV] - best baseline: {np.nanmean(zp - bb):+.3f}, wins {int((zp > bb).sum())}/{len(zp)}, one-sided Wilcoxon p = {p:.2e}")
    for b in BASE:
        vb = arr(e, b); print(f"       vs {b:13s}: {np.nanmean(zp - vb):+.3f}  p = {wilcoxon(zp, vb, alternative='greater').pvalue:.1e}")
    zc = arr(e, "Z[CONST]")
    print(f"   H2  Z[PROV] - Z[CONST]: {np.nanmean(zp - zc):+.3f}, wins {int((zp > zc).sum())}/{len(zp)}, p = {wilcoxon(zp, zc, alternative='greater').pvalue:.2e}")
    za = arr(e, "Z[PROV-AUD]")
    print(f"       Z[PROV-AUD] - best baseline: {np.nanmean(za - bb):+.3f}, p = {wilcoxon(za, bb, alternative='greater').pvalue:.2e}")
    for m in ["PROV", "PROV-AUD", "CL", "ORACLE"]:
        c = np.array([r["cal"][(e, m)] for r in runs])
        print(f"   calib {m:9s}: meanP {c[:,0].mean():.3f} vs rate {c[:,1].mean():.3f}  |gap| {np.abs(c[:,0]-c[:,1]).mean():.3f}  Brier {c[:,2].mean():.4f}")
    # Friedman + mean ranks (methods as treatments, runs as blocks)
    meths = BASE + ["Z[CL]", "Z[PROV]", "Z[PROV-AUD]", "Z[CONST]"]
    M = np.column_stack([arr(e, m) for m in meths]); M = M[~np.isnan(M).any(1)]
    st = friedmanchisquare(*M.T)
    ranks = np.mean([rankdata(-row) for row in M], 0)
    k, n = M.shape[1], M.shape[0]; cd = 3.031 * np.sqrt(k * (k + 1) / (6 * n))   # q_0.05 for k=8 (Demsar 2006)
    print(f"   Friedman chi2 = {st.statistic:.1f}, p = {st.pvalue:.1e}; Nemenyi CD = {cd:.2f}; mean ranks: " +
          ", ".join(f"{m}={rk:.2f}" for m, rk in sorted(zip(meths, ranks), key=lambda t: t[1])))
    for stt in ["S1_wed", "S2_thu_pm", "S3_fri_ddos", "S4_clean"]:
        rate = np.mean([r["strata"][(e, stt, "rate")] for r in runs])
        zs = np.nanmean([r["strata"][(e, stt, "Z[PROV]")] for r in runs]); bs = np.nanmean([r["strata"][(e, stt, "best_base")] for r in runs])
        print(f"   stratum {stt:12s}: event rate {rate:.3f} | Z[PROV] {zs:.3f} vs best base {bs:.3f}")
# Holm over the two primary events
ps = sorted([(pvals["rel"], "rel"), (pvals["sign"], "sign")])
print("\nHolm (primary rel, sign):", [(e, min(1, p * (2 - i))) for i, (p, e) in enumerate(ps)])
# sensitivity (seeds 100-109, FS-A)
sens = [r for r in runs if any(k[1].startswith("Z[PROV_") for k in r["auc"])]
if sens:
    print(f"\nSensitivity (n = {len(sens)} runs):")
    for e in ["rel", "sign"]:
        keys = sorted({k[1] for r in sens for k in r["auc"] if k[0] == e and k[1].startswith("Z[PROV")})
        print(f"  [{e}] " + "  ".join(f"{k}={np.nanmean([r['auc'][(e, k)] for r in sens]):.3f}" for k in keys))
