"""Revision checks on the 40 E1c runs (exact reproduction, verified against saved phi_n):
 (a) M1.5 threshold sensitivity: kappa in {0.25,0.5,1} x tau0 in {0.025,0.05,0.1} (rel), dead zone tau0 (sign);
     scores: Gaussian BLK-FS (two moments) vs best label-blind baseline (baselines from saved npz; they do not depend on thresholds).
 (b) M2.2 Attempted -> attack policy: clean labels redefined; evidence and noisy labels unchanged; Gaussian and saved lattice scores.
 (c) M3.1 interval B: Gaussian P over the evidence box rho_win in [0.1,0.3], rho_pay in [0.3,0.7] (5x3 grid);
     does the width of B predict the error of its midpoint?
Usage: python3 e7_revision.py <seed> ..."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import json, math, os, sys
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pilot_controlled as P
import e1_full as E, e1_block as B
from block_noise import make_units, aggregate
OUT = f"{ROOT}/results/e7"; os.makedirs(OUT, exist_ok=True)
KAPPA, TAU0 = [0.25, 0.5, 1.0], [0.025, 0.05, 0.1]

def auc(y, s):
    a = [roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].mean() < 1]
    return float(np.mean(a)) if a else float("nan")

def gauss_rel(mu, sd, f0, tau):
    return norm.sf((tau - mu) / sd) + norm.cdf((-tau - mu) / sd)
def gauss_sign(mu, sd, f0, dz):
    return norm.cdf((-dz - f0 - mu) / sd) if f0 > 0 else norm.sf((dz - f0 - mu) / sd)

def events(phi_n, phi_c, sdf, kappa, tau0):
    delta = phi_c - phi_n
    rel = np.abs(delta) > kappa * np.abs(phi_n) + tau0 * sdf
    sign = (phi_c * phi_n < 0) & (np.maximum(np.abs(phi_c), np.abs(phi_n)) > tau0 * sdf)
    return rel, sign

def run(seed, fsname, d):
    feats = E.FS[fsname]; rng = np.random.default_rng(seed)
    tr_idx = rng.choice(d.index.values, E.NTRAIN, replace=False); tr = d.loc[tr_idx]; rest = d.drop(index=tr_idx)
    te_parts = []
    for s, fn in E.STRATA.items():
        cand = rest.index[fn(rest["file"]).values]; te_parts.append(rest.loc[rng.choice(cand, E.NPER, replace=False)])
    te = pd.concat(te_parts)
    sc = StandardScaler().fit(E.tf(tr[feats].values))
    Xtr, Xte = sc.transform(E.tf(tr[feats].values)), sc.transform(E.tf(te[feats].values))
    yn = tr["yo"].values.astype(float); yc = tr["yc"].values.astype(float)
    yc_att = (~tr["cc"].isin(["BENIGN"])).values.astype(float)          # Attempted -> attack
    bg = Xtr[rng.choice(E.NTRAIN, E.NBG, replace=False)]
    C, S = P.fit_antecedents(Xtr, seed); H = P.H_matrix(Xtr, C, S); sgn = 1 - 2 * yn
    cat = tr["cat"].values
    unit, rho_u0 = make_units(tr["blk_fs"].values, cat, E.prov(tr))
    ucat = np.zeros(len(rho_u0), np.int8)
    for j in range(E.NTRAIN): ucat[unit[j]] = cat[j]
    n_te, dd = len(te), len(feats)
    phi_n = np.zeros((n_te, dd)); phi_c = np.zeros_like(phi_n); phi_ca = np.zeros_like(phi_n); D = []
    for t, x in enumerate(Xte):
        Cm = P.shap_firing(x, bg, C, S) @ H
        phi_n[t], phi_c[t], phi_ca[t] = Cm @ yn, Cm @ yc, Cm @ yc_att
        D.append(np.stack([aggregate(Cm[i] * sgn, unit, len(rho_u0)) for i in range(dd)]))
    z = np.load(f"{ROOT}/results/e1c/e1_{seed}_{fsname}.npz")
    assert np.allclose(z["phi_n"], phi_n, atol=1e-10) and np.allclose(z["phi_c"], phi_c, atol=1e-10), "not reproduced"
    sdf = phi_n.std(0) + 1e-12
    base = {"S": z["S"], "bootSD": z["bootSD"], "|phi|": np.abs(phi_n)}
    def moments(rho_u):
        mu = np.array([[D[t][i] @ rho_u for i in range(dd)] for t in range(n_te)])
        sd = np.array([[math.sqrt(float((D[t][i] ** 2 * rho_u * (1 - rho_u)).sum())) + 1e-15 for i in range(dd)] for t in range(n_te)])
        return mu, sd
    mu, sd = moments(rho_u0)
    out = {}
    # (a) threshold sensitivity
    for kappa in KAPPA:
        for tau0 in TAU0:
            rel, sign = events(phi_n, phi_c, sdf, kappa, tau0)
            g_rel = np.vectorize(gauss_rel)(mu, sd, phi_n, kappa * np.abs(phi_n) + tau0 * sdf)
            bases = dict(base); bases["bootSD/|phi|"] = z["bootSD"] / (np.abs(phi_n) + tau0 * sdf)
            out[f"thr|rel|{kappa}|{tau0}|Z"] = auc(rel, g_rel)
            out[f"thr|rel|{kappa}|{tau0}|best"] = float(np.nanmax([auc(rel, s) for s in bases.values()]))
            out[f"thr|rel|{kappa}|{tau0}|fixed"] = auc(rel, bases["bootSD/|phi|"])
            out[f"thr|rel|{kappa}|{tau0}|rate"] = float(rel.mean())
            if kappa == 0.5:
                g_sign = np.vectorize(gauss_sign)(mu, sd, phi_n, tau0 * sdf + 0 * phi_n)
                out[f"thr|sign|{tau0}|Z"] = auc(sign, g_sign)
                out[f"thr|sign|{tau0}|best"] = float(np.nanmax([auc(sign, s) for s in bases.values()]))
                out[f"thr|sign|{tau0}|fixed"] = auc(sign, bases["bootSD/|phi|"])
                out[f"thr|sign|{tau0}|rate"] = float(sign.mean())
    # (b) Attempted -> attack
    rel_a, sign_a = events(phi_n, phi_ca, sdf, 0.5, 0.05)
    g_rel = np.vectorize(gauss_rel)(mu, sd, phi_n, 0.5 * np.abs(phi_n) + 0.05 * sdf)
    g_sign = np.vectorize(gauss_sign)(mu, sd, phi_n, 0.05 * sdf + 0 * phi_n)
    bases = dict(base); bases["bootSD/|phi|"] = z["bootSD"] / (np.abs(phi_n) + 0.05 * sdf)
    for e, y, g, k in (("rel", rel_a, g_rel, 1), ("sign", sign_a, g_sign, 2)):
        out[f"att|{e}|Zgauss"] = auc(y, g); out[f"att|{e}|Zlattice"] = auc(y, z["P_BLK-FS"][k])
        out[f"att|{e}|best"] = float(np.nanmax([auc(y, s) for s in bases.values()])); out[f"att|{e}|rate"] = float(y.mean())
        out[f"att|{e}|brier_gauss"] = float(((g - y) ** 2).mean())
    out["att|train_flip"] = float((yn != yc_att).mean()); out["main|train_flip"] = float((yn != yc).mean())
    # (c) interval B over the evidence box
    rel, sign = events(phi_n, phi_c, sdf, 0.5, 0.05)
    P_rel, P_sign = [], []
    for rw in (0.1, 0.15, 0.2, 0.25, 0.3):
        for rp in (0.3, 0.5, 0.7):
            r_u = np.where(ucat == 0, rw, np.where(ucat == 2, rp, rho_u0))
            m_, s_ = moments(r_u)
            P_rel.append(np.vectorize(gauss_rel)(m_, s_, phi_n, 0.5 * np.abs(phi_n) + 0.05 * sdf))
            P_sign.append(np.vectorize(gauss_sign)(m_, s_, phi_n, 0.05 * sdf + 0 * phi_n))
    for e, Ps, y in (("rel", np.array(P_rel), rel), ("sign", np.array(P_sign), sign)):
        lo, hi = Ps.min(0), Ps.max(0); mid = (lo + hi) / 2; w = hi - lo
        err = np.abs(mid - y)
        out[f"int|{e}|width_mean"] = float(w.mean())
        out[f"int|{e}|spearman_width_err"] = float(spearmanr(w.ravel(), err.ravel()).correlation)
        # width vs error WITHIN deciles of the midpoint (removes the trivial dependence of both on the midpoint)
        mv, wv, ev = mid.ravel(), w.ravel(), err.ravel()
        qs = np.quantile(mv, np.linspace(0, 1, 11)); rs = []
        for k_ in range(10):
            msk = (mv >= qs[k_]) & (mv <= qs[k_ + 1])
            if msk.sum() > 20 and np.std(wv[msk]) > 0 and np.std(ev[msk]) > 0: rs.append(spearmanr(wv[msk], ev[msk]).correlation)
        out[f"int|{e}|spearman_within_mid"] = float(np.mean(rs)) if rs else float("nan")
        out[f"int|{e}|n_bins"] = len(rs)
        q = np.quantile(w, [1 / 3, 2 / 3])
        for name, msk in (("low", w <= q[0]), ("mid", (w > q[0]) & (w <= q[1])), ("high", w > q[1])):
            out[f"int|{e}|brier_{name}"] = float(((mid[msk] - y[msk]) ** 2).mean()) if msk.any() else float("nan")
    json.dump(out, open(f"{OUT}/e7_{seed}_{fsname}.json", "w"))
    print(seed, fsname, "ok", flush=True)

if __name__ == "__main__":
    d = B.load()
    for s in [int(x) for x in sys.argv[1:]]:
        for fs in ("A", "B"):
            if not os.path.exists(f"{OUT}/e7_{s}_{fs}.json"): run(s, fs, d)
