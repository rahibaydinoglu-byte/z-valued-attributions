"""M2.3: 30-feature experiment with KernelSHAP on a FIXED coalition sample (linear in the explained function,
so Proposition 1 holds exactly). New subsamples (seeds 140-149), feature set FS-K30 (label-free rule, fs_k30.txt).
Scores: Z[BLK-FS] Gaussian (two moments) vs label-blind baselines (S, bootstrap s.d. over 8 refits, |phi|, bootSD/|phi|).
Usage: python3 e8_kernelshap.py <seed> ..."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import json, math, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import pilot_controlled as P
import e1_full as E, e1_block as B
from block_noise import make_units, aggregate
FEATS = open(f"{ROOT}/experiments/fs_k30.txt").read().split("\n")
M_COAL, NBG, NPER, NBOOT = 1024, 30, 30, 8
OUT = f"{ROOT}/results/e8"; os.makedirs(OUT, exist_ok=True)

def coalitions(d, M, rng):
    """Paired sample from the Shapley kernel over sizes 1..d-1 (complements included)."""
    sizes = np.arange(1, d); w = (d - 1) / (sizes * (d - sizes)); w /= w.sum()
    Z = []
    for _ in range(M // 2):
        s = rng.choice(sizes, p=w); z = np.zeros(d, bool); z[rng.choice(d, s, replace=False)] = True
        Z.append(z); Z.append(~z)
    return np.array(Z)

def kernelshap_matrix(Z):
    """Return a function mapping (v_S for S in Z, v_empty, v_full) -> phi (d,), linear in its inputs.
    Constrained least squares with uniform weights (coalitions already sampled from the kernel)."""
    M, d = Z.shape; Zf = Z.astype(float)
    # eliminate phi_d via efficiency: phi_d = tot - sum_{k<d} phi_k
    A = Zf[:, :-1] - Zf[:, [-1]]
    G = np.linalg.solve(A.T @ A + 1e-10 * np.eye(d - 1), A.T)      # (d-1) x M
    zl = Zf[:, -1]
    def phi(vS, v0, v1):   # vS: (M, K) ; v0, v1: (K,)
        tot = v1 - v0
        y = vS - v0[None] - zl[:, None] * tot[None]
        p = G @ y                                                  # (d-1, K)
        return np.vstack([p, tot[None] - p.sum(0, keepdims=True)])  # (d, K)
    return phi

def shap_firing_ks(x, bg, C, S, Z, phi_fn):
    m = bg.shape[0]
    Xs = np.repeat(bg[None], len(Z), 0)                             # (M, m, d)
    Xs[np.broadcast_to(Z[:, None, :], Xs.shape)] = np.broadcast_to(x[None, None, :], Xs.shape)[np.broadcast_to(Z[:, None, :], Xs.shape)]
    vS = P.firing(Xs.reshape(-1, len(x)), C, S).reshape(len(Z), m, -1).mean(1)   # (M, R)
    v0 = P.firing(bg, C, S).mean(0); v1 = P.firing(x[None], C, S)[0]
    return phi_fn(vS, v0, v1)                                       # (d, R)

def auc(y, s):
    a = [roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].mean() < 1]
    return float(np.mean(a)) if a else float("nan")

def run(seed, d):
    t0 = time.time(); rng = np.random.default_rng(seed)
    tr_idx = rng.choice(d.index.values, E.NTRAIN, replace=False); tr = d.loc[tr_idx]; rest = d.drop(index=tr_idx)
    te = pd.concat([rest.loc[rng.choice(rest.index[fn(rest["file"]).values], NPER, replace=False)] for fn in E.STRATA.values()])
    sc = StandardScaler().fit(E.tf(tr[FEATS].values))
    Xtr, Xte = sc.transform(E.tf(tr[FEATS].values)), sc.transform(E.tf(te[FEATS].values))
    yn, yc = tr["yo"].values.astype(float), tr["yc"].values.astype(float)
    bg = Xtr[rng.choice(E.NTRAIN, NBG, replace=False)]
    Z = coalitions(len(FEATS), M_COAL, rng); phi_fn = kernelshap_matrix(Z)
    C, S = P.fit_antecedents(Xtr, seed); H = P.H_matrix(Xtr, C, S); sgn = 1 - 2 * yn
    ev_idx = rng.choice(rest.index.values, 20000, replace=False)
    Xev = sc.transform(E.tf(d.loc[ev_idx, FEATS].values)); yev = d.loc[ev_idx, "yc"].values
    tsk_auc = roc_auc_score(yev, P.firing(Xev, C, S) @ (H @ yn))
    gb_auc = roc_auc_score(yev, HistGradientBoostingClassifier(random_state=seed).fit(Xtr, yn).predict_proba(Xev)[:, 1])
    unit, rho_u = make_units(tr["blk_fs"].values, tr["cat"].values, E.prov(tr))
    boots = []
    for b in range(NBOOT):
        ii = rng.choice(E.NTRAIN, E.NTRAIN, replace=True)
        Cb, Sb = P.fit_antecedents(Xtr[ii], seed + 1000 + b); boots.append((Cb, Sb, P.H_matrix(Xtr[ii], Cb, Sb) @ yn[ii]))
    n_te, dd = len(te), len(FEATS)
    phi_n = np.zeros((n_te, dd)); phi_c = np.zeros_like(phi_n); Ssens = np.zeros_like(phi_n); sdb = np.zeros_like(phi_n)
    mu = np.zeros_like(phi_n); sd = np.zeros_like(phi_n)
    for t, x in enumerate(Xte):
        Cm = shap_firing_ks(x, bg, C, S, Z, phi_fn) @ H
        phi_n[t], phi_c[t], Ssens[t] = Cm @ yn, Cm @ yc, np.abs(Cm).sum(1)
        for i in range(dd):
            Du = aggregate(Cm[i] * sgn, unit, len(rho_u)); mu[t, i] = Du @ rho_u
            sd[t, i] = math.sqrt(float((Du ** 2 * rho_u * (1 - rho_u)).sum())) + 1e-15
        sdb[t] = np.stack([shap_firing_ks(x, bg, Cb, Sb, Z, phi_fn) @ th for Cb, Sb, th in boots]).std(0)
    sdf = phi_n.std(0) + 1e-12; delta = phi_c - phi_n
    rel = np.abs(delta) > 0.5 * np.abs(phi_n) + 0.05 * sdf
    sign = (phi_c * phi_n < 0) & (np.maximum(np.abs(phi_c), np.abs(phi_n)) > 0.05 * sdf)
    tau = 0.5 * np.abs(phi_n) + 0.05 * sdf; dz = 0.05 * sdf
    g_rel = norm.sf((tau - mu) / sd) + norm.cdf((-tau - mu) / sd)
    g_sign = np.where(phi_n > 0, norm.cdf((-dz - phi_n - mu) / sd), norm.sf((dz - phi_n - mu) / sd))
    base = {"S": Ssens, "bootSD": sdb, "|phi|": np.abs(phi_n), "bootSD/|phi|": sdb / (np.abs(phi_n) + 0.05 * sdf)}
    out = dict(seed=seed, tsk_auc=tsk_auc, gb_auc=gb_auc, seconds=time.time() - t0)
    for e, y, g in (("rel", rel, g_rel), ("sign", sign, g_sign)):
        out[f"{e}|rate"] = float(y.mean()); out[f"{e}|Z"] = auc(y, g); out[f"{e}|brier"] = float(((g - y) ** 2).mean()); out[f"{e}|meanP"] = float(g.mean())
        for b, s in base.items(): out[f"{e}|{b}"] = auc(y, s)
        out[f"{e}|best"] = float(np.nanmax([out[f"{e}|{b}"] for b in base]))
    json.dump(out, open(f"{OUT}/e8_{seed}.json", "w")); print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)

if __name__ == "__main__":
    d = B.load()
    extra = pd.read_parquet(f"{ROOT}/data/matched/matched.parquet", columns=[c for c in FEATS if c not in d.columns])
    for c in extra.columns: d[c] = extra[c].values
    for s in [int(x) for x in sys.argv[1:]]:
        if not os.path.exists(f"{OUT}/e8_{s}.json"): run(s, d)
