"""Z-XAI pilot A (controlled planted noise), 24.10.2026.
Question H1: does label unreliability 1-B (lattice, Thm 5) predict which attributions change
between the noisy-label model and the clean-label model better than label-blind baselines?
Data here are NOT intrusion data (offline sklearn sets); this tests the mechanism only.
"""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import itertools, math, sys, time
import rho_estimators as RE
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

D_FEAT, R, LAM, NBG = 8, 8, 1e-2, 30

def firing(X, C, S):
    z = -0.5 * (((X[:, None, :] - C[None]) / S[None]) ** 2).sum(-1)
    z -= z.max(1, keepdims=True); w = np.exp(z)
    return w / w.sum(1, keepdims=True)

def fit_antecedents(X, seed):
    km = KMeans(R, n_init=5, random_state=seed).fit(X)
    return km.cluster_centers_, np.full((R, X.shape[1]), X.std(0).mean())

def H_matrix(X, C, S):
    W = firing(X, C, S)
    return np.linalg.solve(W.T @ W + LAM * np.eye(R), W.T)

COAL = None
def shap_firing(x, bg, C, S):
    """Exact interventional SHAP of all R normalized firing functions at x -> (d, R)."""
    d = x.size
    vals = {}
    for k in range(d + 1):
        for Sset in itertools.combinations(range(d), k):
            Z = bg.copy(); Z[:, list(Sset)] = x[list(Sset)]
            vals[Sset] = firing(Z, C, S).mean(0)
    phi = np.zeros((d, R))
    for i in range(d):
        rest = [j for j in range(d) if j != i]
        for k in range(d):
            w = math.factorial(k) * math.factorial(d - k - 1) / math.factorial(d)
            for Sset in itertools.combinations(rest, k):
                phi[i] += w * (vals[tuple(sorted(Sset + (i,)))] - vals[Sset])
    return phi

def lattice_lower(d, rho, tau, h):
    keep = np.abs(d) > 1e-12
    d, rho = d[keep], rho[keep]
    if d.size == 0: return 1.0
    q = np.rint(d / h).astype(int); off = int(np.abs(q).sum())
    dist = np.zeros(2 * off + 1); dist[off] = 1.0
    for qj, pj in zip(q, rho):
        new = dist * (1 - pj)
        if qj > 0: new[qj:] += dist[:-qj] * pj
        elif qj < 0: new[:qj] += dist[-qj:] * pj
        else: new += dist * pj
        dist = new
    vals = (np.arange(dist.size) - off) * h
    pk = np.zeros(rho.size + 1); pk[0] = 1.0
    for pj in rho:
        pk[1:] = pk[1:] * (1 - pj) + pk[:-1] * pj; pk[0] *= 1 - pj
    tailK = 1 - np.cumsum(pk)
    cum_abs = None
    best = 0.0
    order = np.argsort(np.abs(vals)); absv = np.abs(vals)[order]; cdf = np.cumsum(dist[order])
    for k in range(rho.size + 1):
        t = tau - h * k / 2
        if t < 0: break
        idx = np.searchsorted(absv, t + 1e-12, side="right")
        p = cdf[idx - 1] if idx > 0 else 0.0
        best = max(best, p - max(tailK[k], 0.0))
        if tailK[k] < 1e-6: break
    return best

def lattice_dist(d, rho, h):
    keep = np.abs(d) > 1e-12
    d, rho = d[keep], rho[keep]
    if d.size == 0: return np.array([0.0]), np.array([1.0])
    q = np.rint(d / h).astype(int); off = int(np.abs(q).sum())
    dist = np.zeros(2 * off + 1); dist[off] = 1.0
    for qj, pj in zip(q, rho):
        new = dist * (1 - pj)
        if qj > 0: new[qj:] += dist[:-qj] * pj
        elif qj < 0: new[:qj] += dist[-qj:] * pj
        else: new += dist * pj
        dist = new
    return (np.arange(dist.size) - off) * h, dist

def load(name, seed):
    rng = np.random.default_rng(seed)
    if name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
    elif name == "wine":
        X, y = load_wine(return_X_y=True); y = (y == 0).astype(int)
    else:
        X, y = make_classification(800, 12, n_informative=6, n_redundant=2, flip_y=0.0,
                                   class_sep=1.0, random_state=seed)
    X = StandardScaler().fit_transform(X)
    cols = rng.choice(X.shape[1], D_FEAT, replace=False)
    return X[:, cols], y.astype(float)

def plant_noise(Xtr, ytr, kind, rate, seed):
    rng = np.random.default_rng(seed)
    yn = ytr.copy(); n = len(ytr)
    if kind == "uniform":
        flip = rng.random(n) < rate
    else:  # systematic block noise: flip a contiguous region (mimics window/attack-level labelling errors)
        km = KMeans(6, n_init=3, random_state=seed).fit(Xtr)
        flip = np.zeros(n, bool)
        for c in rng.permutation(6):
            if flip.mean() >= rate: break
            m = km.labels_ == c
            flip |= m & (rng.random(n) < 0.8)
    yn[flip] = 1 - yn[flip]
    return yn, flip

def estimate_rho(X, yn, seed):
    rho = np.zeros(len(yn))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yn):
        gb = GradientBoostingClassifier(random_state=seed).fit(X[tr], yn[tr])
        p1 = gb.predict_proba(X[te])[:, 1]
        p_obs = np.where(yn[te] == 1, p1, 1 - p1)
        rho[te] = np.clip(1 - p_obs, 1e-3, 0.999)
    return rho

def run(name, kind, rate, seed, n_test=80, n_boot=15, rho_mode='lr_cl'):
    X, y = load(name, seed)
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    Xte = Xte[:n_test]
    yn, flip = plant_noise(Xtr, ytr, kind, rate, seed)
    rng = np.random.default_rng(seed)
    bg = Xtr[rng.choice(len(Xtr), NBG, replace=False)]
    C, S = fit_antecedents(Xtr, seed)
    H = H_matrix(Xtr, C, S)
    if rho_mode == "lr_cl":
        rho = RE.rho_cl(RE.oof_proba(Xtr, yn, seed, "lr"), yn)
    elif rho_mode == "gb_bayes":
        rho = RE.rho_bayes(RE.oof_proba(Xtr, yn, seed, "gb"), yn)
    elif rho_mode == "provenance":
        # SIMULATED imperfect provenance prior: analyst flags a suspect region that covers ~60% of the truly
        # flipped points plus an equal number of clean points; rho=0.5 inside the flagged set, 0.02 outside,
        # combined with the data-based estimate by taking the maximum.
        prng = np.random.default_rng(seed + 999)
        flagged = (flip & (prng.random(len(flip)) < 0.6))
        clean_idx = np.flatnonzero(~flip); extra = prng.choice(clean_idx, min(flagged.sum(), clean_idx.size), replace=False)
        flagged[extra] = True
        rho = np.maximum(np.where(flagged, 0.5, 0.02), RE.rho_cl(RE.oof_proba(Xtr, yn, seed, "lr"), yn))
    else:
        rho = estimate_rho(Xtr, yn, seed)
    sgn = 1 - 2 * yn

    # bootstrap baseline (label-blind, refit everything on resamples of the noisy data)
    boots = []
    for b in range(n_boot):
        idx = rng.choice(len(Xtr), len(Xtr), replace=True)
        Cb, Sb = fit_antecedents(Xtr[idx], seed + 100 + b)
        Hb = H_matrix(Xtr[idx], Cb, Sb)
        boots.append((Cb, Sb, Hb @ yn[idx]))

    phi_n, phi_c, sens, unrel, sdboot, rhobar_sens = [], [], [], [], [], []
    for x in Xte:
        Afir = shap_firing(x, bg, C, S)            # d x R
        Cm = Afir @ H                               # d x n  (label-attribution kernel)
        phi_n.append(Cm @ yn); phi_c.append(Cm @ ytr)
        sens.append(np.abs(Cm).sum(1))
        rhobar_sens.append(np.abs(Cm).sum(1) * rho.mean())
        bphi = np.stack([shap_firing(x, bg, Cb, Sb) @ th for Cb, Sb, th in boots])
        sdboot.append(bphi.std(0))
        unrel.append(Cm * sgn)                      # d_ij, store for B
    phi_n, phi_c = np.array(phi_n), np.array(phi_c)
    sdf = phi_n.std(0) + 1e-9
    delta = phi_c - phi_n
    events = {
        "abs":  np.abs(delta) > 0.25 * sdf,
        "rel":  np.abs(delta) > 0.5 * np.abs(phi_n) + 0.05 * sdf,
        "sign": (phi_c * phi_n < 0) & (np.maximum(np.abs(phi_c), np.abs(phi_n)) > 0.05 * sdf),
    }
    Pchg = {e: np.zeros_like(phi_n) for e in events}     # point estimate P(event) from lattice law
    Bcert = np.zeros_like(phi_n)                         # certified lower bound for abs event
    EabsD = np.zeros_like(phi_n)
    for t in range(len(Xte)):
        for i in range(D_FEAT):
            dvec = unrel[t][i]
            sd = math.sqrt((dvec ** 2 * rho * (1 - rho)).sum()) + 1e-12
            h = min(0.05 * sdf[i], sd) / 40
            v, p = lattice_dist(dvec, rho, h)
            f0 = phi_n[t, i]
            Pchg["abs"][t, i] = p[np.abs(v) > 0.25 * sdf[i]].sum()
            Pchg["rel"][t, i] = p[np.abs(v) > 0.5 * abs(f0) + 0.05 * sdf[i]].sum()
            fc = f0 + v
            Pchg["sign"][t, i] = p[(fc * f0 < 0) & (np.maximum(np.abs(fc), abs(f0)) > 0.05 * sdf[i])].sum()
            EabsD[t, i] = (p * np.abs(v)).sum()
            Bcert[t, i] = lattice_lower(dvec, rho, 0.25 * sdf[i], h)
    base = {
        "label sensitivity S": np.array(sens),
        "bootstrap SD": np.array(sdboot),
        "|phi| magnitude": np.abs(phi_n),
        "bootSD/|phi|": np.array(sdboot) / (np.abs(phi_n) + 0.05 * sdf),
    }
    out = {}
    for e, ev in events.items():
        sc = dict(base); sc["Z: P(change)"] = Pchg[e]
        for k, s in sc.items():
            aucs = [roc_auc_score(ev[:, i], s[:, i]) for i in range(D_FEAT) if 0 < ev[:, i].mean() < 1]
            out[f"{e}:{k}"] = np.mean(aucs) if aucs else np.nan
        out[f"_rate_{e}"] = ev.mean()
        out[f"_brier_{e}"] = np.mean((Pchg[e] - ev) ** 2)
        out[f"_meanP_{e}"] = Pchg[e].mean()
    out["_Bcert_mean"] = Bcert.mean(); out["_unchanged_abs"] = 1 - events["abs"].mean()
    out["_flip_rate"] = flip.mean(); out["_rho_auc"] = roc_auc_score(flip, rho)
    return out

if __name__ == "__main__":
    t0 = time.time()
    configs = [(n, k, r) for n in ["breast_cancer", "wine", "synthetic"]
               for (k, r) in [("uniform", 0.10), ("uniform", 0.20), ("block", 0.15)]]
    mode = sys.argv[1] if len(sys.argv) > 1 else "lr_cl"
    print("rho_mode =", mode)
    seeds = [0, 1, 2, 3, 4]
    import json
    allres = {}
    for cfg in configs:
        res = [run(*cfg, s, rho_mode=mode) for s in seeds]
        allres["|".join(map(str, cfg))] = res
        print(f"{cfg}  flip={np.mean([r['_flip_rate'] for r in res]):.2f} rhoAUC={np.mean([r['_rho_auc'] for r in res]):.2f} Bcert={np.mean([r['_Bcert_mean'] for r in res]):.2f} unchanged_abs={np.mean([r['_unchanged_abs'] for r in res]):.2f}", flush=True)
        for e in ["abs", "rel", "sign"]:
            keys = [k for k in res[0] if k.startswith(e + ":")]
            line = "  ".join(f"{k.split(':')[1]}={np.nanmean([r[k] for r in res]):.3f}±{np.nanstd([r[k] for r in res]):.3f}" for k in keys)
            print(f"   [{e}] rate={np.mean([r['_rate_'+e] for r in res]):.2f} brier={np.mean([r['_brier_'+e] for r in res]):.3f} meanP={np.mean([r['_meanP_'+e] for r in res]):.2f} | {line}", flush=True)
    json.dump({k: v for k, v in allres.items()}, open(f"pilot_controlled_results_{mode}.json", "w"), indent=1, default=float)
    print(f"total {time.time()-t0:.0f}s")
