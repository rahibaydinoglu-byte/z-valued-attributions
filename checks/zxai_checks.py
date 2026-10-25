"""Numerical checks for Z-XAI-IDS Stage 3 (24.10.2026)."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import itertools, math
import numpy as np
from sklearn.cluster import KMeans

rng = np.random.default_rng(0)

# ---------- linear-in-labels TSK (label-free antecedents) ----------
def firing(X, C, S):
    # product t-norm of Gaussian MFs, normalized
    z = -0.5 * (((X[:, None, :] - C[None]) / S[None]) ** 2).sum(-1)
    z -= z.max(1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(1, keepdims=True)

def fit_tsk(X, y, R=6, lam=1e-2, seed=0):
    km = KMeans(R, n_init=5, random_state=seed).fit(X)       # uses X only
    C = km.cluster_centers_
    S = np.full_like(C, X.std(0).mean())
    W = firing(X, C, S)
    H = np.linalg.solve(W.T @ W + lam * np.eye(R), W.T)       # theta = H y
    return C, S, H

def shap_exact(fun, x, bg):
    """Exact interventional SHAP for a scalar function, d small."""
    d = x.size
    phi = np.zeros(d)
    def v(Sset):
        Z = bg.copy()
        idx = list(Sset)
        Z[:, idx] = x[idx]
        return fun(Z).mean()
    cache = {}
    for k in range(d + 1):
        for Sset in itertools.combinations(range(d), k):
            cache[Sset] = v(Sset)
    for i in range(d):
        rest = [j for j in range(d) if j != i]
        for k in range(d):
            for Sset in itertools.combinations(rest, k):
                wgt = math.factorial(k) * math.factorial(d - k - 1) / math.factorial(d)
                Si = tuple(sorted(Sset + (i,)))
                phi[i] += wgt * (cache[Si] - cache[Sset])
    return phi

# ---------- Check 1: attribution is exactly linear in labels ----------
n, d = 300, 5
X = rng.normal(size=(n, d))
ytrue = (X[:, 0] + 0.5 * X[:, 1] - 0.7 * X[:, 2] > 0).astype(float)
C, S, H = fit_tsk(X, ytrue)
bg = X[rng.choice(n, 40, replace=False)]
x0 = X[0]

# a_i(x): SHAP of each normalized firing function w_r
R = C.shape[0]
A = np.stack([shap_exact(lambda Z, r=r: firing(Z, C, S)[:, r], x0, bg) for r in range(R)], 1)  # d x R
Cmat = A @ H                    # d x n : c_ij(x0)

max_err = 0.0
for t in range(20):
    ynew = ytrue.copy()
    flip = rng.random(n) < 0.15
    ynew[flip] = 1 - ynew[flip]
    theta = H @ ynew
    phi_direct = shap_exact(lambda Z: firing(Z, C, S) @ theta, x0, bg)   # same antecedents, refit consequents
    phi_lin = Cmat @ ynew
    max_err = max(max_err, np.abs(phi_direct - phi_lin).max())
print(f"[Check 1] max |phi(retrained) - C y| over 20 label flips: {max_err:.2e}")

# ---------- Check 2: is exact B monotone in rho_k? (search for counterexample) ----------
def exact_PA(dvec, rho, tau):
    """P(|Delta| <= tau)-style fuzzy event with triangular mu_A(t)=max(0,1-|t|/tau)."""
    m = len(dvec)
    tot = 0.0
    for F in itertools.product([0, 1], repeat=m):
        F = np.array(F)
        p = np.prod(np.where(F == 1, rho, 1 - rho))
        delta = dvec @ F
        tot += p * max(0.0, 1 - abs(delta) / tau)
    return tot

viol = []
for trial in range(20000):
    m = rng.integers(2, 7)
    dvec = rng.normal(size=m)
    rho = rng.uniform(0, 0.5, size=m)
    tau = rng.uniform(0.2, 2.0)
    k = rng.integers(m)
    h = 1e-4
    rho2 = rho.copy(); rho2[k] = min(0.5, rho[k] + h)
    diff = exact_PA(dvec, rho2, tau) - exact_PA(dvec, rho, tau)
    if diff > 1e-10:
        viol.append((diff, m, dvec.round(3), rho.round(3), round(tau, 3), k))
print(f"[Check 2] monotonicity violations (rho<=0.5, symmetric triangular A): {len(viol)}/20000")
if viol:
    v = max(viol, key=lambda z: z[0])
    print("  largest violation:", v)

# sign-coherent case must never violate
vc = 0
for trial in range(5000):
    m = rng.integers(2, 7)
    dvec = np.abs(rng.normal(size=m))
    rho = rng.uniform(0, 1, size=m)
    tau = rng.uniform(0.2, 2.0)
    k = rng.integers(m)
    rho2 = rho.copy(); rho2[k] = min(1, rho[k] + 1e-4)
    if exact_PA(dvec, rho2, tau) - exact_PA(dvec, rho, tau) > 1e-12:
        vc += 1
print(f"[Check 2b] violations in sign-coherent case: {vc}/5000")

# ---------- Check 3: Bernstein certified lower bound is sound ----------
def bernstein_lower(dvec, rho, tau):
    mu = dvec @ rho
    v = (dvec ** 2 * rho * (1 - rho)).sum()
    M = np.abs(dvec).max()
    t = tau - abs(mu)
    if t <= 0:
        return 0.0
    return max(0.0, 1 - 2 * math.exp(-t * t / (2 * v + 2 * M * t / 3)))

bad, gaps = 0, []
for trial in range(3000):
    m = rng.integers(20, 200)
    dvec = rng.normal(scale=1 / np.sqrt(m), size=m) * rng.exponential(size=m)
    rho = rng.uniform(0, 0.3, size=m)
    tau = rng.uniform(0.05, 1.0)
    F = rng.random((4000, m)) < rho
    delta = F.astype(float) @ dvec
    p_mc = (np.abs(delta) <= tau).mean()
    se = math.sqrt(max(p_mc * (1 - p_mc), 1e-6) / 4000)
    lb = bernstein_lower(dvec, rho, tau)
    if lb > p_mc + 4 * se:
        bad += 1
    gaps.append(p_mc - lb)
print(f"[Check 3] Bernstein lower bound above MC estimate (+4 SE): {bad}/3000; "
      f"median gap {np.median(gaps):.3f}, 90th pct {np.quantile(gaps, 0.9):.3f}")

# ---------- Check 4: Markov-type certified bound is monotone and sound ----------
def markov_lower(dvec, rho, tau):
    return max(0.0, 1 - (np.abs(dvec) * rho).sum() / tau)
bad4 = 0
for trial in range(3000):
    m = rng.integers(5, 100)
    dvec = rng.normal(size=m) / m
    rho = rng.uniform(0, 0.5, size=m)
    tau = rng.uniform(0.01, 0.5)
    F = rng.random((4000, m)) < rho
    p_mc = (np.abs(F.astype(float) @ dvec) <= tau).mean()
    se = math.sqrt(max(p_mc * (1 - p_mc), 1e-6) / 4000)
    if markov_lower(dvec, rho, tau) > p_mc + 4 * se:
        bad4 += 1
print(f"[Check 4] Markov lower bound above MC estimate: {bad4}/3000")
