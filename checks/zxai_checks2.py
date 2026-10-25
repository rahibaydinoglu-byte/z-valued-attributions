"""Tighter certified lower bounds for B = P(|Delta| <= tau), Delta = sum_j d_j F_j, F_j ~ Bern(rho_j)."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import math
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(1)
C0 = 0.5600  # Berry-Esseen constant, non-identical summands (Shevtsova 2010) [YOXLANMALI]

def bernstein_lower(d, rho, tau):
    mu = d @ rho; v = (d**2 * rho * (1 - rho)).sum(); M = np.abs(d).max()
    t = tau - abs(mu)
    if t <= 0: return 0.0
    return max(0.0, 1 - 2 * math.exp(-t * t / (2 * v + 2 * M * t / 3)))

def be_lower(d, rho, tau):
    mu = d @ rho; v = (d**2 * rho * (1 - rho)).sum()
    if v <= 0: return float(abs(mu) <= tau)
    # E|X - EX|^3 for X = d*F: |d|^3 * rho(1-rho)(rho^2 + (1-rho)^2)
    L = (np.abs(d)**3 * rho * (1 - rho) * (rho**2 + (1 - rho)**2)).sum()
    eps = C0 * L / v**1.5
    s = math.sqrt(v)
    p = norm.cdf((tau - mu) / s) - norm.cdf((-tau - mu) / s)
    return max(0.0, p - 2 * eps)

def lattice_lower(d, rho, tau, h):
    """Exact DP on lattice for rounded weights + rounding error control.
    Delta = Delta_r + E, |E| <= (h/2) K, K = number of flips.
    P(|Delta|<=tau) >= max_k [ P(|Delta_r| <= tau - h k/2, K <= k) ]  (joint DP over (value, count) too heavy)
    -> use P(|Delta_r| <= tau - h k/2) - P(K > k)."""
    q = np.rint(d / h).astype(int)
    off = int(np.abs(q).sum())
    dist = np.zeros(2 * off + 1); dist[off] = 1.0
    for qj, pj in zip(q, rho):
        new = dist * (1 - pj)
        if qj >= 0: new[qj:] += dist[:len(dist) - qj] * pj if qj > 0 else dist * pj
        else: new[:qj] += dist[-qj:] * pj
        dist = new
    vals = (np.arange(len(dist)) - off) * h
    # Poisson-binomial distribution of K
    pk = np.zeros(len(rho) + 1); pk[0] = 1.0
    for pj in rho:
        pk[1:] = pk[1:] * (1 - pj) + pk[:-1] * pj; pk[0] *= (1 - pj)
    tailK = 1 - np.cumsum(pk)            # P(K > k)
    best = 0.0
    for k in range(len(rho) + 1):
        t = tau - h * k / 2
        if t < 0: break
        best = max(best, dist[np.abs(vals) <= t + 1e-12].sum() - max(tailK[k], 0.0))
    return best

rows = []
for trial in range(600):
    m = int(rng.integers(50, 400))
    d = rng.normal(scale=1 / np.sqrt(m), size=m) * rng.exponential(size=m)
    rho = rng.uniform(0, 0.3, size=m)
    sd = math.sqrt((d**2 * rho * (1 - rho)).sum())
    tau = abs(d @ rho) + rng.uniform(0.5, 3.0) * sd     # tau on the scale of the actual spread
    F = rng.random((20000, m)) < rho
    p_mc = (np.abs(F.astype(float) @ d) <= tau).mean()
    se = math.sqrt(max(p_mc * (1 - p_mc), 1e-6) / 20000)
    h = sd / 200
    lb = dict(bern=bernstein_lower(d, rho, tau), be=be_lower(d, rho, tau), lat=lattice_lower(d, rho, tau, h))
    rows.append((p_mc, se, lb))

for key in ["bern", "be", "lat"]:
    viol = sum(lb[key] > p + 4 * se for p, se, lb in rows)
    gaps = np.array([p - lb[key] for p, se, lb in rows])
    print(f"{key:5s}: violations {viol}/{len(rows)}  median gap {np.median(gaps):.3f}  90th pct {np.quantile(gaps, .9):.3f}")
