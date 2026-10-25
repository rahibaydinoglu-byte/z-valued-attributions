"""Robust certified bound over a box rho in [rhoL, rhoU] via the coupling F = G OR E."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import math, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("c2", "zxai_checks2.py")
src = open("zxai_checks2.py").read().split("rows = []")[0]
exec(src)
rng = np.random.default_rng(7)

def lattice_cdf_nonneg(w, e, h):
    """P(sum w_j E_j <= s) on a grid, w>=0, rounding UP (conservative)."""
    q = np.ceil(w / h).astype(int)
    dist = np.zeros(int(q.sum()) + 1); dist[0] = 1.0
    for qj, ej in zip(q, e):
        new = dist * (1 - ej)
        if qj > 0: new[qj:] += dist[:len(dist) - qj] * ej
        else: new += dist * ej
        dist = new
    return np.cumsum(dist), h

def robust_lower(d, rL, rU, tau, h):
    ebar = (rU - rL) / (1 - rL)
    cdfE, _ = lattice_cdf_nonneg(np.abs(d), ebar, h)
    best = 0.0
    for k in range(0, len(cdfE), max(1, len(cdfE) // 60)):
        s = k * h
        if s >= tau: break
        best = max(best, lattice_lower(d, rL, tau - s, h) - (1 - cdfE[k]))
    return best

viol, gaps = 0, []
for trial in range(150):
    m = int(rng.integers(50, 250))
    d = rng.normal(scale=1/np.sqrt(m), size=m) * rng.exponential(size=m)
    rL = rng.uniform(0, 0.2, size=m); rU = np.minimum(rL + rng.uniform(0, 0.08, size=m), 0.5)
    sd = math.sqrt((d**2 * rU * (1 - rU)).sum())
    tau = abs(d @ rU) + rng.uniform(1, 3) * sd
    lb = robust_lower(d, rL, rU, tau, sd / 150)
    # worst case over random points in the box (and corners) by Monte Carlo
    worst = 1.0
    for _ in range(8):
        rho = rL + rng.random(m) * (rU - rL)
        F = rng.random((10000, m)) < rho
        worst = min(worst, (np.abs(F.astype(float) @ d) <= tau).mean())
    if lb > worst + 0.02: viol += 1
    gaps.append(worst - lb)
print(f"robust box bound: violations {viol}/150, median gap to sampled worst case {np.median(gaps):.3f}, 90th pct {np.quantile(gaps,.9):.3f}")
