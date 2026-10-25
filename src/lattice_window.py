"""Windowed exact lattice DP for Delta = sum_j d_j F_j, F_j ~ Bern(rho_j).
Keeps only a sliding window around the running mean (± nsd·sd_total + max|q|); mass pushed outside is accumulated in
`lost` and returned, so a certified bound can subtract it. Zero-q terms are skipped (they do not move mass).
"""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import math
import numpy as np

def lattice_dist_window(d, rho, h, nsd=12.0):
    q = np.rint(d / h).astype(np.int64)
    keep = q != 0
    q, r = q[keep], rho[keep]
    if q.size == 0:
        return np.array([0.0]), np.array([1.0]), 0.0
    sd = math.sqrt(float((q.astype(float) ** 2 * r * (1 - r)).sum()))
    half = int(math.ceil(nsd * sd + np.abs(q).max())) + 2
    W = 2 * half + 1
    # process in an order that makes the running mean move monotonically (positive drifts first, then negative)
    order = np.argsort(-(q * r))
    q, r = q[order], r[order]
    dist = np.zeros(W); lo = -half; dist[half] = 1.0      # dist[k] = P(S = lo + k)
    run_mu = 0.0; lost = 0.0
    for qj, pj in zip(q, r):
        new = dist * (1 - pj)
        if qj > 0:
            new[qj:] += dist[:-qj] * pj; lost += dist[-qj:].sum() * pj
        else:
            new[:qj] += dist[-qj:] * pj; lost += dist[:-qj].sum() * pj
        dist = new
        run_mu += qj * pj
        shift = int(round(run_mu)) - (lo + half)          # keep window centred on the running mean
        if abs(shift) > half // 4:
            if shift > 0:
                lost += dist[:shift].sum(); dist = np.concatenate([dist[shift:], np.zeros(shift)])
            else:
                lost += dist[shift:].sum(); dist = np.concatenate([np.zeros(-shift), dist[:shift]])
            lo += shift
    vals = (lo + np.arange(W)) * h
    return vals, dist, lost
