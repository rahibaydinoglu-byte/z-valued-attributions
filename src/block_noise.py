"""Block-noise model A3': flips are shared within blocks of training flows.
Units = blocks (within a provenance category) restricted to the training sample; a unit flips all its members with prob rho_b.
Delta = sum_units D_u G_u, D_u = sum_{j in u} d_j, G_u ~ Bern(rho_u)  ->  same lattice machinery (Prop. 1, Thm 5 with units).
"""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import numpy as np

def make_units(block_id, category, rho_flow, block_cats=(0, 2)):
    """block_id: array (n,) of hashable block labels; category: provenance category per flow
    (0 benign in window, 1 benign outside, 2 attack zero-payload, 3 attack other).
    Flows in block_cats are grouped by (category, block_id); others stay independent units.
    Returns unit index per flow (n,), unit rho (U,)."""
    n = len(block_id)
    unit = np.empty(n, dtype=np.int64)
    keys = {}
    rho_u = []
    for j in range(n):
        if category[j] in block_cats:
            k = (category[j], block_id[j])
            u = keys.get(k)
            if u is None:
                u = len(rho_u); keys[k] = u; rho_u.append(rho_flow[j])
        else:
            u = len(rho_u); rho_u.append(rho_flow[j])
        unit[j] = u
    return unit, np.asarray(rho_u, dtype=float)

def aggregate(dvec, unit, n_units):
    return np.bincount(unit, weights=dvec, minlength=n_units)
