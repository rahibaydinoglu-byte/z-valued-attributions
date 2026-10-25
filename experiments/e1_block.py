"""Block-noise modes on E1 runs. Reproduces the E1 sample of (seed, FS) exactly (same RNG call order),
checks phi_n against the saved E1 npz, and adds P(event) under block-noise models.
Usage: python3 e1_block.py <outdir> <seed> [<seed> ...]
"""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import json, math, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pilot_controlled as P
import e1_full as E
from lattice_window import lattice_dist_window
from block_noise import make_units, aggregate

BLOCKS = {"fm": ["file", "mi"], "fms": ["file", "mi", "src"], "fmh": ["file", "mi", "hostpair"], "fh": ["file", "hostpair"], "fs": ["file", "src"]}
import os as _os
if _os.environ.get("BLOCKS_ONLY"): BLOCKS = {k: BLOCKS[k] for k in _os.environ["BLOCKS_ONLY"].split(",")}

def load():
    d = E.load()
    h = pd.read_parquet(f"{ROOT}/data/matched/hosts.parquet")
    d["hostpair"] = h["hostpair"].values; d["src"] = h["sip"].str.strip().values
    d["cat"] = np.select([(d.yo == 0) & d.pad2, d.yo == 0, d.zeropay], [0, 1, 2], 3)
    for k, cols in BLOCKS.items():
        d[f"blk_{k}"] = pd.factorize(pd.MultiIndex.from_frame(d[cols].astype(str)))[0]
    return d

def probs(dv, rho, f0, sdf_i):
    sd = math.sqrt(float((dv ** 2 * rho * (1 - rho)).sum())) + 1e-15
    v, p, lost = lattice_dist_window(dv, rho, min(0.05 * sdf_i, sd) / 40)
    fc = f0 + v
    return (p[np.abs(v) > 0.25 * sdf_i].sum(), p[np.abs(v) > 0.5 * abs(f0) + 0.05 * sdf_i].sum(),
            p[(fc * f0 < 0) & (np.maximum(np.abs(fc), abs(f0)) > 0.05 * sdf_i)].sum(), lost)

def run(seed, fsname, d, outdir, check=True):
    t0 = time.time(); feats = E.FS[fsname]
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(d.index.values, E.NTRAIN, replace=False)
    tr = d.loc[tr_idx]; rest = d.drop(index=tr_idx)
    te_parts, strata = [], []
    for s, fn in E.STRATA.items():
        cand = rest.index[fn(rest["file"]).values]
        te_parts.append(rest.loc[rng.choice(cand, E.NPER, replace=False)]); strata += [s] * E.NPER
    te = pd.concat(te_parts); strata = np.array(strata)
    sc = StandardScaler().fit(E.tf(tr[feats].values))
    Xtr, Xte = sc.transform(E.tf(tr[feats].values)), sc.transform(E.tf(te[feats].values))
    yn, yc = tr["yo"].values.astype(float), tr["yc"].values.astype(float)
    bg = Xtr[rng.choice(E.NTRAIN, E.NBG, replace=False)]
    C, S = P.fit_antecedents(Xtr, seed); H = P.H_matrix(Xtr, C, S); sgn = 1 - 2 * yn
    rho_flow = E.prov(tr)                     # same marginals as PROV
    cat = tr["cat"].values
    phi_n = np.zeros((len(te), len(feats))); phi_c = np.zeros_like(phi_n); Dv = []
    for t, x in enumerate(Xte):
        Cm = P.shap_firing(x, bg, C, S) @ H
        phi_n[t], phi_c[t] = Cm @ yn, Cm @ yc; Dv.append(Cm * sgn)
    if check:
        saved = np.load(f"{ROOT}/results/e1/e1_{seed}_{fsname}.npz")
        assert np.allclose(saved["phi_n"], phi_n, atol=1e-10), "sample not reproduced"
    sdf = phi_n.std(0) + 1e-12; delta = phi_c - phi_n
    events = {"abs": np.abs(delta) > 0.25 * sdf, "rel": np.abs(delta) > 0.5 * np.abs(phi_n) + 0.05 * sdf,
              "sign": (phi_c * phi_n < 0) & (np.maximum(np.abs(phi_c), np.abs(phi_n)) > 0.05 * sdf)}
    modes = {"IND": (np.arange(E.NTRAIN), rho_flow)}
    unit_stats = {}
    for k in BLOCKS:
        unit, rho_u = make_units(tr[f"blk_{k}"].values, cat, rho_flow)
        modes[f"BLK_{k}"] = (unit, rho_u)
        sizes = np.bincount(unit)[np.isin(np.arange(len(rho_u)), np.unique(unit[np.isin(cat, (0, 2))]))]
        unit_stats[k] = dict(units=int(len(rho_u)), flagged_units=int(len(sizes)), mean_flagged_size=float(sizes.mean()) if len(sizes) else 0.0,
                             max_size=int(sizes.max()) if len(sizes) else 0)
    out = {}
    for mname, (unit, rho_u) in modes.items():
        arr = np.zeros((3, len(te), len(feats)))
        for t in range(len(te)):
            for i in range(len(feats)):
                dv = aggregate(Dv[t][i], unit, len(rho_u)) if mname != "IND" else Dv[t][i]
                arr[:, t, i] = probs(dv, rho_u, phi_n[t, i], sdf[i])[:3]
        out[mname] = arr
    np.savez_compressed(f"{outdir}/blk_{seed}_{fsname}.npz", strata=strata, **{f"ev_{k}": v for k, v in events.items()},
                        **{f"P_{k}": v for k, v in out.items()})
    meta = dict(seed=seed, fs=fsname, unit_stats=unit_stats, seconds=time.time() - t0)
    json.dump(meta, open(f"{outdir}/blk_{seed}_{fsname}.json", "w"), indent=1)
    print(json.dumps(meta), flush=True)

if __name__ == "__main__":
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    d = load()
    for s in [int(x) for x in sys.argv[2:]]:
        for fs in ("A", "B"):
            if not os.path.exists(f"{outdir}/blk_{s}_{fs}.json"):
                run(s, fs, d, outdir, check=os.path.exists(f"{ROOT}/results/e1/e1_{s}_{fs}.npz"))
