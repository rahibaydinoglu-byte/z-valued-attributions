"""E3 evidence-quality sweep (protocol: protocols/E3_protocol.md). Reproduces each E2 run up to SHAP, verifies phi_n,
then propagates synthetic evidence of controlled lift. Usage: python3 e3_sweep.py <dataset> <seed> [...]"""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import pilot_controlled as P
import e2_labeller as L
import e2_run as E2
from e1_full import lattice_probs
from pilot_cicids import tf

LEVELS = [1, 1.5, 2, 3, 5, 10, 20]
SENS = [(0.5, 3), (0.5, 10)]
OUT = f"{ROOT}/results/e3"; os.makedirs(OUT, exist_ok=True)

def evidence(flip, recall, lift, rng):
    n, nf = len(flip), int(flip.sum())
    noise = nf / n
    prec = min(0.95, lift * noise); capped = lift * noise > 0.95
    fl = np.flatnonzero(flip); cl = np.flatnonzero(~flip)
    tp = fl[rng.random(nf) < recall]
    n_fp = int(round(len(tp) * (1 - prec) / prec)); n_fp = min(n_fp, len(cl))
    fp = rng.choice(cl, n_fp, replace=False)
    flagged = np.zeros(n, bool); flagged[tp] = True; flagged[fp] = True
    rest = (nf - len(tp)) / max(1, n - flagged.sum())
    rho = np.where(flagged, prec, max(rest, 1e-4))
    return rho, dict(lift=lift, recall=recall, precision=prec, capped=bool(capped), flagged=int(flagged.sum()))

def run(name, seed, df):
    t0 = time.time()
    yt, mech, par = L.label(df, seed); y = df["y"].values
    near = L.provenance(df, yt, pad=30)[2]
    rng = np.random.default_rng(seed)
    tr = rng.choice(len(df), E2.NTRAIN, replace=False)
    pool = np.setdiff1d(np.arange(len(df)), tr)
    te = np.r_[rng.choice(pool[near[pool]], E2.NPER, replace=False), rng.choice(pool[~near[pool]], E2.NPER, replace=False)]
    Xraw = df[E2.FEATS].values
    sc = StandardScaler().fit(tf(Xraw[tr]))
    Xtr, Xte = sc.transform(tf(Xraw[tr])), sc.transform(tf(Xraw[te]))
    yn, yc = yt[tr].astype(float), y[tr].astype(float)
    bg = Xtr[rng.choice(E2.NTRAIN, E2.NBG, replace=False)]
    C, S = P.fit_antecedents(Xtr, seed); H = P.H_matrix(Xtr, C, S); sgn = 1 - 2 * yn
    saved = np.load(f"{ROOT}/results/e2/e2_{name}_{seed}.npz")
    phi_n = np.zeros((len(te), len(E2.FEATS))); Dv = []
    for t, x in enumerate(Xte):
        Cm = P.shap_firing(x, bg, C, S) @ H
        phi_n[t] = Cm @ yn; Dv.append(Cm * sgn)
    assert np.allclose(phi_n, saved["phi_n"], atol=1e-10), "E2 sample not reproduced"
    sdf = phi_n.std(0) + 1e-12
    flip = yn != yc
    erng = np.random.default_rng(seed + 50000)
    configs = [(0.9, l) for l in LEVELS] + SENS
    out, metas = {}, []
    for r, l in configs:
        rho, meta = evidence(flip, r, l, erng); metas.append(meta)
        arr = np.zeros((3, len(te), len(E2.FEATS)))
        for t in range(len(te)):
            for i in range(len(E2.FEATS)):
                a, b, c, _ = lattice_probs(Dv[t][i], rho, phi_n[t, i], sdf[i]); arr[:, t, i] = (a, b, c)
        out[f"P_r{r}_L{l}"] = arr
    np.savez_compressed(f"{OUT}/e3_{name}_{seed}.npz", **out)
    m = dict(dataset=name, seed=seed, train_noise=float(flip.mean()), evidence=metas, seconds=time.time() - t0)
    json.dump(m, open(f"{OUT}/e3_{name}_{seed}.json", "w"), indent=1); print(json.dumps(m)[:300], flush=True)

if __name__ == "__main__":
    name = sys.argv[1]
    df = pd.read_parquet(f"{ROOT}/data/nfv3/pool_{name}.parquet"); df["cls"] = df["cls"].astype(str)
    for s in [int(x) for x in sys.argv[2:]]:
        if not os.path.exists(f"{OUT}/e3_{name}_{s}.json"): run(name, s, df)
