"""Stage 6 pre-emptive check (R1/R3): do simpler scores using the SAME evidence (BLK-FS, E1c) match the lattice Z?
Reproduces each E1c run up to the label-attribution kernel (verifies phi_n vs saved npz), then scores:
  ED      : |E[Delta]| relative to the event threshold (first moment only)
  GAUSS   : P(event) under a normal approximation with block variance (two moments)
  MARKOV  : 1 - Markov surrogate of Theorem 1(c)
and evaluates AUROC/Brier against the saved events. Usage: python3 e6_simple.py <seed> ..."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import json, math, os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pilot_controlled as P
import e1_full as E, e1_block as B
from block_noise import make_units, aggregate
OUT = f"{ROOT}/results/e6"; os.makedirs(OUT, exist_ok=True)

def run(seed, fsname, d):
    feats = E.FS[fsname]; rng = np.random.default_rng(seed)
    tr_idx = rng.choice(d.index.values, E.NTRAIN, replace=False); tr = d.loc[tr_idx]; rest = d.drop(index=tr_idx)
    te_parts = []
    for s, fn in E.STRATA.items():
        cand = rest.index[fn(rest["file"]).values]; te_parts.append(rest.loc[rng.choice(cand, E.NPER, replace=False)])
    te = pd.concat(te_parts)
    sc = StandardScaler().fit(E.tf(tr[feats].values))
    Xtr, Xte = sc.transform(E.tf(tr[feats].values)), sc.transform(E.tf(te[feats].values))
    yn = tr["yo"].values.astype(float)
    bg = Xtr[rng.choice(E.NTRAIN, E.NBG, replace=False)]
    C, S = P.fit_antecedents(Xtr, seed); H = P.H_matrix(Xtr, C, S); sgn = 1 - 2 * yn
    rho_flow = E.prov(tr)
    unit, rho_u = make_units(tr["blk_fs"].values, tr["cat"].values, rho_flow)
    phi_n = np.zeros((len(te), len(feats))); D = []
    for t, x in enumerate(Xte):
        Cm = P.shap_firing(x, bg, C, S) @ H
        phi_n[t] = Cm @ yn; D.append(np.stack([aggregate(Cm[i] * sgn, unit, len(rho_u)) for i in range(len(feats))]))
    z = np.load(f"{ROOT}/results/e1c/e1_{seed}_{fsname}.npz")
    assert np.allclose(z["phi_n"], phi_n, atol=1e-10), "not reproduced"
    sdf = phi_n.std(0) + 1e-12
    out = {}
    for e in ("rel", "sign"):
        y = z[f"ev_{e}"]; k = {"rel": 1, "sign": 2}[e]
        sc_ = {"ED": np.zeros_like(phi_n), "GAUSS": np.zeros_like(phi_n), "MARKOV": np.zeros_like(phi_n)}
        for t in range(len(te)):
            for i in range(len(feats)):
                Du = D[t][i]; mu = float(Du @ rho_u); sd = math.sqrt(float((Du ** 2 * rho_u * (1 - rho_u)).sum())) + 1e-15
                f0 = phi_n[t, i]
                if e == "rel":
                    tau = 0.5 * abs(f0) + 0.05 * sdf[i]
                    sc_["ED"][t, i] = abs(mu) / tau
                    sc_["GAUSS"][t, i] = norm.sf((tau - mu) / sd) + norm.cdf((-tau - mu) / sd)
                    sc_["MARKOV"][t, i] = min(1.0, float(np.abs(Du) @ rho_u) / tau)
                else:   # sign change: phi* = f0 + Delta crosses zero beyond the dead zone
                    dz = 0.05 * sdf[i]
                    sc_["ED"][t, i] = -np.sign(f0) * mu / (abs(f0) + dz)
                    sc_["GAUSS"][t, i] = norm.cdf((-dz - f0 - mu) / sd) if f0 > 0 else norm.sf((dz - f0 - mu) / sd)
                    sc_["MARKOV"][t, i] = min(1.0, float(np.abs(Du) @ rho_u) / (abs(f0) + dz))
        sc_["LATTICE"] = z["P_BLK-FS"][k]
        for name, s in sc_.items():
            a = [roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1]) if 0 < y[:, i].mean() < 1]
            out[f"{e}|{name}|auc"] = float(np.mean(a)) if a else float("nan")
            if name in ("GAUSS", "LATTICE"):
                out[f"{e}|{name}|brier"] = float(((s - y) ** 2).mean()); out[f"{e}|{name}|meanP"] = float(s.mean())
                # 10-bin ECE
                p, yy = s.ravel(), y.ravel().astype(float); bins = np.clip((p * 10).astype(int), 0, 9)
                out[f"{e}|{name}|ece"] = float(sum(abs(p[bins == b].mean() - yy[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
                np.save(f"{OUT}/cal_{e}_{name}_{seed}_{fsname}.npy", np.stack([p, yy]))
        out[f"{e}|rate"] = float(y.mean())
    json.dump(out, open(f"{OUT}/e6_{seed}_{fsname}.json", "w"))
    print(seed, fsname, {k: round(v, 3) for k, v in out.items() if "auc" in k}, flush=True)

if __name__ == "__main__":
    d = B.load()
    for s in [int(x) for x in sys.argv[1:]]:
        for fs in ("A", "B"):
            if not os.path.exists(f"{OUT}/e6_{s}_{fs}.json"): run(s, fs, d)
