"""E2 runner (protocol: protocols/E2_protocol.md, frozen 25.10.2026).
Usage: python3 e2_run.py <dataset> <seed> [<seed> ...]
"""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import json, math, os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pilot_controlled as P
import rho_estimators as RE
import e2_labeller as L
from e1_full import lattice_probs
from pilot_cicids import tf
from block_noise import make_units, aggregate

FEATS = ["L4_DST_PORT", "FLOW_DURATION_MILLISECONDS", "IN_PKTS", "OUT_PKTS", "IN_BYTES",
         "SRC_TO_DST_AVG_THROUGHPUT", "TCP_WIN_MAX_IN", "TCP_FLAGS"]
R, LAM, NBG, NTRAIN, NPER, NBOOT = 16, 1e-2, 30, 6000, 60, 8
P.R = R; P.LAM = LAM
OUT = f"{ROOT}/results/e2"; os.makedirs(OUT, exist_ok=True)

def run(name, seed, df):
    t0 = time.time()
    yt, mech, par = L.label(df, seed)
    y = df["y"].values
    rho_p, cat, _ = L.provenance(df, yt)                       # rule (i) + (ii) categories
    rho_prov = np.where(cat == 2, 1e-3, rho_p)                 # PROV = rule (i) only
    rho_prov2 = rho_p                                          # PROV+ii
    near = L.provenance(df, yt, pad=30)[2]
    flip_all = yt != y
    prior_auc = float(roc_auc_score(flip_all, rho_prov)) if 0 < flip_all.mean() < 1 else float("nan")
    blk = pd.factorize(pd.MultiIndex.from_arrays([df["day"].values, df["src"].values]))[0]

    rng = np.random.default_rng(seed)
    tr = rng.choice(len(df), NTRAIN, replace=False)
    pool = np.setdiff1d(np.arange(len(df)), tr)
    te = np.r_[rng.choice(pool[near[pool]], NPER, replace=False), rng.choice(pool[~near[pool]], NPER, replace=False)]
    strata = np.array(["near"] * NPER + ["far"] * NPER)
    Xraw = df[FEATS].values
    sc = StandardScaler().fit(tf(Xraw[tr]))
    Xtr, Xte = sc.transform(tf(Xraw[tr])), sc.transform(tf(Xraw[te]))
    yn, yc = yt[tr].astype(float), y[tr].astype(float)
    bg = Xtr[rng.choice(NTRAIN, NBG, replace=False)]
    C, S = P.fit_antecedents(Xtr, seed); H = P.H_matrix(Xtr, C, S); sgn = 1 - 2 * yn

    ev_idx = rng.choice(pool, 20000, replace=False)
    Xev = sc.transform(tf(Xraw[ev_idx])); yev = y[ev_idx]
    tsk_auc = roc_auc_score(yev, P.firing(Xev, C, S) @ (H @ yn))
    gb_auc = roc_auc_score(yev, HistGradientBoostingClassifier(random_state=seed).fit(Xtr, yn).predict_proba(Xev)[:, 1])

    rhos = {}
    p1 = np.zeros(NTRAIN)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, yn):
        p1[b] = HistGradientBoostingClassifier(random_state=seed).fit(Xtr[a], yn[a]).predict_proba(Xtr[b])[:, 1]
    rhos["CL"] = RE.rho_cl(np.clip(p1, 1e-4, 1 - 1e-4), yn)
    rhos["PROV"] = rho_prov[tr]
    rhos["PROV+ii"] = rho_prov2[tr]
    rhos["CONST"] = np.full(NTRAIN, float(flip_all.mean()))
    rhos["ORACLE"] = np.where(yn != yc, 0.999, 0.001)
    unit, rho_u = make_units(blk[tr], cat[tr], rho_prov2[tr])  # BLK-FS: units over categories 0 and 2 (as E1c)
    rho_u_prim = rho_u.copy()
    # BLK-FS uses the PROV marginal (rule i): units of category 2 get 1e-3
    cat_unit = np.zeros(len(rho_u), np.int8)
    for j in range(NTRAIN): cat_unit[unit[j]] = cat[tr][j]
    rho_u_prim[cat_unit == 2] = 1e-3
    rhos["BLK-FS"] = rho_u_prim
    flip = yn != yc
    rho_auc_tr = {k: float(roc_auc_score(flip, v)) for k, v in rhos.items() if len(v) == NTRAIN and k != "CONST" and 0 < flip.mean() < 1}

    boots = []
    for b in range(NBOOT):
        ii = rng.choice(NTRAIN, NTRAIN, replace=True)
        Cb, Sb = P.fit_antecedents(Xtr[ii], seed + 1000 + b)
        boots.append((Cb, Sb, P.H_matrix(Xtr[ii], Cb, Sb) @ yn[ii]))
    n_te, dd = len(te), len(FEATS)
    phi_n = np.zeros((n_te, dd)); phi_c = np.zeros_like(phi_n); sens_S = np.zeros_like(phi_n); sdb = np.zeros_like(phi_n); Dv = []
    for t, x in enumerate(Xte):
        Cm = P.shap_firing(x, bg, C, S) @ H
        phi_n[t], phi_c[t], sens_S[t] = Cm @ yn, Cm @ yc, np.abs(Cm).sum(1); Dv.append(Cm * sgn)
        sdb[t] = np.stack([P.shap_firing(x, bg, Cb, Sb) @ th for Cb, Sb, th in boots]).std(0)
    sdf = phi_n.std(0) + 1e-12; delta = phi_c - phi_n
    events = {"abs": np.abs(delta) > 0.25 * sdf, "rel": np.abs(delta) > 0.5 * np.abs(phi_n) + 0.05 * sdf,
              "sign": (phi_c * phi_n < 0) & (np.maximum(np.abs(phi_c), np.abs(phi_n)) > 0.05 * sdf)}
    Pz, lost_max = {}, 0.0
    for mode, rho in rhos.items():
        arr = np.zeros((3, n_te, dd))
        for t in range(n_te):
            for i in range(dd):
                dv = aggregate(Dv[t][i], unit, len(rho)) if mode == "BLK-FS" else Dv[t][i]
                a, r_, s_, lost = lattice_probs(dv, rho, phi_n[t, i], sdf[i])
                arr[:, t, i] = (a, r_, s_); lost_max = max(lost_max, lost)
        Pz[mode] = arr
    np.savez_compressed(f"{OUT}/e2_{name}_{seed}.npz", phi_n=phi_n, phi_c=phi_c, S=sens_S, bootSD=sdb, strata=strata,
                        **{f"ev_{k}": v for k, v in events.items()}, **{f"P_{k}": v for k, v in Pz.items()})
    meta = dict(dataset=name, seed=seed, labeller=par, noise=float(flip_all.mean()),
                mech={k: float(np.mean(mech == v)) for k, v in (("M1a", 1), ("M1b", 2), ("M2", 3))},
                prior_rho_auc_pool=prior_auc, train_flip=float(flip.mean()), rho_auc_train=rho_auc_tr,
                tsk_auc_clean=tsk_auc, gb_auc_clean=gb_auc, lattice_lost_max=lost_max, seconds=time.time() - t0)
    json.dump(meta, open(f"{OUT}/e2_{name}_{seed}.json", "w"), indent=1)
    print(json.dumps(meta), flush=True)

if __name__ == "__main__":
    name = sys.argv[1]
    df = pd.read_parquet(f"{ROOT}/data/nfv3/pool_{name}.parquet")
    df["cls"] = df["cls"].astype(str)
    for s in [int(x) for x in sys.argv[2:]]:
        if not os.path.exists(f"{OUT}/e2_{name}_{s}.json"):
            run(name, s, df)
