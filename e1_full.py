"""E1 full study (protocol: protocols/E1_E1c_protocol.md, frozen 24.10.2026).
Usage: python3 e1_full.py <seed> [<seed> ...]   -> writes results/e1_<seed>_<FS>.npz + .json per run
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
from lattice_window import lattice_dist_window
from pilot_cicids import provenance_rho, tf

FS = {
    "A": ["destination port", "flow duration", "total fwd packets", "total backward packets",
          "total length of fwd packets", "flow bytes/s", "init_win_bytes_forward", "psh flag count"],
    "B": ["flow iat mean", "fwd packet length max", "bwd packet length mean", "total length of bwd packets",
          "fin flag count", "syn flag count", "init_win_bytes_backward", "flow packets/s"],
}
R, LAM, NBG, NTRAIN, NPER, NBOOT = 16, 1e-2, 30, 6000, 30, 8
P.R = R; P.LAM = LAM
OUT = f"{ROOT}/results/e1"; os.makedirs(OUT, exist_ok=True)
STRATA = {"S1_wed": lambda f: f.str.startswith("Wednesday"),
          "S2_thu_pm": lambda f: f.str.contains("Thursday-WorkingHours-Afternoon"),
          "S3_fri_ddos": lambda f: f.str.contains("Afternoon-DDos"),
          "S4_clean": lambda f: f.str.contains("Monday|Tuesday|Thursday-WorkingHours-Morning|Friday-WorkingHours-Morning|Afternoon-PortScan")}

def load():
    cols = ["file", "mi", "co", "cc"] + sorted(set(FS["A"] + FS["B"]))
    d = pd.read_parquet(f"{ROOT}/data/matched/matched.parquet", columns=cols)
    d["file"] = d["file"].astype(str)
    d["yo"] = (~d["co"].isin(["BENIGN", "Attempted"])).astype(np.int8)
    d["yc"] = (~d["cc"].isin(["BENIGN", "Attempted"])).astype(np.int8)
    d["pad2"] = provenance_rho(d, pad=2)[1]
    for pad in (0, 1, 5):
        d[f"pad{pad}"] = provenance_rho(d, pad=pad)[1]
    d["zeropay"] = (d["yo"] == 1) & (d["total length of fwd packets"].fillna(0) == 0)
    return d

def prov(df, pad=2, rw=0.2, rp=0.5):
    rho = np.full(len(df), 1e-3)
    rho[(df["yo"].values == 0) & df[f"pad{pad}"].values] = rw
    rho[df["zeropay"].values] = rp
    return rho

def prov_audited(d, train_idx, df, rng, frac=0.01):
    pool = d.index.difference(train_idx)
    aud = d.loc[rng.choice(pool, int(frac * len(d)), replace=False)]
    cat = lambda x: np.select([(x["yo"] == 0) & x["pad2"], x["yo"] == 0, x["zeropay"]], [0, 1, 2], 3)
    ca, flip = cat(aud), (aud["yo"] != aud["yc"]).values
    rates = {c: (flip[ca == c].sum() + 1) / ((ca == c).sum() + 2) for c in range(4)}
    return np.vectorize(rates.get)(cat(df)).astype(float), {int(k): float(v) for k, v in rates.items()}

def lattice_probs(dv, rho, f0, sdf_i):
    sd = math.sqrt(float((dv ** 2 * rho * (1 - rho)).sum())) + 1e-15
    v, p, lost = lattice_dist_window(dv, rho, min(0.05 * sdf_i, sd) / 40)
    fc = f0 + v
    return (p[np.abs(v) > 0.25 * sdf_i].sum(), p[np.abs(v) > 0.5 * abs(f0) + 0.05 * sdf_i].sum(),
            p[(fc * f0 < 0) & (np.maximum(np.abs(fc), abs(f0)) > 0.05 * sdf_i)].sum(), lost)

def run(seed, fsname, d, sens=False, blocks=None):
    t0 = time.time(); feats = FS[fsname]
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(d.index.values, NTRAIN, replace=False)
    tr = d.loc[tr_idx]
    rest = d.drop(index=tr_idx)
    te_parts, strata = [], []
    for s, fn in STRATA.items():
        cand = rest.index[fn(rest["file"]).values]
        pick = rng.choice(cand, NPER, replace=False)
        te_parts.append(rest.loc[pick]); strata += [s] * NPER
    te = pd.concat(te_parts); strata = np.array(strata)
    sc = StandardScaler().fit(tf(tr[feats].values))
    Xtr, Xte = sc.transform(tf(tr[feats].values)), sc.transform(tf(te[feats].values))
    yn, yc = tr["yo"].values.astype(float), tr["yc"].values.astype(float)
    bg = Xtr[rng.choice(NTRAIN, NBG, replace=False)]
    C, S = P.fit_antecedents(Xtr, seed); H = P.H_matrix(Xtr, C, S); sgn = 1 - 2 * yn

    # model quality on clean labels, independent eval sample
    ev_idx = rng.choice(rest.index.values, 20000, replace=False)
    Xev = sc.transform(tf(d.loc[ev_idx, feats].values)); yev = d.loc[ev_idx, "yc"].values
    tsk_auc = roc_auc_score(yev, P.firing(Xev, C, S) @ (H @ yn))
    gb = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, yn)
    gb_auc = roc_auc_score(yev, gb.predict_proba(Xev)[:, 1])

    rhos = {}
    p1 = np.zeros(NTRAIN)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, yn):
        p1[b] = HistGradientBoostingClassifier(random_state=seed).fit(Xtr[a], yn[a]).predict_proba(Xtr[b])[:, 1]
    rhos["CL"] = RE.rho_cl(np.clip(p1, 1e-4, 1 - 1e-4), yn)
    rhos["PROV"] = prov(tr)
    rhos["PROV-AUD"], aud_rates = prov_audited(d, tr_idx, tr, rng)
    rhos["CONST"] = np.full(NTRAIN, float((d["yo"] != d["yc"]).mean()))
    rhos["ORACLE"] = np.where(yn != yc, 0.999, 0.001)
    if sens:
        for pad in (0, 1, 5): rhos[f"PROV_pad{pad}"] = prov(tr, pad=pad)
        for rw in (0.1, 0.3): rhos[f"PROV_rw{rw}"] = prov(tr, rw=rw)
        for rp in (0.3, 0.7): rhos[f"PROV_rp{rp}"] = prov(tr, rp=rp)
    flip = yn != yc
    rho_auc = {k: float(roc_auc_score(flip, v)) for k, v in rhos.items() if k not in ("CONST",) and len(v) == len(flip)}

    boots = []
    for b in range(NBOOT):
        ii = rng.choice(NTRAIN, NTRAIN, replace=True)
        Cb, Sb = P.fit_antecedents(Xtr[ii], seed + 1000 + b)
        boots.append((Cb, Sb, P.H_matrix(Xtr[ii], Cb, Sb) @ yn[ii]))
    n_te, dd = len(te), len(feats)
    phi_n = np.zeros((n_te, dd)); phi_c = np.zeros((n_te, dd)); sens_S = np.zeros((n_te, dd)); sdb = np.zeros((n_te, dd))
    Dv = []
    for t, x in enumerate(Xte):
        Cm = P.shap_firing(x, bg, C, S) @ H
        phi_n[t], phi_c[t], sens_S[t] = Cm @ yn, Cm @ yc, np.abs(Cm).sum(1)
        Dv.append(Cm * sgn)
        sdb[t] = np.stack([P.shap_firing(x, bg, Cb, Sb) @ th for Cb, Sb, th in boots]).std(0)
    sdf = phi_n.std(0) + 1e-12; delta = phi_c - phi_n
    events = {"abs": np.abs(delta) > 0.25 * sdf,
              "rel": np.abs(delta) > 0.5 * np.abs(phi_n) + 0.05 * sdf,
              "sign": (phi_c * phi_n < 0) & (np.maximum(np.abs(phi_c), np.abs(phi_n)) > 0.05 * sdf)}
    Pz, lost_max = {}, 0.0
    unit_of = {}
    if blocks:
        from block_noise import make_units, aggregate
        for mname, col in blocks.items():
            unit, rho_u = make_units(tr[col].values, tr["cat"].values, rhos["PROV"])
            unit_of[mname] = unit; rhos[mname] = rho_u
    for mode, rho in rhos.items():
        arr = np.zeros((3, n_te, dd))
        for t in range(n_te):
            for i in range(dd):
                dv = aggregate(Dv[t][i], unit_of[mode], len(rho)) if mode in unit_of else Dv[t][i]
                a, r_, s_, lost = lattice_probs(dv, rho, phi_n[t, i], sdf[i])
                arr[:, t, i] = (a, r_, s_); lost_max = max(lost_max, lost)
        Pz[mode] = arr
    np.savez_compressed(f"{OUT}/e1_{seed}_{fsname}.npz", phi_n=phi_n, phi_c=phi_c, S=sens_S, bootSD=sdb, strata=strata,
                        **{f"ev_{k}": v for k, v in events.items()}, **{f"P_{k}": v for k, v in Pz.items()})
    meta = dict(seed=seed, fs=fsname, tsk_auc_clean=tsk_auc, gb_auc_clean=gb_auc, train_flip=float(flip.mean()),
                rho_auc=rho_auc, audit_rates=aud_rates, lattice_lost_max=lost_max, seconds=time.time() - t0)
    json.dump(meta, open(f"{OUT}/e1_{seed}_{fsname}.json", "w"), indent=1)
    print(json.dumps(meta), flush=True)

if __name__ == "__main__":
    d = load()
    for s in [int(x) for x in sys.argv[1:]]:
        for fsname in ("A", "B"):
            if os.path.exists(f"{OUT}/e1_{s}_{fsname}.json"):
                continue
            run(s, fsname, d, sens=(fsname == "A" and s < 110))
