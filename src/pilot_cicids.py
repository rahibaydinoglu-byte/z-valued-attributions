"""Z-XAI pilot B: REAL label noise on CIC-IDS2017 (original labels = noisy, corrected labels = clean),
on matched flows with identical features. Binary task: attack vs benign ("Attempted" = benign, following the corrected release).
Pre-declared before running: events rel & sign primary (abs reported); estimators CL, provenance, CL+provenance;
baselines S, bootstrap SD, |phi|, bootSD/|phi|; 5 independent subsamples.
"""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import itertools, math, sys, time, json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pilot_controlled as P          # firing, shap_firing, lattice_dist
import rho_estimators as RE

FEATS = ["destination port", "flow duration", "total fwd packets", "total backward packets",
         "total length of fwd packets", "flow bytes/s", "init_win_bytes_forward", "psh flag count"]
R, LAM, NBG, NTRAIN, NTEST, NBOOT = 16, 1e-2, 30, 6000, 150, 8
P.R = R; P.LAM = LAM

def tf(X):
    X = np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(X) * np.log1p(np.abs(X))

def provenance_rho(df, pad=2, rho_win=0.2, rho_payload=0.5):
    """Uses ONLY original (noisy) labels + documented labelling rules; never the corrected labels.
    (i) benign-labelled flows within +-pad minutes of an originally attack-labelled minute of the same capture file;
    (ii) attack-labelled flows with zero forward payload (unsuccessful attempts, Engelen et al. 2021 rule)."""
    yo = df["yo"].values
    att = df.loc[yo == 1, ["file", "mi"]].drop_duplicates()
    win = set((f, mi + d) for f, mi in att.itertuples(index=False) for d in range(-pad, pad + 1))
    inwin = np.fromiter(((f, mi) in win for f, mi in zip(df["file"], df["mi"])), bool, len(df))
    rho = np.full(len(df), 0.0)
    rho[(yo == 0) & inwin] = rho_win
    rho[(yo == 1) & (df["total length of fwd packets"].fillna(0).values == 0)] = rho_payload
    return rho, inwin

def run(seed, data, rho_modes=("cl", "prov", "cl+prov"), nboot=None):
    nboot = NBOOT if nboot is None else nboot
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    tr, te = data.iloc[idx[:NTRAIN]], data.iloc[idx[NTRAIN:NTRAIN + 4 * NTEST]]
    sc = StandardScaler().fit(tf(tr[FEATS].values))
    Xtr, Xte_all = sc.transform(tf(tr[FEATS].values)), sc.transform(tf(te[FEATS].values))
    yn, yc = tr["yo"].values.astype(float), tr["yc"].values.astype(float)
    # test points: half from noisy capture windows, half elsewhere (so both regimes are represented; declared)
    hot = te["file"].str.contains("Thursday-WorkingHours-Afternoon|Afternoon-DDos").values
    pick = np.r_[np.flatnonzero(hot)[:NTEST // 2], np.flatnonzero(~hot)[:NTEST // 2]]
    Xte = Xte_all[pick]
    bg = Xtr[rng.choice(len(Xtr), NBG, replace=False)]
    C, S = P.fit_antecedents(Xtr, seed); H = P.H_matrix(Xtr, C, S)
    sgn = 1 - 2 * yn

    # model quality (scope risk R1): TSK vs gradient boosting, both trained on noisy labels, scored on clean labels
    W_te = P.firing(Xte_all, C, S)
    tsk_auc = roc_auc_score(te["yc"].values, W_te @ (H @ yn))
    gb = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, yn)
    gb_auc = roc_auc_score(te["yc"].values, gb.predict_proba(Xte_all)[:, 1])

    rhos = {}
    p1 = np.zeros(len(yn))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, yn):
        p1[b] = HistGradientBoostingClassifier(random_state=seed).fit(Xtr[a], yn[a]).predict_proba(Xtr[b])[:, 1]
    p1 = np.clip(p1, 1e-4, 1 - 1e-4)
    rhos["cl"] = RE.rho_cl(p1, yn)
    pr = data.loc[tr.index, 'prov_rho'].values
    rhos["prov"] = np.maximum(pr, 1e-3)
    rhos["cl+prov"] = np.maximum(pr, rhos["cl"])
    rhos["const"] = np.full(len(yn), float(data["yo"].ne(data["yc"]).mean()))   # structure-only ablation (global rate)
    rhos["oracle"] = np.where(yn != yc, 0.999, 0.001)                            # upper bound (uses clean labels)
    flip = (yn != yc)
    rho_auc = {k: roc_auc_score(flip, v) for k, v in rhos.items()}
    fp_tp = {k: float(v[~flip].sum() / max(v[flip].sum(), 1e-9)) for k, v in rhos.items()}

    boots = []
    for b in range(nboot):
        ii = rng.choice(len(Xtr), len(Xtr), replace=True)
        Cb, Sb = P.fit_antecedents(Xtr[ii], seed + 100 + b)
        boots.append((Cb, Sb, P.H_matrix(Xtr[ii], Cb, Sb) @ yn[ii]))

    phi_n, phi_c, sens, sdb, Dvec = [], [], [], [], []
    for x in Xte:
        A = P.shap_firing(x, bg, C, S); Cm = A @ H
        phi_n.append(Cm @ yn); phi_c.append(Cm @ yc); sens.append(np.abs(Cm).sum(1)); Dvec.append(Cm * sgn)
        sdb.append(np.stack([P.shap_firing(x, bg, Cb, Sb) @ th for Cb, Sb, th in boots]).std(0) if boots else np.zeros(len(FEATS)))
    phi_n, phi_c = np.array(phi_n), np.array(phi_c)
    d = len(FEATS); sdf = phi_n.std(0) + 1e-9; delta = phi_c - phi_n
    ev = {"abs": np.abs(delta) > 0.25 * sdf,
          "rel": np.abs(delta) > 0.5 * np.abs(phi_n) + 0.05 * sdf,
          "sign": (phi_c * phi_n < 0) & (np.maximum(np.abs(phi_c), np.abs(phi_n)) > 0.05 * sdf)}
    base = {"S": np.array(sens), "bootSD": np.array(sdb), "|phi|": np.abs(phi_n),
            "bootSD/|phi|": np.array(sdb) / (np.abs(phi_n) + 0.05 * sdf)}
    out = {"seed": seed, "tsk_auc_clean": tsk_auc, "gb_auc_clean": gb_auc, "train_flip_rate": float(flip.mean()),
           "rho_auc": rho_auc, "fp_tp_mass": fp_tp, "event_rate": {e: float(v.mean()) for e, v in ev.items()}}
    for mode in rho_modes:
        rho = rhos[mode]
        Pz = {e: np.zeros_like(phi_n) for e in ev}
        for t in range(len(Xte)):
            for i in range(d):
                dv = Dvec[t][i]
                sd = math.sqrt((dv ** 2 * rho * (1 - rho)).sum()) + 1e-12
                v, p = P.lattice_dist(dv, rho, min(0.05 * sdf[i], sd) / 40)
                f0 = phi_n[t, i]; fc = f0 + v
                Pz["abs"][t, i] = p[np.abs(v) > 0.25 * sdf[i]].sum()
                Pz["rel"][t, i] = p[np.abs(v) > 0.5 * abs(f0) + 0.05 * sdf[i]].sum()
                Pz["sign"][t, i] = p[(fc * f0 < 0) & (np.maximum(np.abs(fc), abs(f0)) > 0.05 * sdf[i])].sum()
        for e in ev:
            out[f"Z[{mode}]:{e}"] = Pz[e]
    # AUROC per feature, averaged
    res = {k: v for k, v in out.items() if not k.startswith("Z[")}
    for e, y in ev.items():
        scores = dict(base); scores.update({k.split(":")[0]: v for k, v in out.items() if k.startswith("Z[") and k.endswith(":" + e)})
        for name, s in scores.items():
            a = [roc_auc_score(y[:, i], s[:, i]) for i in range(d) if 0 < y[:, i].mean() < 1]
            res[f"{e}|{name}"] = float(np.mean(a)) if a else float("nan")
        for mode in rho_modes:
            res[f"{e}|meanP[{mode}]"] = float(out[f"Z[{mode}]:{e}"].mean())
    return res

if __name__ == "__main__":
    t0 = time.time()
    cols = ["file", "mi", "co", "cc"] + FEATS
    data = pd.read_parquet(f"{ROOT}/data/matched/matched.parquet", columns=cols)
    data["yo"] = (~data["co"].isin(["BENIGN", "Attempted"])).astype(int)
    data["yc"] = (~data["cc"].isin(["BENIGN", "Attempted"])).astype(int)
    print(f"data {len(data):,} rows; binary noise rate {(data.yo != data.yc).mean():.4f}", flush=True)
    data["prov_rho"], _ = provenance_rho(data)   # computed on the FULL noisy corpus (original labels only)
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
    allr = []
    for s in seeds:
        r = run(s, data); allr.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, default=str), flush=True)
        json.dump(allr, open(f"{ROOT}/pilots/pilot_cicids_results_{'_'.join(map(str, seeds))}.json", "w"), indent=1, default=float)
    print(f"total {time.time()-t0:.0f}s")
