"""Memory-lean flow matching: original CIC-IDS2017 TrafficLabelling vs corrected release (Zenodo 22016274).
Key (after Sadath P et al. 2026): sorted endpoint pair (ip<<16|port), protocol, minute; 12h repair; shift search on Monday;
keys repeated within a version are dropped. Pass 2 reads features only for matched original rows.
"""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import glob, json, os, time, gc
import numpy as np, pandas as pd

ORIG_DIR = f"{ROOT}/data/cic_original/extracted/TrafficLabelling "
CORR = f"{ROOT}/data/zenodo_huang2026/cicids2017_corrected_huang2026.csv"
OUT = f"{ROOT}/data/matched"; os.makedirs(OUT, exist_ok=True)
t0 = time.time(); log = {}
def norm(c): return " ".join(str(c).strip().split()).lower()

def ip2int(s):
    p = s.astype(str).str.strip().str.split(".", expand=True)
    p = p.iloc[:, :4].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (p[:, 0] << 24) | (p[:, 1] << 16) | (p[:, 2] << 8) | p[:, 3]

def compact(df, ts):
    a = (ip2int(df["sip"]) << 16) | pd.to_numeric(df["sport"], errors="coerce").fillna(0).astype("int64").values
    b = (ip2int(df["dip"]) << 16) | pd.to_numeric(df["dport"], errors="coerce").fillna(0).astype("int64").values
    return pd.DataFrame({
        "e1": np.minimum(a, b), "e2": np.maximum(a, b),
        "pr": pd.to_numeric(df["proto"], errors="coerce").fillna(-1).astype("int16").values,
        "mi": ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(minutes=1)).fillna(-1).astype("int64").values,
        "hour": ts.dt.hour.astype("int8").values,
    })

# ---------- original, per file ----------
O = []
for f in sorted(glob.glob(ORIG_DIR + "/*.csv")):
    hdr = pd.read_csv(f, encoding="latin-1", nrows=0).columns
    w = {norm(c): c for c in hdr}
    use = {w["source ip"]: "sip", w["destination ip"]: "dip", w["source port"]: "sport", w["destination port"]: "dport",
           w["protocol"]: "proto", w["timestamp"]: "ts", w["label"]: "label"}
    df = pd.read_csv(f, encoding="latin-1", usecols=list(use), dtype=str).rename(columns=use)
    df["row"] = np.arange(len(df), dtype="int32")
    df = df.dropna(subset=["sip"])
    ts = pd.to_datetime(df["ts"].str.strip(), format="mixed", dayfirst=True, errors="coerce")
    k = compact(df, ts)
    k["row"] = df["row"].values; k["file"] = os.path.basename(f)
    k["label"] = df["label"].astype("category").values
    O.append(k); del df, ts; gc.collect()
    print(f"orig {os.path.basename(f)[:40]:40s} {len(k):,}  NaT-min={int((k['mi']<0).sum())}", flush=True)
O = pd.concat(O, ignore_index=True)
O["file"] = O["file"].astype("category"); O["label"] = O["label"].astype("category")
log["orig_rows"] = len(O)

# 12h repair (original only shows the 12h signature)
def repair(k, name):
    early, late = float(k["hour"].between(1, 7).mean()), float(k["hour"].between(13, 17).mean())
    fix = early > 0.02 and late < 0.005
    log[f"{name}_12h_fix"] = fix
    if fix:
        amb = k["hour"].between(1, 7).values
        k.loc[amb, "mi"] += 12 * 60
        log[f"{name}_12h_rows"] = int(amb.sum())
    print(f"{name}: hours1-7 {early:.2%} hours13-17 {late:.2%} fix={fix}", flush=True)
repair(O, "orig")

# ---------- corrected, chunked ----------
C = []
cols = {"Src IP": "sip", "Src Port": "sport", "Dst IP": "dip", "Dst Port": "dport", "Protocol": "proto", "Timestamp": "ts", "Label": "label"}
for ch in pd.read_csv(CORR, usecols=list(cols), dtype=str, chunksize=400_000):
    ch = ch.rename(columns=cols)
    ts = pd.to_datetime(ch["ts"], format="mixed", errors="coerce")
    k = compact(ch, ts); k["label"] = ch["label"].values
    C.append(k); del ch, ts; gc.collect()
C = pd.concat(C, ignore_index=True); C["label"] = C["label"].astype("category")
log["corr_rows"] = len(C)
repair(C, "corr")
print(f"loaded both ({time.time()-t0:.0f}s)  orig {len(O):,} corr {len(C):,}", flush=True)

KEY = ["e1", "e2", "pr", "mi"]
cmon = C.loc[C["mi"].between(*np.percentile(C["mi"], [0, 100])) & (pd.to_datetime(C["mi"].values * 60, unit="s").day == 3), KEY].drop_duplicates()
omon = O.loc[O["file"].astype(str).str.startswith("Monday"), KEY]
res = {}
for s in range(-12, 13):
    t = omon.copy(); t["mi"] += 60 * s
    res[s] = int(len(t.merge(cmon, on=KEY)))
best = max(res, key=res.get); log["shift_search"] = res; log["shift_applied_h"] = best
print("shift:", res, "-> best", best, flush=True)
O["mi"] += 60 * best

ou = O[~O.duplicated(KEY, keep=False)]; cu = C[~C.duplicated(KEY, keep=False)]
log["orig_unique"] = len(ou); log["corr_unique"] = len(cu)
m = ou.rename(columns={"label": "label_original"}).merge(
    cu[KEY + ["label"]].rename(columns={"label": "label_corrected"}), on=KEY)
log["matched"] = len(m); log["rate_vs_unique_orig"] = len(m) / len(ou); log["rate_vs_all_orig"] = len(m) / len(O)
print(f"matched {len(m):,} rate(unique) {len(m)/len(ou):.3f} rate(all) {len(m)/len(O):.3f}", flush=True)
del O, C, ou, cu; gc.collect()

def coarse(s):
    s = pd.Series(s).astype(str).str.strip()
    conds = [s.str.contains("Attempted", case=False), s.str.upper().eq("BENIGN"), s.str.contains("Infilt", case=False),
             s.str.contains("DDoS", case=False), s.str.contains("DoS", case=False), s.str.contains("Port", case=False),
             s.str.contains("Bot", case=False), s.str.contains("Web", case=False), s.str.contains("Patator", case=False),
             s.str.contains("Heartbleed", case=False)]
    names = ["Attempted", "BENIGN", "Infiltration", "DDoS", "DoS", "PortScan", "Bot", "WebAttack", "BruteForce", "Heartbleed"]
    return np.select(conds, names, "OTHER")
m["co"], m["cc"] = coarse(m["label_original"]), coarse(m["label_corrected"])
xt = pd.crosstab(m["co"], m["cc"]); xt.to_csv(f"{OUT}/transition_matrix.csv")
print(xt.to_string(), flush=True)
log["changed"] = int((m["co"] != m["cc"]).sum()); log["change_rate"] = float((m["co"] != m["cc"]).mean())
print(f"changed {log['changed']:,} ({log['change_rate']:.3%})", flush=True)

# ---------- pass 2: features of matched original rows ----------
parts = []
for f in sorted(glob.glob(ORIG_DIR + "/*.csv")):
    fn = os.path.basename(f); sel = m.index[m["file"].astype(str) == fn]
    if len(sel) == 0: continue
    rows = set(m.loc[sel, "row"].tolist())
    hdr = pd.read_csv(f, encoding="latin-1", nrows=0).columns
    drop = {"flow id", "source ip", "destination ip", "source port", "timestamp", "label", "external ip"}
    fc = [c for c in hdr if norm(c) not in drop]
    acc = []
    start = 0
    for ch in pd.read_csv(f, encoding="latin-1", usecols=fc, chunksize=200_000, low_memory=False):
        idx = np.arange(start, start + len(ch)); start += len(ch)
        keep = np.fromiter((i in rows for i in idx), bool, len(idx))
        sub = ch[keep].apply(pd.to_numeric, errors="coerce").astype("float32")
        sub["row"] = idx[keep].astype("int32"); acc.append(sub)
    acc = pd.concat(acc); acc.columns = [norm(c) for c in acc.columns]; acc["file"] = fn
    parts.append(acc); print(f"features {fn[:40]}: {len(acc):,}", flush=True); gc.collect()
F = pd.concat(parts, ignore_index=True)
m["file"] = m["file"].astype(str)
full = m[["file", "row", "mi", "label_original", "label_corrected", "co", "cc"]].merge(F, on=["file", "row"], how="inner")
full["label_original"] = full["label_original"].astype(str); full["label_corrected"] = full["label_corrected"].astype(str)
full.to_parquet(f"{OUT}/matched.parquet", index=False)
log["final_rows"] = len(full)
json.dump(log, open(f"{OUT}/match_log.json", "w"), indent=1, default=str)
print(f"saved {len(full):,} rows; total {time.time()-t0:.0f}s", flush=True)
