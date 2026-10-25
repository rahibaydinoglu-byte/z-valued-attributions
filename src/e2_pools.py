"""E2: build fixed, time-preserving random pools from NetFlow-v3 (uniform flow subsample, fixed seed).
UNSW-NB15-v3: all flows; ToN-IoT-v3, CICIDS2018-v3: ~2.0 M flows each. NF-BoT-IoT-v3 excluded (99.7% attacks).
"""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import zipfile, sys, numpy as np, pandas as pd

FEATS = ["L4_DST_PORT", "FLOW_DURATION_MILLISECONDS", "IN_PKTS", "OUT_PKTS", "IN_BYTES",
         "SRC_TO_DST_AVG_THROUGHPUT", "TCP_WIN_MAX_IN", "TCP_FLAGS"]
META = ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"]
TARGET = {"NF-UNSW-NB15-v3": None, "NF-ToN-IoT-v3": 2_000_000, "NF-CICIDS2018-v3": 2_000_000}
TOTAL = {"NF-UNSW-NB15-v3": 2_365_424, "NF-ToN-IoT-v3": 27_520_260, "NF-CICIDS2018-v3": 20_115_529}

for name in sys.argv[1:] or TARGET:
    zf = zipfile.ZipFile(f"{ROOT}/data/nfv3/{name}.zip")
    csvn = [n for n in zf.namelist() if n.endswith(f"{name}.csv")][0]
    frac = 1.0 if TARGET[name] is None else TARGET[name] / TOTAL[name]
    rng = np.random.default_rng(20261025)
    parts = []
    for ch in pd.read_csv(zf.open(csvn), usecols=META + FEATS, chunksize=1_000_000):
        if frac < 1: ch = ch[rng.random(len(ch)) < frac]
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    df["src"] = pd.factorize(df["IPV4_SRC_ADDR"])[0].astype(np.int32)
    df["dst"] = pd.factorize(df["IPV4_DST_ADDR"])[0].astype(np.int32)
    df["minute"] = (df["FLOW_START_MILLISECONDS"] // 60000).astype(np.int64)
    df["day"] = (df["FLOW_START_MILLISECONDS"] // 86_400_000).astype(np.int32)
    df["y"] = (df["Label"] != 0).astype(np.int8)
    df["cls"] = df["Attack"].astype("category")
    df = df.drop(columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"])
    for c in FEATS: df[c] = df[c].astype(np.float32)
    df.to_parquet(f"{ROOT}/data/nfv3/pool_{name}.parquet", index=False)
    print(f"{name}: pool {len(df):,} | attack share {df.y.mean():.3f} | days {df.day.nunique()} | src hosts {df.src.nunique()} | dst hosts {df.dst.nunique()}", flush=True)
