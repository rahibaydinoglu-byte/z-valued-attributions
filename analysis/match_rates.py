"""Match rate of original CIC-IDS2017 flows by original label (Sec. 6.1, selection caveat)."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import glob, pandas as pd
tot = {}
for f in glob.glob(f"{ROOT}/data/cic_original/extracted/TrafficLabelling */*.csv"):
    hdr = pd.read_csv(f, encoding="latin1", nrows=0).columns; lab = [c for c in hdr if c.strip() == "Label"][0]
    for k, v in pd.read_csv(f, usecols=[lab], encoding="latin1", dtype=str)[lab].astype(str).str.strip().value_counts().items(): tot[k] = tot.get(k, 0) + int(v)
m = pd.read_parquet(f"{ROOT}/data/matched/matched.parquet", columns=["label_original"])["label_original"].str.strip().value_counts()
df = pd.DataFrame({"orig": pd.Series(tot), "matched": m}).fillna(0); df = df[df.orig > 0]; df["rate"] = (df.matched / df.orig).round(3)
print(df.sort_values("orig", ascending=False).to_string()); print("overall %.3f" % (df.matched.sum() / df.orig.sum()))
