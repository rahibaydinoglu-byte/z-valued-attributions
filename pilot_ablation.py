import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import sys, json, time, pandas as pd
import pilot_cicids as Q
cols = ["file", "mi", "co", "cc"] + Q.FEATS
d = pd.read_parquet(f"{ROOT}/data/matched/matched.parquet", columns=cols)
d["yo"] = (~d.co.isin(["BENIGN", "Attempted"])).astype(int); d["yc"] = (~d.cc.isin(["BENIGN", "Attempted"])).astype(int)
d["prov_rho"], _ = Q.provenance_rho(d)
seeds = [int(x) for x in sys.argv[1:]]
out = []
for s in seeds:
    r = Q.run(s, d, rho_modes=("const", "oracle"), nboot=0); out.append(r)
    print(json.dumps({k: v for k, v in r.items() if "|" in k and ("Z[" in k or "meanP" in k)}), flush=True)
    json.dump(out, open(f"{ROOT}/pilots/pilot_ablation_{'_'.join(map(str, seeds))}.json", "w"), indent=1, default=float)
