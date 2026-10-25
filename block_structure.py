"""Table 2: block structure of the real labelling errors among benign-labelled flows within +-2 min of an
originally attack-labelled minute (provenance category 0). Blocks are defined from metadata only."""
import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import numpy as np, pandas as pd
import e1_block as B
d = B.load(); w = d[d["cat"] == 0]; f = (w["yo"] != w["yc"]).astype(float)
print(f"flows {len(w)}  flip rate {f.mean():.3f}")
for name, k in (("file x minute", "fm"), ("file x minute x source host", "fms"), ("file x minute x host pair", "fmh"), ("file x host pair", "fh")):
    g = f.groupby(w[f"blk_{k}"].values); m, n = g.mean(), g.size()
    aon = ((m[n >= 2] == 0) | (m[n >= 2] == 1)).mean()
    between = (n * (m - f.mean()) ** 2).sum() / ((f - f.mean()) ** 2).sum()
    print(f"{name:32s} blocks {len(m):6d}  all-or-none (size>=2) {aon:.3f}  variance share between blocks {between:.3f}")
