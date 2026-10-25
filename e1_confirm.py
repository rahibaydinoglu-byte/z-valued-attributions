import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import os, sys
import e1_full as E, e1_block as B
E.OUT = f"{ROOT}/results/e1c"; os.makedirs(E.OUT, exist_ok=True)
d = B.load()
for s in [int(x) for x in sys.argv[1:]]:
    for fs in ("A", "B"):
        if not os.path.exists(f"{E.OUT}/e1_{s}_{fs}.json"):
            E.run(s, fs, d, sens=False, blocks={"BLK-FS": "blk_fs", "BLK-FMH": "blk_fmh"})
