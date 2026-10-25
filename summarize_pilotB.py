import os as _os, sys as _sys
ROOT = _os.environ.get("ZXAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
import json, numpy as np
from scipy.stats import wilcoxon
A = json.load(open(f"{ROOT}/pilots/pilot_ablation_0_1_2_3_4.json")) + json.load(open(f"{ROOT}/pilots/pilot_ablation_3.json")) + json.load(open(f"{ROOT}/pilots/pilot_ablation_4.json"))
A = sorted(A, key=lambda a: a["seed"])
R = {r["seed"]: r for r in json.load(open(f"{ROOT}/pilots/pilot_cicids_results_0_1_2.json")) + json.load(open(f"{ROOT}/pilots/pilot_cicids_results_3_4.json"))}
print("ablation seeds", [a["seed"] for a in A])
base = ["S", "bootSD", "|phi|", "bootSD/|phi|"]
for e in ["rel", "sign", "abs"]:
    rows = {}
    for k in ["Z[const]", "Z[cl]", "Z[prov]", "Z[cl+prov]", "Z[oracle]"]:
        src = A if k in ("Z[const]", "Z[oracle]") else [R[a["seed"]] for a in A]
        rows[k] = np.array([r[f"{e}|{k}"] for r in src])
    bb = np.array([max(R[a["seed"]][f"{e}|{k}"] for k in base) for a in A])
    print(f"[{e}] best-baseline {bb.mean():.3f} | " + "  ".join(f"{k}={v.mean():.3f}±{v.std():.3f}" for k, v in rows.items()))
    print(f"      prov-const {np.mean(rows['Z[prov]']-rows['Z[const]']):+.3f} (wins {int((rows['Z[prov]']>rows['Z[const]']).sum())}/5) | "
          f"cl-const {np.mean(rows['Z[cl]']-rows['Z[const]']):+.3f} (wins {int((rows['Z[cl]']>rows['Z[const]']).sum())}/5) | "
          f"const-bestbase {np.mean(rows['Z[const]']-bb):+.3f} (wins {int((rows['Z[const]']>bb).sum())}/5)")
    print("      meanP const/oracle:", round(np.mean([a[f'{e}|meanP[const]'] for a in A]), 3), round(np.mean([a[f'{e}|meanP[oracle]'] for a in A]), 3),
          " observed", round(np.mean([R[a['seed']]['event_rate'][e] for a in A]), 3))
