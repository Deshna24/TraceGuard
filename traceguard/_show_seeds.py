import json
from pathlib import Path

BASE    = Path(r"c:\Users\DESHNA\TraceGuard\traceguard")
METRICS = BASE / "outputs" / "metrics"
REPORTS = BASE / "reports"

# Load five-seed results
sr  = json.loads((METRICS / "robustness_five_seeds.json").read_text())
agg = sr["aggregate"]
per = sr["per_seed"]

# Print per-seed table
print("\n=== PER-SEED RESULTS ===")
print(f"{'Seed':>6} {'Acc':>7} {'MacF1':>7} {'B-F1':>7} {'R-F1':>7} {'H-F1':>7} {'PreAct':>8}")
print("-" * 55)
for r in per:
    print(f"{r['seed']:>6} {r['accuracy']:>7.4f} {r['macro_f1']:>7.4f} {r['benign_f1']:>7.4f} {r['resisted_f1']:>7.4f} {r['hijacked_f1']:>7.4f} {r['pre_action_rate']:>8.2%}")
print("-" * 55)
for m in ["accuracy","macro_f1","benign_f1","resisted_f1","hijacked_f1","pre_action_rate"]:
    a = agg[m]
    print(f"{'Mean±Std':>6} {a['mean']:>7.4f}±{a['std']:.4f}  [{a['min']:.4f},{a['max']:.4f}]  ({m})")
