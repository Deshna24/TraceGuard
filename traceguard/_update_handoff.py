import json, shutil
from pathlib import Path

BASE    = Path(r"c:\Users\DESHNA\TraceGuard\traceguard")
METRICS = BASE / "outputs" / "metrics"
HANDOFF = BASE / "outputs" / "handoff"
REPORTS = BASE / "reports"
PLOTS   = BASE / "outputs" / "plots"
CM_DIR  = BASE / "outputs" / "confusion_matrices"
TABLES  = REPORTS / "tables"

# Copy five-seed results + updated report
for src, dst in [
    (METRICS / "robustness_five_seeds.json",   HANDOFF / "robustness_five_seeds.json"),
    (REPORTS / "final_lstm_report.md",          HANDOFF / "final_lstm_report.md"),
    (REPORTS / "error_analysis.md",             HANDOFF / "error_analysis.md"),
]:
    shutil.copy2(src, dst)

# Copy tables folder
tables_handoff = HANDOFF / "tables"
tables_handoff.mkdir(exist_ok=True)
for f in TABLES.glob("*.csv"):
    shutil.copy2(f, tables_handoff / f.name)

# Copy all seed metrics
for f in METRICS.glob("lstm_seed*_metrics.json"):
    shutil.copy2(f, HANDOFF / f.name)
for f in METRICS.glob("latency_stats_lstm_seed*.json"):
    shutil.copy2(f, HANDOFF / f.name)

# Update plots in handoff
plots_handoff = HANDOFF / "plots"
plots_handoff.mkdir(exist_ok=True)
for f in PLOTS.glob("*.png"):
    shutil.copy2(f, plots_handoff / f.name)
for f in CM_DIR.glob("*.png"):
    cm_h = HANDOFF / "confusion_matrices"
    cm_h.mkdir(exist_ok=True)
    shutil.copy2(f, cm_h / f.name)

# Copy all split files
splits_handoff = HANDOFF / "splits"
splits_handoff.mkdir(exist_ok=True)
for f in (BASE / "outputs" / "splits").glob("*.json"):
    shutil.copy2(f, splits_handoff / f.name)

print("Handoff package updated.")
files = list(HANDOFF.rglob("*"))
print(f"Total handoff files: {sum(1 for f in files if f.is_file())}")
