import os
from pathlib import Path

BASE = Path(r"c:\Users\DESHNA\TraceGuard\traceguard")

def check_dir(d, label):
    p = BASE / d
    files = list(p.glob("*")) if p.exists() else []
    print(f"\n[{label}] ({len(files)} files)")
    for f in sorted(files):
        size = f.stat().st_size if f.is_file() else "-"
        print(f"  {'OK' if f.is_file() else 'DIR':3s}  {size:>8}  {f.name}")

check_dir("outputs/models",             "MODELS")
check_dir("outputs/metrics",            "METRICS")
check_dir("outputs/confusion_matrices", "CONFUSION MATRICES")
check_dir("outputs/plots",              "PLOTS")
check_dir("outputs/prefix_results",     "PREFIX RESULTS")
check_dir("outputs/splits",             "SPLITS")
check_dir("outputs/handoff",            "HANDOFF")
check_dir("reports",                    "REPORTS")
