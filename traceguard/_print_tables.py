import json, csv
from pathlib import Path

BASE    = Path(r"c:\Users\DESHNA\TraceGuard\traceguard")
REPORTS = BASE / "reports"
TABLES  = REPORTS / "tables"

# Print all tables
for t in sorted(TABLES.glob("*.csv")):
    print(f"\n=== {t.name} ===")
    with open(t, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            print("  " + " | ".join(str(x) for x in row))
