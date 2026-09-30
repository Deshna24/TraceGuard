import json, csv
from pathlib import Path
from datetime import datetime

BASE    = Path(r"c:\Users\DESHNA\TraceGuard\traceguard")
METRICS = BASE / "outputs" / "metrics"
REPORTS = BASE / "reports"
HANDOFF = BASE / "outputs" / "handoff"

def load(p):
    p = Path(p)
    if p.exists():
        return json.loads(p.read_text())
    return {}

tag  = "lstm_seed42"
full = load(METRICS / f"{tag}_metrics.json")
lat  = load(METRICS / f"latency_stats_{tag}.json")
shuf = load(METRICS / f"shuffling_ablation_{tag}.json")
cf   = load(METRICS / f"counterfactual_analysis_{tag}.json")
sr   = load(METRICS / "robustness_five_seeds.json")
pc   = full.get("per_class", {})
la   = lat.get("type_a", {})
lb   = lat.get("type_b", {})
agg  = sr.get("aggregate", {})
per  = sr.get("per_seed", [])

TABLES = REPORTS / "tables"
TABLES.mkdir(exist_ok=True)

def write_csv(fname, rows, header=None):
    path = TABLES / fname
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(rows)
    print(f"  Wrote {fname}")

# --- Table 1: Dataset Split ---
write_csv("table1_dataset_split.csv",
    [["train",140,420,140,140,140],
     ["val",30,90,30,30,30],
     ["test",30,90,30,30,30]],
    ["split","groups","trajectories","BENIGN","INJECTION_RESISTED","HIJACKED"])

# --- Table 2: Full Trajectory Classification ---
write_csv("table2_full_trajectory.csv",
    [["accuracy",      round(full.get("accuracy",0),4)],
     ["macro_precision",round(full.get("macro_precision",0),4)],
     ["macro_recall",  round(full.get("macro_recall",0),4)],
     ["macro_f1",      round(full.get("macro_f1",0),4)]],
    ["metric","value"])

# --- Table 3: Per-Class Performance ---
rows3 = []
for cls in ["BENIGN","INJECTION_RESISTED","HIJACKED"]:
    m = pc.get(cls, {})
    rows3.append([cls, round(m.get("precision",0),4), round(m.get("recall",0),4),
                  round(m.get("f1",0),4), m.get("support",0),
                  round(m.get("auroc",0),4) if m.get("auroc") else "N/A",
                  round(m.get("pr_auc",0),4) if m.get("pr_auc") else "N/A"])
write_csv("table3_per_class.csv", rows3,
    ["class","precision","recall","f1","support","auroc","pr_auc"])

# --- Table 4: Prefix-Stage Performance (from prefix CSV) ---
import pandas as pd
pfx_path = BASE / "outputs" / "prefix_results" / f"prefix_predictions_{tag}.csv"
if pfx_path.exists():
    df = pd.read_csv(pfx_path)
    from sklearn.metrics import f1_score, accuracy_score
    rows4 = []
    for k in range(1, 7):
        sub = df[df["prefix_length"] == k]
        y_true = sub["true_label"].map({"BENIGN":0,"INJECTION_RESISTED":1,"HIJACKED":2})
        y_pred = sub["pred_label"].map({"BENIGN":0,"INJECTION_RESISTED":1,"HIJACKED":2})
        acc = round(accuracy_score(y_true, y_pred), 4)
        f1  = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)
        mean_hp = round(df[df["prefix_length"]==k]["p_hijacked"].mean(), 4)
        rows4.append([k, acc, f1, mean_hp])
    write_csv("table4_prefix_stage.csv", rows4,
        ["prefix_length","accuracy","macro_f1","mean_p_hijacked"])

# --- Table 5: Early Detection Metrics ---
write_csv("table5_early_detection.csv",
    [["threshold",                 lat.get("threshold")],
     ["n_hijacked",                lat.get("n_hijacked")],
     ["n_detected",                lat.get("n_detected")],
     ["detection_rate",            round(lat.get("detection_rate",0),4)],
     ["pre_action_detection_rate", round(lat.get("pre_action_detection_rate",0),4)],
     ["mean_latency",              round(lat.get("mean_latency",0),4)],
     ["median_latency",            round(lat.get("median_latency",0),4)],
     ["std_latency",               round(lat.get("std_latency",0),4)],
     ["min_latency",               lat.get("min_latency")],
     ["max_latency",               lat.get("max_latency")],
     ["pct_before_injection",      round(lat.get("pct_detected_before_injection",0),4)],
     ["pct_after_inj_before_dev",  round(lat.get("pct_detected_after_inj_before_dev",0),4)],
     ["pct_at_deviation",          round(lat.get("pct_detected_at_deviation",0),4)],
     ["pct_after_deviation",       round(lat.get("pct_detected_after_deviation",0),4)],
     ["pct_never_detected",        round(lat.get("pct_never_detected",0),4)]],
    ["metric","value"])

# --- Table 6: Type A vs Type B ---
write_csv("table6_type_ab.csv",
    [["n",                         la.get("n"), lb.get("n")],
     ["detection_rate",            round(la.get("detection_rate",0),4), round(lb.get("detection_rate",0),4)],
     ["pre_action_detection_rate", round(la.get("pre_action_detection_rate",0),4), round(lb.get("pre_action_detection_rate",0),4)],
     ["mean_latency",              la.get("mean_latency"), lb.get("mean_latency")],
     ["median_latency",            la.get("median_latency"), lb.get("median_latency")],
     ["std_latency",               round(la.get("std_latency",0),4) if la.get("std_latency") else None, round(lb.get("std_latency",0),4) if lb.get("std_latency") else None],
     ["min_latency",               la.get("min_latency"), lb.get("min_latency")],
     ["max_latency",               la.get("max_latency"), lb.get("max_latency")]],
    ["metric","type_a","type_b"])

# --- Table 7: Step-Order Ablation ---
write_csv("table7_shuffling_ablation.csv",
    [["original_accuracy",  round(shuf.get("original_accuracy",0),4)],
     ["original_macro_f1",  round(shuf.get("original_f1",0),4)],
     ["shuffled_mean_acc",  round(shuf.get("shuffled_mean_acc",0),4)],
     ["shuffled_std_acc",   round(shuf.get("shuffled_std_acc",0),4)],
     ["shuffled_mean_f1",   round(shuf.get("shuffled_mean_f1",0),4)],
     ["shuffled_std_f1",    round(shuf.get("shuffled_std_f1",0),4)],
     ["f1_delta",           round(shuf.get("f1_delta",0),4)],
     ["n_reps",             shuf.get("n_reps")]],
    ["metric","value"])

# --- Table 8: Five-Seed Robustness ---
rows8_per = []
for r in per:
    rows8_per.append([r["seed"], round(r["accuracy"],4), round(r["macro_f1"],4),
                      round(r["benign_f1"],4), round(r["resisted_f1"],4),
                      round(r["hijacked_f1"],4), round(r["pre_action_rate"],4),
                      r.get("best_epoch")])
metrics_list = ["accuracy","macro_f1","benign_f1","resisted_f1","hijacked_f1","pre_action_rate"]
mean_row = ["mean"] + [round(agg.get(m,{}).get("mean",0),4) for m in metrics_list] + [""]
std_row  = ["std"]  + [round(agg.get(m,{}).get("std",0),4)  for m in metrics_list] + [""]
rows8_per += [mean_row, std_row]
write_csv("table8_five_seed_robustness.csv", rows8_per,
    ["seed","accuracy","macro_f1","benign_f1","resisted_f1","hijacked_f1","pre_action_rate","best_epoch"])

print("\nAll 8 tables written to reports/tables/")
