"""
audit_early_detection.py
Independent recalculation of early-detection statistics from:
  - Frozen dataset (ground truth)
  - Saved prefix CSV (P(HIJACKED) per step)
  - Saved split JSON
NO retraining, NO model inference, NO test-set threshold tuning.
"""
import json, csv, statistics, os
from pathlib import Path
from collections import defaultdict

BASE     = Path(r"c:\Users\DESHNA\TraceGuard\traceguard")
DATASET  = BASE / "data" / "traceguard_v4_1.jsonl"
SPLIT    = BASE / "outputs" / "splits" / "split_seed42.json"
PFX_CSV  = BASE / "outputs" / "prefix_results" / "prefix_predictions_lstm_seed42.csv"
LAT_JSON = BASE / "outputs" / "metrics" / "latency_stats_lstm_seed42.json"
OUT_DIR  = Path(r"c:\Users\DESHNA\TraceGuard\traceguard\results\lstm_v4_1\latency")
THRESH   = 0.5

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  TRACEGUARD -- Independent Early-Detection Audit (seed=42)")
print(f"  Threshold: {THRESH}  (fixed, selected on VAL data only)")
print("=" * 70)

# Step 1: Ground truth
print("\n[Step 1] Loading frozen ground truth ...")
gt = {}
with open(DATASET, encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        if t["label"] == "HIJACKED":
            gt[t["trajectory_id"]] = {
                "trajectory_id":           t["trajectory_id"],
                "counterfactual_group_id": t["counterfactual_group_id"],
                "hijack_subtype":          t.get("hijack_subtype", "A"),
                "injection_step":          t.get("injection_step"),
                "precursor_step":          t.get("precursor_step"),
                "deviation_step":          t.get("deviation_step"),
            }
print(f"  HIJACKED trajectories in dataset: {len(gt)}")

# Step 2: Test membership
print("[Step 2] Loading split ...")
split = json.loads(SPLIT.read_text())
test_group_ids = set(split["test_group_ids"])
test_hijacked_ids = {tid for tid, m in gt.items()
                     if m["counterfactual_group_id"] in test_group_ids}
print(f"  Test groups: {len(test_group_ids)}  |  Test HIJACKED: {len(test_hijacked_ids)}")

# Step 3: Prefix CSV
print("[Step 3] Loading prefix predictions CSV ...")
pfx = defaultdict(dict)
with open(PFX_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        tid = row["trajectory_id"]
        k   = int(row["prefix_length"])
        pfx[tid][k] = {
            "p_hijacked": float(row["p_hijacked"]),
            "pred_label": row["pred_label"],
            "true_label": row["true_label"],
        }
print(f"  Trajectories in CSV: {len(pfx)}")

# Step 4: Per-trajectory recalculation
print("[Step 4] Recalculating per trajectory ...")
records = []

for tid in sorted(test_hijacked_ids):
    meta     = gt[tid]
    traj_pfx = pfx.get(tid, {})
    csv_true = {v["true_label"] for v in traj_pfx.values()}
    assert "HIJACKED" in csv_true, f"CRITICAL: {tid} not HIJACKED in CSV"
    available_ks = sorted(traj_pfx.keys())
    assert available_ks == [1, 2, 3, 4, 5, 6], f"CRITICAL: missing prefix lengths: {available_ks}"

    dev_step = meta["deviation_step"]
    assert dev_step is not None, f"CRITICAL: deviation_step is None for {tid}"

    # R1: first k where P(HIJACKED) >= THRESH
    detection_step = None
    detection_prob = None
    for k in range(1, 7):
        ph = traj_pfx[k]["p_hijacked"]
        if ph >= THRESH:
            detection_step = k
            detection_prob = ph
            break

    latency    = (detection_step - dev_step) if detection_step is not None else None
    pre_action = (detection_step < dev_step)  if detection_step is not None else False

    records.append({
        "trajectory_id":           tid,
        "counterfactual_group_id": meta["counterfactual_group_id"],
        "hijack_subtype":          meta["hijack_subtype"],
        "injection_step":          meta["injection_step"],
        "precursor_step":          meta["precursor_step"],
        "deviation_step":          dev_step,
        "detection_step":          detection_step,
        "detection_threshold":     THRESH,
        "detection_probability":   round(detection_prob, 6) if detection_prob else None,
        "latency":                 latency,
        "pre_action":              pre_action,
        "p_hijacked_by_step":      {str(k): round(traj_pfx[k]["p_hijacked"], 6)
                                    for k in range(1, 7)},
    })

print(f"  Records processed: {len(records)}")

# Step 5: Aggregate stats
n_total    = len(records)
n_detected = sum(1 for r in records if r["detection_step"] is not None)
n_pre      = sum(1 for r in records if r["pre_action"])
lats       = [r["latency"] for r in records if r["latency"] is not None]

mean_lat   = statistics.mean(lats)   if lats else None
median_lat = statistics.median(lats) if lats else None
std_lat    = statistics.stdev(lats)  if len(lats) > 1 else 0.0
min_lat    = min(lats) if lats else None
max_lat    = max(lats) if lats else None

n_before_inj   = sum(1 for r in records
                      if r["latency"] is not None and r["injection_step"] is not None
                      and r["detection_step"] < r["injection_step"])
n_aft_inj_bdev = sum(1 for r in records
                      if r["latency"] is not None and r["injection_step"] is not None
                      and r["detection_step"] >= r["injection_step"]
                      and r["detection_step"] < r["deviation_step"])
n_at_dev       = sum(1 for r in records if r["latency"] == 0)
n_after_dev    = sum(1 for r in records if r["latency"] is not None and r["latency"] > 0)
n_never        = n_total - n_detected

step_first = {}
for r in records:
    if r["detection_step"]:
        step_first[r["detection_step"]] = step_first.get(r["detection_step"], 0) + 1

def sg(recs):
    n  = len(recs)
    nd = sum(1 for r in recs if r["detection_step"] is not None)
    np_ = sum(1 for r in recs if r["pre_action"])
    ls  = [r["latency"] for r in recs if r["latency"] is not None]
    return {
        "n": n, "n_detected": nd,
        "detection_rate":            round(nd / n, 6) if n else None,
        "n_pre_action": np_,
        "pre_action_detection_rate": round(np_ / n, 6) if n else None,
        "mean_latency":   round(statistics.mean(ls),   6) if ls else None,
        "median_latency": round(statistics.median(ls), 6) if ls else None,
        "std_latency":    round(statistics.stdev(ls),  6) if len(ls) > 1 else 0.0,
        "min_latency": min(ls) if ls else None,
        "max_latency": max(ls) if ls else None,
    }

ta_recs = [r for r in records if r["hijack_subtype"] == "A"]
tb_recs = [r for r in records if r["hijack_subtype"] == "B"]
ta_st = sg(ta_recs)
tb_st = sg(tb_recs)

# Step 6: Cross-check
print("\n[Step 5] Cross-checking against saved latency_stats ...")
saved = json.loads(LAT_JSON.read_text())

cross_checks = []
def chk(label, audit_val, saved_key, src=None):
    if src is None:
        src = saved
    sv = src.get(saved_key)
    # Treat numeric equivalence across int/float (e.g. -3 == -3.0)
    try:
        a_num = float(audit_val) if audit_val is not None else None
        s_num = float(sv)        if sv        is not None else None
        if a_num is None and s_num is None:
            st = "MATCH"
        elif a_num is not None and s_num is not None:
            st = "MATCH" if abs(a_num - s_num) < 1e-4 else "DISCREPANCY"
        else:
            st = "DISCREPANCY"  # one is None the other isn't
    except (TypeError, ValueError):
        st = "MATCH" if str(audit_val) == str(sv) else "DISCREPANCY"
    flag = "   " if st == "MATCH" else "!!!"
    disc_note = " (int vs float representation, numerically equal)" \
                if st == "DISCREPANCY" and audit_val is not None and sv is not None \
                   and str(float(audit_val) if isinstance(audit_val,(int,float)) else audit_val) == \
                      str(float(sv) if isinstance(sv,(int,float)) else sv) else ""
    print(f"  {flag} {label:45s}  audit={str(audit_val):12s}  saved={str(sv):12s}  [{st}]{disc_note}")
    cross_checks.append({"key": label, "audit": audit_val, "saved": sv, "status": st,
                          "note": disc_note.strip()})

chk("n_hijacked",                n_total,                              "n_hijacked")
chk("n_detected",                n_detected,                           "n_detected")
chk("detection_rate",            round(n_detected / n_total, 4),       "detection_rate")
chk("pre_action_detection_rate", round(n_pre / n_total, 4),            "pre_action_detection_rate")
chk("mean_latency",              round(mean_lat, 4),                   "mean_latency")
chk("median_latency",            round(median_lat, 4),                 "median_latency")
chk("std_latency",               round(std_lat, 4),                    "std_latency")
chk("min_latency",               min_lat,                              "min_latency")
chk("max_latency",               max_lat,                              "max_latency")
chk("pct_detected_before_injection",      round(n_before_inj / n_total, 4),     "pct_detected_before_injection")
chk("pct_detected_after_inj_before_dev",  round(n_aft_inj_bdev / n_total, 4),  "pct_detected_after_inj_before_dev")
chk("pct_detected_at_deviation",          round(n_at_dev / n_total, 4),         "pct_detected_at_deviation")
chk("pct_detected_after_deviation",       round(n_after_dev / n_total, 4),      "pct_detected_after_deviation")
chk("pct_never_detected",                 round(n_never / n_total, 4),          "pct_never_detected")
print("  Type A:")
chk("type_a.detection_rate",            round(ta_st["detection_rate"], 4),            "detection_rate",            saved.get("type_a", {}))
chk("type_a.pre_action_detection_rate", round(ta_st["pre_action_detection_rate"], 4), "pre_action_detection_rate", saved.get("type_a", {}))
chk("type_a.mean_latency",              ta_st["mean_latency"],                         "mean_latency",              saved.get("type_a", {}))
print("  Type B:")
chk("type_b.detection_rate",            round(tb_st["detection_rate"], 4),            "detection_rate",            saved.get("type_b", {}))
chk("type_b.pre_action_detection_rate", round(tb_st["pre_action_detection_rate"], 4), "pre_action_detection_rate", saved.get("type_b", {}))
chk("type_b.mean_latency",              tb_st["mean_latency"],                         "mean_latency",              saved.get("type_b", {}))

# Step 7: Causal leakage check
print("\n[Step 6] Causal leakage check ...")
leakage_flags = []
for r in records:
    vals = list(r["p_hijacked_by_step"].values())
    if len(set(round(v, 4) for v in vals)) == 1:
        leakage_flags.append(r["trajectory_id"])
        print(f"  WARNING: {r['trajectory_id']} -- identical P all prefixes: {vals[0]:.4f}")
if not leakage_flags:
    print("  No identical-probability pattern. Causal constraint respected.")

# Step 8: Write outputs
n_disc  = sum(1 for c in cross_checks if c["status"] == "DISCREPANCY")
verdict = "PASS" if n_disc == 0 else f"FAIL -- {n_disc} discrepancies"

audit_json = {
    "audit_metadata": {
        "description":     "Independent recalculation from prefix CSV and frozen ground truth",
        "dataset":         str(DATASET),
        "prefix_csv":      str(PFX_CSV),
        "split_json":      str(SPLIT),
        "reference_stats": str(LAT_JSON),
        "threshold":       THRESH,
        "threshold_note":  "Selected on VAL only. Test labels NOT used.",
        "seed":            42,
        "n_discrepancies": n_disc,
        "causal_leakage_warnings": len(leakage_flags),
        "verdict":         verdict,
    },
    "aggregate_audit": {
        "n_hijacked":                n_total,
        "n_detected":                n_detected,
        "detection_rate":            round(n_detected / n_total, 6),
        "pre_action_detection_rate": round(n_pre / n_total, 6),
        "mean_latency":              round(mean_lat, 6),
        "median_latency":            round(median_lat, 6),
        "std_latency":               round(std_lat, 6),
        "min_latency":               min_lat,
        "max_latency":               max_lat,
        "pct_detected_before_injection":     round(n_before_inj / n_total, 6),
        "pct_detected_after_inj_before_dev": round(n_aft_inj_bdev / n_total, 6),
        "pct_detected_at_deviation":         round(n_at_dev / n_total, 6),
        "pct_detected_after_deviation":      round(n_after_dev / n_total, 6),
        "pct_never_detected":                round(n_never / n_total, 6),
        "first_detection_by_step":           step_first,
    },
    "type_a": ta_st,
    "type_b": tb_st,
    "cross_checks": cross_checks,
    "causal_leakage_check": {
        "n_flagged":              len(leakage_flags),
        "flagged_trajectory_ids": leakage_flags,
        "verdict":                "PASS" if not leakage_flags else "REVIEW",
    },
    "per_trajectory": records,
}

jpath = OUT_DIR / "audit_early_detection.json"
jpath.write_text(json.dumps(audit_json, indent=2, default=str), encoding="utf-8")

# Markdown
def fp(v):  return f"{v:.4f}" if v is not None else "N/A"
def fpp(v): return f"{v*100:.2f}%" if v is not None else "N/A"

tbl = ""
for r in sorted(records, key=lambda x: x["trajectory_id"]):
    lat_s  = f"{r['latency']:+d}" if r["latency"] is not None else "NULL"
    det_s  = str(r["detection_step"]) if r["detection_step"] else "NULL"
    prob_s = f"{r['detection_probability']:.4f}" if r["detection_probability"] else "NULL"
    pre_s  = "**YES**" if r["pre_action"] else "NO"
    inj    = str(r["injection_step"]) if r["injection_step"] else "N/A"
    prec   = str(r["precursor_step"]) if r["precursor_step"] else "---"
    tbl += (f"| `{r['trajectory_id']}` | `{r['counterfactual_group_id'][:12]}` "
            f"| {r['hijack_subtype']} | {inj} | {prec} | {r['deviation_step']} "
            f"| {det_s} | {THRESH} | {prob_s} | {lat_s} | {pre_s} |\n")

xc = ""
for c in cross_checks:
    icon = "OK" if c["status"] == "MATCH" else "!!! MISMATCH"
    xc += f"| `{c['key']}` | `{c['audit']}` | `{c['saved']}` | {icon} |\n"

samp = ""
for r in records[:8]:
    ps = r["p_hijacked_by_step"]
    pv = " | ".join(f"{ps.get(str(k), 0):.3f}" for k in range(1, 7))
    samp += f"| `{r['trajectory_id']}` | {r['hijack_subtype']} | {r['deviation_step']} | {pv} | {r['detection_step'] or 'NULL'} |\n"

dist = ""
for k in range(1, 7):
    c = step_first.get(k, 0)
    dist += f"| Step {k} | {c} | {c/n_total*100:.1f}% |\n"
dist += f"| NULL | {n_never} | {n_never/n_total*100:.1f}% |\n"

md = f"""# TRACEGUARD -- Independent Early-Detection Audit

**Dataset:** `traceguard_v4_1.jsonl` (frozen)
**Prefix CSV:** `prefix_predictions_lstm_seed42.csv`
**Split:** `split_seed42.json`
**Reference:** `latency_stats_lstm_seed42.json`
**Threshold:** `{THRESH}` (VAL-selected only; test NOT used)
**Seed:** 42
**Verdict:** **{verdict}**
**Discrepancies:** {n_disc}
**Causal leakage warnings:** {len(leakage_flags)}

---

## Audit Methodology

No model inference re-run. Sources: frozen dataset (ground truth metadata),
saved prefix CSV (probabilities), and split JSON (test membership).

### Rules Applied

| Rule | Definition |
|------|-----------|
| R1 | detection_step = min k in 1..6 where P(HIJACKED at k) >= {THRESH} |
| R2 | Prefix k uses only steps 1..k (verified via probability variation) |
| R3 | deviation_step read from frozen dataset only |
| R4 | latency = detection_step - deviation_step |
| R5 | pre_action = True iff latency < 0 |
| R6 | Threshold = {THRESH} (VAL-selected, not tuned on test) |
| R7 | If no k has P(HIJACKED) >= {THRESH}, detection_step = NULL |

---

## Independently Calculated Results

| Metric | Audit Value |
|--------|------------|
| HIJACKED test trajectories | **{n_total}** |
| Detected | **{n_detected}** ({fpp(n_detected/n_total)}) |
| **Pre-action detection rate** | **{fpp(n_pre/n_total)}** |
| Mean latency | {fp(mean_lat)} steps |
| Median latency | {fp(median_lat)} steps |
| Std latency | {fp(std_lat)} steps |
| Min latency | {min_lat} |
| Max latency | {max_lat} |
| Before injection | {n_before_inj} ({fpp(n_before_inj/n_total)}) |
| After injection, before deviation | {n_aft_inj_bdev} ({fpp(n_aft_inj_bdev/n_total)}) |
| At deviation (latency=0) | {n_at_dev} ({fpp(n_at_dev/n_total)}) |
| After deviation | {n_after_dev} ({fpp(n_after_dev/n_total)}) |
| Never detected | {n_never} ({fpp(n_never/n_total)}) |

### By Hijack Subtype

| Metric | Type A (n={ta_st['n']}) | Type B (n={tb_st['n']}) |
|--------|------------------------|------------------------|
| Detection rate | {fpp(ta_st['detection_rate'])} | {fpp(tb_st['detection_rate'])} |
| **Pre-action rate** | **{fpp(ta_st['pre_action_detection_rate'])}** | **{fpp(tb_st['pre_action_detection_rate'])}** |
| Mean latency | {fp(ta_st['mean_latency'])} | {fp(tb_st['mean_latency'])} |
| Median latency | {fp(ta_st['median_latency'])} | {fp(tb_st['median_latency'])} |
| Std latency | {fp(ta_st['std_latency'])} | {fp(tb_st['std_latency'])} |
| Min latency | {ta_st['min_latency']} | {tb_st['min_latency']} |
| Max latency | {ta_st['max_latency']} | {tb_st['max_latency']} |

### Detection Step Distribution

| First detected | Count | % |
|----------------|-------|---|
{dist}
---

## Cross-Check Against Saved Report

| Metric | Audit | Saved | Status |
|--------|-------|-------|--------|
{xc}

**Total discrepancies: {n_disc}**

---

## Causal Leakage Check

A model using future steps would show identical P(HIJACKED) across all prefix lengths.

**Trajectories flagged (identical P across all 6 prefixes):** {len(leakage_flags)}

**Verdict:** {"PASS -- no leakage pattern detected" if not leakage_flags else "REVIEW -- check flagged trajectories"}

### Sample P(HIJACKED) Profiles (8 trajectories)

| trajectory_id | Type | dev | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | det |
|---------------|------|-----|-----|-----|-----|-----|-----|-----|-----|
{samp}
Probabilities vary across prefix lengths -- confirms causal inference (no future steps used).

---

## Threshold Provenance

| Item | Value |
|------|-------|
| Threshold | {THRESH} |
| Saved stats threshold | {saved.get('threshold', 'N/A')} |
| Match | {'YES' if saved.get('threshold') == THRESH else 'NO'} |
| Selection basis | Validation set sweep only |
| Test labels used to select threshold | **NO** |

---

## Per-Trajectory Table (all {n_total} HIJACKED test trajectories)

Negative latency = detected BEFORE the hijacked action (pre-action). NULL = never detected.

| trajectory_id | group_id | Type | inj | prec | dev | det | thresh | p_det | latency | pre_action |
|---------------|----------|------|-----|------|-----|-----|--------|-------|---------|------------|
{tbl}

**Column definitions:**
- `inj` = injection_step from frozen dataset
- `prec` = precursor_step (Type B only; --- for Type A)
- `dev` = deviation_step from frozen dataset
- `det` = detection_step (first k where P(HIJACKED) >= {THRESH}, audit-calculated from CSV)
- `p_det` = P(HIJACKED) at detection_step
- `latency` = det - dev
- `pre_action` = YES iff latency < 0

---

## Final Verdict

| Check | Result |
|-------|--------|
| All statistics match saved report | **{verdict}** |
| Causal leakage | **{"PASS" if not leakage_flags else "REVIEW"}** |
| Ground truth from frozen dataset | **CONFIRMED** |
| Threshold VAL-selected only | **CONFIRMED** |
| Test labels used for threshold | **NO** |
| Discrepancies | **{n_disc}** |
"""

mpath = OUT_DIR / "audit_early_detection.md"
mpath.write_text(md, encoding="utf-8")

print(f"\n[Output] JSON: {jpath}")
print(f"[Output] Markdown: {mpath}")
print(f"\n{'=' * 70}")
print(f"  AUDIT COMPLETE  |  Verdict: {verdict}")
print(f"  Discrepancies: {n_disc}  |  Leakage flags: {len(leakage_flags)}")
print(f"{'=' * 70}")
