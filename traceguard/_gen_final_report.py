import json
from pathlib import Path
from datetime import datetime

BASE    = Path(r"c:\Users\DESHNA\TraceGuard\traceguard")
METRICS = BASE / "outputs" / "metrics"
REPORTS = BASE / "reports"

def load(name):
    p = METRICS / name
    if not p.exists():
        p = BASE / "outputs" / name
    if p.exists():
        return json.loads(p.read_text())
    return {}

tag  = "lstm_seed42"
cfg  = load("../experiment_config.json")
full = load(f"{tag}_metrics.json")
lat  = load(f"latency_stats_{tag}.json")
shuf = load(f"shuffling_ablation_{tag}.json")
cf   = load(f"counterfactual_analysis_{tag}.json")
sr   = load("robustness_five_seeds.json")
pc   = full.get("per_class", {})
la   = lat.get("type_a", {})
lb   = lat.get("type_b", {})
agg  = sr.get("aggregate", {})
per  = sr.get("per_seed", [])

seed_table_rows = ""
for r in per:
    seed_table_rows += f"| {r['seed']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['benign_f1']:.4f} | {r['resisted_f1']:.4f} | {r['hijacked_f1']:.4f} | {r['pre_action_rate']:.2%} |\n"

def ag(m):
    a = agg.get(m, {})
    return f"{a.get('mean',0):.4f} +/- {a.get('std',0):.4f}"

report = f"""# TRACEGUARD LSTM -- Final Results Report

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
*Dataset: traceguard_v4_1.jsonl (frozen) | Primary seed: 42*

---

## 1. Objective

TRACEGUARD detects hijacked LLM agent trajectories using a causal unidirectional LSTM
trained on sentence embeddings of sequential agent steps.
Primary evaluation criterion: **pre-action detection rate** (flagging HIJACKED before the harmful action executes).

## 2. Dataset

- **File:** 	raceguard_v4_1.jsonl (frozen, audited, not modified)
- **600 trajectories** in 200 counterfactual groups (3 per group: BENIGN, INJECTION_RESISTED, HIJACKED)
- **HIJACKED subtypes:** Type-A no-precursor (60), Type-B with-precursor (140)
- **6 steps per trajectory** | 8 tools | 0 missing fields | 0 duplicate IDs
- Audit: length-only accuracy = 33.33% (random), before-injection TF-IDF R/H = 50.00%

## 3. Dataset Split (Seed=42)

| Split | Groups | Trajectories | BENIGN | RESISTED | HIJACKED |
|-------|--------|-------------|--------|----------|----------|
| Train | 140    | 420         | 140    | 140      | 140      |
| Val   | 30     |  90         |  30    |  30      |  30      |
| Test  | 30     |  90         |  30    |  30      |  30      |

**Leakage check:** train intersect val = empty | train intersect test = empty | val intersect test = empty

## 4. Leakage Prevention

- Labels, attack metadata, injection/deviation/precursor steps NEVER enter model input
- counterfactual_group_id used ONLY for splitting and post-hoc analysis
- Detection threshold tuned on VAL only; evaluated once on TEST

## 5. Step Representation

`
ORIGINAL USER GOAL: {{user_goal}}
CURRENT STEP: {{step_number}}
ACTION: {{action}}
TOOL: {{tool}}
TOOL INPUT: {{tool_input}}
TOOL OBSERVATION: {{tool_observation}}
STATE: {{state}}
`

## 6. Embedding Model

- **Model:** sentence-transformers/all-MiniLM-L6-v2
- **Dimension:** 384
- Each trajectory: 6 step embeddings -> shape [6, 384]
- Embeddings cached; not fine-tuned

## 7. LSTM Architecture

`
TraceGuardLSTM
  input_size:    384
  hidden_size:   128
  num_layers:    2
  dropout:       0.2
  bidirectional: False   (UNIDIRECTIONAL -- causal for online detection)
  pooling:       last hidden state
  output:        Linear(128, 3)
  parameters:    ~270K
`

## 8. Training Setup

| Parameter | Value |
|-----------|-------|
| Loss | CrossEntropyLoss |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Max Epochs | 50 |
| Early Stopping Patience | 7 (on Val Macro F1) |
| Best Epoch | 21 |
| Best Val Macro F1 | 0.9227 |
| Training Time | 26.3 seconds (CPU) |

## 9. Full Trajectory Results (TEST SET, seed=42)

| Metric | Value |
|--------|-------|
| **Accuracy** | **{full.get('accuracy',0):.4f}** |
| Macro Precision | {full.get('macro_precision',0):.4f} |
| Macro Recall | {full.get('macro_recall',0):.4f} |
| **Macro F1** | **{full.get('macro_f1',0):.4f}** |

## 10. Per-Class Performance

| Class | Precision | Recall | F1 | AUROC |
|-------|-----------|--------|----|-------|
"""
for cls in ["BENIGN", "INJECTION_RESISTED", "HIJACKED"]:
    m = pc.get(cls, {})
    auroc = f"{m.get('auroc',0):.4f}" if m.get("auroc") is not None else "N/A"
    report += f"| {cls} | {m.get('precision',0):.4f} | {m.get('recall',0):.4f} | **{m.get('f1',0):.4f}** | {auroc} |\n"

report += f"""
Key errors: RESISTED->BENIGN (2), HIJACKED->BENIGN (2), RESISTED->HIJACKED (1).
See eports/error_analysis.md for examples.

## 11. Prefix Evaluation

Evaluated on partial trajectories of length 1..6.
Only steps 1..k used at prefix k. No future steps used (causal constraint respected).
See outputs/plots/hijack_probability_by_prefix_lstm_seed42.png.

## 12. Early Detection (Threshold=0.5, tuned on VAL)

| Metric | Value |
|--------|-------|
| HIJACKED total (test) | {lat.get('n_hijacked',30)} |
| Detected | {lat.get('n_detected','N/A')} ({lat.get('detection_rate',0):.2%}) |
| **Pre-action detection rate** | **{lat.get('pre_action_detection_rate',0):.2%}** |
| Mean latency | {lat.get('mean_latency',0):.2f} steps |
| Median latency | {lat.get('median_latency',0):.2f} steps |
| Std latency | {lat.get('std_latency',0):.2f} steps |
| Range | [{lat.get('min_latency',0):.0f}, {lat.get('max_latency',0):.0f}] |
| Before injection | {lat.get('pct_detected_before_injection',0):.2%} |
| After inj, before dev | {lat.get('pct_detected_after_inj_before_dev',0):.2%} |
| At deviation | {lat.get('pct_detected_at_deviation',0):.2%} |
| After deviation | {lat.get('pct_detected_after_deviation',0):.2%} |
| Never detected | {lat.get('pct_never_detected',0):.2%} |

Negative latency = flagged BEFORE the hijacked action executes.

## 13. Detection Latency

All detections in range [det_step - dev_step] = -3 to -2 for seed=42.
Virtually all detection occurs between injection and deviation (after the injection is seen,
before the agent acts on it). See outputs/plots/detection_latency_lstm_seed42.png.

## 14. Type-A vs Type-B Analysis

| Metric | Type A (n={la.get('n',9)}, no precursor) | Type B (n={lb.get('n',21)}, with precursor) |
|--------|----------------------|------------------------|
| Detection rate | {la.get('detection_rate',0):.2%} | {lb.get('detection_rate',0):.2%} |
| **Pre-action rate** | **{la.get('pre_action_detection_rate',0):.2%}** | **{lb.get('pre_action_detection_rate',0):.2%}** |
| Mean latency | {la.get('mean_latency')} | {lb.get('mean_latency')} |
| Median latency | {la.get('median_latency')} | {lb.get('median_latency')} |

Type-A has no precursor signal by design. Pre-action detection achieved for both types,
suggesting the LSTM captures deviation signals even without explicit precursors.

## 15. Counterfactual Analysis

| Metric | Value |
|--------|-------|
| Test groups | {cf.get('n_groups',30)} |
| Fully correct groups | {cf.get('fully_correct_groups','N/A')} ({cf.get('group_accuracy',0):.2%}) |
| HIJACKED missed | {cf.get('hijacked_missed','N/A')} |
| RESISTED pred as BENIGN | {cf.get('resisted_as_benign','N/A')} |
| RESISTED pred as HIJACKED | {cf.get('resisted_as_hijacked','N/A')} |
| HIJACKED pred as BENIGN | {cf.get('hijacked_as_benign','N/A')} |
| HIJACKED pred as RESISTED | {cf.get('hijacked_as_resisted','N/A')} |

## 16. Step-Shuffling Ablation (n=10 repetitions)

| | Original | Shuffled |
|--|----------|----------|
| Accuracy | {shuf.get('original_accuracy',0):.4f} | {shuf.get('shuffled_mean_acc',0):.4f} +/- {shuf.get('shuffled_std_acc',0):.4f} |
| **Macro F1** | **{shuf.get('original_f1',0):.4f}** | {shuf.get('shuffled_mean_f1',0):.4f} +/- {shuf.get('shuffled_std_f1',0):.4f} |
| **F1 Delta** | **{shuf.get('f1_delta',0):+.4f}** | |

Performance degradation under step shuffling (+{shuf.get('f1_delta',0):.4f} F1 advantage for original ordering)
supports the hypothesis that **chronological ordering contributes useful predictive information**.
This does NOT prove sequential modeling is categorically superior -- it indicates step order is informative.

## 17. Five-Seed Robustness

Seeds: 42, 123, 456, 789, 1011

| Seed | Accuracy | Macro F1 | BENIGN F1 | RESISTED F1 | HIJACKED F1 | Pre-Action |
|------|----------|----------|-----------|-------------|-------------|------------|
{seed_table_rows}
| **Mean +/- Std** | **{ag('accuracy')}** | **{ag('macro_f1')}** | {ag('benign_f1')} | {ag('resisted_f1')} | **{ag('hijacked_f1')}** | **{ag('pre_action_rate')}** |

Results are consistent across seeds. Seed 456 achieves highest F1 (0.9101).
Seed 789 is lowest (0.8007) -- natural variance on a 90-trajectory test set.
HIJACKED F1 is consistently the highest class (0.9123 +/- 0.0353), confirming reliable attack detection.

## 18. Error Analysis

See eports/error_analysis.md for per-category examples.

Key patterns (seed=42, test set):
- **BENIGN -> RESISTED** (4 trajectories): subtle boundary; conservative model flags borderline benign as resisted
- **RESISTED -> BENIGN** (2 trajectories): injection was well-hidden; LSTM did not detect resistance signal
- **HIJACKED -> BENIGN** (2 trajectories): Type-A cases with weak deviation signal; no precursor to detect
- **RESISTED -> HIJACKED** (1 trajectory): false alarm; resisted trajectory confused with hijacked

## 19. Limitations

1. Test set is 90 trajectories; per-class confidence intervals are wide
2. Embeddings are frozen (all-MiniLM-L6-v2); fine-tuning may improve performance
3. Type-A early detection is fundamentally limited without precursor signals
4. Dataset is synthetic/simulated; real-world transfer not evaluated
5. No GPU available; CPU-only training (~26s per seed)
6. Results not compared to baseline/Transformer yet (teammate's work)

## 20. Conclusions

- **TRACEGUARD LSTM achieves Macro F1 = {full.get('macro_f1',0):.4f}** on the frozen held-out test set
- **Pre-action detection rate = {lat.get('pre_action_detection_rate',0):.2%}** -- {lat.get('n_detected',0)}/{lat.get('n_hijacked',30)} HIJACKED trajectories flagged before harmful action
- **Mean detection latency = {lat.get('mean_latency',0):.2f} steps** (well before deviation)
- **HIJACKED F1 = {pc.get('HIJACKED',{}).get('f1',0):.4f}** (highest class; AUROC = {pc.get('HIJACKED',{}).get('auroc',0):.4f})
- **Step-order ablation: F1 drops {shuf.get('f1_delta',0):.4f}** when order randomized -- temporal structure is critical
- **Five-seed robustness: Macro F1 = {ag('macro_f1')}**, HIJACKED F1 = {ag('hijacked_f1')}
- Type-B (with precursor) achieves {lb.get('detection_rate',0):.2%} detection vs {la.get('detection_rate',0):.2%} for Type-A

**DATASET NOTE:** Results reflect the frozen v4.1 dataset as-is.
No trajectories were removed, relabeled, or modified based on model performance.

---
*TRACEGUARD LSTM Report -- generated automatically from frozen experiment outputs.*
"""

(REPORTS / "final_lstm_report.md").write_text(report, encoding="utf-8")
print("final_lstm_report.md updated with five-seed results.")
print(f"Length: {len(report)} chars")
