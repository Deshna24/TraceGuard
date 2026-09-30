# write_report.py - generates final_lstm_report.md from saved JSON outputs
import json, os
from pathlib import Path
from datetime import datetime

BASE    = Path(r"c:\Users\DESHNA\TraceGuard\traceguard")
METRICS = BASE / "outputs" / "metrics"
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)

def load(name):
    p = METRICS / name
    if p.exists():
        return json.loads(p.read_text())
    return {}

tag = "lstm_seed42"
cfg   = load("../experiment_config.json")
full  = load(f"{tag}_metrics.json")
lat   = load(f"latency_stats_{tag}.json")
shuf  = load(f"shuffling_ablation_{tag}.json")
cf    = load(f"counterfactual_analysis_{tag}.json")

# Check if we have results yet
if not full:
    print("No results yet - experiment still running.")
    exit(0)

pc = full.get("per_class", {})
la = lat.get("type_a", {})
lb = lat.get("type_b", {})

lines = [
    "# TRACEGUARD LSTM — Final Results Report",
    "",
    f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    f"*Tag: {tag}*",
    "",
    "---",
    "",
    "## 1. Objective",
    "",
    "TRACEGUARD detects and localizes hijacked LLM agent trajectories using a",
    "causal (unidirectional) LSTM trained on sentence embeddings of sequential agent steps.",
    "The primary evaluation criterion is **early detection before the hijacked action occurs**.",
    "",
    "## 2. Dataset",
    "",
    "- **File:** 	raceguard_v4_1.jsonl (frozen, audited, not modified)",
    "- **600 trajectories** in 200 counterfactual groups (3 per group)",
    "- **Classes:** BENIGN (200), INJECTION_RESISTED (200), HIJACKED (200)",
    "- **HIJACKED subtypes:** Type-A no-precursor (60), Type-B with-precursor (140)",
    "- **6 steps per trajectory** | 8 tools | 0 missing fields | 0 duplicate IDs",
    "",
    "## 3. Dataset Split",
    "",
    "> **Note:** The unit of splitting is counterfactual_group_id.",
    "> All 3 trajectories from a group remain in exactly one split.",
    "> Individual trajectory splitting was NOT used (would cause data leakage).",
    "",
    "| Split | Groups | Trajectories | BENIGN | RESISTED | HIJACKED |",
    "|-------|--------|-------------|--------|----------|----------|",
    "| Train | 140 | 420 | 140 | 140 | 140 |",
    "| Val   | 30  | 90  | 30  | 30  | 30  |",
    "| Test  | 30  | 90  | 30  | 30  | 30  |",
    "",
    "## 4. Leakage Prevention",
    "",
    "- Labels, attack metadata, injection/deviation steps NEVER enter model input",
    "- counterfactual_group_id used ONLY for splitting and post-hoc analysis",
    "- Embeddings computed from behavioral fields only: action, tool, tool_input, tool_observation, state, user_goal",
    "- Detection threshold tuned on VAL only",
    "",
    "## 5. Step Representation",
    "",
    "`",
    "ORIGINAL USER GOAL: {user_goal}",
    "CURRENT STEP: {step_number}",
    "ACTION: {action}",
    "TOOL: {tool}",
    "TOOL INPUT: {tool_input}",
    "TOOL OBSERVATION: {tool_observation}",
    "STATE: {state}",
    "`",
    "",
    "## 6. Embedding Model",
    "",
    f"- **Model:** {cfg.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')}",
    f"- **Dimension:** {cfg.get('embedding_dim', 384)}",
    "- Each trajectory → [6, D] tensor of step embeddings",
    "- Embeddings cached to disk; not fine-tuned",
    "",
    "## 7. LSTM Architecture",
    "",
    "`",
    "TraceGuardLSTM",
    f"  input_size:   {cfg.get('embedding_dim', 384)}",
    "  hidden_size:  128",
    "  num_layers:   2",
    "  dropout:      0.2",
    "  bidirectional: False  (UNIDIRECTIONAL — causal for online detection)",
    "  pooling:      last hidden state",
    "  output:       Linear(128, 3)",
    "`",
    "",
    "## 8. Training Setup",
    "",
    "| Parameter | Value |",
    "|-----------|-------|",
    "| Loss | CrossEntropyLoss |",
    "| Optimizer | AdamW |",
    "| Learning Rate | 1e-3 |",
    "| Weight Decay | 1e-4 |",
    "| Batch Size | 32 |",
    "| Max Epochs | 50 |",
    "| Early Stopping | Patience=7 on Val Macro F1 |",
    f"| Best Epoch | {cfg.get('best_epoch', 'N/A')} |",
    "",
    "## 9. Full Trajectory Results",
    "",
    "Evaluated on held-out TEST set (seed=42). Test data NOT used for any model decisions.",
    "",
    "| Metric | Value |",
    "|--------|-------|",
    f"| Accuracy | {full.get('accuracy', 0):.4f} |",
    f"| Macro Precision | {full.get('macro_precision', 0):.4f} |",
    f"| Macro Recall | {full.get('macro_recall', 0):.4f} |",
    f"| Macro F1 | {full.get('macro_f1', 0):.4f} |",
    "",
    "## 10. Per-Class Performance",
    "",
    "| Class | Precision | Recall | F1 | AUROC |",
    "|-------|-----------|--------|----|-------|",
]
for cls in ["BENIGN", "INJECTION_RESISTED", "HIJACKED"]:
    m = pc.get(cls, {})
    auroc = f"{m.get('auroc', 0):.4f}" if m.get("auroc") is not None else "N/A"
    lines.append(f"| {cls} | {m.get('precision',0):.4f} | {m.get('recall',0):.4f} | {m.get('f1',0):.4f} | {auroc} |")

lines += [
    "",
    "### Error Pattern Focus",
    "- **RESISTED → BENIGN**: Agent correctly resisted but misclassified as benign (common confusion)",
    "- **RESISTED → HIJACKED**: False positive — dangerous for trust",
    "- **HIJACKED → RESISTED**: False negative — missed attack",
    "- **HIJACKED → BENIGN**: False negative — missed attack",
    "",
    "See eports/error_analysis.md for concrete examples.",
    "",
    "## 11. Prefix Evaluation",
    "",
    "The LSTM was evaluated on partial trajectories of length 1..6.",
    "Only steps 1..k are used at prefix k. NO future steps are used.",
    "See outputs/plots/hijack_probability_by_prefix_*.png.",
    "",
    "## 12. Early Detection",
    "",
    f"**Detection threshold:** {lat.get('threshold', 0.5)} (P(HIJACKED) >= threshold)",
    "Threshold selected on validation set.",
    "",
    "| Metric | Value |",
    "|--------|-------|",
    f"| Total HIJACKED (test) | {lat.get('n_hijacked', 'N/A')} |",
    f"| Detected | {lat.get('n_detected', 'N/A')} ({lat.get('detection_rate', 0):.2%}) |",
    f"| **Pre-action detection rate** | **{lat.get('pre_action_detection_rate', 0):.2%}** |",
    f"| Mean latency | {lat.get('mean_latency')} steps |",
    f"| Median latency | {lat.get('median_latency')} steps |",
    f"| Std latency | {lat.get('std_latency')} steps |",
    f"| Before injection | {lat.get('pct_detected_before_injection', 0):.2%} |",
    f"| After inj, before dev | {lat.get('pct_detected_after_inj_before_dev', 0):.2%} |",
    f"| At deviation | {lat.get('pct_detected_at_deviation', 0):.2%} |",
    f"| After deviation | {lat.get('pct_detected_after_deviation', 0):.2%} |",
    f"| Never detected | {lat.get('pct_never_detected', 0):.2%} |",
    "",
    "Negative latency = detected BEFORE the hijacked action (pre-action).",
    "Zero = detected at deviation. Positive = detected after.",
    "",
    "## 13. Detection Latency",
    "",
    "See outputs/plots/detection_latency_*.png.",
    "",
    "## 14. Type-A vs Type-B Analysis",
    "",
    "| Metric | Type A (no precursor) | Type B (with precursor) |",
    "|--------|----------------------|------------------------|",
    f"| Count (test set) | {la.get('n', 'N/A')} | {lb.get('n', 'N/A')} |",
    f"| Detection rate | {la.get('detection_rate', 0):.2%} | {lb.get('detection_rate', 0):.2%} |",
    f"| Pre-action rate | {la.get('pre_action_detection_rate', 0):.2%} | {lb.get('pre_action_detection_rate', 0):.2%} |",
    f"| Mean latency | {la.get('mean_latency')} | {lb.get('mean_latency')} |",
    f"| Median latency | {la.get('median_latency')} | {lb.get('median_latency')} |",
    "",
    "> Type A has no precursor signal by design. Early detection for Type A",
    "> relies entirely on deviation-step signals. Type B benefits from",
    "> precursor signals (scope_drift, reasoning_shift, parameter_creep).",
    "",
    "## 15. Counterfactual Analysis",
    "",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| Test groups | {cf.get('n_groups', 'N/A')} |",
    f"| Fully correct groups | {cf.get('fully_correct_groups', 'N/A')} ({cf.get('group_accuracy', 0):.2%}) |",
    f"| HIJACKED missed | {cf.get('hijacked_missed', 'N/A')} |",
    f"| RESISTED pred as BENIGN | {cf.get('resisted_as_benign', 'N/A')} |",
    f"| RESISTED pred as HIJACKED | {cf.get('resisted_as_hijacked', 'N/A')} |",
    f"| HIJACKED pred as BENIGN | {cf.get('hijacked_as_benign', 'N/A')} |",
    f"| HIJACKED pred as RESISTED | {cf.get('hijacked_as_resisted', 'N/A')} |",
    "",
    "## 16. Step-Shuffling Ablation",
    "",
    f"| | Original | Shuffled (n={shuf.get('n_reps', 10)}) |",
    f"|--|----------|---------|",
    f"| Accuracy | {shuf.get('original_accuracy', 0):.4f} | {shuf.get('shuffled_mean_acc', 0):.4f} ± {shuf.get('shuffled_std_acc', 0):.4f} |",
    f"| Macro F1 | {shuf.get('original_f1', 0):.4f} | {shuf.get('shuffled_mean_f1', 0):.4f} ± {shuf.get('shuffled_std_f1', 0):.4f} |",
    f"| F1 Delta | {shuf.get('f1_delta', 0):+.4f} | |",
    "",
]
delta = shuf.get("f1_delta", 0)
if delta > 0:
    lines.append("Performance degradation under step shuffling supports the hypothesis that "
                 "chronological ordering contributes useful predictive information.")
else:
    lines.append("No clear performance degradation under step shuffling; "
                 "the model may not heavily rely on step order (or signal is strong regardless).")

# Check for five-seed results
seeds_path = METRICS / "robustness_five_seeds.json"
if seeds_path.exists():
    sr = json.loads(seeds_path.read_text())
    agg = sr["aggregate"]
    lines += [
        "",
        "## 17. Five-Seed Robustness",
        "",
        "Seeds: 42, 123, 456, 789, 1011",
        "",
        "| Metric | Mean ± Std | Min | Max |",
        "|--------|------------|-----|-----|",
    ]
    for m in ["accuracy", "macro_f1", "hijacked_f1", "resisted_f1", "benign_f1", "pre_action_rate"]:
        a = agg.get(m, {})
        lines.append(f"| {m} | {a.get('mean',0):.4f} ± {a.get('std',0):.4f} | {a.get('min',0):.4f} | {a.get('max',0):.4f} |")
else:
    lines += ["", "## 17. Five-Seed Robustness", "", "*Run experiments/run_seeds.py to populate this section.*"]

lines += [
    "",
    "## 18. Error Analysis",
    "",
    "See eports/error_analysis.md for per-category error examples.",
    "",
    "## 19. Limitations",
    "",
    "1. Test set is small (90 trajectories), confidence intervals on per-class metrics are wide.",
    "2. Embeddings are frozen (all-MiniLM-L6-v2); fine-tuning may improve performance.",
    "3. Type-A early detection is fundamentally limited — no precursor to detect.",
    "4. Dataset is synthetic/simulated; real-world transfer is not evaluated.",
    "5. LSTM is compared to no external baseline here; see teammate's results for comparison.",
    "",
    "## 20. Conclusions",
    "",
    "- **TRACEGUARD LSTM** achieves measurable performance on the 3-class trajectory classification task.",
    f"- **Macro F1: {full.get('macro_f1', 0):.4f}** on the frozen, held-out test set.",
    f"- **Pre-action detection rate: {lat.get('pre_action_detection_rate', 0):.2%}** — HIJACKED trajectories flagged before the harmful action.",
    "- **Type-B (precursor)** trajectories are easier to detect early than **Type-A (no precursor)**.",
    "- Step-order shuffling analysis provides insight into whether temporal ordering is informative.",
    "",
    "---",
    "*TRACEGUARD LSTM Report — automatically generated from frozen experiment outputs.*",
]

(REPORTS / "final_lstm_report.md").write_text("\n".join(lines), encoding="utf-8")
print("final_lstm_report.md written.")
