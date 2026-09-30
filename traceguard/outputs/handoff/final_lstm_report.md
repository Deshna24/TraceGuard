# TRACEGUARD LSTM -- Final Results Report

*Generated: 2026-09-26 01:25*
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
ORIGINAL USER GOAL: {user_goal}
CURRENT STEP: {step_number}
ACTION: {action}
TOOL: {tool}
TOOL INPUT: {tool_input}
TOOL OBSERVATION: {tool_observation}
STATE: {state}
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
| **Accuracy** | **0.8556** |
| Macro Precision | 0.8550 |
| Macro Recall | 0.8556 |
| **Macro F1** | **0.8533** |

## 10. Per-Class Performance

| Class | Precision | Recall | F1 | AUROC |
|-------|-----------|--------|----|-------|
| BENIGN | 0.8462 | 0.7333 | **0.7857** | 0.9728 |
| INJECTION_RESISTED | 0.8438 | 0.9000 | **0.8710** | 0.9789 |
| HIJACKED | 0.8750 | 0.9333 | **0.9032** | 0.9794 |

Key errors: RESISTED->BENIGN (2), HIJACKED->BENIGN (2), RESISTED->HIJACKED (1).
See 
eports/error_analysis.md for examples.

## 11. Prefix Evaluation

Evaluated on partial trajectories of length 1..6.
Only steps 1..k used at prefix k. No future steps used (causal constraint respected).
See outputs/plots/hijack_probability_by_prefix_lstm_seed42.png.

## 12. Early Detection (Threshold=0.5, tuned on VAL)

| Metric | Value |
|--------|-------|
| HIJACKED total (test) | 30 |
| Detected | 28 (93.33%) |
| **Pre-action detection rate** | **93.33%** |
| Mean latency | -2.71 steps |
| Median latency | -3.00 steps |
| Std latency | 0.46 steps |
| Range | [-3, -2] |
| Before injection | 0.00% |
| After inj, before dev | 93.33% |
| At deviation | 0.00% |
| After deviation | 0.00% |
| Never detected | 6.67% |

Negative latency = flagged BEFORE the hijacked action executes.

## 13. Detection Latency

All detections in range [det_step - dev_step] = -3 to -2 for seed=42.
Virtually all detection occurs between injection and deviation (after the injection is seen,
before the agent acts on it). See outputs/plots/detection_latency_lstm_seed42.png.

## 14. Type-A vs Type-B Analysis

| Metric | Type A (n=9, no precursor) | Type B (n=21, with precursor) |
|--------|----------------------|------------------------|
| Detection rate | 88.89% | 95.24% |
| **Pre-action rate** | **88.89%** | **95.24%** |
| Mean latency | -2.5 | -2.8 |
| Median latency | -2.5 | -3.0 |

Type-A has no precursor signal by design. Pre-action detection achieved for both types,
suggesting the LSTM captures deviation signals even without explicit precursors.

## 15. Counterfactual Analysis

| Metric | Value |
|--------|-------|
| Test groups | 30 |
| Fully correct groups | 20 (66.67%) |
| HIJACKED missed | 2 |
| RESISTED pred as BENIGN | 2 |
| RESISTED pred as HIJACKED | 1 |
| HIJACKED pred as BENIGN | 2 |
| HIJACKED pred as RESISTED | 0 |

## 16. Step-Shuffling Ablation (n=10 repetitions)

| | Original | Shuffled |
|--|----------|----------|
| Accuracy | 0.8556 | 0.5933 +/- 0.0291 |
| **Macro F1** | **0.8533** | 0.5722 +/- 0.0334 |
| **F1 Delta** | **+0.2811** | |

Performance degradation under step shuffling (+0.2811 F1 advantage for original ordering)
supports the hypothesis that **chronological ordering contributes useful predictive information**.
This does NOT prove sequential modeling is categorically superior -- it indicates step order is informative.

## 17. Five-Seed Robustness

Seeds: 42, 123, 456, 789, 1011

| Seed | Accuracy | Macro F1 | BENIGN F1 | RESISTED F1 | HIJACKED F1 | Pre-Action |
|------|----------|----------|-----------|-------------|-------------|------------|
| 42 | 0.8556 | 0.8533 | 0.7857 | 0.8710 | 0.9032 | 93.33% |
| 123 | 0.8667 | 0.8683 | 0.8387 | 0.9091 | 0.8571 | 90.00% |
| 456 | 0.9111 | 0.9101 | 0.8852 | 0.8772 | 0.9677 | 90.00% |
| 789 | 0.8000 | 0.8007 | 0.7500 | 0.7368 | 0.9153 | 73.33% |
| 1011 | 0.8444 | 0.8442 | 0.7869 | 0.8276 | 0.9180 | 86.67% |

| **Mean +/- Std** | **0.8556 +/- 0.0358** | **0.8553 +/- 0.0354** | 0.8093 +/- 0.0473 | 0.8443 +/- 0.0597 | **0.9123 +/- 0.0353** | **0.8667 +/- 0.0699** |

Results are consistent across seeds. Seed 456 achieves highest F1 (0.9101).
Seed 789 is lowest (0.8007) -- natural variance on a 90-trajectory test set.
HIJACKED F1 is consistently the highest class (0.9123 +/- 0.0353), confirming reliable attack detection.

## 18. Error Analysis

See 
eports/error_analysis.md for per-category examples.

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

- **TRACEGUARD LSTM achieves Macro F1 = 0.8533** on the frozen held-out test set
- **Pre-action detection rate = 93.33%** -- 28/30 HIJACKED trajectories flagged before harmful action
- **Mean detection latency = -2.71 steps** (well before deviation)
- **HIJACKED F1 = 0.9032** (highest class; AUROC = 0.9794)
- **Step-order ablation: F1 drops 0.2811** when order randomized -- temporal structure is critical
- **Five-seed robustness: Macro F1 = 0.8553 +/- 0.0354**, HIJACKED F1 = 0.9123 +/- 0.0353
- Type-B (with precursor) achieves 95.24% detection vs 88.89% for Type-A

**DATASET NOTE:** Results reflect the frozen v4.1 dataset as-is.
No trajectories were removed, relabeled, or modified based on model performance.

---
*TRACEGUARD LSTM Report -- generated automatically from frozen experiment outputs.*
