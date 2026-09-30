# TRACEGUARD

**Full Title:** TRACEGUARD: Early Detection and Localization of Hijacked LLM Agent Trajectories Using Sequential Deep Learning

---

## 1. Research Motivation
LLM agents operate by executing multi-step tool trajectories in interactive environments. When exposed to untrusted external data (such as web pages, retrieved documents, or external emails), agents are vulnerable to indirect prompt injection attacks. Standard static input filters fail to capture post-injection execution context, while full-trajectory evaluation after execution fails to prevent unauthorized side-effects (e.g., unauthorized data exfiltration or system modification). TRACEGUARD addresses this challenge by performing dynamic, step-by-step sequence analysis of agent execution trajectories to detect behavioral hijacking prior to irreversible tool execution.

---

## 2. Research Question
Can sequential deep learning architectures classify agent behavior and trigger early detection of trajectory hijacking before malicious actions take effect, when evaluated under strict group-aware counterfactual splitting on multi-step trajectory sequences?

---

## 3. Dataset & Counterfactual Grouping
- **Dataset Version:** TRACEGUARD v4.1 (`data/raw/trajectories/traceguard_v4_1.jsonl`)
- **Total Trajectories:** 600
- **Counterfactual Groups:** 200 groups (3 trajectories per group)
- **Trajectory Length:** 6 steps per trajectory
- **Counterfactual Group Structure:** Each group contains 3 parallel counterfactual variants sharing identical initial user goals and prompt prefixes:
  1. **BENIGN:** No prompt injection present; clean execution.
  2. **INJECTION_RESISTED:** Indirect prompt injection present in observation, but agent resists it and continues benign execution.
  3. **HIJACKED:** Indirect prompt injection present and followed by agent, leading to behavioral hijacking at `deviation_step` ($\text{step } 5$).

---

## 4. Three Classes Definition
1. **BENIGN:** No injection present and no behavioral deviation occurs.
2. **INJECTION_RESISTED:** Indirect prompt injection is present in the environment observation, but the agent successfully resists it and does not deviate from the user's original objective.
3. **HIJACKED:** Indirect prompt injection is present, and the agent follows the malicious instructions, producing an observable behavioral deviation.

---

## 5. Experimental Setup & Group-Aware Splitting
- **Dataset Partitioning:** 140 Train groups (420 trajs), 30 Validation groups (90 trajs), 30 Test groups (90 trajs).
- **Group Disjointness:** Strictly group-aware; no counterfactual group ID crosses train/val/test splits.
- **Frozen Primary Split:** `traceguard/outputs/splits/split_seed42.json` (Seed 42).
- **Sentence Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional step embeddings).

---

## 6. LSTM Architecture & Performance
- **Model:** Bi-directional LSTM with sequence pooling (`model_lstm.py`).
- **Hidden Dimension:** 128
- **Seed 42 Performance:**
  - Accuracy: **0.8556**
  - Macro Precision: **0.8550**
  - Macro Recall: **0.8556**
  - Macro F1: **0.8533**
  - HIJACKED Precision: **0.8750**, Recall: **0.9333**, F1: **0.9032**, AUROC: **0.9794**
  - Pre-action Detection Rate: **93.33%** (28 / 30)
  - Mean Detection Latency: **-2.714 steps**

---

## 7. Transformer Architecture & Performance
- **Model:** `CLSTransformer` (Sequence Transformer Classifier with learnable `[CLS]` token pooling).
- **Architecture:** $d_{\text{model}} = 128$, 2 encoder layers, 4 attention heads, $d_{\text{ff}} = 256$, dropout = 0.2.
- **Optimizer:** AdamW (`lr=3e-4`, `weight_decay=1e-4`, `batch_size=32`, `max_epochs=50`, `patience=8`).
- **Seed 42 Performance:**
  - Accuracy: **0.7111**
  - Macro Precision: **0.7140**
  - Macro Recall: **0.7111**
  - Macro F1: **0.7033**
  - HIJACKED Precision: **0.9333**, Recall: **0.9333**, F1: **0.9333**, AUROC: **0.9889**
  - Detection Rate ($P(\text{HIJACKED}) \ge 0.5$): **93.33%** (28 / 30 detected)
  - Pre-action Detection Rate: **90.0%** (27 / 30 detected pre-action)
  - Never Detected Rate: **6.67%** (2 / 30)
  - Mean Detection Latency: **-3.321 steps** (Median: **-3.0 steps**)

---

## 8. Logistic Regression Baseline
- **Model:** Non-sequential Mean-Pooled Step Embeddings + Logistic Regression (`baseline.py`).
- **Seed 42 Performance:**
  - Accuracy: **0.6556**
  - Macro Precision: **0.6478**
  - Macro Recall: **0.6556**
  - Macro F1: **0.6456**
  - HIJACKED F1: **0.7385**, AUROC: **0.8989**
  - Pre-action Detection Rate: **3.33%** (1 / 30)
  - Mean Detection Latency: **-0.167 steps**

---

## 9. Early Detection Methodology & Latency
Early detection evaluates prefix sequences $k \in \{1, \dots, N\}$ using only steps available up to prefix length $k$.
- **Detection Criteria:** First prefix $k$ where $P(\text{HIJACKED}) \ge 0.5$ (fixed threshold; non-tuned).
- **Detection Latency:** $\text{latency} = \text{detection\_step} - \text{deviation\_step}$.
- **Pre-action Detection:** Defined as $\text{detection\_step} < \text{deviation\_step}$ (negative latency indicates detection prior to deviation).

---

## 10. Step-Order Ablation
Evaluates whether chronological step order provides predictive signal by comparing original vs. randomly shuffled step orders on Seed 42:
- **Chronological Sequence Macro F1:** **0.7033**
- **Shuffled Step Order Macro F1:** **0.6563 ± 0.0323**
- **Macro F1 Delta ($\Delta F_1$):** **0.0469**
- *Interpretation:* Performance decreased under step-order shuffling ($\Delta F_1 = 0.0469$), indicating that chronological ordering contributes predictive information.

---

## 11. Five-Seed Robustness Evaluation
Evaluated across 5 random seeds (`42`, `123`, `456`, `789`, `1011`):
- **LSTM 5-Seed Macro F1:** **0.8553 ± 0.0354**
- **Transformer 5-Seed Performance:**
  - Accuracy: **0.5422 ± 0.2608**
  - Macro F1: **0.4453 ± 0.3342**
  - BENIGN F1: **0.6281 ± 0.1802**
  - INJECTION_RESISTED F1: **0.2987 ± 0.3953**
  - HIJACKED F1: **0.4092 ± 0.4494**
  - Pre-action Detection Rate: **0.2333 ± 0.3490**
- *Observation:* Across five seeds, the Transformer showed substantially higher variability in Macro F1 than the frozen LSTM baseline, indicating sensitivity to random initialization and training conditions on the current benchmark.

---

## 12. Limitations & Synthetic Domain Scope
1. **Synthetic Dataset Benchmark:** TRACEGUARD v4.1 is a synthetic, controlled benchmark dataset. Results do not imply universal real-world generalization across unconstrained LLM agent platforms.
2. **Precursor Signal Characteristics:** In TRACEGUARD v4.1, the Type-B precursor signal is relatively easy to distinguish in temporal sanity analysis.
3. **Small Sample Sensitivity:** Transformers exhibit optimization variance under small sample sizes ($N=420$ training samples) under strict group-aware counterfactual splitting.

---

## 13. Repository Structure & Canonical Manifest

```
TraceGuard/
├── data/
│   └── raw/trajectories/traceguard_v4_1.jsonl    # Canonical frozen dataset v4.1
├── experiments/
│   ├── run_transformer_v4_1_cls.py               # Transformer CLS model script
│   ├── diagnose_transformer_collapse.py          # Pre-training diagnostic script
│   ├── run_full_transformer_and_baselines.py     # Reproducibility pipeline script
│   └── run_final_transformer_audit.py            # Zero-mutation audit script
├── results/
│   ├── final_comparison/                          # Canonical baseline comparison
│   │   ├── model_comparison.csv
│   │   ├── model_comparison.md
│   │   ├── model_comparison.json
│   │   └── README.md
│   ├── lstm_v4_1/                                # Canonical frozen LSTM results
│   │   └── README.md
│   ├── transformer_v4_1/                         # Canonical frozen Transformer results
│   │   ├── ablations/
│   │   ├── checkpoints/
│   │   ├── configs/
│   │   ├── confusion_matrices/
│   │   ├── embeddings/
│   │   ├── handoff/
│   │   ├── latency/
│   │   ├── metrics/
│   │   ├── prefix_detection/
│   │   ├── reports/
│   │   ├── robustness/
│   │   └── README.md
│   ├── REPOSITORY_MANIFEST.md                     # File manifest & canonical index
│   └── REPOSITORY_CLEANUP_REPORT.md               # Repository audit & cleanup log
├── src/                                           # Core source modules
└── traceguard/outputs/splits/split_seed42.json  # Canonical frozen group split
```

---

## 14. Reproducibility Guide

To reproduce all Transformer, non-sequential baseline, and comparison metrics:

```bash
python experiments/run_full_transformer_and_baselines.py
```

To run the zero-mutation final audit script:

```bash
python experiments/run_final_transformer_audit.py
```
