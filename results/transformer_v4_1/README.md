# TRACEGUARD Transformer Baseline (v4.1) — Frozen Documentation

**STATUS:** **FROZEN AFTER FINAL AUDIT**  
**AUDIT STATUS:** **PASS** (0 Failures, 0 Warnings)  

---

## 1. Overview
This directory contains the frozen, audited artifacts, checkpoints, metrics, ablations, and reports for the Transformer sequence classifier baseline on TRACEGUARD dataset v4.1 (`data/raw/trajectories/traceguard_v4_1.jsonl`).

> **Methodological Note on Previous Experiment:** An earlier initial Transformer run under mean-pooling collapsed `INJECTION_RESISTED` predictions into `BENIGN` ($F_1 = 0.000$). That earlier run was an obsolete, failed exploratory experiment and is **NOT** used in any final comparison table or paper evaluation. All canonical Transformer results are based on the audited `CLSTransformer` architecture documented herein.

---

## 2. Model & Training Configuration
- **Dataset:** TRACEGUARD v4.1 (`data/raw/trajectories/traceguard_v4_1.jsonl`)
- **Group Split File:** `traceguard/outputs/splits/split_seed42.json` (FROZEN seed 42 split)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional step embeddings)
- **Architecture:** `CLSTransformer` (Learnable `[CLS]` token prepended to step sequence)
  - `input_dim`: 384
  - `model_dim`: 128
  - `num_layers`: 2
  - `num_heads`: 4
  - `ff_dim`: 256
  - `dropout`: 0.2
  - `pooling`: `[CLS]` token output at index 0 $\to$ Linear($128 \to 3$)
- **Optimizer:** AdamW (`lr=3e-4`, `weight_decay=1e-4`, `batch_size=32`, `max_epochs=50`, `patience=8`)

---

## 3. Verified Seed 42 Results Summary

| Metric | Verified Transformer (CLS) Value |
| :--- | :--- |
| **Accuracy** | 0.7111 |
| **Macro Precision** | 0.7140 |
| **Macro Recall** | 0.7111 |
| **Macro F1** | 0.7033 |
| **BENIGN F1** | 0.6667 |
| **INJECTION_RESISTED F1** | 0.5098 |
| **HIJACKED Precision** | 0.9333 |
| **HIJACKED Recall** | 0.9333 |
| **HIJACKED F1** | 0.9333 |
| **HIJACKED AUROC** | 0.9889 |

### Early Detection Stats ($P(\text{HIJACKED}) \ge 0.5$)
- **Detection Rate:** **93.33%** (28 / 30 HIJACKED trajectories detected)
- **Pre-action Detection Rate:** **90.0%** (27 / 30 detected prior to deviation)
- **Never Detected Rate:** **6.67%** (2 / 30)
- **Mean Detection Latency:** **-3.321 steps**
- **Median Detection Latency:** **-3.0 steps**

---

## 4. Five-Seed Robustness Summary
Across 5 group-aware splits (`42`, `123`, `456`, `789`, `1011`):
- **Accuracy:** $0.5422 \pm 0.2608$
- **Macro F1:** $0.4453 \pm 0.3342$
- **BENIGN F1:** $0.6281 \pm 0.1802$
- **INJECTION_RESISTED F1:** $0.2987 \pm 0.3953$
- **HIJACKED F1:** $0.4092 \pm 0.4494$
- **Pre-action Detection Rate:** $0.2333 \pm 0.3490$
- *Analysis:* Across five seeds, the Transformer showed substantially higher variability in Macro F1 than the frozen LSTM baseline, indicating sensitivity to random initialization and training conditions on the current benchmark.

---

## 5. Step-Order Shuffling Ablation
- **Chronological Sequence Macro F1:** **0.7033**
- **Shuffled Step Order Macro F1:** **0.6563 ± 0.0323**
- **Macro F1 Delta ($\Delta F_1$):** **0.0469**
- *Interpretation:* Performance decreased under step-order shuffling ($\Delta F_1 = 0.0469$), indicating that chronological ordering contributes predictive information.

---

## 6. Audit Reports Directory
- [`reports/final_audit.md`](file:///c:/Users/DESHNA/TraceGuard/results/transformer_v4_1/reports/final_audit.md) — Comprehensive zero-mutation final audit report.
- [`reports/pretraining_diagnostics.md`](file:///c:/Users/DESHNA/TraceGuard/results/transformer_v4_1/reports/pretraining_diagnostics.md) — LDA & embedding separability diagnostics.
- [`reports/experiment_report.md`](file:///c:/Users/DESHNA/TraceGuard/results/transformer_v4_1/reports/experiment_report.md) — Detailed experiment report.
- [`reports/quality_check.md`](file:///c:/Users/DESHNA/TraceGuard/results/transformer_v4_1/reports/quality_check.md) — Quality control verification checklist.
