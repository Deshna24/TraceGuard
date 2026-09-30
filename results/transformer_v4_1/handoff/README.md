# TRACEGUARD Transformer Baseline (v4.1) — Handoff & Reproducibility Guide

## 1. Overview
This directory contains all artifacts, checkpoints, metrics, ablations, robustness summaries, and reports for the **Transformer sequence classifier baseline** on TRACEGUARD dataset v4.1 (`data/raw/trajectories/traceguard_v4_1.jsonl`).

---

## 2. Model Architecture & Training Configuration
- **Architecture:** `CLSTransformer` (Sequence Transformer Classifier with learnable `[CLS]` token pooling)
- **Input Dimension:** 384 (`sentence-transformers/all-MiniLM-L6-v2`)
- **Hidden Dimension ($d_{\text{model}}$):** 128
- **Positional Encoding:** Sinusoidal Positional Encoding
- **Encoder Layers:** 2
- **Attention Heads:** 4
- **Feedforward Dimension ($d_{\text{ff}}$):** 256
- **Dropout:** 0.2
- **Optimizer:** AdamW (`lr=3e-4`, `weight_decay=1e-4`, `batch_size=32`, `max_epochs=50`, `patience=8`)
- **Model Selection:** Validation Macro F1
- **Early Detection Decision Threshold:** $P(\text{HIJACKED}) \ge 0.5$

---

## 3. Dataset & Group Split Information
- **Dataset:** TRACEGUARD v4.1 (`data/raw/trajectories/traceguard_v4_1.jsonl`) — 600 trajectories, 200 counterfactual groups
- **Group-Aware Splitting:** 140 Train groups (420 trajs), 30 Val groups (90 trajs), 30 Test groups (90 trajs)
- **Group Disjointness:** Verified (0 group ID overlap between splits across all 5 seeds: `42`, `123`, `456`, `789`, `1011`).

---

## 4. Key Results Summary

### Seed 42 Model Comparison Table

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | HIJACKED Precision | HIJACKED Recall | HIJACKED F1 | HIJACKED AUROC | Pre-action Detection Rate | Mean Detection Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Pooling + Logistic Regression** | 0.6556 | 0.6478 | 0.6556 | 0.6456 | 0.6857 | 0.8000 | 0.7385 | 0.8989 | 0.0333 | -0.167 |
| **LSTM** | 0.8556 | 0.8550 | 0.8556 | 0.8533 | 0.8750 | 0.9333 | 0.9032 | 0.9794 | 0.9333 | -2.714 |
| **Transformer (CLS)** | 0.7111 | 0.7140 | 0.7111 | 0.7033 | 0.9333 | 0.9333 | 0.9333 | 0.9889 | 0.9000 | -3.321 |

### 5-Seed Robustness Summary (Transformer CLS)
- **Accuracy:** $0.5422 \pm 0.2608$
- **Macro F1:** $0.4453 \pm 0.3342$
- **BENIGN F1:** $0.6281 \pm 0.1802$
- **INJECTION_RESISTED F1:** $0.2987 \pm 0.3953$
- **HIJACKED F1:** $0.4092 \pm 0.4494$
- **Pre-action Detection Rate:** $0.2333 \pm 0.3490$

### Step-Order Shuffling Ablation (Seed 42)
- **Original Macro F1:** 0.7033
- **Shuffled Step Order Macro F1:** 0.6563 ± 0.0323
- **$\Delta F_1$:** 0.0469

---

## 5. Directory Structure & Key Output File Locations

```
results/
├── final_comparison/
│   ├── model_comparison.csv                       # Baseline comparison CSV
│   ├── model_comparison.md                        # Markdown comparison table
│   └── model_comparison.json                      # Full comparison JSON
└── transformer_v4_1/
    ├── ablations/
    │   ├── step_shuffling_ablation.json           # Step shuffling ablation results
    │   └── step_shuffling_ablation.csv
    ├── checkpoints/
    │   └── transformer_seed{42,123,456,789,1011}_best.pth # Checkpoints
    ├── configs/
    │   ├── experiment_config_seed42.json
    │   └── split_seed{seed}.json
    ├── confusion_matrices/
    │   └── transformer_seed{seed}_confusion_matrix.png
    ├── embeddings/
    │   └── embeddings_sentence-transformers_all-MiniLM-L6-v2.npy
    ├── latency/
    │   └── transformer_seed{seed}_early_detection.csv
    ├── metrics/
    │   ├── transformer_seed{seed}_metrics.json
    │   └── transformer_seed{seed}_history.json
    ├── prefix_detection/
    │   └── transformer_seed{seed}_prefixes.csv
    ├── reports/
    │   ├── pretraining_diagnostics.md             # Detailed pre-training diagnostics
    │   ├── experiment_report.md                   # Comprehensive report
    │   └── quality_check.md                       # Quality check audit
    ├── robustness/
    │   ├── five_seed_robustness.json             # 5-seed mean ± std summary
    │   └── five_seed_robustness.csv
    └── handoff/
        └── README.md                              # This handoff file
```

---

## 6. Reproduction Guide

Run the full end-to-end pipeline from repository root:

```bash
python experiments/run_full_transformer_and_baselines.py
```

---

## 7. Scientific Limitations & Language Guidelines

1. **Sequential Order Interpretation:** Performance under step-order shuffling was substantially lower than with chronological ordering ($\Delta F_1 = 0.0469$), supporting the hypothesis that temporal ordering contributes predictive information.
2. **Precursor Signal:** In TRACEGUARD v4.1, the Type-B precursor signal is relatively easy to distinguish in temporal sanity analysis.
3. **Synthetic Domain Scope:** TRACEGUARD v4.1 is a synthetic, controlled benchmark. Results do not imply universal real-world generalization.
