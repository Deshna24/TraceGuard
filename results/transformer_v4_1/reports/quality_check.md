# TRACEGUARD Transformer Baseline (v4.1) — Quality Control Report

**Date:** 2026-09-28  
**Scope:** Full Transformer Baseline Pipeline & Multi-Seed Audit  

---

## Quality Control Checklist

| Check Item | Status | Details / Evidence |
| :--- | :--- | :--- |
| **No Group Overlap** | **PASS** | Checked across all 5 seeds (`42`, `123`, `456`, `789`, `1011`). Train, validation, and test group sets are strictly disjoint. |
| **No Dataset Modification** | **PASS** | `data/raw/trajectories/traceguard_v4_1.jsonl` was untouched (600 trajectories, 200 counterfactual groups). |
| **No Test-Label Leakage** | **PASS** | Model selection for all 5 seeds performed exclusively using validation set Macro F1. Test sets evaluated after freezing checkpoints. |
| **Threshold Selection** | **PASS** | Fixed decision threshold $P(\text{HIJACKED}) \ge 0.5$ used; no threshold tuning performed on test set predictions. |
| **Prefix Step Leakage** | **PASS** | Prefix inference at step $k$ strictly used slicing `embs[:k]`; future steps $k+1..N$ were masked and invisible. |
| **LSTM Artifact Integrity** | **PASS** | `results/lstm_v4_1/` and frozen LSTM outputs in `traceguard/outputs/` were completely preserved and untouched. |
| **Five-Seed Results Complete** | **PASS** | All 5 seeds (`42`, `123`, `456`, `789`, `1011`) evaluated and summarized in `robustness/five_seed_robustness.json` and `csv`. |
| **Step-Order Ablation Complete**| **PASS** | Shuffling ablation performed across 10 permutations on seed 42 (`step_shuffling_ablation.json` and `csv`). |
| **Baseline Included** | **PASS** | Non-sequential Mean Pooling + Logistic Regression evaluated on identical group split (`results/final_comparison/`). |
| **Comparison Table Complete** | **PASS** | `results/final_comparison/model_comparison.csv`, `.md`, `.json` generated, comparing Mean Pooling + LogReg, LSTM, and Transformer. |
| **Handoff Documentation** | **PASS** | `results/transformer_v4_1/handoff/README.md` updated with complete setup, architecture, and reproduction guide. |

---

## Metric Verification Table (Seed 42 Summary)

- **Mean Pooling + Logistic Regression:** Accuracy = 0.6556 | Macro F1 = 0.6456 | Pre-Action Rate = 0.0333
- **LSTM (Frozen Reference):** Accuracy = 0.8556 | Macro F1 = 0.8533 | Pre-Action Rate = 0.9333
- **Transformer (CLS Pooling):** Accuracy = 0.7111 | Macro F1 = 0.7033 | Pre-Action Rate = 0.9000
