# TRACEGUARD Model Comparison — Canonical Results (v4.1)

**Status:** **FROZEN & VERIFIED**  

---

## 1. Overview
This directory contains the canonical baseline and model comparison artifacts for TRACEGUARD v4.1 on the frozen Seed 42 group-aware counterfactual split (`traceguard/outputs/splits/split_seed42.json`).

---

## 2. Canonical Model Comparison Table (Seed 42)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | HIJACKED Precision | HIJACKED Recall | HIJACKED F1 | HIJACKED AUROC | Pre-action Detection Rate | Mean Detection Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Pooling + Logistic Regression** | 0.6556 | 0.6478 | 0.6556 | 0.6456 | 0.6857 | 0.8000 | 0.7385 | 0.8989 | 3.33% | -0.167 steps |
| **LSTM (Frozen)** | **0.8556** | **0.8550** | **0.8556** | **0.8533** | 0.8750 | **0.9333** | 0.9032 | 0.9794 | **93.33%** | -2.714 steps |
| **Transformer (CLS)** | 0.7111 | 0.7140 | 0.7111 | 0.7033 | **0.9333** | **0.9333** | **0.9333** | **0.9889** | 90.0% | **-3.321 steps** |

---

## 3. Comparison Files
- [`model_comparison.csv`](file:///c:/Users/DESHNA/TraceGuard/results/final_comparison/model_comparison.csv) — CSV format
- [`model_comparison.md`](file:///c:/Users/DESHNA/TraceGuard/results/final_comparison/model_comparison.md) — Markdown format
- [`model_comparison.json`](file:///c:/Users/DESHNA/TraceGuard/results/final_comparison/model_comparison.json) — Full structured JSON format

---

## 4. Multi-Seed Variance Note
Five-seed robustness metrics are evaluated and reported separately in each model directory because the Transformer baseline exhibited higher optimization variance across counterfactual group splits ($0.4453 \pm 0.3342$ Macro F1) than the recurrent LSTM baseline ($0.8553 \pm 0.0354$ Macro F1).
