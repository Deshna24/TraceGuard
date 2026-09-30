# TRACEGUARD Baseline & Model Comparison Table (v4.1 — Seed 42)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | HIJACKED Precision | HIJACKED Recall | HIJACKED F1 | HIJACKED AUROC | Pre-action Detection Rate | Mean Detection Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Pooling + Logistic Regression** | 0.6556 | 0.6478 | 0.6556 | 0.6456 | 0.6857 | 0.8000 | 0.7385 | 0.8989 | 0.0333 | -0.167 |
| **LSTM** | 0.8556 | 0.8550 | 0.8556 | 0.8533 | 0.8750 | 0.9333 | 0.9032 | 0.9794 | 0.9333 | -2.714 |
| **Transformer (CLS)** | 0.7111 | 0.7140 | 0.7111 | 0.7033 | 0.9333 | 0.9333 | 0.9333 | 0.9889 | 0.9000 | -3.321 |
