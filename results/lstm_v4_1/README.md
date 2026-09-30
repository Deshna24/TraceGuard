# TRACEGUARD LSTM Baseline (v4.1) — Frozen Documentation

**Status:** **FROZEN & READ-ONLY**  
**Audit Status:** VERIFIED  

---

## 1. Overview
This directory contains the frozen experimental results, metrics, and documentation for the Bi-directional LSTM baseline on TRACEGUARD dataset v4.1 (`data/raw/trajectories/traceguard_v4_1.jsonl`). All artifacts under `results/lstm_v4_1/` are frozen and read-only.

---

## 2. Configuration & Architecture
- **Dataset:** `data/raw/trajectories/traceguard_v4_1.jsonl` (600 trajectories, 200 counterfactual groups)
- **Split File:** `traceguard/outputs/splits/split_seed42.json` (FROZEN seed 42 split)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional step embeddings)
- **Model Architecture:** Bi-directional LSTM
  - `input_dim`: 384
  - `hidden_dim`: 128
  - `num_layers`: 2
  - `dropout`: 0.2
  - `pooling`: Masked mean sequence pooling
- **Training Config:** AdamW, `lr=1e-3`, `weight_decay=1e-4`, `batch_size=32`, `max_epochs=50`, `patience=7`

---

## 3. Verified Seed 42 Performance

| Metric | Frozen Value |
| :--- | :--- |
| **Accuracy** | 0.8556 |
| **Macro Precision** | 0.8550 |
| **Macro Recall** | 0.8556 |
| **Macro F1** | 0.8533 |
| **BENIGN F1** | 0.7857 |
| **INJECTION_RESISTED F1** | 0.8710 |
| **HIJACKED Precision** | 0.8750 |
| **HIJACKED Recall** | 0.9333 |
| **HIJACKED F1** | 0.9032 |
| **HIJACKED AUROC** | 0.9794 |
| **Pre-action Detection Rate** | 93.33% (28 / 30) |
| **Mean Detection Latency** | -2.714 steps |

---

## 4. Five-Seed Robustness Summary
- **Five-Seed Macro F1 (Mean ± Std):** **0.8553 ± 0.0354**
- **Tested Seeds:** `42`, `123`, `456`, `789`, `1011`
- *Observation:* The LSTM architecture maintained low variance across all 5 counterfactual group splits.

---

## 5. Early Detection & Ablation Summary
- **Early Detection Criteria:** First prefix $k$ where $P(\text{HIJACKED}) \ge 0.5$.
- **Pre-action Detection Rate:** 93.33% (28 / 30 HIJACKED test trajectories detected prior to deviation).
- **Mean Latency:** -2.714 steps.
- **Step Shuffling Ablation:** Chronological ordering provided positive predictive gain over step-shuffled sequences.
