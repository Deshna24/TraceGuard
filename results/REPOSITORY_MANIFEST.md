# TRACEGUARD Repository Manifest & Canonical Index

**Date:** 2026-09-28  
**Status:** **FROZEN & READ-ONLY**  

---

## 1. Canonical Data & Splits
- **Canonical Dataset:** [`data/raw/trajectories/traceguard_v4_1.jsonl`](file:///c:/Users/DESHNA/TraceGuard/data/raw/trajectories/traceguard_v4_1.jsonl)
- **Canonical Seed 42 Split:** [`traceguard/outputs/splits/split_seed42.json`](file:///c:/Users/DESHNA/TraceGuard/traceguard/outputs/splits/split_seed42.json)
- **Multi-Seed Split Directory:** `traceguard/outputs/splits/split_seed{42,123,456,789,1011}.json`

---

## 2. Canonical Model Directories
- **LSTM Baseline:** [`results/lstm_v4_1/`](file:///c:/Users/DESHNA/TraceGuard/results/lstm_v4_1/)
- **Transformer Baseline:** [`results/transformer_v4_1/`](file:///c:/Users/DESHNA/TraceGuard/results/transformer_v4_1/)
- **Final Model Comparison:** [`results/final_comparison/`](file:///c:/Users/DESHNA/TraceGuard/results/final_comparison/)

---

## 3. Key Canonical Reports
- **Final Audit Report:** [`results/transformer_v4_1/reports/final_audit.md`](file:///c:/Users/DESHNA/TraceGuard/results/transformer_v4_1/reports/final_audit.md)
- **Pre-Training Diagnostics:** [`results/transformer_v4_1/reports/pretraining_diagnostics.md`](file:///c:/Users/DESHNA/TraceGuard/results/transformer_v4_1/reports/pretraining_diagnostics.md)
- **Experiment Report:** [`results/transformer_v4_1/reports/experiment_report.md`](file:///c:/Users/DESHNA/TraceGuard/results/transformer_v4_1/reports/experiment_report.md)
- **Quality Control Audit:** [`results/transformer_v4_1/reports/quality_check.md`](file:///c:/Users/DESHNA/TraceGuard/results/transformer_v4_1/reports/quality_check.md)
- **Comparison Report:** [`results/final_comparison/model_comparison.md`](file:///c:/Users/DESHNA/TraceGuard/results/final_comparison/model_comparison.md)

---

## 4. Pipeline Execution Scripts
- [`experiments/run_full_transformer_and_baselines.py`](file:///c:/Users/DESHNA/TraceGuard/experiments/run_full_transformer_and_baselines.py) — Reproduces 5-seed Transformer, LogReg baseline, step ablation, and final comparison table.
- [`experiments/run_final_transformer_audit.py`](file:///c:/Users/DESHNA/TraceGuard/experiments/run_final_transformer_audit.py) — Zero-mutation audit script.
- [`experiments/run_transformer_v4_1_cls.py`](file:///c:/Users/DESHNA/TraceGuard/experiments/run_transformer_v4_1_cls.py) — Transformer CLS model implementation.
- [`experiments/diagnose_transformer_collapse.py`](file:///c:/Users/DESHNA/TraceGuard/experiments/diagnose_transformer_collapse.py) — Pre-training diagnostic script.

---

## 5. Removed Disposable Artifacts
- Temporary Python bytecode caches (`__pycache__/` directories)
- Temporary unreferenced test scratch script (`test_ollama.py`)

---

## 6. Historical Files Retained (MANUAL REVIEW)
- `traceguard/_*.py` — Historical helper scripts retained for record.
- `src/v3_*.py` & `src/diagnose_seed789.py` — Historical pilot diagnostic scripts retained.
- `results/reports/seed789_error_analysis.md` & `v3_*.md` — Historical pilot audit documentation retained.
