# TRACEGUARD-X Research Plan

**Status:** COMPLETE (Phase 12 — 2026-10-27). All 12 implementation and experimentation phases are complete, 100% of tests pass (74/74), and protected artifact integrity is hash-verified (166/166 intact). Accompanying documentation:
- [TRACEGUARD_X_ARCHITECTURE.md](TRACEGUARD_X_ARCHITECTURE.md)
- [TRACEGUARD_X_EXPERIMENTS.md](TRACEGUARD_X_EXPERIMENTS.md)
- [TRACEGUARD_X_EVALUATION.md](TRACEGUARD_X_EVALUATION.md)
- [TRACEGUARD_X_LIMITATIONS.md](TRACEGUARD_X_LIMITATIONS.md)

---

## 1. Repository audit summary (existing architecture)

The repository contains two layers:

1. **Frozen v4.1 research pipeline** (`traceguard/src/`, `traceguard/data/`, `traceguard/outputs/`):
   - Dataset: `traceguard/data/traceguard_v4_1.jsonl` — 600 trajectories, 200 counterfactual groups × 3 variants (`_B` benign, `_R` injection-resisted, `_H` hijacked), exactly 6 steps each, 200 per class. HIJACKED splits into 60 Type-A (no precursor) / 140 Type-B (precursor present).
   - Canonical representation: `traceguard/src/preprocessing.py:build_step_text()` — fixed template `ORIGINAL USER GOAL / CURRENT STEP / ACTION / TOOL / TOOL INPUT / TOOL OBSERVATION / STATE`, ordered by `step["step"]`, no truncation (invariant: exactly 6 steps), leakage assertion rejecting label-bearing literals; forbidden metadata fields enumerated in `traceguard/src/config.py:90-96`.
   - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, batch 64, **not normalized**, per-trajectory shape `[6, 384]`; precomputed cache at `traceguard/outputs/embeddings_cache/`.
   - Class mapping (`traceguard/src/config.py:38-40`): `BENIGN=0, INJECTION_RESISTED=1, HIJACKED=2`.
   - Model: `TraceGuardLSTM` (`traceguard/src/model_lstm.py:20-89`) — unidirectional 2-layer LSTM, input 384, hidden 128, dropout 0.2, packed sequences, last-hidden-state pooling, `Linear(128→3)`.
   - Training: AdamW lr 1e-3, wd 1e-4, batch 32, ≤50 epochs, early stopping patience 7 on validation Macro F1 (`traceguard/src/train_lstm.py`).
   - Split: group-aware on `counterfactual_group_id`, stratified by HIJACKED subtype, 140/30/30 groups → 420/90/90 trajectories (`traceguard/src/split.py`); official split `traceguard/outputs/splits/split_seed42.json` plus seeds 123/456/789/1011.
   - Evaluation: full metrics (`traceguard/src/evaluate.py`), prefix evaluation over k=1..6 (`traceguard/src/prefix_detection.py`), detection latency (`traceguard/src/latency.py`). Official threshold: **P(HIJACKED) ≥ 0.5 inclusive** (`DETECTION_THRESHOLD`, `traceguard/src/config.py:79`).
   - Checkpoint loading: strict `model_config` equality check then `load_state_dict` (`traceguard/runtime/detector.py:89-131`); no silent fallback.
   - Comparison models (frozen results only): mean-pooled Logistic Regression (acc 0.6556 / macro F1 0.6456), CLS Transformer (0.7111 / 0.7033) vs LSTM (0.8556 / 0.8533) — `results/final_comparison/model_comparison.md`.

2. **Runtime demo system** (`traceguard/runtime/`, `traceguard/agent/`, `traceguard/web/`, `frontend/`):
   - FastAPI backend (`traceguard/web/server.py`) with deterministic scripted runs and live Ollama runs (Granite `granite4.1:8b-q4_K_M`, temperature 0, JSON mode, loopback only).
   - `PreActionGate` (`traceguard/runtime/action_gate.py`): pre-action, BLOCK iff P(HIJACKED) ≥ 0.5, **fail-closed** on any detector error; `execute_if_allowed` raises `BlockedActionError`; unknown tools never execute.
   - Controlled tools: `search` (fixed in-memory KB), `calculator` (AST-restricted), `database` (fictional records) — `traceguard/agent/tools.py`.
   - WebSocket event stream with typed `EventEnvelope`s; React 19/Vite frontend with live trajectory, risk graph, gate panel, event feed.

## 2. Original frozen baseline (label: ORIGINAL FROZEN BASELINE)

From `traceguard/outputs/handoff/final_lstm_report.md` and `traceguard/outputs/metrics/lstm_seed42_metrics.json`:

| Metric | Seed 42 (test, 90 trajectories) |
|---|---|
| Accuracy | 0.8556 |
| Macro F1 | 0.8533 |
| HIJACKED F1 | 0.9032 (AUROC 0.9794) |
| Pre-action detection rate | 93.33% |
| Mean detection latency | −2.71 steps (median −3.0) |
| Step-shuffle ablation ΔF1 | +0.2811 (ordering matters) |
| Five-seed Macro F1 | 0.8553 ± 0.0354 |

Five-seed macro F1 range: 0.80–0.911 (seed 789 collapse is documented in `traceguard/reports/error_analysis.md`: all errors BENIGN→INJECTION_RESISTED).

## 3. Research question

Can temporal deep-learning models improve **early detection** of LLM-agent behavioral hijacking while maintaining reliable calibration, controlling false alarms, and generalizing to unseen trajectory patterns?

## 4. Hypotheses (to be tested, not assumed)

- **H1 (GRU, B1):** a GRU with comparable capacity matches or exceeds the frozen LSTM's test Macro F1 with fewer parameters and equal or better latency.
- **H2 (Temporal attention, B2):** learned step-level attention improves prefix-time detection (earlier flagging at k < deviation step) without increasing benign false-positive rate; attention weights are *diagnostic, not causal*.
- **H3 (Multi-task early warning, B3):** an auxiliary "deviation within horizon h" head, supervised by derived `deviation_step` metadata (which **exists** in the dataset), improves pre-deviation detection vs the single-task model.
- **H4 (Calibration, D):** the frozen model's raw probabilities are imperfectly calibrated; validation-fit calibration (e.g., temperature scaling) improves Brier/ECE without changing ranking.
- **H5 (Generalization, C):** group-aware splits prevent leakage, but models may still exploit template-level shortcuts; held-out-attack-template evaluation will show reduced performance. *(Feasibility depends on template metadata inspection — `attack_family`/`attack_subtype` exist.)*
- **H6 (Gate, F):** a three-way gate (ALLOW/ESCALATE/BLOCK) with an abstain band can reduce false blocks at equal harm-blocking, at the cost of escalations. Policy parameters fit on validation only.

## 5. Proposed experiments

| Track | Experiment | Output location |
|---|---|---|
| A | Baseline reproduction from frozen checkpoints (no retraining) | `traceguard_x/experiments/track_a_baseline/` |
| B1 | GRU training (seeds 42 first, then 5-seed) | `traceguard_x/experiments/track_b1_gru/` |
| B2 | Temporal-attention model | `traceguard_x/experiments/track_b2_attention/` |
| B3 | Multi-task early-warning head (derived supervision from `deviation_step`/`precursor_step`) | `traceguard_x/experiments/track_b3_earlywarn/` |
| B4 | Optional Temporal CNN (only if B1–B3 complete) | `traceguard_x/experiments/track_b4_tcn/` |
| C | Group-split integrity tests, counterfactual consistency, held-out `attack_subtype` split, controlled perturbations (paraphrase, formatting, field noise) | `traceguard_x/experiments/track_c_generalization/` |
| D | Brier, ECE, reliability data, classwise calibration, validation-fit temperature scaling | `traceguard_x/experiments/track_d_calibration/` |
| E | Prefix-by-prefix early detection with paired comparisons vs baseline detection records | `traceguard_x/experiments/track_e_early/` |
| F | Experimental three-way gate + offline policy evaluation | `traceguard_x/experiments/track_f_gate/` |

## 6. Data leakage risks and controls

- **Known controlled:** counterfactual groups never cross splits (asserted in `split.py`); forbidden metadata fields excluded from step text (asserted in `preprocessing.py`); embedding texts carry no label literals.
- **Risks to test in Track C:** (a) shared `user_goal` text between counterfactual variants is by design but could let a model memorize goals; (b) `attack_subtype` templates appear in both train and test under the official split — a held-out-template split is a *new experimental* split, clearly versioned, never replacing the official one; (c) any calibration/normalization must be fit on train/val only; (d) runtime demonstration logs (`traceguard/runtime_logs/`) must never be mixed into benchmark data.
- **Early-warning supervision risk (B3):** `deviation_step` is metadata; prefix models must only receive steps ≤ k, and the auxiliary label at prefix k must be derivable from *documented* metadata (deviation occurs at step ≤ k+h) without using step contents > k.

## 7. Evaluation methodology

- Official split (`split_seed42.json`) and threshold (0.5) preserved for all baseline reproduction.
- Metrics: accuracy, macro P/R/F1, per-class P/R/F1, confusion matrix, HIJACKED AUROC/PR-AUC, first detection step, pre-action detection rate, never-detected rate, latency, FPR on BENIGN and INJECTION_RESISTED.
- Early detection: all valid prefixes k=1..6, no future-step access (enforced by construction and tested).
- Uncertainty: Brier, ECE (15-bin), reliability data, classwise-ECE, high-confidence error rate.
- Statistical rigor: paired comparisons over identical test groups; five-seed mean ± sd for new models; paired tests (e.g., Wilcoxon signed-rank over per-group detection records) only where n permits; no significance claims from point estimates alone.
- Labels used on every reported number: **Original frozen baseline / Reproduced baseline / New experimental result / Runtime demonstration / Unverified hypothesis / NOT EVALUATED**.

## 8. Implementation phases

Phases 0–12 as executed in order (audit+protection → baseline → framework → GRU → attention → early-warning → experiments/ablations → generalization+calibration → experimental gate → runtime integration → frontend → full test suite → final report). Each phase ends with a protected-artifact hash re-verification against `traceguard_x/integrity/protected_hashes_phase0.json`.

## 9. Expected deliverables

- `docs/TRACEGUARD_X_{RESEARCH_PLAN,ARCHITECTURE,EXPERIMENTS,EVALUATION,LIMITATIONS}.md`
- `traceguard_x/` package: models (GRU, attention, multi-task), training pipeline, calibration, experimental gate, all with tests.
- Machine-readable metrics JSONs per experiment + human-readable summaries.
- Runtime: optional experimental model/gate selection via config, default unchanged.
- Frontend: research dashboard additions showing real outputs, unavailable states when data is missing.

## 10. Acceptance criteria

See §18 of the project specification; operationally: original runtime works, frozen artifacts byte-identical (hash-verified), ≥1 new sequential model trained and evaluated on the official split, temporal attention implemented or blocked with cause, early-warning supervision feasibility documented, evaluation beyond accuracy, leakage/group tests pass, isolated artifact dirs, experimental gate cannot bypass execution boundary, automated tests for new components, no fabricated results anywhere.

## 11. Resource requirements

CPU-only training (frozen config used device cpu). Small models (≈0.5–1M params), 600×6×384 dataset → minutes per training run on CPU. No cloud services, no GPU assumption, no new heavy dependencies beyond what exists in `requirements.txt` (torch, sentence-transformers, scikit-learn, numpy, pandas).
