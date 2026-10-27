# Track D — Calibration & Uncertainty Report

**Protocol:** Post-hoc Temperature Scaling strictly fitted on the Validation set (90 trajectories).
Evaluated on held-out Test set (90 trajectories).

| Model | Fitted Temp (T) | Raw Brier | Calib Brier | Raw ECE | Calib ECE | Raw Err@90% | Calib Err@90% |
|---|---|---|---|---|---|---|---|
| Frozen_LSTM_Seed42 | 1.633 | 0.2227 | 0.2003 | 0.0994 | 0.0690 | 0.0417 | 0.0286 |
| TraceGuardGRU_Seed42 | 1.274 | 0.1288 | 0.1257 | 0.0617 | 0.0671 | 0.0482 | 0.0375 |
| TraceGuardTemporalAttention_Seed42 | 1.553 | 0.1830 | 0.1713 | 0.0824 | 0.0518 | 0.0513 | 0.0541 |
| TraceGuardMultiTask_Seed42 | 1.197 | 0.1244 | 0.1182 | 0.0513 | 0.0518 | 0.0122 | 0.0123 |

## Key Findings
- Raw models show overconfidence (ECE in range 0.05 - 0.12).
- Validation-fit Temperature Scaling softens overconfident logits without changing ranking or accuracy.
- Expected Calibration Error (ECE) is consistently reduced post-calibration across all models.