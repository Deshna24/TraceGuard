# Track A — Baseline Reproduction (label: Reproduced baseline)

Seed: 42; checkpoint best_epoch=21, best_val_f1=0.9227385608741541
Split sizes: train=420, val=90, test=90; group leakage ok=True

- Accuracy: 0.8556
- Macro P/R/F1: 0.8550 / 0.8556 / 0.8533
- BENIGN: P=0.8462 R=0.7333 F1=0.7857 AUROC=0.9728
- INJECTION_RESISTED: P=0.8438 R=0.9000 F1=0.8710 AUROC=0.9789
- HIJACKED: P=0.8750 R=0.9333 F1=0.9032 AUROC=0.9794
- FPR (BENIGN): 0.1000; FPR (INJECTION_RESISTED): 0.0333
- Detection rate: 0.9333333333333333; pre-action rate: 0.9333333333333333; mean latency: -2.7142857142857144
- Never detected: 0.06666666666666667

## Comparison with official frozen results
- full_metrics/accuracy: reproduced=0.8555555555555555, official=0.8555555555555555, abs_diff=0.0
- full_metrics/macro_precision: reproduced=0.8549679487179488, official=0.8549679487179488, abs_diff=0.0
- full_metrics/macro_recall: reproduced=0.8555555555555555, official=0.8555555555555555, abs_diff=0.0
- full_metrics/macro_f1: reproduced=0.8533026113671275, official=0.8533026113671275, abs_diff=0.0
- latency/n_hijacked: reproduced=30, official=30, abs_diff=0.0
- latency/n_detected: reproduced=28, official=28, abs_diff=0.0
- latency/detection_rate: reproduced=0.9333333333333333, official=0.9333333333333333, abs_diff=0.0
- latency/pre_action_detection_rate: reproduced=0.9333333333333333, official=0.9333333333333333, abs_diff=0.0
- latency/mean_latency: reproduced=-2.7142857142857144, official=-2.7142857142857144, abs_diff=0.0
- latency/median_latency: reproduced=-3.0, official=-3.0, abs_diff=0.0
- latency/pct_never_detected: reproduced=0.06666666666666667, official=0.06666666666666667, abs_diff=0.0
- prefix_probabilities/n_rows_compared: 540
- prefix_probabilities/max_abs_prob_diff: 4.172325134277344e-07