# Track B2 — Temporal Attention Experiment (Seed 123)

**Label:** New experimental result

- Parameters: 305,155 (encoder: 296,448, attention: 8,320)
- Best epoch: 14 (val Macro F1: 0.8893)
- Test Accuracy: 0.8111
- Test Macro F1: 0.8117
- HIJACKED F1: 0.8667 (AUROC: 0.9439)
- Pre-action detection rate: 86.67%
- Mean detection latency: -2.57 steps (median: -3.0)
- Never detected rate: 6.67%
- FPR Benign: 0.0667, FPR Resisted: 0.0667
- Prefix inference latency: 0.403 ms / prefix

> [!NOTE]
> Attention weights reflect how the pooling layer weights sequence representations. They do not constitute causal proof of model decisions.