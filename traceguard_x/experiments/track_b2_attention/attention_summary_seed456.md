# Track B2 — Temporal Attention Experiment (Seed 456)

**Label:** New experimental result

- Parameters: 305,155 (encoder: 296,448, attention: 8,320)
- Best epoch: 43 (val Macro F1: 0.9114)
- Test Accuracy: 0.9000
- Test Macro F1: 0.8992
- HIJACKED F1: 0.9355 (AUROC: 0.9922)
- Pre-action detection rate: 96.67%
- Mean detection latency: -2.66 steps (median: -3.0)
- Never detected rate: 3.33%
- FPR Benign: 0.0667, FPR Resisted: 0.0333
- Prefix inference latency: 0.518 ms / prefix

> [!NOTE]
> Attention weights reflect how the pooling layer weights sequence representations. They do not constitute causal proof of model decisions.