# Track B2 — Temporal Attention Experiment (Seed 42)

**Label:** New experimental result

- Parameters: 305,155 (encoder: 296,448, attention: 8,320)
- Best epoch: 20 (val Macro F1: 0.9330)
- Test Accuracy: 0.8889
- Test Macro F1: 0.8885
- HIJACKED F1: 0.9492 (AUROC: 0.9833)
- Pre-action detection rate: 93.33%
- Mean detection latency: -3.04 steps (median: -3.0)
- Never detected rate: 6.67%
- FPR Benign: 0.0333, FPR Resisted: 0.0000
- Prefix inference latency: 0.455 ms / prefix

> [!NOTE]
> Attention weights reflect how the pooling layer weights sequence representations. They do not constitute causal proof of model decisions.