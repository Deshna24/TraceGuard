# Track B3 — Multi-Task Early Warning Experiment (Seed 42)

**Label:** New experimental result

- Parameters: 296,964 (encoder: 296,448, main: 387, warn: 129)
- Best epoch: 18 (val Macro F1: 0.9778)
- Test Accuracy: 0.9222
- Test Macro F1: 0.9225
- HIJACKED F1: 0.9032 (AUROC: 0.9894)
- Pre-action detection rate: 90.00%
- Mean detection latency: -2.46 steps (median: -3.0)
- Never detected rate: 6.67%
- Auxiliary Imminent-Deviation AUROC: 0.7805
- Mean warning score: HIJACKED=0.7175, RESISTED=0.4573, BENIGN=0.1529
- Prefix inference latency: 0.368 ms / prefix

> [!IMPORTANT]
> The auxiliary early-warning score is an experimental research indicator. It is strictly separated from the official P(HIJACKED) safety gate decision.