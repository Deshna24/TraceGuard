# TRACEGUARD â€” LSTM Results Handoff Package

This package contains all primary LSTM results for the TRACEGUARD project.

## Dataset
- **File:** `traceguard_v4_1.jsonl` (frozen, do not modify)
- **600 trajectories** | 200 BENIGN | 200 INJECTION_RESISTED | 200 HIJACKED
- **200 counterfactual groups** (unit of splitting)

## Split
- **Seed:** 42
- Train: 140 groups / 420 trajectories
- Val:   30 groups / 90 trajectories
- Test:  30 groups / 90 trajectories
- Split group IDs: `dataset_split.json`

## Embedding Model
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension:** 384
- Each trajectory = 6 step embeddings â†’ shape [6, 384]

## Step Representation
Each step text is constructed as:
```
ORIGINAL USER GOAL:
{user_goal}

CURRENT STEP:
{step_number}

ACTION:
{action}

TOOL:
{tool}

TOOL INPUT:
{tool_input}

TOOL OBSERVATION:
{tool_observation}

STATE:
{state}
```

**Ground-truth metadata is NEVER included in step text.**

## LSTM Configuration
- architecture: TraceGuardLSTM
- input_size: 384
- hidden_size: 128
- num_layers: 2
- dropout: 0.2
- num_classes: 3
- bidirectional: False
- pooling: last_hidden_state

## Training
- Optimizer: AdamW (lr=0.001, wd=0.0001)
- Batch size: 32
- Max epochs: 50
- Early stopping patience: 7 (val Macro F1)
- Best epoch: 21

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 0.8556 |
| Macro F1 | 0.8533 |
| BENIGN F1 | 0.7857 |
| INJECTION_RESISTED F1 | 0.8710 |
| HIJACKED F1 | 0.9032 |

## Detection
- **Threshold:** 0.5
- Pre-action detection rate: 93.33%
- Mean latency: -2.7142857142857144

| | Type A | Type B |
|-|--------|--------|
| Pre-action rate | 88.89% | 95.24% |
| Mean latency | -2.5 | -2.8 |

## Files
| File | Description |
|------|-------------|
| `dataset_split.json` | All group IDs per split |
| `experiment_config.json` | Full experiment configuration |
| `lstm_metrics.json` | Full-trajectory test metrics |
| `latency_stats.json` | Detection latency metrics |
| `plots/` | All figures |
| `confusion_matrices/` | Confusion matrix PNG |

## For Teammate
When implementing your baseline/Transformer, keep identical:
- Dataset file (traceguard_v4_1.jsonl, unchanged)
- Split group IDs (use dataset_split.json, seed=42)
- Embedding model (all-MiniLM-L6-v2)
- Step text format (see above)
- Detection threshold (0.5)
- Class encoding: BENIGN=0, INJECTION_RESISTED=1, HIJACKED=2
