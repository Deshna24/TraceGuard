# TRACEGUARD — LSTM Experiment Pipeline

**Early Detection and Localization of Hijacked LLM Agent Trajectories Using Sequential Deep Learning**

## Overview

This directory contains the complete LSTM implementation for the TRACEGUARD research project.
Dataset: 	raceguard_v4_1.jsonl (FROZEN — do not modify)

## Directory Structure

`
traceguard/
├── data/
│   └── traceguard_v4_1.jsonl          # Frozen dataset (600 trajectories)
├── src/
│   ├── config.py                      # All hyperparameters and paths
│   ├── data_loader.py                 # Dataset loading + validation
│   ├── split.py                       # Group-aware 70/15/15 split
│   ├── preprocessing.py               # Step text construction
│   ├── embeddings.py                  # Sentence embedding + caching
│   ├── model_lstm.py                  # TraceGuardLSTM architecture + Dataset
│   ├── train_lstm.py                  # Training loop (AdamW + early stopping)
│   ├── evaluate.py                    # Full-trajectory metrics
│   ├── prefix_detection.py            # Online/prefix inference
│   ├── latency.py                     # Detection latency analysis
│   ├── plots.py                       # All publication figures
│   └── utils.py                       # Seed, device, IO utilities
├── experiments/
│   ├── run_lstm.py                    # Main experiment (all 18 phases)
│   ├── run_seeds.py                   # Five-seed robustness experiment
│   └── run_prefix_evaluation.py      # Standalone prefix re-evaluation
├── outputs/
│   ├── splits/                        # Saved group split IDs per seed
│   ├── models/                        # Saved LSTM checkpoints (.pth)
│   ├── metrics/                       # JSON metrics + CSV predictions
│   ├── confusion_matrices/            # Confusion matrix PNGs
│   ├── prefix_results/                # Per-prefix prediction CSV/JSON
│   ├── plots/                         # All publication figures
│   ├── embeddings_cache/              # Cached sentence embeddings
│   ├── handoff/                       # Teammate handoff package
│   └── experiment_config.json         # Full reproducibility config
├── reports/
│   ├── error_analysis.md              # Per-category error examples
│   └── final_lstm_report.md           # Complete LSTM results report
└── requirements.txt
`

## Quick Start

`ash
cd traceguard

# Install dependencies
pip install -r requirements.txt

# Run primary experiment (seed=42)
python experiments/run_lstm.py

# Run with GPU + force embedding recomputation
python experiments/run_lstm.py --force-embed

# Skip training, re-evaluate from checkpoint
python experiments/run_lstm.py --skip-train

# Five-seed robustness experiment (run after primary)
python experiments/run_seeds.py

# Standalone prefix evaluation
python experiments/run_prefix_evaluation.py --threshold 0.5
`

## Key Design Decisions

### Group-Aware Split (CRITICAL)
The dataset contains 200 counterfactual groups (3 trajectories each: BENIGN, INJECTION_RESISTED, HIJACKED).
**All 3 trajectories from a group MUST stay in the same split.**
Individual trajectory splitting would cause data leakage (audit found 387 pairs with cosine similarity >0.95).

Split: 70% groups → TRAIN | 15% → VAL | 15% → TEST
= 140 groups (420 trajs) | 30 groups (90 trajs) | 30 groups (90 trajs)

### Step Text Construction
Each step is encoded as:
`
ORIGINAL USER GOAL:
{user_goal}

CURRENT STEP:
{step_number}

ACTION: {action}
TOOL: {tool}
TOOL INPUT: {tool_input}
TOOL OBSERVATION: {tool_observation}
STATE: {state}
`
**Labels, attack metadata, and counterfactual group ID NEVER enter the model.**

### LSTM Architecture (Unidirectional)
- Input: [batch, 6, 384]  (384 = all-MiniLM-L6-v2 dim)
- LSTM: hidden=128, layers=2, dropout=0.2, **unidirectional** (causal)
- Output: last hidden state → Dropout → Linear(128, 3)
- Unidirectionality is MANDATORY for online/early detection

### Detection Rule
A trajectory is flagged HIJACKED at the earliest prefix k where:
P(HIJACKED) >= 0.5
Threshold tuned on VAL only, evaluated once on TEST.

## Experiment Phases
1. Load + validate dataset (enforce all invariants)
2. Group-aware split
3. Generate/cache embeddings (all-MiniLM-L6-v2)
4. Train LSTM (AdamW, CrossEntropyLoss, early stopping on val Macro F1)
5. Full trajectory evaluation
6. Prefix/online evaluation (prefixes 1..6, causal)
7. Detection latency analysis
8. Type-A vs Type-B analysis
9. Counterfactual group analysis
10. Step-shuffling ablation
11. Error analysis
12. Five-seed robustness
13. Plots, tables, final report, handoff package

## Reproducibility
- Primary seed: 42
- Robustness seeds: [42, 123, 456, 789, 1011]
- All outputs saved under outputs/
- Split IDs saved to outputs/splits/split_seed{seed}.json
- Checkpoint saved to outputs/models/lstm_seed{seed}_best.pth
- Experiment config saved to outputs/experiment_config.json

## For Teammate
See outputs/handoff/README.md for the complete handoff package.
