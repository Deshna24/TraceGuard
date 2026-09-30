"""
config.py - Central configuration for TRACEGUARD LSTM pipeline.
All hyperparameters, paths, and constants live here.
"""

import os
from pathlib import Path

# -------------------------------------------------
# Paths
# -------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "traceguard_v4_1.jsonl"

OUTPUTS_DIR    = ROOT_DIR / "outputs"
SPLITS_DIR     = OUTPUTS_DIR / "splits"
MODELS_DIR     = OUTPUTS_DIR / "models"
METRICS_DIR    = OUTPUTS_DIR / "metrics"
CM_DIR         = OUTPUTS_DIR / "confusion_matrices"
PREFIX_DIR     = OUTPUTS_DIR / "prefix_results"
PLOTS_DIR      = OUTPUTS_DIR / "plots"
HANDOFF_DIR    = OUTPUTS_DIR / "handoff"
EMB_CACHE_DIR  = OUTPUTS_DIR / "embeddings_cache"
REPORTS_DIR    = ROOT_DIR / "reports"

for _d in [SPLITS_DIR, MODELS_DIR, METRICS_DIR, CM_DIR, PREFIX_DIR,
           PLOTS_DIR, HANDOFF_DIR, EMB_CACHE_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Dataset constants (FROZEN -- do not change)
# -------------------------------------------------
TOTAL_TRAJECTORIES   = 600
TOTAL_GROUPS         = 200
N_CLASSES            = 3
STEPS_PER_TRAJECTORY = 6

LABEL_MAP     = {"BENIGN": 0, "INJECTION_RESISTED": 1, "HIJACKED": 2}
INV_LABEL_MAP = {0: "BENIGN", 1: "INJECTION_RESISTED", 2: "HIJACKED"}
CLASS_NAMES   = ["BENIGN", "INJECTION_RESISTED", "HIJACKED"]

# -------------------------------------------------
# Split configuration
# -------------------------------------------------
TRAIN_GROUPS = 140   # 70%
VAL_GROUPS   = 30    # 15%
TEST_GROUPS  = 30    # 15%

TRAIN_TRAJ = 420
VAL_TRAJ   = 90
TEST_TRAJ  = 90

# -------------------------------------------------
# Embedding model
# -------------------------------------------------
EMBEDDING_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_SHORT = "all-MiniLM-L6-v2"

# -------------------------------------------------
# LSTM hyperparameters
# -------------------------------------------------
LSTM_HIDDEN_SIZE   = 128
LSTM_NUM_LAYERS    = 2
LSTM_DROPOUT       = 0.2
LSTM_BIDIRECTIONAL = False   # Unidirectional for causal/online detection

# -------------------------------------------------
# Training hyperparameters
# -------------------------------------------------
LEARNING_RATE           = 1e-3
WEIGHT_DECAY            = 1e-4
BATCH_SIZE              = 32
MAX_EPOCHS              = 50
EARLY_STOPPING_PATIENCE = 7   # based on val Macro F1

# -------------------------------------------------
# Detection
# -------------------------------------------------
DETECTION_THRESHOLD = 0.5   # P(HIJACKED) >= threshold

# -------------------------------------------------
# Seeds for robustness experiment
# -------------------------------------------------
PRIMARY_SEED     = 42
ROBUSTNESS_SEEDS = [42, 123, 456, 789, 1011]

# -------------------------------------------------
# Fields MUST NEVER enter model input
# -------------------------------------------------
FORBIDDEN_INPUT_FIELDS = [
    "label", "attack_family", "attack_subtype",
    "injection", "injection_step", "precursor_type",
    "precursor_step", "deviation_step", "attack_success",
    "counterfactual_group_id", "hijack_subtype",
    "generation_method", "ground_truth_explanation",
]
