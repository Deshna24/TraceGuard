"""Central paths and experiment conventions for TRACEGUARD-X.

New artifacts are written only inside ``traceguard_x/experiments/<track>``;
frozen directories are never written to.
"""

from pathlib import Path

TRACEGUARD_X_ROOT = Path(__file__).resolve().parent
INTEGRITY_DIR = TRACEGUARD_X_ROOT / "integrity"
EXPERIMENTS_DIR = TRACEGUARD_X_ROOT / "experiments"
MODELS_DIR = TRACEGUARD_X_ROOT / "models"
TESTS_DIR = TRACEGUARD_X_ROOT / "tests"

# Frozen artifact locations (read-only from the perspective of TRACEGUARD-X).
# The frozen traceguard/ package is a sibling of traceguard_x/ in the repo root.
REPO_ROOT = TRACEGUARD_X_ROOT.parent

# Frozen artifact locations (read-only from the perspective of TRACEGUARD-X).
FROZEN_DATA_PATH = REPO_ROOT / "traceguard" / "data" / "traceguard_v4_1.jsonl"
FROZEN_SPLITS_DIR = REPO_ROOT / "traceguard" / "outputs" / "splits"
FROZEN_MODELS_DIR = REPO_ROOT / "traceguard" / "outputs" / "models"
FROZEN_EMB_CACHE_DIR = REPO_ROOT / "traceguard" / "outputs" / "embeddings_cache"
FROZEN_METRICS_DIR = REPO_ROOT / "traceguard" / "outputs" / "metrics"
FROZEN_PREFIX_DIR = REPO_ROOT / "traceguard" / "outputs" / "prefix_results"
FROZEN_HANDOFF_DIR = REPO_ROOT / "traceguard" / "outputs" / "handoff"
FROZEN_TRANSFORMER_DIR = REPO_ROOT / "results" / "transformer_v4_1"
FROZEN_COMPARISON_DIR = REPO_ROOT / "results" / "final_comparison"


def experiment_dir(track: str) -> Path:
    """Return (and create) the artifact directory for an experiment track."""
    path = EXPERIMENTS_DIR / track
    path.mkdir(parents=True, exist_ok=True)
    return path
