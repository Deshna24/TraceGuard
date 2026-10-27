"""Experiment bookkeeping for TRACEGUARD-X: metadata, hashing, reproducibility."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_hashes(paths: list[Path]) -> dict[str, str]:
    """SHA-256 map for a list of files (missing files recorded as null)."""
    return {
        str(p): (sha256_file(p) if Path(p).is_file() else None) for p in paths
    }


def software_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }


def save_experiment_metadata(
    experiment_dir: Path,
    experiment_id: str,
    model_architecture: str,
    hyperparameters: dict[str, Any],
    seed: int,
    dataset_version: str,
    split_id: str,
    feature_representation: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the standard experiment metadata record and return its path."""
    record = {
        "experiment_id": experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_architecture": model_architecture,
        "hyperparameters": hyperparameters,
        "seed": seed,
        "dataset_version": dataset_version,
        "split_id": split_id,
        "feature_representation": feature_representation,
        "software": software_versions(),
    }
    if extra:
        record.update(extra)
    path = Path(experiment_dir) / f"{experiment_id}_metadata.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def save_json(data: Any, path: Path) -> Path:
    """Save JSON (converting numpy scalars/arrays automatically)."""
    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_default), encoding="utf-8")
    return path
