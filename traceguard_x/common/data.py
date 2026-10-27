"""Data loading helpers for TRACEGUARD-X experiments.

Reuses the canonical frozen loaders (``src.data_loader``, ``src.preprocessing``,
``src.embeddings``) and the frozen split files. Nothing here writes to frozen
directories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import src.data_loader as canonical_data_loader  # noqa: E402  (path set by package __init__)
import src.config as canonical_config  # noqa: E402

from traceguard_x.paths import (
    FROZEN_DATA_PATH,
    FROZEN_EMB_CACHE_DIR,
    FROZEN_SPLITS_DIR,
)


def load_dataset() -> list[dict[str, Any]]:
    """Load and validate the frozen TRACEGUARD v4.1 dataset (600 trajectories)."""
    return canonical_data_loader.load_dataset(FROZEN_DATA_PATH)


def load_split(seed: int = 42) -> dict[str, Any]:
    """Load a frozen group-aware split JSON by seed."""
    path = FROZEN_SPLITS_DIR / f"split_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Frozen split not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def split_trajectories(
    trajectories: list[dict[str, Any]], split_info: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Map a split's group-id lists onto trajectory records.

    Returns a dict with ``train``/``val``/``test`` trajectory lists. Raises if
    any group id is unknown or if a trajectory appears in more than one
    partition (group-level leakage guard).
    """
    group_map = canonical_data_loader.get_group_map(trajectories)
    partitions: dict[str, list[dict[str, Any]]] = {}
    seen_groups: set[str] = set()
    for part in ("train", "val", "test"):
        ids = split_info[f"{part}_group_ids"]
        unknown = [gid for gid in ids if gid not in group_map]
        if unknown:
            raise KeyError(f"Split references unknown group ids: {unknown[:5]}")
        overlap = seen_groups.intersection(ids)
        if overlap:
            raise ValueError(f"Group-level leakage: {sorted(overlap)[:5]}")
        seen_groups.update(ids)
        trajectories_for_part: list[dict[str, Any]] = []
        for gid in ids:
            trajectories_for_part.extend(group_map[gid])
        partitions[part] = trajectories_for_part
    return partitions


def load_embeddings() -> dict[str, np.ndarray]:
    """Load the frozen precomputed embedding cache ({trajectory_id: [6, 384]})."""
    cache_path = FROZEN_EMB_CACHE_DIR / "embeddings_all-MiniLM-L6-v2.npy"
    meta_path = FROZEN_EMB_CACHE_DIR / "embeddings_all-MiniLM-L6-v2_meta.json"
    if not cache_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Frozen embedding cache missing; refusing to regenerate embeddings "
            "for reproduction purposes."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("model_name") != canonical_config.EMBEDDING_MODEL_NAME:
        raise ValueError(
            f"Embedding cache model mismatch: {meta.get('embedding_model')!r}"
        )
    embeddings = np.load(cache_path, allow_pickle=True).item()
    if meta.get("embedding_dim") != 384:
        raise ValueError(f"Unexpected embedding dim in cache meta: {meta}")
    return embeddings


def verify_no_group_leakage(
    trajectories: list[dict[str, Any]], split_info: dict[str, Any]
) -> dict[str, Any]:
    """Assert no counterfactual group spans partitions; return a report dict."""
    partitions = split_trajectories(trajectories, split_info)
    group_of = {t["trajectory_id"]: t["counterfactual_group_id"] for t in trajectories}
    report: dict[str, Any] = {"ok": True, "partitions": {}}
    part_groups: dict[str, set[str]] = {}
    for part, trajs in partitions.items():
        gids = {group_of[t["trajectory_id"]] for t in trajs}
        part_groups[part] = gids
        report["partitions"][part] = {
            "n_trajectories": len(trajs),
            "n_groups": len(gids),
            "label_counts": {
                label: sum(1 for t in trajs if t["label"] == label)
                for label in ("BENIGN", "INJECTION_RESISTED", "HIJACKED")
            },
        }
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        overlap = part_groups[a] & part_groups[b]
        if overlap:
            report["ok"] = False
            report[f"overlap_{a}_{b}"] = sorted(overlap)
    return report
