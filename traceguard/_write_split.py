"""
split.py - Group-aware train/val/test split for TRACEGUARD.

RULE: The unit of splitting is counterfactual_group_id.
All 3 trajectories from the same group MUST stay together in one split.
NO individual trajectory splitting.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import (
    TRAIN_GROUPS, VAL_GROUPS, TEST_GROUPS,
    TRAIN_TRAJ, VAL_TRAJ, TEST_TRAJ,
    SPLITS_DIR,
)
from src.utils import save_json


def create_group_aware_split(trajs: list[dict], seed: int = 42):
    """
    Create group-aware 70/15/15 split.

    Returns:
        train_trajs, val_trajs, test_trajs — lists of trajectory dicts
        split_info — dict with group IDs per split (for reproducibility)
    """
    # Build group_id -> [traj, ...] mapping
    group_map: dict[str, list] = {}
    for t in trajs:
        gid = t["counterfactual_group_id"]
        group_map.setdefault(gid, []).append(t)

    group_ids = sorted(group_map.keys())  # deterministic ordering

    # For stratification: represent each group by the label of its HIJACKED member
    # (all groups have exactly one of each class, so we can use any stable label)
    # We stratify on the HIJACKED subtype (A/B) to keep distribution balanced.
    group_strata = []
    for gid in group_ids:
        members = group_map[gid]
        hj = next((t for t in members if t["label"] == "HIJACKED"), None)
        stratum = hj.get("hijack_subtype", "A") if hj else "A"
        group_strata.append(stratum)

    group_ids_arr = np.array(group_ids)
    strata_arr    = np.array(group_strata)

    # Step 1: split off 30 test groups
    n_total = len(group_ids)
    test_frac = TEST_GROUPS / n_total       # 0.15

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(group_ids_arr, strata_arr))

    trainval_ids    = group_ids_arr[trainval_idx]
    trainval_strata = strata_arr[trainval_idx]
    test_ids        = group_ids_arr[test_idx]

    # Step 2: split trainval into 140 train + 30 val
    val_frac_of_trainval = VAL_GROUPS / len(trainval_ids)  # 30/170 ~ 0.1765

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=seed)
    train_idx, val_idx = next(sss2.split(trainval_ids, trainval_strata))

    train_ids = trainval_ids[train_idx]
    val_ids   = trainval_ids[val_idx]

    # Convert to trajectory lists
    def groups_to_trajs(gids):
        out = []
        for gid in gids:
            out.extend(group_map[gid])
        return out

    train_trajs = groups_to_trajs(train_ids)
    val_trajs   = groups_to_trajs(val_ids)
    test_trajs  = groups_to_trajs(test_ids)

    _verify_split(train_ids, val_ids, test_ids,
                  train_trajs, val_trajs, test_trajs)

    split_info = {
        "seed": seed,
        "train_group_ids": sorted(train_ids.tolist()),
        "val_group_ids":   sorted(val_ids.tolist()),
        "test_group_ids":  sorted(test_ids.tolist()),
        "train_n_groups":  len(train_ids),
        "val_n_groups":    len(val_ids),
        "test_n_groups":   len(test_ids),
        "train_n_trajs":   len(train_trajs),
        "val_n_trajs":     len(val_trajs),
        "test_n_trajs":    len(test_trajs),
        "train_label_dist": dict(Counter(t["label"] for t in train_trajs)),
        "val_label_dist":   dict(Counter(t["label"] for t in val_trajs)),
        "test_label_dist":  dict(Counter(t["label"] for t in test_trajs)),
    }

    return train_trajs, val_trajs, test_trajs, split_info


def _verify_split(train_ids, val_ids, test_ids,
                  train_trajs, val_trajs, test_trajs):
    """STOP if any group appears in multiple splits."""
    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set   & test_set

    if tv or tt or vt:
        raise AssertionError(
            f"STOP: Group leakage detected!\n"
            f"  train ∩ val  = {tv}\n"
            f"  train ∩ test = {tt}\n"
            f"  val   ∩ test = {vt}"
        )

    print("[Split] Leakage check passed — no group appears in multiple splits.")

    # Size checks
    assert len(train_ids) == TRAIN_GROUPS, \
        f"STOP: Expected {TRAIN_GROUPS} train groups, got {len(train_ids)}"
    assert len(val_ids) == VAL_GROUPS, \
        f"STOP: Expected {VAL_GROUPS} val groups, got {len(val_ids)}"
    assert len(test_ids) == TEST_GROUPS, \
        f"STOP: Expected {TEST_GROUPS} test groups, got {len(test_ids)}"
    assert len(train_trajs) == TRAIN_TRAJ, \
        f"STOP: Expected {TRAIN_TRAJ} train trajs, got {len(train_trajs)}"
    assert len(val_trajs) == VAL_TRAJ, \
        f"STOP: Expected {VAL_TRAJ} val trajs, got {len(val_trajs)}"
    assert len(test_trajs) == TEST_TRAJ, \
        f"STOP: Expected {TEST_TRAJ} test trajs, got {len(test_trajs)}"

    print(f"[Split] Train: {len(train_ids)} groups / {len(train_trajs)} trajs  "
          f"| Val: {len(val_ids)} groups / {len(val_trajs)} trajs  "
          f"| Test: {len(test_ids)} groups / {len(test_trajs)} trajs")

    for split_name, split_trajs in [("Train", train_trajs), ("Val", val_trajs), ("Test", test_trajs)]:
        dist = Counter(t["label"] for t in split_trajs)
        print(f"  {split_name} class dist: {dict(dist)}")


def save_split(split_info: dict, seed: int):
    path = SPLITS_DIR / f"split_seed{seed}.json"
    save_json(split_info, path)
    print(f"[Split] Saved split info to {path}")
    return path
