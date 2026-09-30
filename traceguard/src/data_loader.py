"""
data_loader.py - Load and validate traceguard_v4_1.jsonl.
STOP conditions are enforced on invariant violations.
"""

import json
from pathlib import Path
from collections import Counter
from src.config import (
    DATA_PATH, TOTAL_TRAJECTORIES, TOTAL_GROUPS, STEPS_PER_TRAJECTORY,
    LABEL_MAP, INV_LABEL_MAP, CLASS_NAMES,
)


def load_dataset(path=None) -> list[dict]:
    """Load JSONL, validate all invariants, return list of trajectory dicts."""
    path = Path(path) if path else DATA_PATH
    print(f"\n[DataLoader] Loading dataset from: {path}")

    if not path.exists():
        raise FileNotFoundError(f"STOP: Dataset not found at {path}")

    trajectories = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"STOP: JSON parse error on line {i+1}: {e}")
            trajectories.append(traj)

    _validate_dataset(trajectories)
    return trajectories


def _validate_dataset(trajs: list[dict]):
    """Enforce all dataset invariants. STOP on any violation."""
    print(f"[DataLoader] Validating {len(trajs)} trajectories ...")

    # 1. Total count
    if len(trajs) != TOTAL_TRAJECTORIES:
        raise AssertionError(
            f"STOP: Expected {TOTAL_TRAJECTORIES} trajectories, found {len(trajs)}"
        )

    # 2. Required fields
    required = [
        "trajectory_id", "counterfactual_group_id", "label",
        "task_category", "user_goal", "steps",
    ]
    for traj in trajs:
        for field in required:
            if field not in traj:
                raise AssertionError(
                    f"STOP: Missing required field '{field}' in trajectory {traj.get('trajectory_id')}"
                )

    # 3. Label distribution
    label_counts = Counter(t["label"] for t in trajs)
    for cls in CLASS_NAMES:
        if label_counts[cls] != 200:
            raise AssertionError(
                f"STOP: Expected 200 trajectories for class {cls}, found {label_counts[cls]}"
            )

    # 4. Group count and sizes
    group_map: dict[str, list] = {}
    for t in trajs:
        gid = t["counterfactual_group_id"]
        group_map.setdefault(gid, []).append(t)

    if len(group_map) != TOTAL_GROUPS:
        raise AssertionError(
            f"STOP: Expected {TOTAL_GROUPS} groups, found {len(group_map)}"
        )
    for gid, members in group_map.items():
        if len(members) != 3:
            raise AssertionError(f"STOP: Group {gid} has {len(members)} members (expected 3)")

    # 5. Unique trajectory IDs
    ids = [t["trajectory_id"] for t in trajs]
    if len(ids) != len(set(ids)):
        dup = [tid for tid, cnt in Counter(ids).items() if cnt > 1]
        raise AssertionError(f"STOP: Duplicate trajectory IDs found: {dup}")

    # 6. Step count
    for t in trajs:
        n = len(t["steps"])
        if n != STEPS_PER_TRAJECTORY:
            raise AssertionError(
                f"STOP: Trajectory {t['trajectory_id']} has {n} steps (expected {STEPS_PER_TRAJECTORY})"
            )

    # 7. HIJACKED subtype distribution
    hijacked = [t for t in trajs if t["label"] == "HIJACKED"]
    type_a = [t for t in hijacked if t.get("hijack_subtype") == "A"
              or (t.get("precursor_type") is None and t.get("precursor_step") is None)]
    type_b = [t for t in hijacked if t not in type_a]
    if len(type_a) != 60:
        raise AssertionError(f"STOP: Expected 60 Type-A HIJACKED, found {len(type_a)}")
    if len(type_b) != 140:
        raise AssertionError(f"STOP: Expected 140 Type-B HIJACKED, found {len(type_b)}")

    # 8. Label not in step text (check step keys)
    leaked_keys = {"label", "attack_family", "attack_subtype", "injection",
                   "injection_step", "precursor_type", "precursor_step",
                   "deviation_step", "attack_success", "counterfactual_group_id"}
    for t in trajs:
        step_keys = set(t["steps"][0].keys()) if t["steps"] else set()
        overlap = leaked_keys & step_keys
        if overlap:
            raise AssertionError(
                f"STOP: Ground-truth metadata found in step keys: {overlap}"
            )

    # Summary
    print(f"  Total trajectories : {len(trajs)}")
    print(f"  Label distribution : {dict(label_counts)}")
    print(f"  Total groups       : {len(group_map)}")
    print(f"  Type-A HIJACKED    : {len(type_a)}")
    print(f"  Type-B HIJACKED    : {len(type_b)}")
    print(f"  Steps per traj     : {STEPS_PER_TRAJECTORY}")
    print(f"[DataLoader] All invariants passed.\n")


def get_group_map(trajs: list[dict]) -> dict[str, list[dict]]:
    """Return {group_id: [traj1, traj2, traj3]}."""
    group_map: dict[str, list] = {}
    for t in trajs:
        gid = t["counterfactual_group_id"]
        group_map.setdefault(gid, []).append(t)
    return group_map
