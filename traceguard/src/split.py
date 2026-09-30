import json, numpy as np
from pathlib import Path
from collections import Counter
from src.config import (
    TRAIN_GROUPS, VAL_GROUPS, TEST_GROUPS,
    TRAIN_TRAJ, VAL_TRAJ, TEST_TRAJ, SPLITS_DIR,
)
from src.utils import save_json


def create_group_aware_split(trajs, seed=42):
    group_map = {}
    for t in trajs:
        gid = t["counterfactual_group_id"]
        group_map.setdefault(gid, []).append(t)

    type_a_groups, type_b_groups = [], []
    for gid in sorted(group_map.keys()):
        members = group_map[gid]
        hj = next((t for t in members if t["label"] == "HIJACKED"), None)
        subtype = hj.get("hijack_subtype", "A") if hj else "A"
        (type_a_groups if subtype == "A" else type_b_groups).append(gid)

    rng = np.random.default_rng(seed)
    rng.shuffle(type_a_groups)
    rng.shuffle(type_b_groups)

    a_total = len(type_a_groups)
    b_total = len(type_b_groups)
    a_test  = round(a_total * TEST_GROUPS / 200)
    a_val   = round(a_total * VAL_GROUPS  / 200)
    a_train = a_total - a_test - a_val
    b_test  = TEST_GROUPS - a_test
    b_val   = VAL_GROUPS  - a_val
    b_train = b_total - b_test - b_val

    assert a_train + b_train == TRAIN_GROUPS
    assert a_val   + b_val   == VAL_GROUPS
    assert a_test  + b_test  == TEST_GROUPS

    test_ids  = np.array(type_a_groups[:a_test]  + type_b_groups[:b_test])
    val_ids   = np.array(type_a_groups[a_test:a_test+a_val] + type_b_groups[b_test:b_test+b_val])
    train_ids = np.array(type_a_groups[a_test+a_val:] + type_b_groups[b_test+b_val:])

    def g2t(gids):
        out = []
        for gid in gids:
            out.extend(group_map[gid])
        return out

    train_trajs = g2t(train_ids)
    val_trajs   = g2t(val_ids)
    test_trajs  = g2t(test_ids)
    _verify_split(train_ids, val_ids, test_ids, train_trajs, val_trajs, test_trajs)

    split_info = {
        "seed":             seed,
        "train_group_ids":  sorted(train_ids.tolist()),
        "val_group_ids":    sorted(val_ids.tolist()),
        "test_group_ids":   sorted(test_ids.tolist()),
        "train_n_groups":   len(train_ids),
        "val_n_groups":     len(val_ids),
        "test_n_groups":    len(test_ids),
        "train_n_trajs":    len(train_trajs),
        "val_n_trajs":      len(val_trajs),
        "test_n_trajs":     len(test_trajs),
        "train_label_dist": dict(Counter(t["label"] for t in train_trajs)),
        "val_label_dist":   dict(Counter(t["label"] for t in val_trajs)),
        "test_label_dist":  dict(Counter(t["label"] for t in test_trajs)),
    }
    return train_trajs, val_trajs, test_trajs, split_info


def _verify_split(train_ids, val_ids, test_ids, train_trajs, val_trajs, test_trajs):
    ts, vs, ss = set(train_ids), set(val_ids), set(test_ids)
    if (ts & vs) or (ts & ss) or (vs & ss):
        raise AssertionError(f"STOP: Group leakage! tv={ts&vs} tt={ts&ss} vt={vs&ss}")
    print("[Split] Leakage check passed.")
    assert len(train_ids)   == TRAIN_GROUPS, f"train groups: {len(train_ids)}"
    assert len(val_ids)     == VAL_GROUPS,   f"val groups: {len(val_ids)}"
    assert len(test_ids)    == TEST_GROUPS,  f"test groups: {len(test_ids)}"
    assert len(train_trajs) == TRAIN_TRAJ,   f"train trajs: {len(train_trajs)}"
    assert len(val_trajs)   == VAL_TRAJ,     f"val trajs: {len(val_trajs)}"
    assert len(test_trajs)  == TEST_TRAJ,    f"test trajs: {len(test_trajs)}"
    print(f"[Split] Train:{len(train_ids)}gr/{len(train_trajs)}tr | "
          f"Val:{len(val_ids)}gr/{len(val_trajs)}tr | "
          f"Test:{len(test_ids)}gr/{len(test_trajs)}tr")
    for name, sp in [("Train", train_trajs), ("Val", val_trajs), ("Test", test_trajs)]:
        dist = Counter(t["label"] for t in sp)
        print(f"  {name} label dist: {dict(dist)}")


def save_split(split_info, seed):
    path = SPLITS_DIR / f"split_seed{seed}.json"
    save_json(split_info, path)
    print(f"[Split] Saved split info: {path}")
    return path
