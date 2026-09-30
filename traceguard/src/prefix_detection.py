"""
prefix_detection.py - Prefix-based (online) inference for TRACEGUARD.

For each test trajectory, evaluate prefixes of length 1..6.
NEVER use future steps. LSTM is unidirectional (causal).
"""

import csv, json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.config import (
    INV_LABEL_MAP, LABEL_MAP, CLASS_NAMES,
    STEPS_PER_TRAJECTORY, PREFIX_DIR, DETECTION_THRESHOLD,
)
from src.utils import save_json, get_device


def run_prefix_evaluation(
    model,
    trajectories: list[dict],
    embeddings_dict: dict,
    device=None,
    threshold: float = DETECTION_THRESHOLD,
    tag: str = "test",
) -> list[dict]:
    """
    For each trajectory, evaluate LSTM on prefixes [e1], [e1,e2], ..., [e1..e6].
    NEVER uses future steps.

    Returns:
        List of per-prefix result dicts.
    """
    if device is None:
        device = get_device()

    model.eval()
    results = []

    with torch.no_grad():
        for traj in tqdm(trajectories, desc="[Prefix] Evaluating"):
            tid = traj["trajectory_id"]
            emb = embeddings_dict[tid]          # [6, D]
            n   = emb.shape[0]

            assert n == STEPS_PER_TRAJECTORY, (
                f"Trajectory {tid} has {n} steps, expected {STEPS_PER_TRAJECTORY}"
            )

            for k in range(1, n + 1):
                # Use only steps 1..k — NO FUTURE STEPS
                prefix_emb = emb[:k]            # [k, D]
                x = torch.tensor(prefix_emb, dtype=torch.float32).unsqueeze(0).to(device)
                lengths = [k]

                logits = model(x, lengths)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()  # [3]

                results.append({
                    "trajectory_id":         tid,
                    "true_label":            traj["label"],
                    "prefix_length":         k,
                    "p_benign":              float(probs[0]),
                    "p_resisted":            float(probs[1]),
                    "p_hijacked":            float(probs[2]),
                    "pred_label":            INV_LABEL_MAP[int(np.argmax(probs))],
                    "flagged_hijacked":      float(probs[2]) >= threshold,
                    # metadata (for analysis — NOT used as model input)
                    "injection_step":        traj.get("injection_step"),
                    "precursor_step":        traj.get("precursor_step"),
                    "deviation_step":        traj.get("deviation_step"),
                    "hijack_subtype":        traj.get("hijack_subtype"),
                    "counterfactual_group_id": traj.get("counterfactual_group_id"),
                })

    # Save CSV
    _save_prefix_csv(results, tag)
    # Save JSON
    save_json(results, PREFIX_DIR / f"prefix_predictions_{tag}.json")
    print(f"[Prefix] Saved {len(results)} prefix rows for {len(trajectories)} trajectories.")
    return results


def _save_prefix_csv(results: list[dict], tag: str):
    path = PREFIX_DIR / f"prefix_predictions_{tag}.csv"
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[Prefix] CSV saved: {path}")
