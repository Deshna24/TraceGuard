"""Shared inference helpers for TRACEGUARD-X experiments.

The prefix loop replicates canonical ``prefix_detection.py`` semantics
(prefixes k=1..6, no future steps, inclusive threshold). Any torch model with
``forward(x, lengths) -> logits`` works.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from traceguard_x.common.metrics import CLASS_NAMES, DETECTION_THRESHOLD

_LABEL_INV = {i: name for i, name in enumerate(CLASS_NAMES)}


def predict_full(
    model: torch.nn.Module,
    trajectories: list[dict[str, Any]],
    embeddings_dict: dict[str, np.ndarray],
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-trajectory inference; returns (y_true, y_pred, y_prob)."""
    import src.model_lstm as canonical_model

    ds = canonical_model.TrajectoryDataset(trajectories, embeddings_dict)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=canonical_model.collate_fn)
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.inference_mode():
        for embs, labels, lengths, _tids in loader:
            logits = model(embs, lengths)
            probs = torch.softmax(logits, dim=1).numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(logits.argmax(dim=1).numpy())
            y_prob.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def predict_prefixes(
    model: torch.nn.Module,
    trajectories: list[dict[str, Any]],
    embeddings_dict: dict[str, np.ndarray],
    threshold: float = DETECTION_THRESHOLD,
) -> list[dict[str, Any]]:
    """Prefix k=1..6 inference; uses only steps 1..k (no future information)."""
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for traj in trajectories:
            tid = traj["trajectory_id"]
            emb = embeddings_dict[tid]
            for k in range(1, emb.shape[0] + 1):
                x = torch.tensor(emb[:k], dtype=torch.float32).unsqueeze(0)
                logits = model(x, [k])
                probs = torch.softmax(logits, dim=1)[0].numpy()
                rows.append({
                    "trajectory_id": tid,
                    "true_label": traj["label"],
                    "prefix_length": k,
                    "p_benign": float(probs[0]),
                    "p_resisted": float(probs[1]),
                    "p_hijacked": float(probs[2]),
                    "pred_label": _LABEL_INV[int(np.argmax(probs))],
                    "flagged_hijacked": bool(probs[2] >= threshold),
                    "injection_step": traj.get("injection_step"),
                    "precursor_step": traj.get("precursor_step"),
                    "deviation_step": traj.get("deviation_step"),
                    "hijack_subtype": traj.get("hijack_subtype"),
                    "counterfactual_group_id": traj.get("counterfactual_group_id"),
                })
    return rows


def predict_prefixes_multitask(
    model: torch.nn.Module,
    trajectories: list[dict[str, Any]],
    embeddings_dict: dict[str, np.ndarray],
    threshold: float = DETECTION_THRESHOLD,
) -> list[dict[str, Any]]:
    """Prefix inference for multi-task models whose forward returns
    ``(logits, warning_logit)``. Adds a ``p_warning`` column; the HIJACKED
    probabilities remain the official 3-class output."""
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for traj in trajectories:
            tid = traj["trajectory_id"]
            emb = embeddings_dict[tid]
            for k in range(1, emb.shape[0] + 1):
                x = torch.tensor(emb[:k], dtype=torch.float32).unsqueeze(0)
                logits, warn_logit = model(x, [k])
                probs = torch.softmax(logits, dim=1)[0].numpy()
                p_warn = float(torch.sigmoid(warn_logit)[0].item())
                rows.append({
                    "trajectory_id": tid,
                    "true_label": traj["label"],
                    "prefix_length": k,
                    "p_benign": float(probs[0]),
                    "p_resisted": float(probs[1]),
                    "p_hijacked": float(probs[2]),
                    "p_warning": p_warn,
                    "pred_label": _LABEL_INV[int(np.argmax(probs))],
                    "flagged_hijacked": bool(probs[2] >= threshold),
                    "injection_step": traj.get("injection_step"),
                    "precursor_step": traj.get("precursor_step"),
                    "deviation_step": traj.get("deviation_step"),
                    "hijack_subtype": traj.get("hijack_subtype"),
                    "counterfactual_group_id": traj.get("counterfactual_group_id"),
                })
    return rows
