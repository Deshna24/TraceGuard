"""Track A: reproduce the frozen baseline from official artifacts (no retraining).

Loads the frozen seed-42 LSTM checkpoint with the same strict checks as the
runtime detector, runs full-trajectory and prefix inference on the official
test split, computes all Track-A metrics, and compares against the official
frozen result files. Writes only inside traceguard_x/experiments/track_a_baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import src.model_lstm as canonical_model
import src.config as canonical_config

from traceguard_x.common import data as xdata
from traceguard_x.common import metrics as xmetrics
from traceguard_x.common.experiment import (
    save_json,
    software_versions,
)
from traceguard_x.paths import (
    FROZEN_METRICS_DIR,
    FROZEN_MODELS_DIR,
    FROZEN_PREFIX_DIR,
    experiment_dir,
)

TRACK = "track_a_baseline"


def load_frozen_checkpoint(seed: int = 42) -> tuple[canonical_model.TraceGuardLSTM, dict]:
    """Load a frozen LSTM checkpoint with strict architecture verification."""
    path = FROZEN_MODELS_DIR / f"lstm_seed{seed}_best.pth"
    if not path.exists():
        raise FileNotFoundError(f"Frozen checkpoint missing: {path}")
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    config = checkpoint["model_config"]
    expected = {
        "architecture": "TraceGuardLSTM",
        "input_size": 384,
        "hidden_size": canonical_config.LSTM_HIDDEN_SIZE,
        "num_layers": canonical_config.LSTM_NUM_LAYERS,
        "dropout": canonical_config.LSTM_DROPOUT,
        "num_classes": 3,
        "bidirectional": False,
        "pooling": "last_hidden_state",
    }
    if config != expected:
        raise ValueError(
            f"Checkpoint config mismatch: {config} != {expected}; refusing to load."
        )
    model = canonical_model.TraceGuardLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        num_classes=config["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"best_epoch": checkpoint.get("best_epoch"),
                   "best_val_f1": checkpoint.get("best_val_f1")}


def predict_full(model, trajectories, embeddings_dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-trajectory inference (canonical padded-batch path, lengths all 6)."""
    from torch.utils.data import DataLoader

    ds = canonical_model.TrajectoryDataset(trajectories, embeddings_dict)
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        collate_fn=canonical_model.collate_fn)
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for embs, labels, lengths, _tids in loader:
            logits = model(embs, lengths)
            probs = torch.softmax(logits, dim=1).numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(logits.argmax(dim=1).numpy())
            y_prob.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def predict_prefixes(model, trajectories, embeddings_dict,
                     threshold: float = xmetrics.DETECTION_THRESHOLD) -> list[dict[str, Any]]:
    """Prefix k=1..6 inference replicating canonical prefix_detection logic.

    Uses only steps 1..k (no future steps); threshold inclusive.
    """
    label_inv = {i: name for i, name in enumerate(xmetrics.CLASS_NAMES)}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
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
                    "pred_label": label_inv[int(np.argmax(probs))],
                    "flagged_hijacked": bool(probs[2] >= threshold),
                    "injection_step": traj.get("injection_step"),
                    "precursor_step": traj.get("precursor_step"),
                    "deviation_step": traj.get("deviation_step"),
                    "hijack_subtype": traj.get("hijack_subtype"),
                    "counterfactual_group_id": traj.get("counterfactual_group_id"),
                })
    return rows


def compare_with_official(repro: dict, official: dict, keys: list[str]) -> dict[str, Any]:
    diffs = {}
    for key in keys:
        rv, ov = repro.get(key), official.get(key)
        if rv is None or ov is None:
            diffs[key] = {"reproduced": rv, "official": ov, "abs_diff": None}
        else:
            diffs[key] = {"reproduced": rv, "official": ov,
                          "abs_diff": abs(float(rv) - float(ov))}
    return diffs


def main(seed: int = 42) -> dict[str, Any]:
    out_dir = experiment_dir(TRACK)

    trajectories = xdata.load_dataset()
    split_info = xdata.load_split(seed)
    partitions = xdata.split_trajectories(trajectories, split_info)
    leakage = xdata.verify_no_group_leakage(trajectories, split_info)
    embeddings = xdata.load_embeddings()

    model, ckpt_info = load_frozen_checkpoint(seed)

    # ---- Full-trajectory classification ----
    test_trajs = partitions["test"]
    y_true, y_pred, y_prob = predict_full(model, test_trajs, embeddings)
    cls_metrics = xmetrics.full_classification_metrics(y_true, y_pred, y_prob)
    fprs = xmetrics.false_positive_rates(y_true, y_pred)

    # ---- Prefix / early-detection ----
    prefix_rows = predict_prefixes(model, test_trajs, embeddings)
    records = xmetrics.detection_records_from_prefixes(prefix_rows)
    latency = xmetrics.latency_statistics(records)
    false_alarms = xmetrics.benign_false_alarm_before_deviation(prefix_rows)

    # ---- Comparison with official frozen results ----
    official_metrics_path = FROZEN_METRICS_DIR / f"lstm_seed{seed}_metrics.json"
    official_latency_path = FROZEN_METRICS_DIR / f"latency_stats_lstm_seed{seed}.json"
    official_prefix_path = FROZEN_PREFIX_DIR / f"prefix_predictions_lstm_seed{seed}.json"

    comparison: dict[str, Any] = {}
    if official_metrics_path.exists():
        official = json.loads(official_metrics_path.read_text(encoding="utf-8"))
        comparison["full_metrics"] = compare_with_official(
            cls_metrics, official,
            ["accuracy", "macro_precision", "macro_recall", "macro_f1"],
        )
    if official_latency_path.exists():
        official_lat = json.loads(official_latency_path.read_text(encoding="utf-8"))
        comparison["latency"] = compare_with_official(
            latency, official_lat,
            ["n_hijacked", "n_detected", "detection_rate",
             "pre_action_detection_rate", "mean_latency", "median_latency",
             "pct_never_detected"],
        )
    if official_prefix_path.exists():
        official_prefix = json.loads(official_prefix_path.read_text(encoding="utf-8"))
        official_rows = official_prefix if isinstance(official_prefix, list) else official_prefix.get("results", [])
        if official_rows:
            key = lambda r: (r["trajectory_id"], r["prefix_length"])
            repro_sorted = sorted(prefix_rows, key=key)
            official_sorted = sorted(official_rows, key=key)
            if len(repro_sorted) == len(official_sorted) and all(
                a["trajectory_id"] == b["trajectory_id"]
                and a["prefix_length"] == b["prefix_length"]
                for a, b in zip(repro_sorted, official_sorted)
            ):
                max_diff = max(
                    max(abs(a["p_hijacked"] - b["p_hijacked"]),
                        abs(a["p_benign"] - b["p_benign"]),
                        abs(a["p_resisted"] - b["p_resisted"]))
                    for a, b in zip(repro_sorted, official_sorted)
                )
                comparison["prefix_probabilities"] = {
                    "n_rows_compared": len(repro_sorted),
                    "max_abs_prob_diff": float(max_diff),
                }

    report = {
        "label": "Reproduced baseline",
        "track": TRACK,
        "seed": seed,
        "checkpoint": ckpt_info,
        "split": {
            "train": len(partitions["train"]),
            "val": len(partitions["val"]),
            "test": len(partitions["test"]),
        },
        "group_leakage_check": leakage,
        "full_metrics": cls_metrics,
        "false_positive_rates": fprs,
        "latency": latency,
        "false_alarms_any_prefix": false_alarms,
        "comparison_with_official": comparison,
        "software": software_versions(),
    }
    save_json(report, out_dir / f"baseline_reproduction_seed{seed}.json")

    # ---- Human-readable summary ----
    lines = [
        "# Track A — Baseline Reproduction (label: Reproduced baseline)",
        "",
        f"Seed: {seed}; checkpoint best_epoch={ckpt_info.get('best_epoch')}, "
        f"best_val_f1={ckpt_info.get('best_val_f1')}",
        f"Split sizes: train={len(partitions['train'])}, val={len(partitions['val'])}, "
        f"test={len(partitions['test'])}; group leakage ok={leakage['ok']}",
        "",
        f"- Accuracy: {cls_metrics['accuracy']:.4f}",
        f"- Macro P/R/F1: {cls_metrics['macro_precision']:.4f} / "
        f"{cls_metrics['macro_recall']:.4f} / {cls_metrics['macro_f1']:.4f}",
    ]
    for cls in xmetrics.CLASS_NAMES:
        m = cls_metrics["per_class"][cls]
        lines.append(
            f"- {cls}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
            f"AUROC={m['auroc'] if m['auroc'] is None else round(m['auroc'], 4)}"
        )
    lines += [
        f"- FPR (BENIGN): {fprs['BENIGN']:.4f}; FPR (INJECTION_RESISTED): {fprs['INJECTION_RESISTED']:.4f}",
        f"- Detection rate: {latency.get('detection_rate')}; pre-action rate: "
        f"{latency.get('pre_action_detection_rate')}; mean latency: {latency.get('mean_latency')}",
        f"- Never detected: {latency.get('pct_never_detected')}",
        "",
        "## Comparison with official frozen results",
    ]
    for section, diffs in comparison.items():
        if section == "prefix_probabilities":
            for key, value in diffs.items():
                lines.append(f"- {section}/{key}: {value}")
            continue
        for key, d in diffs.items():
            lines.append(f"- {section}/{key}: reproduced={d['reproduced']}, "
                         f"official={d['official']}, abs_diff={d['abs_diff']}")
    (out_dir / f"baseline_summary_seed{seed}.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)
