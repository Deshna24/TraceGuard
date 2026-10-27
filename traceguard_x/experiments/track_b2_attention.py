"""Track B2: Temporal Attention experiment for TRACEGUARD-X.

Trains and evaluates TraceGuardTemporalAttention on canonical embeddings and splits.
Captures step-level attention distributions for prefix inference,
respects causal sequence masking, and outputs full classification + latency metrics.
Writes exclusively inside ``traceguard_x/experiments/track_b2_attention/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.model_lstm as canonical_model
from traceguard_x.common import data as xdata
from traceguard_x.common import metrics as xmetrics
from traceguard_x.common.experiment import (
    save_experiment_metadata,
    save_json,
    set_seed,
)
from traceguard_x.common.training import train_model
from traceguard_x.models.attention import TraceGuardTemporalAttention
from traceguard_x.paths import experiment_dir

TRACK = "track_b2_attention"


def predict_prefixes_with_attention(
    model: TraceGuardTemporalAttention,
    trajectories: list[dict[str, Any]],
    embeddings_dict: dict[str, np.ndarray],
    threshold: float = xmetrics.DETECTION_THRESHOLD,
) -> tuple[list[dict[str, Any]], dict[str, list[list[float]]]]:
    """Run prefix inference and collect both prefix predictions and attention distributions."""
    model.eval()
    rows: list[dict[str, Any]] = []
    attn_dict: dict[str, list[list[float]]] = {}

    with torch.no_grad():
        for traj in trajectories:
            tid = traj["trajectory_id"]
            emb = embeddings_dict[tid]
            traj_attns: list[list[float]] = []

            for k in range(1, emb.shape[0] + 1):
                x = torch.tensor(emb[:k], dtype=torch.float32).unsqueeze(0)
                logits, attn = model(x, [k], return_attention=True)
                probs = torch.softmax(logits, dim=1)[0].numpy()
                attn_weights = attn[0, :k].numpy().tolist()
                traj_attns.append(attn_weights)

                rows.append({
                    "trajectory_id": tid,
                    "true_label": traj["label"],
                    "prefix_length": k,
                    "p_benign": float(probs[0]),
                    "p_resisted": float(probs[1]),
                    "p_hijacked": float(probs[2]),
                    "pred_label": xmetrics.CLASS_NAMES[int(np.argmax(probs))],
                    "flagged_hijacked": bool(probs[2] >= threshold),
                    "injection_step": traj.get("injection_step"),
                    "precursor_step": traj.get("precursor_step"),
                    "deviation_step": traj.get("deviation_step"),
                    "hijack_subtype": traj.get("hijack_subtype"),
                    "counterfactual_group_id": traj.get("counterfactual_group_id"),
                    "attention_step_weights": attn_weights,
                })
            attn_dict[tid] = traj_attns

    return rows, attn_dict


def run_attention_seed(seed: int = 42, max_epochs: int = 50, patience: int = 7) -> dict[str, Any]:
    out_dir = experiment_dir(TRACK)
    set_seed(seed)

    dataset = xdata.load_dataset()
    split_info = xdata.load_split(seed)
    partitions = xdata.split_trajectories(dataset, split_info)
    embeddings = xdata.load_embeddings()

    leak_report = xdata.verify_no_group_leakage(dataset, split_info)
    if not leak_report["ok"]:
        raise ValueError(f"Group leakage detected in split seed {seed}: {leak_report}")

    train_ds = canonical_model.TrajectoryDataset(partitions["train"], embeddings)
    val_ds = canonical_model.TrajectoryDataset(partitions["val"], embeddings)
    test_ds = canonical_model.TrajectoryDataset(partitions["test"], embeddings)

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=canonical_model.collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, collate_fn=canonical_model.collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, collate_fn=canonical_model.collate_fn
    )

    model = TraceGuardTemporalAttention(
        input_size=384, hidden_size=128, attention_dim=64, num_layers=2, dropout=0.2
    )
    param_counts = model.count_parameters()

    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_save_dir=out_dir,
        model_tag=f"attention_seed{seed}",
        max_epochs=max_epochs,
        patience=patience,
    )

    # Full trajectory evaluation
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for embs, labels, lengths, _ in test_loader:
            logits = model(embs, lengths)
            probs = torch.softmax(logits, dim=1).numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(logits.argmax(dim=1).numpy())
            y_prob.extend(probs)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    full_m = xmetrics.full_classification_metrics(y_true, y_pred, y_prob)
    fpr = xmetrics.false_positive_rates(y_true, y_pred)

    # Prefix inference with attention extraction
    t_start = time.perf_counter()
    prefix_rows, attn_dict = predict_prefixes_with_attention(
        model, partitions["test"], embeddings
    )
    inference_time = (time.perf_counter() - t_start) / len(prefix_rows) * 1000

    detection_records = xmetrics.detection_records_from_prefixes(prefix_rows)
    latency_stats = xmetrics.latency_statistics(detection_records)

    # Save prefix predictions and attention distributions
    df_prefix = pd.DataFrame([
        {k: v for k, v in r.items() if k != "attention_step_weights"}
        for r in prefix_rows
    ])
    csv_path = out_dir / f"attention_seed{seed}_prefix_predictions.csv"
    df_prefix.to_csv(csv_path, index=False)

    attn_path = out_dir / f"attention_seed{seed}_weights.json"
    save_json(attn_dict, attn_path)

    summary = {
        "experiment_id": f"attention_seed{seed}",
        "track": TRACK,
        "seed": seed,
        "architecture": "TraceGuardTemporalAttention",
        "parameter_counts": param_counts,
        "training": {
            "best_epoch": train_result["best_epoch"],
            "best_val_f1": train_result["best_val_f1"],
            "training_time_seconds": train_result["training_time_seconds"],
        },
        "test_metrics": {
            "accuracy": full_m["accuracy"],
            "macro_precision": full_m["macro_precision"],
            "macro_recall": full_m["macro_recall"],
            "macro_f1": full_m["macro_f1"],
            "per_class": full_m["per_class"],
            "confusion_matrix": full_m["confusion_matrix"],
            "false_positive_rates": fpr,
        },
        "latency_metrics": latency_stats,
        "inference_latency_ms_per_prefix": inference_time,
        "attention_diagnostics_notice": "Attention weights are diagnostic representation pooling indicators, NOT causal explanations.",
    }

    metrics_path = out_dir / f"attention_seed{seed}_metrics.json"
    save_json(summary, metrics_path)

    save_experiment_metadata(
        experiment_dir=out_dir,
        experiment_id=f"attention_seed{seed}",
        model_architecture="TraceGuardTemporalAttention",
        hyperparameters={
            "input_size": 384,
            "hidden_size": 128,
            "attention_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
        },
        seed=seed,
        dataset_version="v4.1",
        split_id=f"split_seed{seed}",
        feature_representation="all-MiniLM-L6-v2 (384-dim unnormalized)",
    )

    md = [
        f"# Track B2 — Temporal Attention Experiment (Seed {seed})",
        "",
        "**Label:** New experimental result",
        "",
        f"- Parameters: {param_counts['total']:,} (encoder: {param_counts['encoder']:,}, attention: {param_counts['attention']:,})",
        f"- Best epoch: {train_result['best_epoch']} (val Macro F1: {train_result['best_val_f1']:.4f})",
        f"- Test Accuracy: {full_m['accuracy']:.4f}",
        f"- Test Macro F1: {full_m['macro_f1']:.4f}",
        f"- HIJACKED F1: {full_m['per_class']['HIJACKED']['f1']:.4f} (AUROC: {full_m['per_class']['HIJACKED']['auroc']:.4f})",
        f"- Pre-action detection rate: {latency_stats['pre_action_detection_rate'] * 100:.2f}%",
        f"- Mean detection latency: {latency_stats['mean_latency']:.2f} steps (median: {latency_stats['median_latency']:.1f})",
        f"- Never detected rate: {latency_stats['pct_never_detected'] * 100:.2f}%",
        f"- FPR Benign: {fpr['BENIGN']:.4f}, FPR Resisted: {fpr['INJECTION_RESISTED']:.4f}",
        f"- Prefix inference latency: {inference_time:.3f} ms / prefix",
        "",
        "> [!NOTE]",
        "> Attention weights reflect how the pooling layer weights sequence representations. They do not constitute causal proof of model decisions.",
    ]
    md_path = out_dir / f"attention_summary_seed{seed}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md))
    return summary


def run_all_seeds() -> dict[str, Any]:
    seeds = [42, 123, 456, 789, 1011]
    results = [run_attention_seed(s) for s in seeds]
    f1s = [r["test_metrics"]["macro_f1"] for r in results]
    accs = [r["test_metrics"]["accuracy"] for r in results]
    pre_actions = [r["latency_metrics"]["pre_action_detection_rate"] for r in results]
    latencies = [r["latency_metrics"]["mean_latency"] for r in results]

    summary_5seeds = {
        "seeds": seeds,
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s, ddof=1)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)),
        "pre_action_mean": float(np.mean(pre_actions)),
        "mean_latency_mean": float(np.mean(latencies)),
        "per_seed_results": results,
    }

    out_dir = experiment_dir(TRACK)
    save_json(summary_5seeds, out_dir / "attention_five_seeds_robustness.json")
    print("\n=== Temporal Attention 5-Seed Robustness Summary ===")
    print(f"Macro F1: {summary_5seeds['macro_f1_mean']:.4f} ± {summary_5seeds['macro_f1_std']:.4f}")
    print(f"Accuracy: {summary_5seeds['accuracy_mean']:.4f} ± {summary_5seeds['accuracy_std']:.4f}")
    return summary_5seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()

    if args.all_seeds:
        run_all_seeds()
    else:
        run_attention_seed(args.seed)
