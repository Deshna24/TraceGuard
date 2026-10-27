"""Track B1: Gated Recurrent Unit (GRU) experiment.

Trains and evaluates TraceGuardGRU on the canonical embeddings and splits,
measuring parameter count, training stability, inference latency, full classification
metrics, and prefix detection latency.
Writes exclusively inside ``traceguard_x/experiments/track_b1_gru/``.
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
from traceguard_x.common import inference as xinf
from traceguard_x.common import metrics as xmetrics
from traceguard_x.common.experiment import (
    save_experiment_metadata,
    save_json,
    set_seed,
)
from traceguard_x.common.training import train_model
from traceguard_x.models.gru import TraceGuardGRU
from traceguard_x.paths import experiment_dir

TRACK = "track_b1_gru"


def run_gru_seed(seed: int = 42, max_epochs: int = 50, patience: int = 7) -> dict[str, Any]:
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

    model = TraceGuardGRU(input_size=384, hidden_size=128, num_layers=2, dropout=0.2)
    param_counts = model.count_parameters()

    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_save_dir=out_dir,
        model_tag=f"gru_seed{seed}",
        max_epochs=max_epochs,
        patience=patience,
    )

    # Full trajectory evaluation on test set
    y_true, y_pred, y_prob = xinf.predict_full(model, partitions["test"], embeddings)
    full_m = xmetrics.full_classification_metrics(y_true, y_pred, y_prob)
    fpr = xmetrics.false_positive_rates(y_true, y_pred)

    # Prefix inference on test set (k=1..6)
    t_start = time.perf_counter()
    prefix_rows = xinf.predict_prefixes(model, partitions["test"], embeddings)
    inference_time = (time.perf_counter() - t_start) / len(prefix_rows) * 1000  # ms per prefix

    detection_records = xmetrics.detection_records_from_prefixes(prefix_rows)
    latency_stats = xmetrics.latency_statistics(detection_records)

    # Save prefix predictions CSV
    df_prefix = pd.DataFrame(prefix_rows)
    csv_path = out_dir / f"gru_seed{seed}_prefix_predictions.csv"
    df_prefix.to_csv(csv_path, index=False)

    summary = {
        "experiment_id": f"gru_seed{seed}",
        "track": TRACK,
        "seed": seed,
        "architecture": "TraceGuardGRU",
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
    }

    metrics_path = out_dir / f"gru_seed{seed}_metrics.json"
    save_json(summary, metrics_path)

    # Save metadata
    save_experiment_metadata(
        experiment_dir=out_dir,
        experiment_id=f"gru_seed{seed}",
        model_architecture="TraceGuardGRU",
        hyperparameters={
            "input_size": 384,
            "hidden_size": 128,
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

    # Generate markdown summary
    md = [
        f"# Track B1 — GRU Experiment (Seed {seed})",
        "",
        "**Label:** New experimental result",
        "",
        f"- Parameters: {param_counts['total']:,} (recurrent: {param_counts['recurrent']:,})",
        f"- Best epoch: {train_result['best_epoch']} (val Macro F1: {train_result['best_val_f1']:.4f})",
        f"- Test Accuracy: {full_m['accuracy']:.4f}",
        f"- Test Macro F1: {full_m['macro_f1']:.4f}",
        f"- HIJACKED F1: {full_m['per_class']['HIJACKED']['f1']:.4f} (AUROC: {full_m['per_class']['HIJACKED']['auroc']:.4f})",
        f"- Pre-action detection rate: {latency_stats['pre_action_detection_rate'] * 100:.2f}%",
        f"- Mean detection latency: {latency_stats['mean_latency']:.2f} steps (median: {latency_stats['median_latency']:.1f})",
        f"- Never detected rate: {latency_stats['pct_never_detected'] * 100:.2f}%",
        f"- FPR Benign: {fpr['BENIGN']:.4f}, FPR Resisted: {fpr['INJECTION_RESISTED']:.4f}",
        f"- Prefix inference latency: {inference_time:.3f} ms / prefix",
    ]
    md_path = out_dir / f"gru_summary_seed{seed}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md))
    return summary


def run_all_seeds() -> dict[str, Any]:
    seeds = [42, 123, 456, 789, 1011]
    results = [run_gru_seed(s) for s in seeds]
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
    save_json(summary_5seeds, out_dir / "gru_five_seeds_robustness.json")
    print("\n=== GRU 5-Seed Robustness Summary ===")
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
        run_gru_seed(args.seed)
