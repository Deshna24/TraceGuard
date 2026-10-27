"""Track B3: Multi-task Early-Warning Experiment for TRACEGUARD-X.

Jointly trains a 3-class classifier and an auxiliary early-warning head.
The auxiliary head predicts whether a behavioral deviation occurs within
the trajectory, supervised strictly from documented trajectory metadata
(deviation_step) without leaking future step tokens.

RESEARCH SEPARATION NOTICE:
The auxiliary warning score is an experimental indicator and is NEVER substituted
for the official P(HIJACKED) safety gate probability.
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
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

from traceguard_x.common import data as xdata
from traceguard_x.common import metrics as xmetrics
from traceguard_x.common.experiment import (
    save_experiment_metadata,
    save_json,
    set_seed,
)
from traceguard_x.common.training import (
    MultiTaskTrajectoryDataset,
    multitask_collate_fn,
    train_model,
)
from traceguard_x.models.multitask import TraceGuardMultiTaskEarlyWarning
from traceguard_x.paths import experiment_dir

TRACK = "track_b3_earlywarn"


def predict_prefixes_multitask(
    model: TraceGuardMultiTaskEarlyWarning,
    trajectories: list[dict[str, Any]],
    embeddings_dict: dict[str, np.ndarray],
    threshold: float = xmetrics.DETECTION_THRESHOLD,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for traj in trajectories:
            tid = traj["trajectory_id"]
            emb = embeddings_dict[tid]
            dev_step = traj.get("deviation_step")

            for k in range(1, emb.shape[0] + 1):
                x = torch.tensor(emb[:k], dtype=torch.float32).unsqueeze(0)
                main_logits, warn_logit = model(x, [k])
                probs = torch.softmax(main_logits, dim=1)[0].numpy()
                p_warn = float(torch.sigmoid(warn_logit)[0].item())

                # Ground truth for auxiliary: is deviation imminent within 2 steps?
                # (k < dev_step <= k + 2)
                imminent_h2 = bool(dev_step is not None and k < dev_step <= k + 2)

                rows.append({
                    "trajectory_id": tid,
                    "true_label": traj["label"],
                    "prefix_length": k,
                    "p_benign": float(probs[0]),
                    "p_resisted": float(probs[1]),
                    "p_hijacked": float(probs[2]),
                    "p_warning_aux": p_warn,
                    "imminent_deviation_h2_true": imminent_h2,
                    "pred_label": xmetrics.CLASS_NAMES[int(np.argmax(probs))],
                    "flagged_hijacked": bool(probs[2] >= threshold),
                    "injection_step": traj.get("injection_step"),
                    "precursor_step": traj.get("precursor_step"),
                    "deviation_step": dev_step,
                    "hijack_subtype": traj.get("hijack_subtype"),
                    "counterfactual_group_id": traj.get("counterfactual_group_id"),
                })

    return rows


def run_multitask_seed(
    seed: int = 42, max_epochs: int = 50, patience: int = 7, aux_weight: float = 0.5
) -> dict[str, Any]:
    out_dir = experiment_dir(TRACK)
    set_seed(seed)

    dataset = xdata.load_dataset()
    split_info = xdata.load_split(seed)
    partitions = xdata.split_trajectories(dataset, split_info)
    embeddings = xdata.load_embeddings()

    leak_report = xdata.verify_no_group_leakage(dataset, split_info)
    if not leak_report["ok"]:
        raise ValueError(f"Group leakage detected in split seed {seed}: {leak_report}")

    train_ds = MultiTaskTrajectoryDataset(partitions["train"], embeddings)
    val_ds = MultiTaskTrajectoryDataset(partitions["val"], embeddings)
    test_ds = MultiTaskTrajectoryDataset(partitions["test"], embeddings)

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=multitask_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, collate_fn=multitask_collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, collate_fn=multitask_collate_fn
    )

    model = TraceGuardMultiTaskEarlyWarning(
        input_size=384, hidden_size=128, num_layers=2, dropout=0.2
    )
    param_counts = model.count_parameters()

    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_save_dir=out_dir,
        model_tag=f"multitask_seed{seed}",
        max_epochs=max_epochs,
        patience=patience,
        is_multitask=True,
        aux_weight=aux_weight,
    )

    # Full trajectory evaluation
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for embs, labels, _, lengths, _ in test_loader:
            main_logits, _ = model(embs, lengths)
            probs = torch.softmax(main_logits, dim=1).numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(main_logits.argmax(dim=1).numpy())
            y_prob.extend(probs)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    full_m = xmetrics.full_classification_metrics(y_true, y_pred, y_prob)
    fpr = xmetrics.false_positive_rates(y_true, y_pred)

    # Prefix inference with auxiliary warning scores
    t_start = time.perf_counter()
    prefix_rows = predict_prefixes_multitask(model, partitions["test"], embeddings)
    inference_time = (time.perf_counter() - t_start) / len(prefix_rows) * 1000

    detection_records = xmetrics.detection_records_from_prefixes(prefix_rows)
    latency_stats = xmetrics.latency_statistics(detection_records)

    # Auxiliary head evaluation on prefix data
    df_prefix = pd.DataFrame(prefix_rows)
    csv_path = out_dir / f"multitask_seed{seed}_prefix_predictions.csv"
    df_prefix.to_csv(csv_path, index=False)

    y_imminent = df_prefix["imminent_deviation_h2_true"].values.astype(int)
    p_warn = df_prefix["p_warning_aux"].values
    if len(np.unique(y_imminent)) > 1:
        aux_auroc = float(roc_auc_score(y_imminent, p_warn))
        aux_pr_auc = float(average_precision_score(y_imminent, p_warn))
    else:
        aux_auroc = None
        aux_pr_auc = None

    # Mean warning score on hijacked vs benign/resisted prefixes
    hijacked_warn_mean = float(df_prefix[df_prefix["true_label"] == "HIJACKED"]["p_warning_aux"].mean())
    benign_warn_mean = float(df_prefix[df_prefix["true_label"] == "BENIGN"]["p_warning_aux"].mean())
    resisted_warn_mean = float(df_prefix[df_prefix["true_label"] == "INJECTION_RESISTED"]["p_warning_aux"].mean())

    summary = {
        "experiment_id": f"multitask_seed{seed}",
        "track": TRACK,
        "seed": seed,
        "architecture": "TraceGuardMultiTaskEarlyWarning",
        "parameter_counts": param_counts,
        "training": {
            "best_epoch": train_result["best_epoch"],
            "best_val_f1": train_result["best_val_f1"],
            "training_time_seconds": train_result["training_time_seconds"],
            "aux_loss_weight": aux_weight,
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
        "auxiliary_head_metrics": {
            "imminent_deviation_auroc": aux_auroc,
            "imminent_deviation_pr_auc": aux_pr_auc,
            "mean_warning_hijacked": hijacked_warn_mean,
            "mean_warning_benign": benign_warn_mean,
            "mean_warning_resisted": resisted_warn_mean,
            "supervision_method": "derived from documented deviation_step metadata without leaking future tokens",
        },
        "inference_latency_ms_per_prefix": inference_time,
    }

    metrics_path = out_dir / f"multitask_seed{seed}_metrics.json"
    save_json(summary, metrics_path)

    save_experiment_metadata(
        experiment_dir=out_dir,
        experiment_id=f"multitask_seed{seed}",
        model_architecture="TraceGuardMultiTaskEarlyWarning",
        hyperparameters={
            "input_size": 384,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "aux_weight": aux_weight,
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
        f"# Track B3 — Multi-Task Early Warning Experiment (Seed {seed})",
        "",
        "**Label:** New experimental result",
        "",
        f"- Parameters: {param_counts['total']:,} (encoder: {param_counts['encoder']:,}, main: {param_counts['main_head']}, warn: {param_counts['warning_head']})",
        f"- Best epoch: {train_result['best_epoch']} (val Macro F1: {train_result['best_val_f1']:.4f})",
        f"- Test Accuracy: {full_m['accuracy']:.4f}",
        f"- Test Macro F1: {full_m['macro_f1']:.4f}",
        f"- HIJACKED F1: {full_m['per_class']['HIJACKED']['f1']:.4f} (AUROC: {full_m['per_class']['HIJACKED']['auroc']:.4f})",
        f"- Pre-action detection rate: {latency_stats['pre_action_detection_rate'] * 100:.2f}%",
        f"- Mean detection latency: {latency_stats['mean_latency']:.2f} steps (median: {latency_stats['median_latency']:.1f})",
        f"- Never detected rate: {latency_stats['pct_never_detected'] * 100:.2f}%",
        f"- Auxiliary Imminent-Deviation AUROC: {aux_auroc:.4f}" if aux_auroc else "- Auxiliary AUROC: N/A",
        f"- Mean warning score: HIJACKED={hijacked_warn_mean:.4f}, RESISTED={resisted_warn_mean:.4f}, BENIGN={benign_warn_mean:.4f}",
        f"- Prefix inference latency: {inference_time:.3f} ms / prefix",
        "",
        "> [!IMPORTANT]",
        "> The auxiliary early-warning score is an experimental research indicator. It is strictly separated from the official P(HIJACKED) safety gate decision.",
    ]
    md_path = out_dir / f"multitask_summary_seed{seed}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    args = parser.parse_args()
    run_multitask_seed(seed=args.seed, aux_weight=args.aux_weight)
