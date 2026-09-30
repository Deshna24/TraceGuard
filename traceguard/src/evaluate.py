"""
evaluate.py - Full trajectory evaluation for TRACEGUARD LSTM.
Reports accuracy, per-class + macro metrics, confusion matrix, AUROC, PR-AUC.
"""

import json, csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score,
    classification_report,
)
from src.config import (
    INV_LABEL_MAP, CLASS_NAMES, LABEL_MAP,
    METRICS_DIR, CM_DIR,
)
from src.utils import save_json, get_device


def predict_full(model, loader, device=None) -> tuple:
    """Run model over all batches, return (y_true, y_pred, y_prob, traj_ids)."""
    if device is None:
        device = get_device()
    model.eval()
    y_true, y_pred, y_prob, tids = [], [], [], []
    with torch.no_grad():
        for embs, labels, lengths, traj_ids in loader:
            embs = embs.to(device)
            logits = model(embs, lengths)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
            tids.extend(traj_ids)
    return np.array(y_true), np.array(y_pred), np.array(y_prob), tids


def compute_full_metrics(y_true, y_pred, y_prob, tag: str = "lstm_test") -> dict:
    """Compute and save all classification metrics."""
    acc = accuracy_score(y_true, y_pred)

    prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # AUROC — one-vs-rest
    auroc = {}
    pr_auc = {}
    for i, cls in enumerate(CLASS_NAMES):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
            auroc[cls]  = float(roc_auc_score(y_bin, y_prob[:, i]))
            pr_auc[cls] = float(average_precision_score(y_bin, y_prob[:, i]))
        else:
            auroc[cls]  = None
            pr_auc[cls] = None

    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        per_class[cls] = {
            "precision": float(prec_per[i]),
            "recall":    float(rec_per[i]),
            "f1":        float(f1_per[i]),
            "support":   int(sup_per[i]),
            "auroc":     auroc[cls],
            "pr_auc":    pr_auc[cls],
        }

    metrics = {
        "accuracy":        float(acc),
        "macro_precision": float(prec_mac),
        "macro_recall":    float(rec_mac),
        "macro_f1":        float(f1_mac),
        "per_class":       per_class,
    }

    save_json(metrics, METRICS_DIR / f"{tag}_metrics.json")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Full-Trajectory Results: {tag}")
    print(f"{'='*60}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Macro Precision: {prec_mac:.4f}")
    print(f"  Macro Recall   : {rec_mac:.4f}")
    print(f"  Macro F1       : {f1_mac:.4f}")
    print()
    for cls in CLASS_NAMES:
        m = per_class[cls]
        auroc_str = f"{m['auroc']:.4f}" if m['auroc'] is not None else "N/A"
        print(f"  {cls:20s}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  AUROC={auroc_str}")
    print()

    return metrics


def plot_confusion_matrix(y_true, y_pred, tag: str = "lstm_test"):
    """Save labelled confusion matrix PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"TRACEGUARD LSTM — Confusion Matrix\n({tag})", fontsize=13)
    plt.tight_layout()
    path = CM_DIR / f"{tag}_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluate] Confusion matrix saved: {path}")
    return path


def save_predictions(y_true, y_pred, y_prob, traj_ids, tag: str = "lstm_test"):
    """Save per-trajectory predictions to CSV."""
    path = METRICS_DIR / f"{tag}_predictions.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trajectory_id", "true_label", "pred_label",
            "p_benign", "p_resisted", "p_hijacked",
        ])
        for tid, yt, yp, ypr in zip(traj_ids, y_true, y_pred, y_prob):
            writer.writerow([
                tid,
                INV_LABEL_MAP[int(yt)],
                INV_LABEL_MAP[int(yp)],
                f"{ypr[0]:.6f}", f"{ypr[1]:.6f}", f"{ypr[2]:.6f}",
            ])
    print(f"[Evaluate] Predictions saved: {path}")
    return path
