"""
plots.py - All TRACEGUARD figures (publication-quality PNGs).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from src.config import CLASS_NAMES, PLOTS_DIR, STEPS_PER_TRAJECTORY

# Consistent style
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi":     150,
})

COLORS = {
    "BENIGN":             "#2196F3",   # blue
    "INJECTION_RESISTED": "#4CAF50",   # green
    "HIJACKED_A":         "#FF9800",   # orange
    "HIJACKED_B":         "#F44336",   # red
    "train":              "#5C6BC0",
    "val":                "#26A69A",
}


def plot_training_curves(history: dict, tag: str = "lstm_seed42"):
    """Figure 1 & 2: train/val loss and val Macro F1."""
    epochs = range(1, len(history["train_loss"]) + 1)
    best_ep = history.get("best_epoch", len(epochs))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color=COLORS["train"], lw=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color=COLORS["val"],   lw=2)
    axes[0].axvline(best_ep, color="gray", ls="--", lw=1.2, label=f"Best epoch ({best_ep})")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropyLoss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Macro F1
    axes[1].plot(epochs, history["val_macro_f1"], color=COLORS["val"], lw=2, label="Val Macro F1")
    axes[1].axvline(best_ep, color="gray", ls="--", lw=1.2, label=f"Best epoch ({best_ep})")
    axes[1].set_title("Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"TRACEGUARD LSTM Training — {tag}", fontsize=15, y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / f"training_curves_{tag}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Plots] Training curves saved: {out}")
    return out


def plot_per_class_f1(metrics: dict, tag: str = "lstm_test"):
    """Figure 4: per-class F1 bar chart."""
    cls_names = CLASS_NAMES
    f1_vals   = [metrics["per_class"][c]["f1"] for c in cls_names]
    prec_vals = [metrics["per_class"][c]["precision"] for c in cls_names]
    rec_vals  = [metrics["per_class"][c]["recall"]    for c in cls_names]

    x = np.arange(len(cls_names))
    w = 0.26

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, prec_vals, w, label="Precision", color="#42A5F5", edgecolor="white")
    ax.bar(x,     rec_vals,  w, label="Recall",    color="#66BB6A", edgecolor="white")
    ax.bar(x + w, f1_vals,   w, label="F1",        color="#FFA726", edgecolor="white")

    for i, (p, r, f) in enumerate(zip(prec_vals, rec_vals, f1_vals)):
        ax.text(i - w, p + 0.01, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i,     r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w, f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(cls_names, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"TRACEGUARD LSTM — Per-Class Metrics ({tag})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / f"per_class_f1_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plots] Per-class F1 saved: {out}")
    return out


def plot_hijack_probability_by_prefix(prefix_results: list[dict], tag: str = "test"):
    """Figure 5: Average P(HIJACKED) per prefix length, split by group."""
    df = pd.DataFrame(prefix_results)

    # Define groups
    def _group(row):
        if row["true_label"] == "BENIGN":
            return "BENIGN"
        elif row["true_label"] == "INJECTION_RESISTED":
            return "INJECTION_RESISTED"
        elif row.get("hijack_subtype") == "A":
            return "HIJACKED_A"
        else:
            return "HIJACKED_B"

    df["group"] = df.apply(_group, axis=1)
    grouped = df.groupby(["group", "prefix_length"])["p_hijacked"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    for grp, col in [
        ("BENIGN",             COLORS["BENIGN"]),
        ("INJECTION_RESISTED", COLORS["INJECTION_RESISTED"]),
        ("HIJACKED_A",         COLORS["HIJACKED_A"]),
        ("HIJACKED_B",         COLORS["HIJACKED_B"]),
    ]:
        sub = grouped[grouped["group"] == grp]
        if sub.empty:
            continue
        label_map = {
            "BENIGN": "Benign",
            "INJECTION_RESISTED": "Injection Resisted",
            "HIJACKED_A": "Hijacked (Type A)",
            "HIJACKED_B": "Hijacked (Type B)",
        }
        ax.plot(sub["prefix_length"], sub["p_hijacked"],
                marker="o", lw=2, label=label_map[grp], color=col)

    ax.axhline(0.5, color="gray", ls="--", lw=1.2, label="Threshold (0.5)")
    ax.set_xlabel("Prefix Length (Steps Seen)")
    ax.set_ylabel("Mean P(HIJACKED)")
    ax.set_title(f"TRACEGUARD — Hijack Probability by Prefix Step ({tag})")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(1, STEPS_PER_TRAJECTORY + 1))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / f"hijack_probability_by_prefix_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plots] Hijack probability plot saved: {out}")
    return out


def plot_detection_latency(latency_df: pd.DataFrame, tag: str = "test"):
    """Figures 6 & 7: latency histogram + Type A vs B comparison."""
    detected = latency_df[latency_df["detected"] & latency_df["latency"].notna()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Histogram ---
    bins = range(
        int(detected["latency"].min()) - 1,
        int(detected["latency"].max()) + 2
    ) if len(detected) > 0 else range(-5, 6)

    axes[0].hist(detected["latency"], bins=bins, color="#7986CB",
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="red", ls="--", lw=1.5, label="Deviation step")
    axes[0].set_xlabel("Detection Latency (det_step - dev_step)")
    axes[0].set_ylabel("Number of Trajectories")
    axes[0].set_title("Detection Latency Distribution")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # --- Type A vs B side-by-side ---
    type_a = detected[detected["hijack_subtype"] == "A"]["latency"]
    type_b = detected[detected["hijack_subtype"] == "B"]["latency"]

    all_bins = range(
        int(min(detected["latency"].min(), 0)) - 1,
        int(detected["latency"].max()) + 2
    ) if len(detected) > 0 else range(-5, 6)

    axes[1].hist(type_a, bins=all_bins, alpha=0.7, label=f"Type A (n={len(type_a)})",
                 color=COLORS["HIJACKED_A"], edgecolor="white")
    axes[1].hist(type_b, bins=all_bins, alpha=0.7, label=f"Type B (n={len(type_b)})",
                 color=COLORS["HIJACKED_B"], edgecolor="white")
    axes[1].axvline(0, color="red", ls="--", lw=1.5, label="Deviation step")
    axes[1].set_xlabel("Detection Latency (det_step - dev_step)")
    axes[1].set_ylabel("Number of Trajectories")
    axes[1].set_title("Type A vs Type B Detection Latency")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle(f"TRACEGUARD — Detection Latency Analysis ({tag})", fontsize=14)
    plt.tight_layout()
    out = PLOTS_DIR / f"detection_latency_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plots] Latency plots saved: {out}")
    return out


def plot_shuffling_ablation(original_f1: float, shuffled_mean: float,
                             shuffled_std: float, tag: str = "test"):
    """Figure 8: Original vs shuffled Macro F1."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Original\n(chronological)", "Shuffled\n(permuted)"],
           [original_f1, shuffled_mean],
           color=["#42A5F5", "#EF5350"],
           edgecolor="white", width=0.5, yerr=[0, shuffled_std],
           capsize=6)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Step-Order Ablation: Original vs Shuffled")
    for i, v in enumerate([original_f1, shuffled_mean]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / f"shuffling_ablation_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plots] Shuffling ablation plot saved: {out}")
    return out


def plot_seed_robustness(seed_results: list[dict]):
    """Figure for five-seed robustness."""
    metrics = ["accuracy", "macro_f1", "hijacked_f1", "resisted_f1", "benign_f1"]
    labels  = ["Accuracy", "Macro F1", "HIJACKED F1", "RESISTED F1", "BENIGN F1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    means = [np.mean([r[m] for r in seed_results]) for m in metrics]
    stds  = [np.std( [r[m] for r in seed_results]) for m in metrics]

    ax.bar(x, means, yerr=stds, capsize=6, color="#5C6BC0",
           edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("TRACEGUARD LSTM — Five-Seed Robustness")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}±{s:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "seed_robustness.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plots] Seed robustness plot saved: {out}")
    return out
