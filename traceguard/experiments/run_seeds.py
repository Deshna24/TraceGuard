"""
run_seeds.py - Five-seed robustness experiment for TRACEGUARD LSTM.

Trains the LSTM from scratch using 5 seeds:
  42, 123, 456, 789, 1011

Reports mean ± std for accuracy, macro F1, per-class F1, and early detection.

Usage:
    cd traceguard
    python experiments/run_seeds.py
"""

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    ROBUSTNESS_SEEDS, DATA_PATH,
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOPPING_PATIENCE, DETECTION_THRESHOLD,
    LEARNING_RATE, WEIGHT_DECAY, METRICS_DIR, PLOTS_DIR,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    EMBEDDING_MODEL_NAME, CLASS_NAMES, LABEL_MAP, INV_LABEL_MAP,
)
from src.utils import set_seed, get_device, save_json
from src.data_loader import load_dataset
from src.split import create_group_aware_split, save_split
from src.embeddings import generate_embeddings, get_embedding_dim
from src.model_lstm import TraceGuardLSTM, make_dataloader
from src.train_lstm import train_lstm
from src.evaluate import predict_full, compute_full_metrics
from src.prefix_detection import run_prefix_evaluation
from src.latency import compute_latency_metrics
from src.plots import plot_seed_robustness
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support


def run_one_seed(seed: int, trajectories: list, emb_dict: dict, emb_dim: int, device) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Seed {seed}")
    print(f"{'─'*60}")
    set_seed(seed)

    # Split
    train_trajs, val_trajs, test_trajs, split_info = \
        create_group_aware_split(trajectories, seed=seed)
    save_split(split_info, seed)

    # Loaders
    train_loader = make_dataloader(train_trajs, emb_dict, BATCH_SIZE, shuffle=True,  seed=seed)
    val_loader   = make_dataloader(val_trajs,   emb_dict, BATCH_SIZE, shuffle=False, seed=seed)
    test_loader  = make_dataloader(test_trajs,  emb_dict, BATCH_SIZE, shuffle=False, seed=seed)

    # Model
    model = TraceGuardLSTM(emb_dim, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT)
    tag   = f"lstm_seed{seed}"

    # Train
    model, history = train_lstm(
        model, train_loader, val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        model_tag=tag,
    )
    save_json(history, METRICS_DIR / f"{tag}_training_history.json")

    # Evaluate
    y_true, y_pred, y_prob, traj_ids = predict_full(model, test_loader, device=device)
    full_metrics = compute_full_metrics(y_true, y_pred, y_prob, tag=tag)

    # Prefix + latency
    prefix_results = run_prefix_evaluation(model, test_trajs, emb_dict, device=device,
                                           threshold=DETECTION_THRESHOLD, tag=tag)
    latency_stats, _ = compute_latency_metrics(prefix_results, DETECTION_THRESHOLD, tag=tag)

    # Collect result
    pc = full_metrics["per_class"]
    result = {
        "seed":                    seed,
        "accuracy":                full_metrics["accuracy"],
        "macro_f1":                full_metrics["macro_f1"],
        "macro_precision":         full_metrics["macro_precision"],
        "macro_recall":            full_metrics["macro_recall"],
        "benign_f1":               pc["BENIGN"]["f1"],
        "resisted_f1":             pc["INJECTION_RESISTED"]["f1"],
        "hijacked_f1":             pc["HIJACKED"]["f1"],
        "pre_action_rate":         latency_stats.get("pre_action_detection_rate", 0),
        "mean_latency":            latency_stats.get("mean_latency"),
        "best_epoch":              history.get("best_epoch"),
        "best_val_f1":             history.get("best_val_f1"),
    }
    return result


def main():
    device = get_device()
    print(f"\n{'='*70}")
    print(f"  TRACEGUARD — Five-Seed Robustness Experiment")
    print(f"  Seeds: {ROBUSTNESS_SEEDS}")
    print(f"{'='*70}\n")

    # Load dataset once
    trajectories = load_dataset()

    # Generate embeddings once (shared across seeds — only text changes with seed,
    # not which trajectories appear)
    emb_dict = generate_embeddings(trajectories, force_recompute=False)
    emb_dim  = get_embedding_dim()

    all_results = []
    for seed in ROBUSTNESS_SEEDS:
        result = run_one_seed(seed, trajectories, emb_dict, emb_dim, device)
        all_results.append(result)
        print(f"  Seed {seed}: acc={result['accuracy']:.4f}  f1={result['macro_f1']:.4f}  "
              f"hijacked_f1={result['hijacked_f1']:.4f}")

    # Aggregate
    metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall",
               "benign_f1", "resisted_f1", "hijacked_f1", "pre_action_rate"]

    summary = {"per_seed": all_results, "aggregate": {}}
    for m in metrics:
        vals = [r[m] for r in all_results if r[m] is not None]
        summary["aggregate"][m] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
        }

    save_json(summary, METRICS_DIR / "robustness_five_seeds.json")
    plot_seed_robustness(all_results)

    print(f"\n{'='*70}")
    print(f"  Five-Seed Robustness Summary")
    print(f"{'='*70}")
    agg = summary["aggregate"]
    for m in metrics:
        a = agg[m]
        print(f"  {m:30s}  {a['mean']:.4f} ± {a['std']:.4f}  "
              f"[{a['min']:.4f}, {a['max']:.4f}]")
    print()


if __name__ == "__main__":
    main()
