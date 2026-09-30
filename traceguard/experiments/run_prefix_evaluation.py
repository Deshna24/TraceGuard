"""
run_prefix_evaluation.py - Standalone prefix evaluation on the test set.
Useful to re-evaluate a trained checkpoint without re-training.

Usage:
    cd traceguard
    python experiments/run_prefix_evaluation.py [--seed 42] [--threshold 0.5]
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import (
    DETECTION_THRESHOLD, PRIMARY_SEED, MODELS_DIR, METRICS_DIR,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT, BATCH_SIZE,
)
from src.utils import set_seed, get_device
from src.data_loader import load_dataset
from src.split import create_group_aware_split
from src.embeddings import generate_embeddings, get_embedding_dim
from src.model_lstm import TraceGuardLSTM
from src.prefix_detection import run_prefix_evaluation
from src.latency import compute_latency_metrics
from src.plots import plot_hijack_probability_by_prefix, plot_detection_latency


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",      type=int,   default=PRIMARY_SEED)
    p.add_argument("--threshold", type=float, default=DETECTION_THRESHOLD)
    return p.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    thr  = args.threshold
    tag  = f"lstm_seed{seed}"
    device = get_device()
    set_seed(seed)

    trajectories = load_dataset()
    _, _, test_trajs, _ = create_group_aware_split(trajectories, seed=seed)
    emb_dict = generate_embeddings(trajectories)
    emb_dim  = get_embedding_dim()

    # Load checkpoint
    ckpt_path = MODELS_DIR / f"{tag}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run run_lstm.py first.")

    model = TraceGuardLSTM(emb_dim, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    prefix_results = run_prefix_evaluation(model, test_trajs, emb_dict,
                                           device=device, threshold=thr, tag=tag)
    latency_stats, latency_df = compute_latency_metrics(prefix_results, thr, tag=tag)
    plot_hijack_probability_by_prefix(prefix_results, tag=tag)
    if len(latency_df) > 0:
        plot_detection_latency(latency_df, tag=tag)

    print(f"\n[Done] Prefix evaluation complete. Threshold={thr}")


if __name__ == "__main__":
    main()
