"""Controlled experiments: 5-seed robustness and step-order ablation for TRACEGUARD-X.

Evaluates:
1. 5-seed robustness across seeds 42, 123, 456, 789, 1011 for GRU and Temporal Attention.
2. Step-order ablation: permuting trajectory steps to test temporal dependency
   (comparing Delta F1 vs the frozen baseline's +0.2811 drop on shuffled order).
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Any

import src.model_lstm as canonical_model
from traceguard_x.common import data as xdata
from traceguard_x.common import inference as xinf
from traceguard_x.common import metrics as xmetrics
from traceguard_x.common.experiment import save_json, set_seed
from traceguard_x.experiments.track_b1_gru import run_gru_seed
from traceguard_x.experiments.track_b2_attention import run_attention_seed
from traceguard_x.models.gru import TraceGuardGRU
from traceguard_x.models.attention import TraceGuardTemporalAttention
from traceguard_x.paths import experiment_dir

SEEDS = [42, 123, 456, 789, 1011]


def run_5seed_gru() -> dict[str, Any]:
    print("\n================ RUNNING GRU 5-SEED ROBUSTNESS ================")
    results = []
    for s in SEEDS:
        print(f"\n--- GRU Seed {s} ---")
        results.append(run_gru_seed(seed=s))

    macro_f1s = [r["test_metrics"]["macro_f1"] for r in results]
    accs = [r["test_metrics"]["accuracy"] for r in results]
    pre_actions = [r["latency_metrics"]["pre_action_detection_rate"] for r in results]
    latencies = [r["latency_metrics"]["mean_latency"] for r in results]

    summary = {
        "model": "TraceGuardGRU",
        "seeds": SEEDS,
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s, ddof=1)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)),
        "pre_action_mean": float(np.mean(pre_actions)),
        "mean_latency_mean": float(np.mean(latencies)),
        "per_seed": results,
    }
    out_dir = experiment_dir("track_b1_gru")
    save_json(summary, out_dir / "gru_five_seeds_robustness.json")
    return summary


def run_5seed_attention() -> dict[str, Any]:
    print("\n================ RUNNING ATTENTION 5-SEED ROBUSTNESS ================")
    results = []
    for s in SEEDS:
        print(f"\n--- Attention Seed {s} ---")
        results.append(run_attention_seed(seed=s))

    macro_f1s = [r["test_metrics"]["macro_f1"] for r in results]
    accs = [r["test_metrics"]["accuracy"] for r in results]
    pre_actions = [r["latency_metrics"]["pre_action_detection_rate"] for r in results]
    latencies = [r["latency_metrics"]["mean_latency"] for r in results]

    summary = {
        "model": "TraceGuardTemporalAttention",
        "seeds": SEEDS,
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s, ddof=1)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)),
        "pre_action_mean": float(np.mean(pre_actions)),
        "mean_latency_mean": float(np.mean(latencies)),
        "per_seed": results,
    }
    out_dir = experiment_dir("track_b2_attention")
    save_json(summary, out_dir / "attention_five_seeds_robustness.json")
    return summary


def run_step_shuffling_ablation(seed: int = 42) -> dict[str, Any]:
    """Evaluate models on permuted sequence steps to verify temporal reliance."""
    print(f"\n================ STEP-SHUFFLE ABLATION (SEED {seed}) ================")
    set_seed(seed)
    dataset = xdata.load_dataset()
    split_info = xdata.load_split(seed)
    partitions = xdata.split_trajectories(dataset, split_info)
    embeddings = xdata.load_embeddings()

    # Create shuffled test embeddings: randomly permute the 6 steps
    rng = np.random.RandomState(seed)
    shuffled_embeddings = {}
    for tid, emb in embeddings.items():
        perm = rng.permutation(emb.shape[0])
        shuffled_embeddings[tid] = emb[perm]

    # Load GRU seed 42 checkpoint
    gru_ckpt_path = experiment_dir("track_b1_gru") / f"gru_seed{seed}_best.pth"
    gru_ckpt = torch.load(gru_ckpt_path, weights_only=False, map_location="cpu")
    gru = TraceGuardGRU(input_size=384, hidden_size=128, num_layers=2)
    gru.load_state_dict(gru_ckpt["model_state_dict"])
    gru.eval()

    # Load Attention seed 42 checkpoint
    attn_ckpt_path = experiment_dir("track_b2_attention") / f"attention_seed{seed}_best.pth"
    attn_ckpt = torch.load(attn_ckpt_path, weights_only=False, map_location="cpu")
    attn = TraceGuardTemporalAttention(input_size=384, hidden_size=128, attention_dim=64, num_layers=2)
    attn.load_state_dict(attn_ckpt["model_state_dict"])
    attn.eval()

    # Clean inference vs Shuffled inference
    # 1. GRU
    y_true, y_pred_clean, y_prob_clean = xinf.predict_full(gru, partitions["test"], embeddings)
    m_clean_gru = xmetrics.full_classification_metrics(y_true, y_pred_clean, y_prob_clean)

    _, y_pred_shuf, y_prob_shuf = xinf.predict_full(gru, partitions["test"], shuffled_embeddings)
    m_shuf_gru = xmetrics.full_classification_metrics(y_true, y_pred_shuf, y_prob_shuf)

    delta_gru_f1 = m_clean_gru["macro_f1"] - m_shuf_gru["macro_f1"]

    # 2. Attention
    y_true, y_pred_clean_att, y_prob_clean_att = xinf.predict_full(attn, partitions["test"], embeddings)
    m_clean_att = xmetrics.full_classification_metrics(y_true, y_pred_clean_att, y_prob_clean_att)

    _, y_pred_shuf_att, y_prob_shuf_att = xinf.predict_full(attn, partitions["test"], shuffled_embeddings)
    m_shuf_att = xmetrics.full_classification_metrics(y_true, y_pred_shuf_att, y_prob_shuf_att)

    delta_att_f1 = m_clean_att["macro_f1"] - m_shuf_att["macro_f1"]

    ablation_results = {
        "seed": seed,
        "gru": {
            "clean_macro_f1": m_clean_gru["macro_f1"],
            "shuffled_macro_f1": m_shuf_gru["macro_f1"],
            "delta_f1_drop": delta_gru_f1,
            "temporal_dependency_verified": bool(delta_gru_f1 > 0.1),
        },
        "attention": {
            "clean_macro_f1": m_clean_att["macro_f1"],
            "shuffled_macro_f1": m_shuf_att["macro_f1"],
            "delta_f1_drop": delta_att_f1,
            "temporal_dependency_verified": bool(delta_att_f1 > 0.1),
        },
        "baseline_lstm_reference": {
            "clean_macro_f1": 0.8533,
            "shuffled_macro_f1": 0.5722,
            "delta_f1_drop": 0.2811,
        },
    }

    out_dir = experiment_dir("track_c_generalization")
    save_json(ablation_results, out_dir / "step_shuffling_ablation.json")

    print("\n--- Ablation Results ---")
    print(f"GRU: Clean F1={m_clean_gru['macro_f1']:.4f} -> Shuffled F1={m_shuf_gru['macro_f1']:.4f} (Delta: {delta_gru_f1:+.4f})")
    print(f"Attention: Clean F1={m_clean_att['macro_f1']:.4f} -> Shuffled F1={m_shuf_att['macro_f1']:.4f} (Delta: {delta_att_f1:+.4f})")
    print(f"Baseline LSTM: Clean F1=0.8533 -> Shuffled F1=0.5722 (Delta: +0.2811)")
    return ablation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--five-seeds", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.five_seeds:
        run_5seed_gru()
        run_5seed_attention()
    if args.all or args.ablation:
        run_step_shuffling_ablation()
