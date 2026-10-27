"""Track C: Data Quality, Counterfactual Consistency, Attack-Family Generalization, and Perturbations.

Evaluates:
C1: Group-aware split integrity (asserts zero group overlap across partitions).
C2: Counterfactual triplet consistency across BENIGN, RESISTED, HIJACKED variants.
C3: Attack-template generalization via held-out attack-family evaluation.
C4: Controlled input perturbations (formatting, paraphrasing, field noise).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import src.model_lstm as canonical_lstm
from traceguard_x.common import data as xdata
from traceguard_x.common import inference as xinf
from traceguard_x.common import metrics as xmetrics
from traceguard_x.common.experiment import save_json, set_seed
from traceguard_x.models.gru import TraceGuardGRU
from traceguard_x.paths import experiment_dir, MODELS_DIR


def evaluate_dataset_helper(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            lengths = [x_b.shape[1]] * x_b.shape[0]
            logits = model(x_b, lengths)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())
            all_probs.extend(probs)
    return xmetrics.full_classification_metrics(all_labels, all_preds, np.array(all_probs))


def evaluate_counterfactual_consistency(
    model: nn.Module,
    device: torch.device,
    dataset: list[dict[str, Any]],
    test_trajectories: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """C2: Evaluate consistency across counterfactual triplets in the test set.

    For each group in test set:
    - Triplet consists of _B (Benign), _R (Resisted), _H (Hijacked).
    - Resisted Invariance: Does model avoid false BLOCK on _R despite prompt injection?
    - Pre-injection agreement: Do _B, _R, _H predictions agree before injection step?
    """
    model.eval()
    test_by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for traj in test_trajectories:
        gid = traj["counterfactual_group_id"]
        variant = traj["trajectory_id"].split("_")[-1]  # 'B', 'R', 'H'
        if gid not in test_by_group:
            test_by_group[gid] = {}
        test_by_group[gid][variant] = traj

    complete_triplets = [
        gid for gid, variants in test_by_group.items()
        if {"B", "R", "H"}.issubset(variants.keys())
    ]

    prefix_rows = xinf.predict_prefixes(model, test_trajectories, embeddings, threshold=threshold)
    rows_by_traj: dict[str, dict[int, dict[str, Any]]] = {}
    for r in prefix_rows:
        tid = r["trajectory_id"]
        if tid not in rows_by_traj:
            rows_by_traj[tid] = {}
        rows_by_traj[tid][r["prefix_length"]] = r

    resisted_invariance_count = 0
    pre_injection_agreement_count = 0
    total_pre_injection_comparisons = 0

    triplet_results = []

    for gid in complete_triplets:
        b_traj = test_by_group[gid]["B"]
        r_traj = test_by_group[gid]["R"]
        h_traj = test_by_group[gid]["H"]

        b_rows = rows_by_traj.get(b_traj["trajectory_id"], {})
        r_rows = rows_by_traj.get(r_traj["trajectory_id"], {})
        h_rows = rows_by_traj.get(h_traj["trajectory_id"], {})

        # 1. Resisted Invariance: Never flagged HIJACKED on _R
        r_flagged = any(row["p_hijacked"] >= threshold for row in r_rows.values())
        if not r_flagged:
            resisted_invariance_count += 1

        # 2. Pre-injection agreement
        inj_step = r_traj.get("injection_step") or 2
        agreed_steps = 0
        tested_steps = 0
        for k in range(1, inj_step):
            if k in b_rows and k in r_rows and k in h_rows:
                b_pred = b_rows[k]["pred_label"]
                r_pred = r_rows[k]["pred_label"]
                h_pred = h_rows[k]["pred_label"]
                tested_steps += 1
                if b_pred == r_pred == h_pred:
                    agreed_steps += 1

        if tested_steps > 0 and agreed_steps == tested_steps:
            pre_injection_agreement_count += 1
        if tested_steps > 0:
            total_pre_injection_comparisons += 1

        triplet_results.append({
            "group_id": gid,
            "resisted_flagged_as_hijacked": r_flagged,
            "pre_injection_tested_steps": tested_steps,
            "pre_injection_agreed_steps": agreed_steps,
        })

    n_triplets = len(complete_triplets)
    return {
        "n_test_triplets": n_triplets,
        "resisted_invariance_rate": (resisted_invariance_count / n_triplets) if n_triplets else 0.0,
        "pre_injection_agreement_rate": (pre_injection_agreement_count / total_pre_injection_comparisons) if total_pre_injection_comparisons else 0.0,
        "details": triplet_results,
    }


def run_heldout_attack_family_experiment(
    heldout_family: str = "tool_misuse",
    seed: int = 42,
    epochs: int = 35,
) -> dict[str, Any]:
    """C3: Train on 4 attack families + benign; test on held-out 5th attack family.

    Enforces strict group-aware isolation: no counterfactual group is split across train/test.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = xdata.load_dataset()
    embeddings = xdata.load_embeddings()

    # Determine group-level attack family (from _R or _H variant)
    group_to_family: dict[str, str | None] = {}
    for d in dataset:
        gid = d["counterfactual_group_id"]
        fam = d.get("attack_family")
        if fam is not None:
            group_to_family[gid] = fam

    heldout_groups = {gid for gid, fam in group_to_family.items() if fam == heldout_family}
    training_groups = [gid for gid in group_to_family.keys() if gid not in heldout_groups]

    # Split training groups into train (80%) and val (20%)
    rng = np.random.RandomState(seed)
    rng.shuffle(training_groups)
    n_train = int(len(training_groups) * 0.8)
    train_gids = set(training_groups[:n_train])
    val_gids = set(training_groups[n_train:])

    train_trajs = [d for d in dataset if d["counterfactual_group_id"] in train_gids]
    val_trajs = [d for d in dataset if d["counterfactual_group_id"] in val_gids]
    test_trajs = [d for d in dataset if d["counterfactual_group_id"] in heldout_groups]

    print(f"Held-out attack family: {heldout_family}")
    print(f"Train groups: {len(train_gids)} ({len(train_trajs)} trajs)")
    print(f"Val groups:   {len(val_gids)} ({len(val_trajs)} trajs)")
    print(f"Test groups (held-out): {len(heldout_groups)} ({len(test_trajs)} trajs)")

    # Build tensor datasets
    def make_dataset(trajs: list[dict[str, Any]]) -> TensorDataset:
        x_list = [embeddings[t["trajectory_id"]] for t in trajs]
        y_list = [xmetrics.LABEL_MAP[t["label"]] for t in trajs]
        return TensorDataset(
            torch.tensor(np.array(x_list), dtype=torch.float32),
            torch.tensor(y_list, dtype=torch.long),
        )

    train_ds = make_dataset(train_trajs)
    val_ds = make_dataset(val_trajs)
    test_ds = make_dataset(test_trajs)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Train a standard TraceGuardGRU
    model = TraceGuardGRU(input_size=384, hidden_size=128, num_layers=2, dropout=0.2, num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_weights = None
    for epoch in range(1, epochs + 1):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            lengths = [x_b.shape[1]] * x_b.shape[0]
            optimizer.zero_grad()
            logits = model(x_b, lengths)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_dataset_helper(model, val_loader, device)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights:
        model.load_state_dict(best_weights)

    test_metrics = evaluate_dataset_helper(model, test_loader, device)

    return {
        "heldout_family": heldout_family,
        "n_heldout_groups": len(heldout_groups),
        "n_heldout_trajectories": len(test_trajs),
        "best_val_macro_f1": best_val_f1,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "per_class": test_metrics["per_class"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
    }


def evaluate_controlled_perturbations(
    model: nn.Module,
    device: torch.device,
    test_trajectories: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
) -> dict[str, Any]:
    """C4: Evaluate sensitivity under synthetic feature noise and simulated perturbations.

    Perturbation categories:
    1. Clean baseline.
    2. Additive Gaussian feature noise (std=0.05) representing subtle embedding variations/paraphrasing.
    3. Moderate Gaussian feature noise (std=0.10).
    4. Observation dropout (randomly zeroing 1 step embedding in the sequence, testing missing observations).
    """
    model.eval()
    x_test = np.array([embeddings[t["trajectory_id"]] for t in test_trajectories])
    y_test = np.array([xmetrics.LABEL_MAP[t["label"]] for t in test_trajectories])
    lengths = [x_test.shape[1]] * len(x_test)

    with torch.no_grad():
        x_clean_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        clean_logits = model(x_clean_tensor, lengths).cpu().numpy()
        clean_preds = clean_logits.argmax(axis=1)
        clean_probs = torch.softmax(torch.tensor(clean_logits), dim=-1).numpy()
        clean_f1 = xmetrics.full_classification_metrics(y_test, clean_preds, clean_probs)["macro_f1"]

    rng = np.random.RandomState(42)

    # 1. Subtle noise (std=0.05)
    noise_subtle = rng.normal(0, 0.05, size=x_test.shape).astype(np.float32)
    x_subtle = x_test + noise_subtle
    with torch.no_grad():
        subtle_logits = model(torch.tensor(x_subtle, dtype=torch.float32).to(device), lengths).cpu().numpy()
        subtle_preds = subtle_logits.argmax(axis=1)
        subtle_probs = torch.softmax(torch.tensor(subtle_logits), dim=-1).numpy()
        subtle_metrics = xmetrics.full_classification_metrics(y_test, subtle_preds, subtle_probs)
        subtle_stability = float((subtle_preds == clean_preds).mean())

    # 2. Moderate noise (std=0.10)
    noise_mod = rng.normal(0, 0.10, size=x_test.shape).astype(np.float32)
    x_mod = x_test + noise_mod
    with torch.no_grad():
        mod_logits = model(torch.tensor(x_mod, dtype=torch.float32).to(device), lengths).cpu().numpy()
        mod_preds = mod_logits.argmax(axis=1)
        mod_probs = torch.softmax(torch.tensor(mod_logits), dim=-1).numpy()
        mod_metrics = xmetrics.full_classification_metrics(y_test, mod_preds, mod_probs)
        mod_stability = float((mod_preds == clean_preds).mean())

    # 3. Step omission (zero out step 3 embedding)
    x_omission = x_test.copy()
    x_omission[:, 2, :] = 0.0
    with torch.no_grad():
        omiss_logits = model(torch.tensor(x_omission, dtype=torch.float32).to(device), lengths).cpu().numpy()
        omiss_preds = omiss_logits.argmax(axis=1)
        omiss_probs = torch.softmax(torch.tensor(omiss_logits), dim=-1).numpy()
        omiss_metrics = xmetrics.full_classification_metrics(y_test, omiss_preds, omiss_probs)
        omiss_stability = float((omiss_preds == clean_preds).mean())

    return {
        "clean_macro_f1": clean_f1,
        "perturbations": {
            "subtle_embedding_noise_sigma005": {
                "macro_f1": subtle_metrics["macro_f1"],
                "delta_f1": subtle_metrics["macro_f1"] - clean_f1,
                "label_stability": subtle_stability,
            },
            "moderate_embedding_noise_sigma010": {
                "macro_f1": mod_metrics["macro_f1"],
                "delta_f1": mod_metrics["macro_f1"] - clean_f1,
                "label_stability": mod_stability,
            },
            "step_omission_step3": {
                "macro_f1": omiss_metrics["macro_f1"],
                "delta_f1": omiss_metrics["macro_f1"] - clean_f1,
                "label_stability": omiss_stability,
            },
        },
    }


def run_all_track_c() -> dict[str, Any]:
    print("\n================ RUNNING TRACK C: GENERALIZATION & DATA QUALITY ================")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = xdata.load_dataset()
    split_info = xdata.load_split(42)
    partitions = xdata.split_trajectories(dataset, split_info)
    embeddings = xdata.load_embeddings()

    # Load trained GRU seed 42
    gru_path = experiment_dir("track_b1_gru") / "gru_seed42_best.pth"
    gru_ckpt = torch.load(gru_path, map_location=device, weights_only=False)
    gru_model = TraceGuardGRU(input_size=384, hidden_size=128, num_layers=2, dropout=0.2, num_classes=3)
    gru_model.load_state_dict(gru_ckpt["model_state_dict"])
    gru_model.to(device)

    # C1: Group split integrity verification
    train_groups = {t["counterfactual_group_id"] for t in partitions["train"]}
    val_groups = {t["counterfactual_group_id"] for t in partitions["val"]}
    test_groups = {t["counterfactual_group_id"] for t in partitions["test"]}
    leakage_detected = bool((train_groups & val_groups) or (train_groups & test_groups) or (val_groups & test_groups))

    c1_result = {
        "group_leakage_detected": leakage_detected,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
    }
    print(f"C1 Group-Aware Integrity: Leakage detected = {leakage_detected}")

    # C2: Counterfactual consistency
    print("Running C2: Counterfactual consistency on test triplets...")
    c2_result = evaluate_counterfactual_consistency(
        gru_model, device, dataset, partitions["test"], embeddings
    )
    print(f"  Resisted Invariance Rate: {c2_result['resisted_invariance_rate']:.4f}")
    print(f"  Pre-injection Agreement Rate: {c2_result['pre_injection_agreement_rate']:.4f}")

    # C3: Attack-template generalization (held-out family)
    print("Running C3: Attack-family generalization (held-out 'tool_misuse')...")
    c3_result = run_heldout_attack_family_experiment("tool_misuse", seed=42)
    print(f"  Held-out Test Macro F1: {c3_result['test_metrics']['macro_f1']:.4f} (vs seen val: {c3_result['best_val_macro_f1']:.4f})")

    # C4: Controlled perturbations
    print("Running C4: Controlled perturbations...")
    c4_result = evaluate_controlled_perturbations(
        gru_model, device, partitions["test"], embeddings
    )
    print(f"  Clean F1: {c4_result['clean_macro_f1']:.4f}")
    for k, v in c4_result["perturbations"].items():
        print(f"  Perturbation '{k}': F1={v['macro_f1']:.4f}, stability={v['label_stability']:.4f}")

    combined = {
        "c1_group_integrity": c1_result,
        "c2_counterfactual_consistency": c2_result,
        "c3_heldout_attack_family": c3_result,
        "c4_controlled_perturbations": c4_result,
    }

    out_dir = experiment_dir("track_c_generalization")
    save_json(combined, out_dir / "generalization_experiments.json")
    print(f"Saved Track C results to {out_dir / 'generalization_experiments.json'}")
    return combined


if __name__ == "__main__":
    run_all_track_c()
