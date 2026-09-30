"""
run_lstm.py - Main TRACEGUARD LSTM experiment script.

Executes all 18 phases in order:
  PHASE 1-4:   Load, validate, split, save
  PHASE 5-6:   Preprocessing, embeddings
  PHASE 7:     Train LSTM
  PHASE 8:     Full trajectory evaluation
  PHASE 9:     Prefix evaluation
  PHASE 10:    Early detection metrics
  PHASE 11:    Type-A vs Type-B
  PHASE 12:    Counterfactual analysis
  PHASE 13:    Step-shuffling ablation
  PHASE 14:    Error analysis
  PHASE 15:    Five-seed robustness (separate script)
  PHASE 16-18: Plots, tables, final report, handoff

Usage:
    cd traceguard
    python experiments/run_lstm.py [--seed 42] [--force-embed]
"""

import sys, os, argparse, json, csv, copy, time
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Make src importable ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_PATH, PRIMARY_SEED, BATCH_SIZE, MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE, DETECTION_THRESHOLD,
    MODELS_DIR, METRICS_DIR, PLOTS_DIR, SPLITS_DIR,
    REPORTS_DIR, HANDOFF_DIR,
    CLASS_NAMES, LABEL_MAP, INV_LABEL_MAP,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY,
    EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_SHORT,
    STEPS_PER_TRAJECTORY,
)
from src.utils import set_seed, get_device, save_json, load_json, pretty_print_dict
from src.data_loader import load_dataset
from src.split import create_group_aware_split, save_split
from src.embeddings import generate_embeddings, get_embedding_dim
from src.model_lstm import TraceGuardLSTM, make_dataloader
from src.train_lstm import train_lstm
from src.evaluate import predict_full, compute_full_metrics, plot_confusion_matrix, save_predictions
from src.prefix_detection import run_prefix_evaluation
from src.latency import compute_latency_metrics
from src.plots import (
    plot_training_curves, plot_per_class_f1,
    plot_hijack_probability_by_prefix, plot_detection_latency,
    plot_shuffling_ablation,
)

from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support


def parse_args():
    p = argparse.ArgumentParser(description="TRACEGUARD LSTM Experiment")
    p.add_argument("--seed",        type=int,  default=PRIMARY_SEED)
    p.add_argument("--force-embed", action="store_true", help="Recompute embeddings")
    p.add_argument("--skip-train",  action="store_true", help="Load existing checkpoint")
    return p.parse_args()


def main():
    args   = parse_args()
    seed   = args.seed
    device = get_device()

    print(f"\n{'='*70}")
    print(f"  TRACEGUARD — LSTM Experiment")
    print(f"  Seed: {seed}   Device: {device}")
    print(f"{'='*70}\n")

    set_seed(seed)
    tag = f"lstm_seed{seed}"

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1-2: Load & Validate
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 1-2] Loading and validating dataset ...")
    trajectories = load_dataset()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3-4: Group-aware split
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 3-4] Creating group-aware train/val/test split ...")
    train_trajs, val_trajs, test_trajs, split_info = \
        create_group_aware_split(trajectories, seed=seed)
    split_info["seed"] = seed
    save_split(split_info, seed)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5-6: Embeddings
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 5-6] Generating / loading embeddings ...")
    emb_dict = generate_embeddings(
        trajectories, force_recompute=args.force_embed
    )
    emb_dim = get_embedding_dim()
    print(f"  Embedding dim: {emb_dim}")

    # Build DataLoaders
    train_loader = make_dataloader(train_trajs, emb_dict, BATCH_SIZE, shuffle=True,  seed=seed)
    val_loader   = make_dataloader(val_trajs,   emb_dict, BATCH_SIZE, shuffle=False, seed=seed)
    test_loader  = make_dataloader(test_trajs,  emb_dict, BATCH_SIZE, shuffle=False, seed=seed)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 7: Train LSTM
    # ─────────────────────────────────────────────────────────────────────────
    model = TraceGuardLSTM(
        input_size  = emb_dim,
        hidden_size = LSTM_HIDDEN_SIZE,
        num_layers  = LSTM_NUM_LAYERS,
        dropout     = LSTM_DROPOUT,
        num_classes = 3,
    )
    print(f"\n[PHASE 7] LSTM Architecture:")
    print(f"  {model.get_config()}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_path = MODELS_DIR / f"{tag}_best.pth"

    if args.skip_train and ckpt_path.exists():
        print(f"\n[PHASE 7] Skipping training — loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        history = {"best_epoch": ckpt["best_epoch"], "best_val_f1": ckpt["best_val_f1"],
                   "train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    else:
        print("\n[PHASE 7] Training LSTM ...")
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

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 8: Full trajectory evaluation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 8] Full trajectory evaluation on TEST set ...")
    y_true, y_pred, y_prob, traj_ids = predict_full(model, test_loader, device=device)
    full_metrics = compute_full_metrics(y_true, y_pred, y_prob, tag=tag)
    plot_confusion_matrix(y_true, y_pred, tag=tag)
    save_predictions(y_true, y_pred, y_prob, traj_ids, tag=tag)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 9: Prefix evaluation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 9] Prefix / online evaluation on TEST set ...")
    prefix_results = run_prefix_evaluation(
        model, test_trajs, emb_dict,
        device=device, threshold=DETECTION_THRESHOLD, tag=tag
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 10-11: Detection latency + Type A/B
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 10-11] Computing detection latency and Type A/B metrics ...")
    latency_stats, latency_df = compute_latency_metrics(
        prefix_results, threshold=DETECTION_THRESHOLD, tag=tag
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 12: Counterfactual analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 12] Counterfactual group analysis ...")
    cf_results = _counterfactual_analysis(
        test_trajs, y_true, y_pred, traj_ids, tag=tag
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 13: Step-shuffling ablation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 13] Step-shuffling ablation ...")
    shuffle_results = _shuffling_ablation(
        model, test_trajs, emb_dict, y_true, device, seed=seed, n_reps=10, tag=tag
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 14: Error analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 14] Error analysis ...")
    _error_analysis(
        test_trajs, y_true, y_pred, y_prob, traj_ids,
        prefix_results, DETECTION_THRESHOLD, tag=tag
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 16: Plots
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 16] Generating plots ...")
    if history.get("train_loss"):
        plot_training_curves(history, tag=tag)
    plot_per_class_f1(full_metrics, tag=tag)
    plot_hijack_probability_by_prefix(prefix_results, tag=tag)
    if len(latency_df) > 0:
        plot_detection_latency(latency_df, tag=tag)
    plot_shuffling_ablation(
        shuffle_results["original_f1"],
        shuffle_results["shuffled_mean_f1"],
        shuffle_results["shuffled_std_f1"],
        tag=tag,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 17: Save experiment config
    # ─────────────────────────────────────────────────────────────────────────
    exp_config = {
        "experiment_tag":       tag,
        "seed":                 seed,
        "dataset":              str(DATA_PATH),
        "n_trajectories":       600,
        "train_groups":         split_info["train_n_groups"],
        "val_groups":           split_info["val_n_groups"],
        "test_groups":          split_info["test_n_groups"],
        "embedding_model":      EMBEDDING_MODEL_NAME,
        "embedding_dim":        emb_dim,
        "lstm_config":          model.get_config(),
        "learning_rate":        LEARNING_RATE,
        "weight_decay":         WEIGHT_DECAY,
        "batch_size":           BATCH_SIZE,
        "max_epochs":           MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "detection_threshold":  DETECTION_THRESHOLD,
        "device":               str(device),
        "best_epoch":           history.get("best_epoch"),
        "best_val_f1":          history.get("best_val_f1"),
    }
    save_json(exp_config, METRICS_DIR.parent / "experiment_config.json")

    # ─────────────────────────────────────────────────────────────────────────
    # Final Summary Print
    # ─────────────────────────────────────────────────────────────────────────
    _print_final_summary(full_metrics, latency_stats, shuffle_results, model)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 18: Teammate handoff package
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 18] Generating teammate handoff package ...")
    _generate_handoff(split_info, exp_config, full_metrics, latency_stats,
                      shuffle_results, seed, tag)

    print("\n[Done] TRACEGUARD LSTM experiment complete.\n")
    return full_metrics, latency_stats


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _counterfactual_analysis(test_trajs, y_true, y_pred, traj_ids, tag):
    """Analyse matched counterfactual groups in the test set."""
    from collections import defaultdict

    # Map traj_id -> pred label
    pred_map = dict(zip(traj_ids, y_pred))

    group_map = defaultdict(list)
    for t in test_trajs:
        group_map[t["counterfactual_group_id"]].append(t)

    results = []
    for gid, members in group_map.items():
        entry = {"group_id": gid, "members": []}
        fully_correct = True
        for m in members:
            true_lbl = LABEL_MAP[m["label"]]
            pred_lbl = pred_map.get(m["trajectory_id"], -1)
            correct  = (true_lbl == pred_lbl)
            if not correct:
                fully_correct = False
            entry["members"].append({
                "trajectory_id": m["trajectory_id"],
                "true_label":    m["label"],
                "pred_label":    INV_LABEL_MAP.get(pred_lbl, "UNKNOWN"),
                "correct":       correct,
            })
        entry["fully_correct"] = fully_correct
        results.append(entry)

    n_total    = len(results)
    n_correct  = sum(r["fully_correct"] for r in results)

    # Error type counting
    hijacked_missed      = 0
    resisted_as_benign   = 0
    resisted_as_hijacked = 0
    hijacked_as_benign   = 0
    hijacked_as_resisted = 0

    for r in results:
        for m in r["members"]:
            tl = m["true_label"]
            pl = m["pred_label"]
            if tl == "HIJACKED"            and pl != "HIJACKED":            hijacked_missed += 1
            if tl == "INJECTION_RESISTED"  and pl == "BENIGN":              resisted_as_benign += 1
            if tl == "INJECTION_RESISTED"  and pl == "HIJACKED":            resisted_as_hijacked += 1
            if tl == "HIJACKED"            and pl == "BENIGN":              hijacked_as_benign += 1
            if tl == "HIJACKED"            and pl == "INJECTION_RESISTED":  hijacked_as_resisted += 1

    summary = {
        "n_groups":              n_total,
        "fully_correct_groups":  n_correct,
        "group_accuracy":        n_correct / n_total if n_total > 0 else 0.0,
        "hijacked_missed":       hijacked_missed,
        "resisted_as_benign":    resisted_as_benign,
        "resisted_as_hijacked":  resisted_as_hijacked,
        "hijacked_as_benign":    hijacked_as_benign,
        "hijacked_as_resisted":  hijacked_as_resisted,
    }
    save_json(summary,  METRICS_DIR / f"counterfactual_analysis_{tag}.json")
    save_json(results,  METRICS_DIR / f"counterfactual_groups_{tag}.json")

    print(f"[CF Analysis] {n_total} groups | Fully correct: {n_correct} ({n_correct/n_total:.2%})")
    print(f"  HIJACKED missed:         {hijacked_missed}")
    print(f"  RESISTED->BENIGN:        {resisted_as_benign}")
    print(f"  RESISTED->HIJACKED:      {resisted_as_hijacked}")
    print(f"  HIJACKED->BENIGN:        {hijacked_as_benign}")
    print(f"  HIJACKED->RESISTED:      {hijacked_as_resisted}")
    return summary


def _shuffling_ablation(model, test_trajs, emb_dict, y_true_orig,
                         device, seed, n_reps=10, tag="test"):
    """Evaluate the same trained LSTM on step-shuffled sequences."""
    from sklearn.metrics import f1_score, accuracy_score

    rng = np.random.default_rng(seed)

    original_preds = []
    for traj in test_trajs:
        emb = torch.tensor(emb_dict[traj["trajectory_id"]], dtype=torch.float32).unsqueeze(0).to(device)
        lengths = [emb.shape[1]]
        model.eval()
        with torch.no_grad():
            logits = model(emb, lengths)
        original_preds.append(logits.argmax(dim=1).item())

    orig_f1  = f1_score(y_true_orig, original_preds, average="macro", zero_division=0)
    orig_acc = accuracy_score(y_true_orig, original_preds)

    shuffled_f1s  = []
    shuffled_accs = []
    for _ in range(n_reps):
        preds = []
        for traj in test_trajs:
            emb = emb_dict[traj["trajectory_id"]].copy()     # [6, D]
            perm = rng.permutation(emb.shape[0])
            emb_shuf = emb[perm]
            x = torch.tensor(emb_shuf, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x, [emb_shuf.shape[0]])
            preds.append(logits.argmax(dim=1).item())
        shuffled_f1s.append(f1_score(y_true_orig, preds, average="macro", zero_division=0))
        shuffled_accs.append(accuracy_score(y_true_orig, preds))

    shuf_mean_f1 = float(np.mean(shuffled_f1s))
    shuf_std_f1  = float(np.std(shuffled_f1s))

    results = {
        "original_accuracy":  orig_acc,
        "original_f1":        orig_f1,
        "n_reps":             n_reps,
        "shuffled_f1s":       shuffled_f1s,
        "shuffled_mean_f1":   shuf_mean_f1,
        "shuffled_std_f1":    shuf_std_f1,
        "f1_delta":           orig_f1 - shuf_mean_f1,
        "shuffled_accs":      shuffled_accs,
        "shuffled_mean_acc":  float(np.mean(shuffled_accs)),
        "shuffled_std_acc":   float(np.std(shuffled_accs)),
    }
    save_json(results, METRICS_DIR / f"shuffling_ablation_{tag}.json")

    print(f"\n[Shuffling Ablation]")
    print(f"  Original Macro F1   : {orig_f1:.4f}")
    print(f"  Shuffled Macro F1   : {shuf_mean_f1:.4f} ± {shuf_std_f1:.4f}  (n={n_reps})")
    print(f"  F1 Delta            : {orig_f1 - shuf_mean_f1:+.4f}")
    if orig_f1 > shuf_mean_f1:
        print("  -> Performance degradation under step shuffling supports the hypothesis "
              "that chronological ordering contributes useful predictive information.")
    else:
        print("  -> No clear degradation; the model may not heavily rely on step order.")
    return results


def _error_analysis(test_trajs, y_true, y_pred, y_prob, traj_ids,
                    prefix_results, threshold, tag):
    """Save example errors for each error category."""
    from collections import defaultdict
    traj_map = {t["trajectory_id"]: t for t in test_trajs}
    pred_map = dict(zip(traj_ids, y_pred))
    prob_map = dict(zip(traj_ids, y_prob))

    # Build detection step map for HIJACKED
    det_step_map = {}
    by_traj = defaultdict(list)
    for row in prefix_results:
        by_traj[row["trajectory_id"]].append(row)
    for tid, rows in by_traj.items():
        rows.sort(key=lambda r: r["prefix_length"])
        for row in rows:
            if row["p_hijacked"] >= threshold:
                det_step_map[tid] = row["prefix_length"]
                break

    categories = {
        "correct_benign":           (0, 0),
        "benign_pred_resisted":     (0, 1),
        "benign_pred_hijacked":     (0, 2),
        "correct_resisted":         (1, 1),
        "resisted_pred_benign":     (1, 0),
        "resisted_pred_hijacked":   (1, 2),
        "correct_hijacked":         (2, 2),
        "hijacked_pred_benign":     (2, 0),
        "hijacked_pred_resisted":   (2, 1),
    }

    examples = {cat: [] for cat in categories}

    for tid, yt, yp in zip(traj_ids, y_true, y_pred):
        for cat, (expected_true, expected_pred) in categories.items():
            if yt == expected_true and yp == expected_pred:
                traj = traj_map.get(tid, {})
                probs = prob_map.get(tid, [0, 0, 0])
                entry = {
                    "trajectory_id":  tid,
                    "true_label":     INV_LABEL_MAP[int(yt)],
                    "pred_label":     INV_LABEL_MAP[int(yp)],
                    "p_benign":       float(probs[0]),
                    "p_resisted":     float(probs[1]),
                    "p_hijacked":     float(probs[2]),
                    "task_category":  traj.get("task_category"),
                    "user_goal":      traj.get("user_goal", "")[:120],
                }
                if yt == 2:  # HIJACKED
                    entry["injection_step"] = traj.get("injection_step")
                    entry["precursor_step"] = traj.get("precursor_step")
                    entry["deviation_step"] = traj.get("deviation_step")
                    entry["hijack_subtype"] = traj.get("hijack_subtype")
                    entry["detection_step"] = det_step_map.get(tid)
                examples[cat].append(entry)
                break

    # Write markdown report
    out = REPORTS_DIR / "error_analysis.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# TRACEGUARD LSTM — Error Analysis\n\n")
        f.write(f"Tag: `{tag}`  |  Threshold: {threshold}\n\n")
        for cat, entries in examples.items():
            f.write(f"\n## {cat.replace('_', ' ').title()} ({len(entries)} examples)\n\n")
            for e in entries[:5]:   # show up to 5 per category
                f.write(f"- **{e['trajectory_id']}** | true={e['true_label']} | "
                        f"pred={e['pred_label']} | "
                        f"P(B/R/H)={e['p_benign']:.3f}/{e['p_resisted']:.3f}/{e['p_hijacked']:.3f}\n")
                f.write(f"  - Task: {e.get('task_category')} | Goal: {e.get('user_goal')}\n")
                if e.get("injection_step") is not None:
                    f.write(f"  - Inj={e['injection_step']} Prec={e.get('precursor_step')} "
                            f"Dev={e['deviation_step']} Det={e.get('detection_step')} "
                            f"Type={e.get('hijack_subtype')}\n")
                f.write("\n")
    print(f"[ErrorAnalysis] Report saved: {out}")


def _print_final_summary(full_metrics, latency_stats, shuffle_results, model):
    pc = full_metrics["per_class"]
    la = latency_stats.get("type_a", {})
    lb = latency_stats.get("type_b", {})

    print(f"\n{'='*70}")
    print("  TRACEGUARD LSTM — FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nDATASET:  600 trajectories")
    print(f"SPLIT:    420 train / 90 validation / 90 test\n")
    print(f"MODEL:    TraceGuardLSTM | hidden={LSTM_HIDDEN_SIZE} | "
          f"layers={LSTM_NUM_LAYERS} | dropout={LSTM_DROPOUT} | unidirectional\n")
    print(f"FULL TEST RESULTS:")
    print(f"  Accuracy       : {full_metrics['accuracy']:.4f}")
    print(f"  Macro F1       : {full_metrics['macro_f1']:.4f}")
    print(f"\nPER-CLASS:")
    for cls in CLASS_NAMES:
        m = pc[cls]
        print(f"  {cls:20s}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
    print(f"\nEARLY DETECTION (threshold={DETECTION_THRESHOLD}):")
    print(f"  Pre-action detection rate : {latency_stats.get('pre_action_detection_rate', 0):.2%}")
    print(f"  Mean latency              : {latency_stats.get('mean_latency')}")
    print(f"  Median latency            : {latency_stats.get('median_latency')}")
    print(f"\nTYPE A:  pre_action={la.get('pre_action_detection_rate', 0):.2%}  "
          f"mean_lat={la.get('mean_latency')}")
    print(f"TYPE B:  pre_action={lb.get('pre_action_detection_rate', 0):.2%}  "
          f"mean_lat={lb.get('mean_latency')}")
    print(f"\nSTEP-ORDER ABLATION:")
    print(f"  Original F1  : {shuffle_results['original_f1']:.4f}")
    print(f"  Shuffled F1  : {shuffle_results['shuffled_mean_f1']:.4f} "
          f"± {shuffle_results['shuffled_std_f1']:.4f}")
    print(f"  Difference   : {shuffle_results['f1_delta']:+.4f}")
    print(f"\n{'='*70}\n")


def _generate_handoff(split_info, exp_config, full_metrics, latency_stats,
                       shuffle_results, seed, tag):
    import shutil

    # Copy key files
    for src_path, dst_name in [
        (METRICS_DIR / f"{tag}_metrics.json",              "lstm_metrics.json"),
        (METRICS_DIR / f"latency_stats_{tag}.json",        "latency_stats.json"),
        (METRICS_DIR / f"shuffling_ablation_{tag}.json",   "shuffling_ablation.json"),
        (METRICS_DIR / f"counterfactual_analysis_{tag}.json", "counterfactual_analysis.json"),
        (SPLITS_DIR / f"split_seed{seed}.json",            "dataset_split.json"),
        (METRICS_DIR.parent / "experiment_config.json",    "experiment_config.json"),
    ]:
        if src_path.exists():
            shutil.copy2(src_path, HANDOFF_DIR / dst_name)

    # Copy plots
    plots_handoff = HANDOFF_DIR / "plots"
    plots_handoff.mkdir(exist_ok=True)
    for png in PLOTS_DIR.glob("*.png"):
        shutil.copy2(png, plots_handoff / png.name)

    # Copy confusion matrix
    cm_handoff = HANDOFF_DIR / "confusion_matrices"
    cm_handoff.mkdir(exist_ok=True)
    from src.config import CM_DIR
    for png in CM_DIR.glob("*.png"):
        shutil.copy2(png, cm_handoff / png.name)

    # Write README
    pc = full_metrics["per_class"]
    la = latency_stats.get("type_a", {})
    lb = latency_stats.get("type_b", {})

    readme = HANDOFF_DIR / "README.md"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("# TRACEGUARD — LSTM Results Handoff Package\n\n")
        f.write("This package contains all primary LSTM results for the TRACEGUARD project.\n\n")
        f.write("## Dataset\n")
        f.write("- **File:** `traceguard_v4_1.jsonl` (frozen, do not modify)\n")
        f.write("- **600 trajectories** | 200 BENIGN | 200 INJECTION_RESISTED | 200 HIJACKED\n")
        f.write("- **200 counterfactual groups** (unit of splitting)\n\n")
        f.write("## Split\n")
        f.write(f"- **Seed:** {seed}\n")
        f.write(f"- Train: {split_info['train_n_groups']} groups / {split_info['train_n_trajs']} trajectories\n")
        f.write(f"- Val:   {split_info['val_n_groups']} groups / {split_info['val_n_trajs']} trajectories\n")
        f.write(f"- Test:  {split_info['test_n_groups']} groups / {split_info['test_n_trajs']} trajectories\n")
        f.write(f"- Split group IDs: `dataset_split.json`\n\n")
        f.write("## Embedding Model\n")
        f.write(f"- **Model:** `{EMBEDDING_MODEL_NAME}`\n")
        f.write(f"- **Dimension:** {exp_config['embedding_dim']}\n")
        f.write(f"- Each trajectory = 6 step embeddings → shape [6, {exp_config['embedding_dim']}]\n\n")
        f.write("## Step Representation\n")
        f.write("Each step text is constructed as:\n```\n")
        f.write("ORIGINAL USER GOAL:\n{user_goal}\n\nCURRENT STEP:\n{step_number}\n\n")
        f.write("ACTION:\n{action}\n\nTOOL:\n{tool}\n\nTOOL INPUT:\n{tool_input}\n\n")
        f.write("TOOL OBSERVATION:\n{tool_observation}\n\nSTATE:\n{state}\n```\n\n")
        f.write("**Ground-truth metadata is NEVER included in step text.**\n\n")
        f.write("## LSTM Configuration\n")
        for k, v in exp_config["lstm_config"].items():
            f.write(f"- {k}: {v}\n")
        f.write(f"\n## Training\n")
        f.write(f"- Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})\n")
        f.write(f"- Batch size: {BATCH_SIZE}\n")
        f.write(f"- Max epochs: {MAX_EPOCHS}\n")
        f.write(f"- Early stopping patience: {EARLY_STOPPING_PATIENCE} (val Macro F1)\n")
        f.write(f"- Best epoch: {exp_config['best_epoch']}\n\n")
        f.write("## Results\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Accuracy | {full_metrics['accuracy']:.4f} |\n")
        f.write(f"| Macro F1 | {full_metrics['macro_f1']:.4f} |\n")
        for cls in CLASS_NAMES:
            m = pc[cls]
            f.write(f"| {cls} F1 | {m['f1']:.4f} |\n")
        f.write(f"\n## Detection\n")
        f.write(f"- **Threshold:** {DETECTION_THRESHOLD}\n")
        f.write(f"- Pre-action detection rate: {latency_stats.get('pre_action_detection_rate', 0):.2%}\n")
        f.write(f"- Mean latency: {latency_stats.get('mean_latency')}\n\n")
        f.write(f"| | Type A | Type B |\n|-|--------|--------|\n")
        f.write(f"| Pre-action rate | {la.get('pre_action_detection_rate',0):.2%} | "
                f"{lb.get('pre_action_detection_rate',0):.2%} |\n")
        f.write(f"| Mean latency | {la.get('mean_latency')} | {lb.get('mean_latency')} |\n\n")
        f.write("## Files\n")
        f.write("| File | Description |\n|------|-------------|\n")
        f.write("| `dataset_split.json` | All group IDs per split |\n")
        f.write("| `experiment_config.json` | Full experiment configuration |\n")
        f.write("| `lstm_metrics.json` | Full-trajectory test metrics |\n")
        f.write("| `latency_stats.json` | Detection latency metrics |\n")
        f.write("| `plots/` | All figures |\n")
        f.write("| `confusion_matrices/` | Confusion matrix PNG |\n\n")
        f.write("## For Teammate\n")
        f.write("When implementing your baseline/Transformer, keep identical:\n")
        f.write("- Dataset file (traceguard_v4_1.jsonl, unchanged)\n")
        f.write("- Split group IDs (use dataset_split.json, seed=42)\n")
        f.write("- Embedding model (all-MiniLM-L6-v2)\n")
        f.write("- Step text format (see above)\n")
        f.write("- Detection threshold (0.5)\n")
        f.write("- Class encoding: BENIGN=0, INJECTION_RESISTED=1, HIJACKED=2\n")
    print(f"[Handoff] Package created at: {HANDOFF_DIR}")


if __name__ == "__main__":
    main()
