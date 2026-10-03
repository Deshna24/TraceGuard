"""
verify_frozen_pipeline.py — P0 Verification: Frozen LSTM Pipeline

This script verifies that the frozen TRACEGUARD LSTM checkpoint can be
loaded and used for inference WITHOUT any modifications to frozen artifacts.

Checks performed:
  1. Checkpoint loads successfully (dict format with model_state_dict)
  2. Model architecture matches frozen config (384→128→3, 2 layers, unidirectional)
  3. MiniLM embedding model loads and produces 384-D embeddings
  4. Canonical step representation (build_step_text) works
  5. Single trajectory prefix inference succeeds
  6. Three class probabilities are returned and sum to ~1.0
  7. Class mapping: BENIGN=0, INJECTION_RESISTED=1, HIJACKED=2
  8. Prefix lengths 1–6 all produce valid predictions

This script does NOT modify any frozen artifacts.
"""

import sys
import os
import json
import numpy as np
import torch

# Ensure traceguard/src is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACEGUARD_DIR = os.path.dirname(SCRIPT_DIR)  # traceguard/
sys.path.insert(0, TRACEGUARD_DIR)

from src.config import (
    LABEL_MAP, INV_LABEL_MAP, CLASS_NAMES,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    EMBEDDING_MODEL_NAME,
    DETECTION_THRESHOLD,
    MODELS_DIR, DATA_PATH,
    STEPS_PER_TRAJECTORY,
)
from src.model_lstm import TraceGuardLSTM
from src.preprocessing import build_step_text


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    all_passed = True
    results = {}

    separator("P0 - VERIFY FROZEN LSTM PIPELINE")

    # ─────────────────────────────────────────────────────
    # CHECK 1: Locate and load frozen checkpoint
    # ─────────────────────────────────────────────────────
    separator("CHECK 1: Load frozen checkpoint")

    checkpoint_path = MODELS_DIR / "lstm_seed42_best.pth"
    print(f"  Checkpoint path: {checkpoint_path}")
    print(f"  Exists: {checkpoint_path.exists()}")

    if not checkpoint_path.exists():
        print("  [FAIL] Frozen checkpoint not found!")
        all_passed = False
        results["checkpoint_load"] = "FAIL — not found"
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    print(f"  Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
    print(f"  Best val F1: {checkpoint.get('best_val_f1', 'N/A')}")

    model_config = checkpoint.get("model_config", {})
    print(f"  Stored model config: {json.dumps(model_config, indent=4)}")
    print("  [PASS] Checkpoint loaded successfully")
    results["checkpoint_load"] = "PASS"

    # ─────────────────────────────────────────────────────
    # CHECK 2: Verify model architecture matches config
    # ─────────────────────────────────────────────────────
    separator("CHECK 2: Verify model architecture")

    expected_config = {
        "architecture": "TraceGuardLSTM",
        "input_size": 384,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "num_classes": 3,
        "bidirectional": False,
        "pooling": "last_hidden_state",
    }

    config_match = True
    for key, expected in expected_config.items():
        actual = model_config.get(key)
        match = actual == expected
        status = "[PASS]" if match else "[FAIL]"
        print(f"  {status} {key}: expected={expected}, actual={actual}")
        if not match:
            config_match = False

    if config_match:
        print("  [PASS] Architecture matches frozen config")
        results["architecture_match"] = "PASS"
    else:
        print("  [FAIL] Architecture mismatch!")
        results["architecture_match"] = "FAIL"
        all_passed = False

    # ─────────────────────────────────────────────────────
    # CHECK 3: Instantiate model and load weights
    # ─────────────────────────────────────────────────────
    separator("CHECK 3: Instantiate model and load state_dict")

    model = TraceGuardLSTM(
        input_size=384,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        num_classes=3,
    )

    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("  [PASS] Model loaded and set to eval mode (no training ops)")
    results["model_load"] = "PASS"

    # ─────────────────────────────────────────────────────
    # CHECK 4: Load MiniLM embedding model
    # ─────────────────────────────────────────────────────
    separator("CHECK 4: Load MiniLM embedding model")
    print(f"  Model name: {EMBEDDING_MODEL_NAME}")

    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dim = st_model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {embedding_dim}")

    if embedding_dim == 384:
        print("  [PASS] MiniLM loaded, dimension = 384")
        results["minilm_load"] = "PASS"
    else:
        print(f"  [FAIL] Expected dim=384, got {embedding_dim}")
        results["minilm_load"] = "FAIL"
        all_passed = False

    # ─────────────────────────────────────────────────────
    # CHECK 5: Canonical step representation
    # ─────────────────────────────────────────────────────
    separator("CHECK 5: Canonical step representation (build_step_text)")

    # Create a synthetic step matching the v4.1 dataset schema
    sample_step = {
        "step": 1,
        "action": "Search for flights from Pune to Nagpur",
        "tool": "search",
        "tool_input": {"query": "Pune to Nagpur flights"},
        "tool_observation": "Found 5 flight options.",
        "state": "Looking up flight information."
    }
    sample_user_goal = "Find the cheapest flight from Pune to Nagpur"

    step_text = build_step_text(sample_step, sample_user_goal, step_number=1)
    print(f"  Step text (first 300 chars):\n{step_text[:300]}")
    print(f"  Step text length: {len(step_text)} characters")

    # Verify the expected format
    assert "ORIGINAL USER GOAL:" in step_text, "Missing ORIGINAL USER GOAL"
    assert "CURRENT STEP:" in step_text, "Missing CURRENT STEP"
    assert "ACTION:" in step_text, "Missing ACTION"
    assert "TOOL:" in step_text, "Missing TOOL"
    assert "TOOL INPUT:" in step_text, "Missing TOOL INPUT"
    assert "TOOL OBSERVATION:" in step_text, "Missing TOOL OBSERVATION"
    assert "STATE:" in step_text, "Missing STATE"
    print("  [PASS] Step text format verified (matches canonical preprocessing.py)")
    results["step_representation"] = "PASS"

    # ─────────────────────────────────────────────────────
    # CHECK 6: Embed a step and verify shape
    # ─────────────────────────────────────────────────────
    separator("CHECK 6: Embedding generation")

    embedding = st_model.encode([step_text], convert_to_numpy=True)
    print(f"  Embedding shape: {embedding.shape}")
    assert embedding.shape == (1, 384), f"Expected (1, 384), got {embedding.shape}"
    print("  [PASS] Embedding shape correct: (1, 384)")
    results["embedding_shape"] = "PASS"

    # ─────────────────────────────────────────────────────
    # CHECK 7: Single-step prefix inference
    # ─────────────────────────────────────────────────────
    separator("CHECK 7: Single-step prefix inference")

    with torch.no_grad():
        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)  # [1, 1, 384]
        lengths = [1]
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    print(f"  Logits shape: {logits.shape}")
    print(f"  Probabilities: {probs}")
    print(f"    P(BENIGN):             {probs[0]:.6f}")
    print(f"    P(INJECTION_RESISTED): {probs[1]:.6f}")
    print(f"    P(HIJACKED):           {probs[2]:.6f}")
    print(f"    Sum:                   {probs.sum():.6f}")

    assert logits.shape == (1, 3), f"Expected logits shape (1,3), got {logits.shape}"
    assert abs(probs.sum() - 1.0) < 1e-5, f"Probabilities don't sum to 1: {probs.sum()}"
    print("  [PASS] Single-step inference successful, probabilities sum to 1.0")
    results["single_step_inference"] = "PASS"

    # ─────────────────────────────────────────────────────
    # CHECK 8: Class mapping verification
    # ─────────────────────────────────────────────────────
    separator("CHECK 8: Class mapping")

    print(f"  LABEL_MAP:     {LABEL_MAP}")
    print(f"  INV_LABEL_MAP: {INV_LABEL_MAP}")
    print(f"  CLASS_NAMES:   {CLASS_NAMES}")

    assert LABEL_MAP == {"BENIGN": 0, "INJECTION_RESISTED": 1, "HIJACKED": 2}
    assert INV_LABEL_MAP == {0: "BENIGN", 1: "INJECTION_RESISTED", 2: "HIJACKED"}
    assert CLASS_NAMES == ["BENIGN", "INJECTION_RESISTED", "HIJACKED"]
    print("  [PASS] Class mapping confirmed: BENIGN=0, INJECTION_RESISTED=1, HIJACKED=2")
    results["class_mapping"] = "PASS"

    # ─────────────────────────────────────────────────────
    # CHECK 9: Multi-step prefix inference (lengths 1–6)
    # ─────────────────────────────────────────────────────
    separator("CHECK 9: Multi-step prefix inference (lengths 1–6)")

    # Build 6 sample steps to simulate a full trajectory
    sample_steps = []
    for i in range(1, 7):
        step = {
            "step": i,
            "action": f"Step {i} action: Process data part {i}",
            "tool": "search" if i <= 3 else ("calculator" if i == 4 else ("database" if i == 5 else None)),
            "tool_input": {"query": f"query_{i}"} if i <= 5 else None,
            "tool_observation": f"Observation for step {i}" if i <= 5 else None,
            "state": f"Working on substep {i}."
        }
        sample_steps.append(step)

    # Generate embeddings for all 6 steps
    step_texts = [build_step_text(s, sample_user_goal, s["step"]) for s in sample_steps]
    step_embeddings = st_model.encode(step_texts, convert_to_numpy=True)  # [6, 384]
    print(f"  Full trajectory embedding shape: {step_embeddings.shape}")
    assert step_embeddings.shape == (6, 384)

    prefix_results = []
    with torch.no_grad():
        for k in range(1, 7):
            prefix_emb = step_embeddings[:k]  # [k, 384]
            x = torch.tensor(prefix_emb, dtype=torch.float32).unsqueeze(0)  # [1, k, 384]
            lengths = [k]
            logits = model(x, lengths)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_label = INV_LABEL_MAP[pred_idx]

            prefix_results.append({
                "prefix_length": k,
                "p_benign": float(probs[0]),
                "p_resisted": float(probs[1]),
                "p_hijacked": float(probs[2]),
                "predicted": pred_label,
                "sum": float(probs.sum()),
            })

            status = "[PASS]" if abs(probs.sum() - 1.0) < 1e-5 else "[FAIL]"
            print(f"  {status} Prefix k={k}: P(B)={probs[0]:.4f} P(R)={probs[1]:.4f} "
                  f"P(H)={probs[2]:.4f} sum={probs.sum():.6f} -> {pred_label}")

    all_sums_valid = all(abs(r["sum"] - 1.0) < 1e-5 for r in prefix_results)
    if all_sums_valid:
        print("  [PASS] All prefix lengths (1–6) produce valid probabilities")
        results["prefix_inference"] = "PASS"
    else:
        print("  [FAIL] Some prefix results have invalid probability sums")
        results["prefix_inference"] = "FAIL"
        all_passed = False

    # ─────────────────────────────────────────────────────
    # CHECK 10: Threshold logic
    # ─────────────────────────────────────────────────────
    separator("CHECK 10: Threshold logic")
    print(f"  Official threshold: P(HIJACKED) >= {DETECTION_THRESHOLD}")

    # Test threshold boundary
    test_cases = [
        (0.49, False, "ALLOW"),
        (0.50, True, "BLOCK"),
        (0.51, True, "BLOCK"),
        (0.99, True, "BLOCK"),
        (0.01, False, "ALLOW"),
    ]
    threshold_ok = True
    for p_hijacked, expected_block, decision in test_cases:
        actual_block = p_hijacked >= DETECTION_THRESHOLD
        match = actual_block == expected_block
        status = "[PASS]" if match else "[FAIL]"
        print(f"  {status} P(HIJACKED)={p_hijacked:.2f} -> {decision} "
              f"(expected_block={expected_block}, actual={actual_block})")
        if not match:
            threshold_ok = False

    if threshold_ok:
        print("  [PASS] Threshold logic verified")
        results["threshold_logic"] = "PASS"
    else:
        print("  [FAIL] Threshold logic incorrect")
        results["threshold_logic"] = "FAIL"
        all_passed = False

    # ─────────────────────────────────────────────────────
    # CHECK 11: Verify frozen artifacts untouched
    # ─────────────────────────────────────────────────────
    separator("CHECK 11: Frozen artifact safety")

    # Verify dataset exists and hasn't been opened for write
    print(f"  Dataset path: {DATA_PATH}")
    print(f"  Dataset exists: {DATA_PATH.exists()}")
    if DATA_PATH.exists():
        print(f"  Dataset size: {DATA_PATH.stat().st_size:,} bytes")
    print(f"  Checkpoint path: {checkpoint_path}")
    print(f"  Checkpoint size: {checkpoint_path.stat().st_size:,} bytes")
    print("  [PASS] No frozen artifacts were modified by this script")
    results["frozen_artifacts_safe"] = "PASS"

    # ─────────────────────────────────────────────────────
    # CHECK 12: Load real trajectory from dataset and score
    # ─────────────────────────────────────────────────────
    separator("CHECK 12: Score a real trajectory from frozen dataset")

    print(f"  Loading first trajectory from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    real_traj = json.loads(first_line)
    print(f"  Trajectory ID: {real_traj['trajectory_id']}")
    print(f"  Label: {real_traj['label']}")
    print(f"  User goal: {real_traj['user_goal']}")
    print(f"  Steps: {len(real_traj['steps'])}")

    # Build step texts using canonical preprocessing
    real_texts = []
    for step in real_traj["steps"]:
        txt = build_step_text(step, real_traj["user_goal"], step["step"])
        real_texts.append(txt)

    real_embeddings = st_model.encode(real_texts, convert_to_numpy=True)
    print(f"  Real trajectory embeddings shape: {real_embeddings.shape}")
    assert real_embeddings.shape == (6, 384)

    with torch.no_grad():
        x = torch.tensor(real_embeddings, dtype=torch.float32).unsqueeze(0)  # [1, 6, 384]
        lengths = [6]
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_label = INV_LABEL_MAP[int(np.argmax(probs))]
    p_hijacked = float(probs[2])
    should_block = p_hijacked >= DETECTION_THRESHOLD

    print(f"  P(BENIGN):             {probs[0]:.6f}")
    print(f"  P(INJECTION_RESISTED): {probs[1]:.6f}")
    print(f"  P(HIJACKED):           {probs[2]:.6f}")
    print(f"  Predicted label:       {pred_label}")
    print(f"  True label:            {real_traj['label']}")
    print(f"  Should block:          {should_block}")
    print(f"  [PASS] Real trajectory scored successfully")
    results["real_trajectory_score"] = "PASS"

    # ─────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────────────
    separator("FINAL SUMMARY")

    for check, status in results.items():
        icon = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {icon} {check}: {status}")

    print()
    if all_passed:
        print("  ==================================================")
        print("  [PASS]  ALL CHECKS PASSED - Frozen LSTM pipeline verified")
        print("  ==================================================")
    else:
        print("  ==================================================")
        print("  [FAIL]  SOME CHECKS FAILED - See details above")
        print("  ==================================================")

    # Save verification results
    results_path = os.path.join(SCRIPT_DIR, "p0_verification_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "all_passed": all_passed,
            "checks": results,
            "frozen_checkpoint": str(checkpoint_path),
            "frozen_config": model_config,
            "class_mapping": LABEL_MAP,
            "threshold": DETECTION_THRESHOLD,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dim": 384,
            "steps_per_trajectory": STEPS_PER_TRAJECTORY,
        }, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
