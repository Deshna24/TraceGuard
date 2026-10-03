"""Focused verification for the inference-only TRACEGUARD detector wrapper."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRACEGUARD_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRACEGUARD_DIR))

from runtime.detector import TraceGuardDetector, threshold_blocks
from src.config import (
    CLASS_NAMES,
    DATA_PATH,
    DETECTION_THRESHOLD,
    LSTM_DROPOUT,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    MODELS_DIR,
)
from src.model_lstm import TraceGuardLSTM
from src.preprocessing import build_step_text


RESULTS_PATH = SCRIPT_DIR / "p0_detector_results.json"
CHECKPOINT_PATH = MODELS_DIR / "lstm_seed42_best.pth"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_first_trajectory() -> dict:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())


def _assert_probabilities(probabilities: dict[str, float]) -> None:
    if list(probabilities) != CLASS_NAMES:
        raise AssertionError(f"Unexpected class ordering: {list(probabilities)}")
    values = np.array(list(probabilities.values()), dtype=np.float64)
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise AssertionError(f"Probability outside [0, 1]: {probabilities}")
    if not np.isclose(values.sum(), 1.0, atol=1e-5):
        raise AssertionError(f"Probabilities do not sum to 1: {probabilities}")


def _direct_model(device: torch.device) -> TraceGuardLSTM:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = TraceGuardLSTM(
        input_size=384,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        num_classes=3,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> int:
    trajectory = _load_first_trajectory()
    user_goal = trajectory["user_goal"]
    steps = trajectory["steps"]
    checkpoint_hash_before = _hash_file(CHECKPOINT_PATH)

    detector = TraceGuardDetector(user_goal=user_goal)
    if detector.model.training:
        raise AssertionError("Detector model is not in evaluation mode")
    if any(parameter.requires_grad for parameter in detector.model.parameters()):
        raise AssertionError("Detector model has trainable parameters")

    model_source = inspect.getsource(TraceGuardDetector)
    for forbidden in (".train(", "backward(", "torch.save(", "optim."):
        if forbidden in model_source:
            raise AssertionError(f"Inference-only detector contains {forbidden}")

    direct_model = _direct_model(detector.device)
    direct_state_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in detector.model.named_parameters()
    }
    prefix_results = {}

    for prefix_length, step in enumerate(steps, start=1):
        detector.add_step(step)
        input_tensor = detector.encode_prefix()

        canonical_texts = [
            build_step_text(item, user_goal, item["step"])
            for item in steps[:prefix_length]
        ]
        canonical_embeddings = detector.embedding_model.encode(
            canonical_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        direct_input = torch.tensor(
            canonical_embeddings,
            dtype=torch.float32,
        ).unsqueeze(0).to(detector.device)
        if not torch.allclose(input_tensor, direct_input, rtol=0.0, atol=1e-6):
            raise AssertionError(f"Input tensor mismatch at prefix {prefix_length}")

        with torch.no_grad():
            direct_logits = direct_model(direct_input, [prefix_length])
            direct_probabilities = torch.softmax(direct_logits, dim=1)[0].cpu().numpy()

        prediction = detector.predict()
        probabilities = detector.get_class_probabilities()
        _assert_probabilities(probabilities)
        wrapper_probabilities = np.array(
            [probabilities[class_name] for class_name in CLASS_NAMES]
        )
        if not np.allclose(
            wrapper_probabilities,
            direct_probabilities,
            rtol=0.0,
            atol=1e-5,
        ):
            raise AssertionError(f"Probability mismatch at prefix {prefix_length}")
        if prediction.predicted_class != CLASS_NAMES[int(np.argmax(direct_probabilities))]:
            raise AssertionError(f"Predicted class mismatch at prefix {prefix_length}")
        if not np.isclose(
            detector.hijack_probability(),
            float(direct_probabilities[2]),
            rtol=0.0,
            atol=1e-5,
        ):
            raise AssertionError(f"P(HIJACKED) mismatch at prefix {prefix_length}")

        repeated = detector.predict()
        if repeated != prediction:
            raise AssertionError(f"Repeated prediction changed at prefix {prefix_length}")

        prefix_results[str(prefix_length)] = {
            "shape": list(input_tensor.shape),
            "probabilities": probabilities,
            "predicted_class": prediction.predicted_class,
            "hijack_probability": detector.hijack_probability(),
            "equivalent_to_direct": True,
        }

    for name, parameter in detector.model.named_parameters():
        if not torch.equal(parameter.detach().cpu(), direct_state_before[name]):
            raise AssertionError(f"Detector weights changed during inference: {name}")

    fresh = TraceGuardDetector(user_goal=user_goal, device=detector.device)
    for step in steps:
        fresh.add_step(step)
    fresh_prediction = fresh.predict()
    if not np.allclose(
        [fresh_prediction.probabilities[name] for name in CLASS_NAMES],
        [detector.predict().probabilities[name] for name in CLASS_NAMES],
        rtol=0.0,
        atol=1e-5,
    ):
        raise AssertionError("Fresh detector output differs from existing detector")

    isolated_a = TraceGuardDetector(user_goal=user_goal, device=detector.device)
    isolated_a.add_step(steps[0])
    isolated_a.add_step(steps[1])
    isolated_b = TraceGuardDetector(user_goal=user_goal, device=detector.device)
    isolated_b.add_step(steps[0])
    isolated_b_prediction = isolated_b.predict()
    fresh_single = TraceGuardDetector(user_goal=user_goal, device=detector.device)
    fresh_single.add_step(steps[0])
    if isolated_b_prediction != fresh_single.predict():
        raise AssertionError("Detector instances share prefix state")
    isolated_a.add_step(steps[2])
    if isolated_a.encode_prefix().shape != (1, 3, 384):
        raise AssertionError("Adding a step did not extend the existing prefix")

    empty = TraceGuardDetector(user_goal=user_goal, device=detector.device)
    try:
        empty.predict()
    except ValueError:
        pass
    else:
        raise AssertionError("Empty detector prediction did not fail clearly")

    caller_step = copy.deepcopy(steps[0])
    caller_step_original = copy.deepcopy(caller_step)
    mutation_check = TraceGuardDetector(user_goal=user_goal, device=detector.device)
    mutation_check.add_step(caller_step)
    caller_step["action"] = "mutated after add"
    if mutation_check.steps[0] != caller_step_original:
        raise AssertionError("Detector mutated by caller-owned step state")

    threshold_results = {
        "below": threshold_blocks(0.49),
        "equal": threshold_blocks(0.50),
        "above": threshold_blocks(0.51),
    }
    if threshold_results != {"below": False, "equal": True, "above": True}:
        raise AssertionError(f"Threshold boundary failed: {threshold_results}")

    checkpoint_hash_after = _hash_file(CHECKPOINT_PATH)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise AssertionError("Frozen checkpoint hash changed during detector verification")

    results = {
        "all_passed": True,
        "checkpoint": str(CHECKPOINT_PATH),
        "device": str(detector.device),
        "model_eval": True,
        "parameters_require_grad": False,
        "prefixes": prefix_results,
        "threshold": DETECTION_THRESHOLD,
        "threshold_results": threshold_results,
        "fresh_detector_equivalent": True,
        "state_isolation": True,
        "caller_step_isolation": True,
        "checkpoint_hash_before": checkpoint_hash_before,
        "checkpoint_hash_after": checkpoint_hash_after,
        "checkpoint_unchanged": True,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Detector verification passed. Results: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())