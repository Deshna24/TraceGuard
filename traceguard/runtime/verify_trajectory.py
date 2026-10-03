"""Focused verification for the TRACEGUARD trajectory state manager."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRACEGUARD_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRACEGUARD_DIR))

from runtime.detector import TraceGuardDetector
from runtime.trajectory import CANONICAL_STEP_FIELDS, TrajectoryState
from src.config import DATA_PATH, MODELS_DIR


RESULTS_PATH = SCRIPT_DIR / "p0_trajectory_results.json"
SPLIT_PATH = TRACEGUARD_DIR / "outputs" / "splits" / "split_seed42.json"
CHECKPOINT_PATH = MODELS_DIR / "lstm_seed42_best.pth"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_hash(path: Path) -> str:
    completed = subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=TRACEGUARD_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _step(number: int) -> dict:
    return {
        "step": number,
        "action": f"Process item {number}",
        "tool": "search" if number < 4 else "calculator",
        "tool_input": {"query": f"item-{number}", "filters": {"rank": number}},
        "tool_observation": {"status": "ok", "items": [number]},
        "state": f"Working on item {number}",
    }


def _expect_error(callable_, error_type: type[BaseException], label: str) -> None:
    try:
        callable_()
    except error_type:
        return
    raise AssertionError(f"{label} did not raise {error_type.__name__}")


def main() -> int:
    checkpoint_hash_before = _git_hash(CHECKPOINT_PATH)
    dataset_hash_before = _git_hash(DATA_PATH)
    split_hash_before = _git_hash(SPLIT_PATH)

    trajectory = TrajectoryState("Process the six requested items")
    if trajectory.user_goal != "Process the six requested items":
        raise AssertionError("User goal was not stored")
    if trajectory.length() != 0 or trajectory.get_current_step() is not None:
        raise AssertionError("New trajectory is not empty")
    if tuple(CANONICAL_STEP_FIELDS) != (
        "step",
        "action",
        "tool",
        "tool_input",
        "tool_observation",
        "state",
    ):
        raise AssertionError("Canonical runtime fields changed")

    steps = [_step(number) for number in range(1, 7)]
    for step in steps:
        trajectory.add_step(step)
    if trajectory.length() != 6:
        raise AssertionError("Six steps were not stored")
    if [step["step"] for step in trajectory.get_steps()] != list(range(1, 7)):
        raise AssertionError("Chronological ordering was not preserved")
    if trajectory.get_current_step()["step"] != 6:
        raise AssertionError("Current step is incorrect")

    returned_steps = trajectory.get_steps()
    returned_steps[0]["tool_input"]["filters"]["rank"] = 999
    returned_steps[0]["tool_observation"]["items"].append(999)
    if trajectory.get_steps()[0]["tool_input"]["filters"]["rank"] != 1:
        raise AssertionError("Returned steps expose internal nested state")
    if trajectory.get_steps()[0]["tool_observation"]["items"] != [1]:
        raise AssertionError("Returned observations expose internal nested state")

    caller_step = _step(1)
    caller_copy = copy.deepcopy(caller_step)
    external_mutation = TrajectoryState("Goal")
    external_mutation.add_step(caller_step)
    caller_step["tool_input"]["filters"]["rank"] = 88
    caller_step["tool_observation"]["items"].append(88)
    if external_mutation.get_current_step() != caller_copy:
        raise AssertionError("Caller mutation changed stored trajectory state")

    _expect_error(lambda: TrajectoryState("Goal").add_step(_step(2)), ValueError, "Skipped first step")
    _expect_error(lambda: trajectory.add_step(_step(6)), ValueError, "Duplicate step")
    _expect_error(lambda: trajectory.add_step(_step(8)), ValueError, "Skipped step")
    _expect_error(lambda: trajectory.add_step(_step(5)), ValueError, "Out-of-order step")
    _expect_error(lambda: trajectory.add_step({"step": True}), ValueError, "Boolean step")

    optional_values = TrajectoryState("Goal with optional values")
    optional_values.add_step({"step": 1})
    optional_text = optional_values.to_canonical_texts()
    if len(optional_text) != 1 or "CURRENT STEP:\n1" not in optional_text[0]:
        raise AssertionError("Missing optional values are not canonical-compatible")

    full_steps_before_prefix = trajectory.get_steps()
    for length in range(1, 7):
        prefix = trajectory.get_prefix(length)
        if [step["step"] for step in prefix] != list(range(1, length + 1)):
            raise AssertionError(f"Prefix {length} is not chronological")
        prefix[0]["action"] = "mutated prefix"
        if trajectory.get_steps() != full_steps_before_prefix:
            raise AssertionError(f"Prefix {length} mutated full trajectory")
        texts = trajectory.to_canonical_texts(length)
        if len(texts) != length or f"CURRENT STEP:\n{length}" not in texts[-1]:
            raise AssertionError(f"Prefix {length} is not canonical-compatible")

    _expect_error(lambda: trajectory.get_prefix(0), ValueError, "Zero prefix")
    _expect_error(lambda: trajectory.get_prefix(7), ValueError, "Future prefix")
    _expect_error(lambda: trajectory.get_prefix("3"), TypeError, "Non-integer prefix")
    _expect_error(
        lambda: setattr(trajectory, "user_goal", "replaced"),
        AttributeError,
        "User-goal replacement",
    )

    detector = TraceGuardDetector(user_goal=trajectory.user_goal)
    for step in trajectory.get_prefix(3):
        detector.add_step(step)
    prediction = detector.predict()
    probabilities = prediction.probabilities
    if prediction.prefix_length != 3:
        raise AssertionError("Detector received the wrong trajectory prefix length")
    if list(probabilities) != ["BENIGN", "INJECTION_RESISTED", "HIJACKED"]:
        raise AssertionError("Detector class ordering is incompatible")
    if abs(sum(probabilities.values()) - 1.0) > 1e-5:
        raise AssertionError("Detector probabilities are invalid")
    if "should_block" in inspect.getsource(TrajectoryState):
        raise AssertionError("Trajectory manager contains safety-gate logic")

    trajectory.reset()
    if trajectory.length() != 0 or trajectory.user_goal != "Process the six requested items":
        raise AssertionError("Reset did not preserve goal and clear steps")
    trajectory.add_step(_step(1))
    if trajectory.get_current_step()["step"] != 1:
        raise AssertionError("Step numbering did not restart after reset")

    checkpoint_hash_after = _git_hash(CHECKPOINT_PATH)
    dataset_hash_after = _git_hash(DATA_PATH)
    split_hash_after = _git_hash(SPLIT_PATH)
    if checkpoint_hash_before != checkpoint_hash_after:
        raise AssertionError("Checkpoint changed during trajectory verification")
    if dataset_hash_before != dataset_hash_after:
        raise AssertionError("Dataset changed during trajectory verification")
    if split_hash_before != split_hash_after:
        raise AssertionError("Split changed during trajectory verification")

    results = {
        "all_passed": True,
        "module": "traceguard/runtime/trajectory.py",
        "class": "TrajectoryState",
        "stored_fields": list(CANONICAL_STEP_FIELDS),
        "step_numbering": "1-based consecutive order; invalid, duplicate, skipped, and out-of-order steps rejected",
        "prefixes": {str(length): True for length in range(1, 7)},
        "external_mutation_protection": True,
        "prefix_isolation": True,
        "user_goal_integrity": True,
        "detector_compatibility": {
            "canonical_texts": True,
            "prefix_length": 3,
            "inference_succeeded": True,
            "probabilities_valid": True,
        },
        "temporary_detector_artifacts_removed": [],
        "frozen_hashes_unchanged": True,
        "checkpoint_git_hash": checkpoint_hash_after,
        "dataset_git_hash": dataset_hash_after,
        "split_git_hash": split_hash_after,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Trajectory verification passed. Results: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())