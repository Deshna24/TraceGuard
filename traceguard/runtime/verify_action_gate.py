"""Focused verification for the fail-closed TRACEGUARD pre-action gate."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRACEGUARD_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRACEGUARD_DIR))

from runtime.action_gate import (  # noqa: E402
    BlockedActionError,
    PreActionGate,
)
from runtime.detector import DetectorPrediction, TraceGuardDetector  # noqa: E402
from runtime.trajectory import TrajectoryState  # noqa: E402
from src.config import DATA_PATH, DETECTION_THRESHOLD, MODELS_DIR  # noqa: E402


RESULTS_PATH = SCRIPT_DIR / "p0_action_gate_results.json"
SPLIT_PATH = TRACEGUARD_DIR / "outputs" / "splits" / "split_seed42.json"
CHECKPOINT_PATH = MODELS_DIR / "lstm_seed42_best.pth"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _step(number: int) -> dict:
    return {
        "step": number,
        "action": f"Inspect item {number}",
        "tool": "search",
        "tool_input": {"query": f"item-{number}"},
        "tool_observation": {"status": "ok"},
        "state": f"Working on item {number}",
    }


class StubDetector:
    """Small detector double for threshold and failure-path checks."""

    def __init__(self, hijack_probability: float):
        self.user_goal = "test goal"
        self.steps: list[dict] = []
        self.hijack_probability = hijack_probability
        self.predict_calls = 0

    def add_step(self, step: dict) -> None:
        self.steps.append(copy.deepcopy(step))

    def predict(self) -> DetectorPrediction:
        self.predict_calls += 1
        p = self.hijack_probability
        return DetectorPrediction(
            probabilities={
                "BENIGN": 1.0 - p,
                "INJECTION_RESISTED": 0.0,
                "HIJACKED": p,
            },
            predicted_class="HIJACKED" if p >= 0.5 else "BENIGN",
            prefix_length=len(self.steps),
        )

    def should_block(self, threshold: float = DETECTION_THRESHOLD) -> bool:
        return self.hijack_probability >= threshold


class FailingDetector(StubDetector):
    def predict(self) -> DetectorPrediction:
        raise RuntimeError("simulated inference failure")


def _action() -> dict:
    return {
        "action": "Search the requested item",
        "tool": "search",
        "tool_input": {"query": "item-1"},
    }


def _assert_threshold_boundaries() -> None:
    trajectory = TrajectoryState("test goal")
    trajectory.add_step(_step(1))
    for probability, expected in ((0.49, "ALLOW"), (0.50, "BLOCK"), (0.51, "BLOCK")):
        result = PreActionGate(StubDetector(probability)).evaluate(_action(), trajectory)
        if result.decision != expected or result.hijack_probability != probability:
            raise AssertionError(f"Threshold {probability} produced {result}")


def _assert_execution_enforcement() -> None:
    trajectory = TrajectoryState("test goal")
    trajectory.add_step(_step(1))
    calls = {"count": 0}

    def executor(_action: dict) -> None:
        calls["count"] += 1

    blocked = PreActionGate(StubDetector(0.51)).evaluate(_action(), trajectory)
    try:
        PreActionGate.execute_if_allowed(blocked, executor)
    except BlockedActionError:
        pass
    else:
        raise AssertionError("Blocked action did not raise")
    if calls["count"] != 0:
        raise AssertionError("Blocked action executed its tool")

    allowed = PreActionGate(StubDetector(0.49)).evaluate(_action(), trajectory)
    PreActionGate.execute_if_allowed(allowed, executor)
    if calls["count"] != 1:
        raise AssertionError("Allowed action was not executable")


def _assert_failure_is_blocked() -> None:
    trajectory = TrajectoryState("test goal")
    trajectory.add_step(_step(1))
    result = PreActionGate(FailingDetector(0.0)).evaluate(_action(), trajectory)
    if result.decision != "BLOCK" or result.error is None or result.pre_action is not True:
        raise AssertionError(f"Detector failure did not fail closed: {result}")


def _assert_real_detector() -> dict:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        source = json.loads(handle.readline())
    trajectory = TrajectoryState(source["user_goal"])
    for step in source["steps"][:3]:
        trajectory.add_step(step)
    detector = TraceGuardDetector(user_goal=trajectory.user_goal)
    gate = PreActionGate(detector)
    results = [
        gate.evaluate(_action(), trajectory.get_prefix(length))
        for length in range(1, 4)
    ]
    for length, result in enumerate(results, start=1):
        if result.error is not None or result.hijack_probability is None:
            raise AssertionError(f"Real detector gate evaluation failed: {result.error}")
        if result.detection_step != length or result.pre_action is not True:
            raise AssertionError("Real detector result has incorrect pre-action metadata")
        if result.proposed_action != _action():
            raise AssertionError("Proposed action was not preserved")
        if result.decision != ("BLOCK" if result.hijack_probability >= 0.5 else "ALLOW"):
            raise AssertionError("Real detector threshold decision is incorrect")
    return {
        "decisions": [result.decision for result in results],
        "hijack_probabilities": [result.hijack_probability for result in results],
        "predicted_classes": [result.predicted_class for result in results],
        "prefix_lengths": [result.detection_step for result in results],
        "pre_action": all(result.pre_action for result in results),
    }


def main() -> int:
    hashes_before = {
        "dataset": _hash_file(DATA_PATH),
        "split": _hash_file(SPLIT_PATH),
        "checkpoint": _hash_file(CHECKPOINT_PATH),
    }
    _assert_threshold_boundaries()
    _assert_execution_enforcement()
    _assert_failure_is_blocked()
    real_detector = _assert_real_detector()

    trajectory = TrajectoryState("test goal")
    steps = [_step(number) for number in range(1, 4)]
    for step in steps:
        trajectory.add_step(step)
    detector = StubDetector(0.1)
    gate = PreActionGate(detector)
    decisions = [
        gate.evaluate(_action(), trajectory.get_prefix(length))
        for length in range(1, 4)
    ]
    if [decision.detection_step for decision in decisions] != [1, 2, 3]:
        raise AssertionError("Repeated prefix evaluation used stale detection steps")
    if [decision.hijack_probability for decision in decisions] != [0.1, 0.1, 0.1]:
        raise AssertionError("Repeated prefix evaluation leaked stale probabilities")
    if trajectory.length() != 3 or [step["step"] for step in trajectory.get_steps()] != [1, 2, 3]:
        raise AssertionError("Gate mutated trajectory state")

    hashes_after = {
        "dataset": _hash_file(DATA_PATH),
        "split": _hash_file(SPLIT_PATH),
        "checkpoint": _hash_file(CHECKPOINT_PATH),
    }
    if hashes_before != hashes_after:
        raise AssertionError("Frozen artifact hash changed during gate verification")

    results = {
        "all_passed": True,
        "module": "traceguard/runtime/action_gate.py",
        "class": "PreActionGate",
        "threshold": DETECTION_THRESHOLD,
        "threshold_results": {"0.49": "ALLOW", "0.50": "BLOCK", "0.51": "BLOCK"},
        "failure_behavior": "BLOCK with explicit error",
        "blocked_action_execution_count": 0,
        "allowed_action_execution_count": 1,
        "real_detector": real_detector,
        "repeated_prefix_evaluation": True,
        "frozen_artifacts_unchanged": True,
        "hashes": {"before": hashes_before, "after": hashes_after},
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Action-gate verification passed. Results: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
