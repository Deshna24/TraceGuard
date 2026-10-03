"""Real-detector verification for the deterministic BENIGN scenario."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TRACEGUARD_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRACEGUARD_DIR))

from agent.agent import ControlledAgent, ScriptedActionModel  # noqa: E402
from agent.scenarios import BENIGN_SCENARIO  # noqa: E402
from agent.tools import build_default_registry  # noqa: E402
from runtime.action_gate import PreActionGate  # noqa: E402
from runtime.detector import TraceGuardDetector  # noqa: E402
from src.config import DATA_PATH, DETECTION_THRESHOLD, MODELS_DIR  # noqa: E402


RESULTS_PATH = SCRIPT_DIR / "p0_benign_results.json"
SPLIT_PATH = TRACEGUARD_DIR / "outputs" / "splits" / "split_seed42.json"
CHECKPOINT_PATH = MODELS_DIR / "lstm_seed42_best.pth"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_once() -> tuple[dict, dict]:
    registry, tools = build_default_registry()
    detector = TraceGuardDetector(user_goal=BENIGN_SCENARIO.user_goal)
    agent = ControlledAgent(
        ScriptedActionModel(BENIGN_SCENARIO.model_outputs),
        registry,
        PreActionGate(detector),
    )
    result = agent.run(BENIGN_SCENARIO.user_goal)
    if result.status != "completed":
        raise AssertionError(f"BENIGN scenario did not complete: {result}")
    if result.user_goal != BENIGN_SCENARIO.user_goal:
        raise AssertionError("BENIGN user goal changed")
    if len(result.gate_decisions) != 1:
        raise AssertionError("BENIGN scenario evaluated an unexpected number of actions")
    decision = result.gate_decisions[0]
    if decision.decision != "ALLOW":
        raise AssertionError(
            f"Real detector blocked the legitimate BENIGN action: {decision}"
        )
    if decision.hijack_probability is None:
        raise AssertionError("BENIGN gate did not expose P(HIJACKED)")
    if decision.threshold != DETECTION_THRESHOLD:
        raise AssertionError("BENIGN gate threshold was overridden")
    if decision.detection_step != 1 or not decision.pre_action:
        raise AssertionError("BENIGN detection metadata is incorrect")
    if tools["calculator"].execution_count != 1:
        raise AssertionError("BENIGN calculator did not execute exactly once")
    if len(result.trajectory) != 1:
        raise AssertionError("BENIGN trajectory has incorrect length")
    step = result.trajectory[0]
    if step["step"] != 1 or step["tool"] != "calculator":
        raise AssertionError("BENIGN trajectory step is incorrect")
    if step["tool_observation"]["output"] != 450:
        raise AssertionError("BENIGN observation has the wrong result")
    if "Injected instruction" in json.dumps(step["tool_observation"]):
        raise AssertionError("BENIGN observation contains injection text")
    if len(result.trajectory) != len(detector.steps):
        # The gate's pre-action candidate is intentionally replaced by the
        # completed observation in the agent trajectory after execution.
        if len(result.trajectory) != 1:
            raise AssertionError("BENIGN trajectory/detector lifecycle is inconsistent")
    result_summary = {
        "status": result.status,
        "user_goal": result.user_goal,
        "trajectory": result.trajectory,
        "gate_decision": decision.decision,
        "hijack_probability": decision.hijack_probability,
        "class_probabilities": decision.class_probabilities,
        "threshold": decision.threshold,
        "detection_step": decision.detection_step,
        "pre_action": decision.pre_action,
    }
    return result_summary, {
        "tool_execution_count": tools["calculator"].execution_count,
        "detector_steps": len(detector.steps),
    }


def main() -> int:
    hashes_before = {
        "dataset": _hash_file(DATA_PATH),
        "split": _hash_file(SPLIT_PATH),
        "checkpoint": _hash_file(CHECKPOINT_PATH),
    }
    runs = [_run_once(), _run_once(), _run_once()]
    first, first_meta = runs[0]
    for run, metadata in runs[1:]:
        if first["trajectory"] != run["trajectory"]:
            raise AssertionError("BENIGN trajectory was not repeatable")
        if first["gate_decision"] != run["gate_decision"]:
            raise AssertionError("BENIGN gate decision was not repeatable")
        if not np.isclose(
            first["hijack_probability"], run["hijack_probability"], rtol=0.0, atol=1e-5
        ):
            raise AssertionError("BENIGN probability was not stable within tolerance")
        if first_meta != metadata:
            raise AssertionError("BENIGN execution metadata was not repeatable")

    hashes_after = {
        "dataset": _hash_file(DATA_PATH),
        "split": _hash_file(SPLIT_PATH),
        "checkpoint": _hash_file(CHECKPOINT_PATH),
    }
    if hashes_before != hashes_after:
        raise AssertionError("Frozen artifact hash changed during BENIGN verification")
    output = {
        "all_passed": True,
        "scenario": BENIGN_SCENARIO.name,
        "user_goal": BENIGN_SCENARIO.user_goal,
        "expected_action": {
            "tool": BENIGN_SCENARIO.expected_tool,
            "tool_input": BENIGN_SCENARIO.expected_tool_input,
        },
        "observed": first,
        "repeatability": {"runs": 3, "stable": True},
        "frozen_artifacts_unchanged": True,
        "hashes": {"before": hashes_before, "after": hashes_after},
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"BENIGN verification passed. Results: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
