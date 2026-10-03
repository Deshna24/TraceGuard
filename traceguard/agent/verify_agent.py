"""Focused verification for the bounded controlled TRACEGUARD agent."""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRACEGUARD_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRACEGUARD_DIR))

from agent.agent import (  # noqa: E402
    AgentOutputError,
    ControlledAgent,
    ScriptedActionModel,
    parse_model_output,
)
from agent.tools import build_default_registry  # noqa: E402
from runtime.action_gate import PreActionGate  # noqa: E402
from runtime.detector import DetectorPrediction  # noqa: E402
from src.config import DATA_PATH, MODELS_DIR  # noqa: E402


SPLIT_PATH = TRACEGUARD_DIR / "outputs" / "splits" / "split_seed42.json"
CHECKPOINT_PATH = MODELS_DIR / "lstm_seed42_best.pth"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StubDetector:
    def __init__(self, hijack_probability: float):
        self.user_goal = "stub"
        self.steps: list[dict] = []
        self.hijack_probability = hijack_probability

    def add_step(self, step: dict) -> None:
        self.steps.append(copy.deepcopy(step))

    def predict(self) -> DetectorPrediction:
        probability = self.hijack_probability
        return DetectorPrediction(
            probabilities={
                "BENIGN": 1.0 - probability,
                "INJECTION_RESISTED": 0.0,
                "HIJACKED": probability,
            },
            predicted_class="HIJACKED" if probability >= 0.5 else "BENIGN",
            prefix_length=len(self.steps),
        )

    def should_block(self, threshold: float = 0.5) -> bool:
        return self.hijack_probability >= threshold


def _agent(probability: float, outputs: list[str]):
    registry, tools = build_default_registry()
    gate = PreActionGate(StubDetector(probability))
    return ControlledAgent(ScriptedActionModel(outputs), registry, gate), tools


def main() -> int:
    hashes_before = {
        "dataset": _hash_file(DATA_PATH),
        "split": _hash_file(SPLIT_PATH),
        "checkpoint": _hash_file(CHECKPOINT_PATH),
    }

    kind, parsed = parse_model_output(
        '{"action":"calculate","tool":"calculator","tool_input":{"expression":"2+2"}}'
    )
    if kind != "action" or parsed.tool != "calculator":
        raise AssertionError("Valid action did not parse")
    for invalid in (
        "not json",
        '{"tool":"calculator","tool_input":{}}',
        '{"action":"x","tool":"calculator","tool_input":"not an object"}',
        '{"action":"x","tool":"calculator","tool_input":{},"extra":true}',
    ):
        try:
            parse_model_output(invalid)
        except AgentOutputError:
            pass
        else:
            raise AssertionError(f"Invalid output was accepted: {invalid}")

    unknown_agent, _ = _agent(
        0.1, ['{"action":"x","tool":"unknown","tool_input":{}}']
    )
    if unknown_agent.run("A task").status != "invalid_action":
        raise AssertionError("Unknown tool did not stop safely")

    normal_outputs = [
        '{"action":"Find the capital","tool":"search","tool_input":{"query":"capital of Japan"}}',
        '{"final":"The capital is Tokyo."}',
    ]
    normal_agent, normal_tools = _agent(0.1, normal_outputs)
    normal = normal_agent.run("Find the capital of Japan")
    if normal.status != "completed" or normal.answer != "The capital is Tokyo.":
        raise AssertionError(f"Normal run did not complete: {normal}")
    if len(normal.trajectory) != 1 or normal_tools["search"].execution_count != 1:
        raise AssertionError("Allowed action did not execute and update trajectory")
    if normal.user_goal != "Find the capital of Japan":
        raise AssertionError("Original goal was changed")
    if normal.gate_decisions[0].decision != "ALLOW":
        raise AssertionError("Allowed action did not pass the gate")

    blocked_outputs = [
        '{"action":"Export records","tool":"database","tool_input":{"operation":"export_records"}}',
    ]
    blocked_agent, blocked_tools = _agent(0.9, blocked_outputs)
    blocked = blocked_agent.run("Summarize the records")
    if blocked.status != "blocked" or blocked_tools["database"].execution_count != 0:
        raise AssertionError("Blocked action reached the tool")
    if blocked.blocked_action["tool"] != "database" or blocked.trajectory:
        raise AssertionError("Blocked action was not preserved safely")

    failing_agent, _ = _agent(0.1, ["malformed"])
    failure = failing_agent.run("A task")
    if failure.status != "model_failure":
        raise AssertionError("Malformed model output did not stop safely")

    hashes_after = {
        "dataset": _hash_file(DATA_PATH),
        "split": _hash_file(SPLIT_PATH),
        "checkpoint": _hash_file(CHECKPOINT_PATH),
    }
    if hashes_before != hashes_after:
        raise AssertionError("Frozen artifact hash changed during agent verification")

    print(
        {
            "all_passed": True,
            "module": "traceguard/agent/agent.py",
            "normal_status": normal.status,
            "blocked_status": blocked.status,
            "blocked_database_execution_count": blocked_tools["database"].execution_count,
            "goal_integrity": normal.user_goal == "Find the capital of Japan",
            "frozen_artifacts_unchanged": True,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
