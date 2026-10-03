"""Read-only, JSON-serializable runtime records for TRACEGUARD demo runs."""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
CLASS_NAMES = ("BENIGN", "INJECTION_RESISTED", "HIJACKED")
DEFAULT_LOG_DIRECTORY = Path(__file__).resolve().parent.parent / "runtime_logs"


def _tool_result_record(result: Any) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "tool": result.tool,
        "output": copy.deepcopy(result.output),
        "error": result.error,
    }


def _gate_record(index: int, decision: Any) -> dict[str, Any]:
    probabilities = (
        None
        if decision.class_probabilities is None
        else {name: float(value) for name, value in decision.class_probabilities.items()}
    )
    return {
        "evaluation_index": index,
        "class_probabilities": probabilities,
        "predicted_class": decision.predicted_class,
        "hijack_probability": decision.hijack_probability,
        "threshold": float(decision.threshold),
        "detection_step": decision.detection_step,
        "pre_action": bool(decision.pre_action),
        "decision": decision.decision,
        "blocked": bool(decision.blocked),
        "proposed_action": copy.deepcopy(decision.proposed_action),
        "error": decision.error,
    }


def build_run_log(
    *,
    scenario: str,
    run_id: str,
    result: Any,
    tool_execution_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Create an immutable-style record from an already completed agent run.

    This function only copies observable run data. It does not invoke tools,
    alter a detector, change a trajectory, or make a gate decision.
    """
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("scenario must be a non-empty string")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(tool_execution_counts, Mapping):
        raise TypeError("tool_execution_counts must be a mapping")
    required = ("user_goal", "trajectory", "gate_decisions", "tool_results", "status")
    if any(not hasattr(result, field) for field in required):
        raise TypeError("result must expose the controlled-agent run result fields")

    evaluations = [
        _gate_record(index, decision)
        for index, decision in enumerate(result.gate_decisions, start=1)
    ]
    crossings = [
        evaluation["detection_step"]
        for evaluation in evaluations
        if evaluation["hijack_probability"] is not None
        and evaluation["hijack_probability"] >= evaluation["threshold"]
    ]
    trajectory = copy.deepcopy(result.trajectory)
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario,
        "run_id": run_id,
        "user_goal": result.user_goal,
        "trajectory": trajectory,
        "proposed_actions": [
            copy.deepcopy(evaluation["proposed_action"]) for evaluation in evaluations
        ],
        "tool_observations": [
            {
                "step": step["step"],
                "tool": step.get("tool"),
                "observation": copy.deepcopy(step.get("tool_observation")),
            }
            for step in trajectory
        ],
        "detector_evaluations": evaluations,
        "gate_decisions": [
            {
                "evaluation_index": evaluation["evaluation_index"],
                "decision": evaluation["decision"],
                "blocked": evaluation["blocked"],
                "proposed_action": copy.deepcopy(evaluation["proposed_action"]),
            }
            for evaluation in evaluations
        ],
        "first_threshold_crossing_step": crossings[0] if crossings else None,
        "actions_blocked": any(evaluation["blocked"] for evaluation in evaluations),
        "blocked_action": copy.deepcopy(result.blocked_action),
        "tool_execution_results": [_tool_result_record(tool_result) for tool_result in result.tool_results],
        "tool_execution_counts": {
            str(tool): int(count) for tool, count in tool_execution_counts.items()
        },
        "final_status": result.status,
        "final_answer": result.answer,
        "error": result.error,
    }


def write_run_log(record: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write one already-built runtime record as UTF-8 JSON."""
    validate_run_log(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def validate_run_log(record: Mapping[str, Any]) -> None:
    """Validate the stable schema without recomputing detector behavior."""
    if not isinstance(record, Mapping):
        raise TypeError("runtime log must be a mapping")
    required = {
        "schema_version", "scenario", "run_id", "user_goal", "trajectory", "proposed_actions",
        "tool_observations", "detector_evaluations", "gate_decisions",
        "first_threshold_crossing_step", "actions_blocked", "blocked_action",
        "tool_execution_results", "tool_execution_counts", "final_status",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"runtime log is missing required fields: {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported runtime log schema: {record['schema_version']!r}")
    if not all(isinstance(record[field], str) and record[field].strip() for field in ("scenario", "run_id", "user_goal", "final_status")):
        raise ValueError("scenario, run_id, user_goal, and final_status must be non-empty strings")
    trajectory = record["trajectory"]
    if not isinstance(trajectory, list):
        raise TypeError("trajectory must be a list")
    for expected_step, step in enumerate(trajectory, start=1):
        if not isinstance(step, Mapping) or step.get("step") != expected_step:
            raise ValueError("trajectory steps must be chronological and 1-based")
    evaluations = record["detector_evaluations"]
    decisions = record["gate_decisions"]
    if not isinstance(evaluations, list) or len(evaluations) != len(decisions):
        raise ValueError("detector evaluations and gate decisions must have equal lengths")
    crossings: list[int] = []
    for index, (evaluation, gate) in enumerate(zip(evaluations, decisions, strict=True), start=1):
        if evaluation.get("evaluation_index") != index or gate.get("evaluation_index") != index:
            raise ValueError("evaluation indexes must be consecutive")
        probabilities = evaluation.get("class_probabilities")
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(CLASS_NAMES):
            raise ValueError("each detector evaluation must contain all class probabilities")
        values = [probabilities[name] for name in CLASS_NAMES]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("class probabilities must be finite values in [0, 1]")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
            raise ValueError("class probabilities must sum to one")
        probability = evaluation.get("hijack_probability")
        threshold = evaluation.get("threshold")
        if probability != probabilities["HIJACKED"] or threshold != 0.5:
            raise ValueError("evaluation probability or fixed threshold is inconsistent")
        expected_decision = "BLOCK" if probability >= threshold else "ALLOW"
        if evaluation.get("decision") != expected_decision or gate.get("decision") != expected_decision:
            raise ValueError("gate decision is inconsistent with the recorded detector probability")
        if evaluation.get("blocked") != (expected_decision == "BLOCK") or gate.get("blocked") != (expected_decision == "BLOCK"):
            raise ValueError("blocked flag is inconsistent with gate decision")
        if evaluation.get("pre_action") is not True:
            raise ValueError("detector evaluations must be pre-action")
        if probability >= threshold:
            crossings.append(evaluation["detection_step"])
    if record["first_threshold_crossing_step"] != (crossings[0] if crossings else None):
        raise ValueError("first threshold crossing step is inconsistent")
    if record["actions_blocked"] != bool(crossings):
        raise ValueError("actions_blocked is inconsistent with gate evaluations")


__all__ = [
    "CLASS_NAMES",
    "DEFAULT_LOG_DIRECTORY",
    "SCHEMA_VERSION",
    "build_run_log",
    "validate_run_log",
    "write_run_log",
]
