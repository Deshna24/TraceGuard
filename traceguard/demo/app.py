"""Streamlit research demo for recorded TRACEGUARD runtime evidence.

The demo is deliberately read-only: it loads validated runtime JSON records and
does not run the detector, agent, or controlled tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traceguard.runtime.logger import CLASS_NAMES, DEFAULT_LOG_DIRECTORY, validate_run_log


SCENARIO_LOGS = {
    "BENIGN": "benign_run.json",
    "INJECTION_RESISTED": "resisted_run.json",
    "HIJACKED": "hijacked_run.json",
}
THRESHOLD = 0.50


class RuntimeLogError(ValueError):
    """Raised when a runtime record cannot be used by the demo."""


def load_runtime_log(scenario: str, log_directory: Path = DEFAULT_LOG_DIRECTORY) -> dict[str, Any]:
    """Load one recorded scenario, rejecting missing or malformed evidence."""
    if scenario not in SCENARIO_LOGS:
        raise RuntimeLogError(f"unknown scenario: {scenario!r}")
    path = Path(log_directory) / SCENARIO_LOGS[scenario]
    if not path.is_file():
        raise RuntimeLogError(f"runtime log is missing: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeLogError(f"runtime log is not valid JSON: {path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeLogError(f"runtime log could not be read: {path}: {exc}") from exc
    try:
        validate_run_log(record)
    except (TypeError, ValueError) as exc:
        raise RuntimeLogError(f"runtime log is invalid: {path}: {exc}") from exc
    if record["scenario"] != scenario:
        raise RuntimeLogError(
            f"runtime log scenario mismatch: expected {scenario}, got {record['scenario']!r}"
        )
    return record


def load_all_runtime_logs(log_directory: Path = DEFAULT_LOG_DIRECTORY) -> dict[str, dict[str, Any]]:
    """Load all supported scenarios without substituting missing records."""
    return {scenario: load_runtime_log(scenario, log_directory) for scenario in SCENARIO_LOGS}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _trajectory_by_step(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {step["step"]: step for step in record["trajectory"]}


def _render_probability_summary(st: Any, evaluation: dict[str, Any]) -> None:
    probabilities = evaluation["class_probabilities"]
    columns = st.columns(4)
    for column, name in zip(columns[:3], CLASS_NAMES, strict=True):
        column.metric(f"P({name})", f"{probabilities[name]:.3f}")
    columns[3].metric("Threshold", f"{evaluation['threshold']:.2f}")


def _render_timeline(st: Any, record: dict[str, Any]) -> None:
    trajectory = _trajectory_by_step(record)
    st.subheader("Ordered trajectory and pre-action evaluations")
    for evaluation in record["detector_evaluations"]:
        index = evaluation["evaluation_index"]
        action = evaluation["proposed_action"]
        step = trajectory.get(index)
        with st.container(border=True):
            st.markdown(f"**Step {index} · {evaluation['decision']}**")
            if step is not None:
                st.write(f"**Action:** {step['action']}")
                st.write(f"**Tool:** `{step['tool']}`")
                st.write(f"**Tool input:** `{_json_text(step['tool_input'])}`")
                observation = step["tool_observation"]
                st.write(f"**Tool observation:** {_json_text(observation)}")
            else:
                st.write(f"**Action:** {action['action']}")
                st.write(f"**Tool:** `{action['tool']}`")
                st.write(f"**Tool input:** `{_json_text(action['tool_input'])}`")
                st.warning("Tool execution did not occur because this proposed action was blocked.")
            _render_probability_summary(st, evaluation)
            status = "ALERT" if evaluation["hijack_probability"] >= evaluation["threshold"] else "SAFE"
            st.write(
                f"**Status:** {status} · **Proposed action:** {action['action']} · "
                f"**Gate:** {evaluation['decision']} · **Pre-action:** {evaluation['pre_action']}"
            )


def _render_selected_record(st: Any, record: dict[str, Any]) -> None:
    scenario = record["scenario"]
    st.title("TRACEGUARD Research Demo")
    st.caption("Recorded runtime evidence; no inference or tool execution is performed by this UI.")
    st.subheader(f"{scenario} scenario")
    st.info(f"**User goal:** {record['user_goal']}")

    latest = record["detector_evaluations"][-1]
    latest_probability = latest["hijack_probability"]
    alert = latest_probability >= THRESHOLD
    first_detection = record["first_threshold_crossing_step"]
    status_column, action_column, detection_column = st.columns(3)
    status_column.metric("Current status", "ALERT" if alert else "SAFE")
    action_column.metric("Final gate", "BLOCK" if record["actions_blocked"] else "ALLOW")
    detection_column.metric(
        "First detection step",
        str(first_detection) if first_detection is not None else "None",
    )
    st.write(
        f"**Fixed P(HIJACKED) threshold:** `{THRESHOLD:.2f}` · "
        f"**Pre-action detection:** `{latest['pre_action']}` · "
        f"**Final status:** `{record['final_status']}`"
    )
    st.metric("P(HIJACKED) at latest evaluation", f"{latest_probability:.3f}")
    if scenario == "HIJACKED":
        blocked = record["blocked_action"]
        st.error(
            f"**BLOCKED before execution:** `{blocked['tool']}` / "
            f"`{_json_text(blocked['tool_input'])}`. "
            f"Database executions recorded: `{record['tool_execution_counts'].get('database', 0)}`."
        )
    _render_timeline(st, record)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="TRACEGUARD Research Demo", layout="wide")
    try:
        records = load_all_runtime_logs()
    except RuntimeLogError as exc:
        st.error(str(exc))
        st.stop()
        return
    scenario = st.sidebar.selectbox("Scenario", list(SCENARIO_LOGS))
    _render_selected_record(st, records[scenario])


if __name__ == "__main__":
    main()
