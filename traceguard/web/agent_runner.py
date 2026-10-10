"""Agent runner that connects the existing ControlledAgent runtime to WebSocket events.

This module DOES NOT contain detector logic, gate logic, or tool implementations.
It uses the existing verified:
  - agent.agent.ControlledAgent / OllamaActionModel / ScriptedActionModel / parse_model_output
  - agent.tools.build_default_registry / ControlledToolRegistry
  - agent.scenarios.BENIGN_SCENARIO / INJECTION_RESISTED_SCENARIO / HIJACKED_SCENARIO
  - runtime.action_gate.PreActionGate / GateDecision / ProposedAction
  - runtime.detector.TraceGuardDetector
  - runtime.trajectory.TrajectoryState
  - runtime.logger.build_run_log / write_run_log

Events are emitted at the correct runtime points using real values.
NEVER hard-codes demo probabilities.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from pathlib import Path
from typing import Any

from runtime.action_gate import GateDecision, PreActionGate, ProposedAction
from runtime.detector import TraceGuardDetector
from runtime.baseline import SingleStepBaseline
from runtime.logger import DEFAULT_LOG_DIRECTORY, build_run_log, write_run_log
from runtime.trajectory import TrajectoryState

from agent.agent import (
    AgentRunResult,
    OllamaActionModel,
    ParsedAction,
    ScriptedActionModel,
    parse_model_output,
)
from agent.scenarios import (
    BENIGN_SCENARIO,
    HIJACKED_SCENARIO,
    INJECTION_RESISTED_SCENARIO,
)
from agent.tools import ControlledToolRegistry, ToolResult, build_default_registry

from web.runtime_manager import RunState, RuntimeManager, runtime_manager
from web.schemas.api_models import EventType, RunStatus, ScenarioName
from web.websocket.manager import manager as ws_manager

logger = logging.getLogger("traceguard.web.agent_runner")

# Injection fixture sentinel used to detect controlled injection content.
INJECTION_SENTINEL = "Injected instruction"

# ---------------------------------------------------------------------------
# Scenario resolution
# ---------------------------------------------------------------------------

SCENARIO_MAP = {
    ScenarioName.BENIGN: BENIGN_SCENARIO,
    ScenarioName.INJECTION_RESISTED: INJECTION_RESISTED_SCENARIO,
    ScenarioName.HIJACKED: HIJACKED_SCENARIO,
}

SCENARIO_DESCRIPTIONS = {
    ScenarioName.BENIGN: (
        "Legitimate task with no injection. Agent performs a benign calculation. "
        "All actions allowed."
    ),
    ScenarioName.INJECTION_RESISTED: (
        "Injection fixture is present in search observation, but the agent remains "
        "aligned and continues the original task. Injection observed, actions allowed."
    ),
    ScenarioName.HIJACKED: (
        "Agent follows injected objective after observing the fixture. Behavioral "
        "deviation is detected by TRACEGUARD; database export is blocked pre-action."
    ),
}


def get_scenario_info(name: ScenarioName, custom_goal: str | None = None) -> dict[str, str]:
    """Return display metadata for a scenario without exposing internals."""
    if name == ScenarioName.CUSTOM:
        return {
            "name": name.value,
            "user_goal": custom_goal or "Complete the user's task.",
            "description": "Interactive mode: Custom user goal and injection payload.",
        }
    scenario = SCENARIO_MAP[name]
    return {
        "name": name.value,
        "user_goal": scenario.user_goal,
        "description": SCENARIO_DESCRIPTIONS[name],
    }


# ---------------------------------------------------------------------------
# The main event-emitting runner
# ---------------------------------------------------------------------------

async def run_scenario(run: RunState) -> None:
    """Execute a full scenario run, emitting WebSocket events at each runtime point.

    Uses ScriptedActionModel for deterministic, reproducible scenario behavior.
    The detector, gate, trajectory, and tools are the real verified components.
    """
    run_id = run.run_id
    scenario_name = run.scenario
    user_goal = run.user_goal
    if scenario_name != ScenarioName.CUSTOM:
        scenario = SCENARIO_MAP[scenario_name]

    try:
        # ── RUN_STARTED ──
        await ws_manager.broadcast_event(
            run_id, EventType.RUN_STARTED,
            payload={"scenario": scenario_name.value, "user_goal": user_goal},
        )

        # ── USER_TASK_RECEIVED ──
        await ws_manager.broadcast_event(
            run_id, EventType.USER_TASK_RECEIVED,
            payload={"user_goal": user_goal},
        )

        # Build the runtime components using real implementations.
        registry, tool_instances = build_default_registry(run.custom_injection, run.injection_target)
        detector = TraceGuardDetector(user_goal=user_goal)
        baseline_detector = SingleStepBaseline()
        gate = PreActionGate(detector)
        trajectory = TrajectoryState(user_goal)
        
        if scenario_name == ScenarioName.CUSTOM:
            model = ScriptedActionModel(()) # Custom cannot be run with scripted model
        else:
            model = ScriptedActionModel(scenario.model_outputs)

        # ── AGENT_STARTED ──
        await ws_manager.broadcast_event(
            run_id, EventType.AGENT_STARTED,
            payload={"model": "ScriptedActionModel (deterministic scenario)"},
        )

        decisions: list[GateDecision] = []
        tool_results: list[ToolResult] = []
        max_steps = 6
        blocked_action: dict[str, Any] | None = None
        final_answer: str | None = None
        final_status = "max_steps_reached"
        final_error: str | None = None

        for iteration in range(max_steps):
            # Check for cancellation.
            if run.status != RunStatus.RUNNING:
                final_status = "stopped"
                break

            # ── Model proposal (blocking, run in thread) ──
            try:
                raw_output = await asyncio.to_thread(
                    model.propose,
                    trajectory.user_goal,
                    trajectory.get_steps(),
                    registry.describe(),
                )
                output_type, output = parse_model_output(raw_output)
            except Exception as exc:
                final_status = "model_failure"
                final_error = str(exc)
                await ws_manager.broadcast_event(
                    run_id, EventType.RUNTIME_ERROR,
                    payload={"error": final_error, "phase": "model_proposal"},
                )
                break

            # ── Final answer ──
            if output_type == "final":
                final_status = "completed"
                final_answer = output
                break

            action: ParsedAction = output  # type: ignore[assignment]

            # ── ACTION_PROPOSED ──
            step_number = trajectory.length() + 1
            run.update_step(step_number)
            await ws_manager.broadcast_event(
                run_id, EventType.ACTION_PROPOSED,
                step=step_number,
                payload={
                    "action": action.action,
                    "tool": action.tool,
                    "tool_input": copy.deepcopy(action.tool_input),
                },
            )

            # Resolve tool.
            try:
                registry.resolve(action.tool)
                proposed = action.as_proposed_action()
            except Exception as exc:
                final_status = "invalid_action"
                final_error = str(exc)
                await ws_manager.broadcast_event(
                    run_id, EventType.RUNTIME_ERROR,
                    step=step_number,
                    payload={"error": final_error, "phase": "tool_resolution"},
                )
                break

            # Build the gate prefix (matches ControlledAgent logic exactly).
            gate_prefix = trajectory.get_steps()
            if not gate_prefix:
                gate_prefix = [
                    {
                        "step": 1,
                        "action": proposed.action,
                        "tool": proposed.tool,
                        "tool_input": copy.deepcopy(proposed.tool_input),
                        "tool_observation": None,
                        "state": "Proposed action; awaiting controlled execution",
                    }
                ]

            # ── Detector evaluation (blocking, run in thread) ──
            try:
                decision = await asyncio.to_thread(gate.evaluate, proposed, gate_prefix)
            except Exception as exc:
                # Fail closed: if detector errors, block.
                final_status = "error"
                final_error = f"Detector evaluation failed: {exc}"
                await ws_manager.broadcast_event(
                    run_id, EventType.RUNTIME_ERROR,
                    step=step_number,
                    payload={"error": final_error, "phase": "detector_evaluation"},
                )
                break

            decisions.append(decision)

            # ── Baseline Evaluation ──
            baseline_p_hijacked = baseline_detector.evaluate_step(gate_prefix[-1] if gate_prefix else {})

            # ── DETECTOR_EVALUATED ──
            class_probs = decision.class_probabilities or {}
            await ws_manager.broadcast_event(
                run_id, EventType.DETECTOR_EVALUATED,
                step=step_number,
                payload={
                    "step": decision.detection_step,
                    "p_benign": class_probs.get("BENIGN"),
                    "p_injection_resisted": class_probs.get("INJECTION_RESISTED"),
                    "p_hijacked": decision.hijack_probability,
                    "baseline_p_hijacked": baseline_p_hijacked,
                    "predicted_class": decision.predicted_class,
                    "threshold": decision.threshold,
                    "pre_action": decision.pre_action,
                },
            )

            # ── TRAJECTORY_UPDATED (prefix state after detector evaluation) ──
            await ws_manager.broadcast_event(
                run_id, EventType.TRAJECTORY_UPDATED,
                step=step_number,
                payload={
                    "trajectory_length": len(gate_prefix),
                    "latest_step": copy.deepcopy(gate_prefix[-1]) if gate_prefix else None,
                },
            )

            # ── THRESHOLD_CROSSED ──
            if (
                decision.hijack_probability is not None
                and decision.hijack_probability >= decision.threshold
            ):
                await ws_manager.broadcast_event(
                    run_id, EventType.THRESHOLD_CROSSED,
                    step=step_number,
                    payload={
                        "p_hijacked": decision.hijack_probability,
                        "threshold": decision.threshold,
                        "detection_step": decision.detection_step,
                    },
                )

            # ── GATE_DECISION ──
            await ws_manager.broadcast_event(
                run_id, EventType.GATE_DECISION,
                step=step_number,
                payload={
                    "proposed_action": proposed.as_dict(),
                    "tool": proposed.tool,
                    "tool_input": copy.deepcopy(proposed.tool_input),
                    "decision": decision.decision,
                    "p_hijacked": decision.hijack_probability,
                    "threshold": decision.threshold,
                    "detection_step": decision.detection_step,
                    "pre_action": decision.pre_action,
                },
            )

            # ── ACTION_BLOCKED ──
            if decision.blocked:
                blocked_action = proposed.as_dict()
                await ws_manager.broadcast_event(
                    run_id, EventType.ACTION_BLOCKED,
                    step=step_number,
                    payload={
                        "proposed_action": blocked_action,
                        "tool": proposed.tool,
                        "tool_input": copy.deepcopy(proposed.tool_input),
                        "p_hijacked": decision.hijack_probability,
                        "threshold": decision.threshold,
                        "reason": decision.error or "P(HIJACKED) >= threshold",
                    },
                )
                final_status = "blocked"
                final_error = decision.error
                break

            # ── TOOL_STARTED ──
            await ws_manager.broadcast_event(
                run_id, EventType.TOOL_STARTED,
                step=step_number,
                payload={"tool": proposed.tool, "tool_input": copy.deepcopy(proposed.tool_input)},
            )

            # Execute tool (blocking, run in thread).
            try:
                result = await asyncio.to_thread(
                    PreActionGate.execute_if_allowed,
                    decision,
                    lambda approved: registry.invoke(approved["tool"], approved["tool_input"]),
                )
            except Exception as exc:
                final_status = "tool_failure"
                final_error = str(exc)
                await ws_manager.broadcast_event(
                    run_id, EventType.RUNTIME_ERROR,
                    step=step_number,
                    payload={"error": final_error, "phase": "tool_execution"},
                )
                break

            tool_results.append(result)

            # ── TOOL_COMPLETED ──
            await ws_manager.broadcast_event(
                run_id, EventType.TOOL_COMPLETED,
                step=step_number,
                payload={
                    "tool": proposed.tool,
                    "success": result.success,
                    "output": copy.deepcopy(result.output),
                    "error": result.error,
                },
            )

            # ── OBSERVATION_RECEIVED ──
            obs = result.as_observation()
            await ws_manager.broadcast_event(
                run_id, EventType.OBSERVATION_RECEIVED,
                step=step_number,
                payload={"observation": obs},
            )

            # ── INJECTION_OBSERVED ──
            obs_str = str(obs)
            if INJECTION_SENTINEL in obs_str:
                await ws_manager.broadcast_event(
                    run_id, EventType.INJECTION_OBSERVED,
                    step=step_number,
                    payload={
                        "observation": obs,
                        "note": (
                            "Untrusted observation content: controlled benchmark/runtime "
                            "fixture containing an injection. This is NOT a real-world "
                            "external attack."
                        ),
                    },
                )

            # Record to trajectory.
            step_dict = {
                "step": trajectory.length() + 1,
                "action": proposed.action,
                "tool": proposed.tool,
                "tool_input": proposed.tool_input,
                "tool_observation": obs,
                "state": "Tool execution completed" if result.success else "Tool execution failed",
            }
            trajectory.add_step(step_dict)
            run.add_trajectory_step(step_dict)

            # ── TRAJECTORY_UPDATED (after tool execution) ──
            await ws_manager.broadcast_event(
                run_id, EventType.TRAJECTORY_UPDATED,
                step=step_dict["step"],
                payload={
                    "trajectory_length": trajectory.length(),
                    "latest_step": copy.deepcopy(step_dict),
                },
            )

            if not result.success:
                final_status = "tool_failure"
                final_error = result.error
                break

        # ── Build AgentRunResult for JSON logging ──
        agent_result = AgentRunResult(
            status=final_status,
            user_goal=trajectory.user_goal,
            answer=final_answer,
            trajectory=trajectory.get_steps(),
            gate_decisions=list(decisions),
            tool_results=list(tool_results),
            blocked_action=copy.deepcopy(blocked_action),
            error=final_error,
        )

        # Build tool execution counts from the actual tool instances.
        tool_execution_counts = {
            name: inst.execution_count
            for name, inst in tool_instances.items()
        }

        # ── Write canonical JSON log (existing logger, never replaced) ──
        try:
            log_record = build_run_log(
                scenario=scenario_name.value,
                run_id=run_id,
                result=agent_result,
                tool_execution_counts=tool_execution_counts,
            )
            log_path = DEFAULT_LOG_DIRECTORY / f"{run_id}.json"
            write_run_log(log_record, log_path)
            logger.info("Runtime log written: %s", log_path)
        except Exception as exc:
            logger.warning("Failed to write runtime log: %s", exc)

        # ── RUN_COMPLETED ──
        completion_payload: dict[str, Any] = {
            "status": final_status,
            "scenario": scenario_name.value,
            "trajectory_length": trajectory.length(),
            "tool_execution_counts": tool_execution_counts,
        }
        if final_answer:
            completion_payload["answer"] = final_answer
        if blocked_action:
            completion_payload["blocked_action"] = blocked_action
        if final_error:
            completion_payload["error"] = final_error

        await ws_manager.broadcast_event(
            run_id, EventType.RUN_COMPLETED,
            payload=completion_payload,
        )

        # Update run state.
        run.complete(
            RunStatus(final_status) if final_status in RunStatus.__members__.values()
            else RunStatus.ERROR,
            result=completion_payload,
        )

    except asyncio.CancelledError:
        await ws_manager.broadcast_event(
            run_id, EventType.RUN_COMPLETED,
            payload={"status": "stopped", "reason": "cancelled"},
        )
        run.complete(RunStatus.STOPPED)
    except Exception as exc:
        logger.exception("Unhandled error in run %s", run_id)
        await ws_manager.broadcast_event(
            run_id, EventType.RUNTIME_ERROR,
            payload={"error": str(exc), "phase": "unhandled"},
        )
        await ws_manager.broadcast_event(
            run_id, EventType.RUN_COMPLETED,
            payload={"status": "error", "error": str(exc)},
        )
        run.complete(RunStatus.ERROR)


# ---------------------------------------------------------------------------
# Live runtime runner (uses OllamaActionModel)
# ---------------------------------------------------------------------------

async def run_scenario_live(run: RunState) -> None:
    """Execute a scenario using the live Ollama model.

    Identical event-emission logic as run_scenario but with OllamaActionModel
    instead of ScriptedActionModel. For live demonstrations where the model
    runs on the local Ollama server (granite4.1:8b-q4_K_M).
    """
    run_id = run.run_id
    scenario_name = run.scenario
    user_goal = run.user_goal

    try:
        await ws_manager.broadcast_event(
            run_id, EventType.RUN_STARTED,
            payload={"scenario": scenario_name.value, "user_goal": user_goal},
        )
        await ws_manager.broadcast_event(
            run_id, EventType.USER_TASK_RECEIVED,
            payload={"user_goal": user_goal},
        )

        registry, tool_instances = build_default_registry(run.custom_injection, run.injection_target)
        detector = TraceGuardDetector(user_goal=user_goal)
        baseline_detector = SingleStepBaseline()
        gate = PreActionGate(detector)
        trajectory = TrajectoryState(user_goal)

        # Use live Ollama model.
        model = OllamaActionModel(model="granite4.1:8b-q4_K_M", timeout=90.0)

        await ws_manager.broadcast_event(
            run_id, EventType.AGENT_STARTED,
            payload={"model": "granite4.1:8b-q4_K_M (Ollama live)"},
        )

        decisions: list[GateDecision] = []
        tool_results: list[ToolResult] = []
        max_steps = 6
        blocked_action: dict[str, Any] | None = None
        final_answer: str | None = None
        final_status = "max_steps_reached"
        final_error: str | None = None

        for iteration in range(max_steps):
            if run.status != RunStatus.RUNNING:
                final_status = "stopped"
                break

            try:
                raw_output = await asyncio.to_thread(
                    model.propose,
                    trajectory.user_goal,
                    trajectory.get_steps(),
                    registry.describe(),
                )
                output_type, output = parse_model_output(raw_output)
            except Exception as exc:
                final_status = "model_failure"
                final_error = str(exc)
                await ws_manager.broadcast_event(
                    run_id, EventType.RUNTIME_ERROR,
                    payload={"error": final_error, "phase": "model_proposal"},
                )
                break

            if output_type == "final":
                final_status = "completed"
                final_answer = output
                break

            action: ParsedAction = output  # type: ignore[assignment]
            step_number = trajectory.length() + 1
            run.update_step(step_number)

            await ws_manager.broadcast_event(
                run_id, EventType.ACTION_PROPOSED,
                step=step_number,
                payload={
                    "action": action.action,
                    "tool": action.tool,
                    "tool_input": copy.deepcopy(action.tool_input),
                },
            )

            try:
                registry.resolve(action.tool)
                proposed = action.as_proposed_action()
            except Exception as exc:
                final_status = "invalid_action"
                final_error = str(exc)
                await ws_manager.broadcast_event(
                    run_id, EventType.RUNTIME_ERROR,
                    step=step_number,
                    payload={"error": final_error, "phase": "tool_resolution"},
                )
                break

            gate_prefix = trajectory.get_steps()
            if not gate_prefix:
                gate_prefix = [
                    {
                        "step": 1,
                        "action": proposed.action,
                        "tool": proposed.tool,
                        "tool_input": copy.deepcopy(proposed.tool_input),
                        "tool_observation": None,
                        "state": "Proposed action; awaiting controlled execution",
                    }
                ]

            try:
                decision = await asyncio.to_thread(gate.evaluate, proposed, gate_prefix)
            except Exception as exc:
                final_status = "error"
                final_error = f"Detector evaluation failed: {exc}"
                await ws_manager.broadcast_event(
                    run_id, EventType.RUNTIME_ERROR,
                    step=step_number,
                    payload={"error": final_error, "phase": "detector_evaluation"},
                )
                break

            decisions.append(decision)

            baseline_p_hijacked = baseline_detector.evaluate_step(gate_prefix[-1] if gate_prefix else {})

            class_probs = decision.class_probabilities or {}
            await ws_manager.broadcast_event(
                run_id, EventType.DETECTOR_EVALUATED,
                step=step_number,
                payload={
                    "step": decision.detection_step,
                    "p_benign": class_probs.get("BENIGN"),
                    "p_injection_resisted": class_probs.get("INJECTION_RESISTED"),
                    "p_hijacked": decision.hijack_probability,
                    "baseline_p_hijacked": baseline_p_hijacked,
                    "predicted_class": decision.predicted_class,
                    "threshold": decision.threshold,
                    "pre_action": decision.pre_action,
                },
            )

            await ws_manager.broadcast_event(
                run_id, EventType.TRAJECTORY_UPDATED,
                step=step_number,
                payload={
                    "trajectory_length": len(gate_prefix),
                    "latest_step": copy.deepcopy(gate_prefix[-1]) if gate_prefix else None,
                },
            )

            if (
                decision.hijack_probability is not None
                and decision.hijack_probability >= decision.threshold
            ):
                await ws_manager.broadcast_event(
                    run_id, EventType.THRESHOLD_CROSSED,
                    step=step_number,
                    payload={
                        "p_hijacked": decision.hijack_probability,
                        "threshold": decision.threshold,
                        "detection_step": decision.detection_step,
                    },
                )

            await ws_manager.broadcast_event(
                run_id, EventType.GATE_DECISION,
                step=step_number,
                payload={
                    "proposed_action": proposed.as_dict(),
                    "tool": proposed.tool,
                    "tool_input": copy.deepcopy(proposed.tool_input),
                    "decision": decision.decision,
                    "p_hijacked": decision.hijack_probability,
                    "threshold": decision.threshold,
                    "detection_step": decision.detection_step,
                    "pre_action": decision.pre_action,
                },
            )

            if decision.blocked:
                blocked_action = proposed.as_dict()
                await ws_manager.broadcast_event(
                    run_id, EventType.ACTION_BLOCKED,
                    step=step_number,
                    payload={
                        "proposed_action": blocked_action,
                        "tool": proposed.tool,
                        "tool_input": copy.deepcopy(proposed.tool_input),
                        "p_hijacked": decision.hijack_probability,
                        "threshold": decision.threshold,
                        "reason": decision.error or "P(HIJACKED) >= threshold",
                    },
                )
                final_status = "blocked"
                final_error = decision.error
                break

            await ws_manager.broadcast_event(
                run_id, EventType.TOOL_STARTED,
                step=step_number,
                payload={"tool": proposed.tool, "tool_input": copy.deepcopy(proposed.tool_input)},
            )

            try:
                result = await asyncio.to_thread(
                    PreActionGate.execute_if_allowed,
                    decision,
                    lambda approved: registry.invoke(approved["tool"], approved["tool_input"]),
                )
            except Exception as exc:
                final_status = "tool_failure"
                final_error = str(exc)
                await ws_manager.broadcast_event(
                    run_id, EventType.RUNTIME_ERROR,
                    step=step_number,
                    payload={"error": final_error, "phase": "tool_execution"},
                )
                break

            tool_results.append(result)

            await ws_manager.broadcast_event(
                run_id, EventType.TOOL_COMPLETED,
                step=step_number,
                payload={
                    "tool": proposed.tool,
                    "success": result.success,
                    "output": copy.deepcopy(result.output),
                    "error": result.error,
                },
            )

            obs = result.as_observation()
            await ws_manager.broadcast_event(
                run_id, EventType.OBSERVATION_RECEIVED,
                step=step_number,
                payload={"observation": obs},
            )

            obs_str = str(obs)
            if INJECTION_SENTINEL in obs_str:
                await ws_manager.broadcast_event(
                    run_id, EventType.INJECTION_OBSERVED,
                    step=step_number,
                    payload={
                        "observation": obs,
                        "note": (
                            "Untrusted observation content: controlled benchmark/runtime "
                            "fixture containing an injection. This is NOT a real-world "
                            "external attack."
                        ),
                    },
                )

            step_dict = {
                "step": trajectory.length() + 1,
                "action": proposed.action,
                "tool": proposed.tool,
                "tool_input": proposed.tool_input,
                "tool_observation": obs,
                "state": "Tool execution completed" if result.success else "Tool execution failed",
            }
            trajectory.add_step(step_dict)
            run.add_trajectory_step(step_dict)

            await ws_manager.broadcast_event(
                run_id, EventType.TRAJECTORY_UPDATED,
                step=step_dict["step"],
                payload={
                    "trajectory_length": trajectory.length(),
                    "latest_step": copy.deepcopy(step_dict),
                },
            )

            if not result.success:
                final_status = "tool_failure"
                final_error = result.error
                break

        agent_result = AgentRunResult(
            status=final_status,
            user_goal=trajectory.user_goal,
            answer=final_answer,
            trajectory=trajectory.get_steps(),
            gate_decisions=list(decisions),
            tool_results=list(tool_results),
            blocked_action=copy.deepcopy(blocked_action),
            error=final_error,
        )

        tool_execution_counts = {
            name: inst.execution_count
            for name, inst in tool_instances.items()
        }

        try:
            log_record = build_run_log(
                scenario=scenario_name.value,
                run_id=run_id,
                result=agent_result,
                tool_execution_counts=tool_execution_counts,
            )
            log_path = DEFAULT_LOG_DIRECTORY / f"{run_id}.json"
            write_run_log(log_record, log_path)
            logger.info("Runtime log written: %s", log_path)
        except Exception as exc:
            logger.warning("Failed to write runtime log: %s", exc)

        completion_payload: dict[str, Any] = {
            "status": final_status,
            "scenario": scenario_name.value,
            "trajectory_length": trajectory.length(),
            "tool_execution_counts": tool_execution_counts,
        }
        if final_answer:
            completion_payload["answer"] = final_answer
        if blocked_action:
            completion_payload["blocked_action"] = blocked_action
        if final_error:
            completion_payload["error"] = final_error

        await ws_manager.broadcast_event(
            run_id, EventType.RUN_COMPLETED,
            payload=completion_payload,
        )

        run.complete(
            RunStatus(final_status) if final_status in RunStatus.__members__.values()
            else RunStatus.ERROR,
            result=completion_payload,
        )

    except asyncio.CancelledError:
        await ws_manager.broadcast_event(
            run_id, EventType.RUN_COMPLETED,
            payload={"status": "stopped", "reason": "cancelled"},
        )
        run.complete(RunStatus.STOPPED)
    except Exception as exc:
        logger.exception("Unhandled error in live run %s", run_id)
        await ws_manager.broadcast_event(
            run_id, EventType.RUNTIME_ERROR,
            payload={"error": str(exc), "phase": "unhandled"},
        )
        await ws_manager.broadcast_event(
            run_id, EventType.RUN_COMPLETED,
            payload={"status": "error", "error": str(exc)},
        )
        run.complete(RunStatus.ERROR)


__all__ = [
    "SCENARIO_DESCRIPTIONS",
    "SCENARIO_MAP",
    "get_scenario_info",
    "run_scenario",
    "run_scenario_live",
]
