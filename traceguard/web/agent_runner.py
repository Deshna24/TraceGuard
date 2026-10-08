import asyncio
import copy
from agent.agent import (
    OllamaActionModel,
    TrajectoryState,
    parse_model_output,
    ParsedAction,
    AgentRunResult,
)
from runtime.action_gate import PreActionGate, GateDecision
from web.websocket.manager import manager


async def run_controlled_agent_with_events(scenario: str, user_goal: str, registry, detector, max_steps: int = 6):
    await manager.broadcast("AGENT_STARTED", {"model": "granite4.1:8b-q4_K_M"})
    
    model = OllamaActionModel(timeout=90)
    gate = PreActionGate(detector)
    trajectory = TrajectoryState(user_goal)
    decisions = []
    tool_results = []
    
    for _ in range(max_steps):
        try:
            # Blocking call, run in thread
            raw_output = await asyncio.to_thread(
                model.propose,
                trajectory.user_goal,
                trajectory.get_steps(),
                registry.describe(),
            )
            output_type, output = parse_model_output(raw_output)
        except Exception as exc:
            await manager.broadcast("RUNTIME_ERROR", {"error": str(exc)})
            await manager.broadcast("RUN_COMPLETED", {"status": "model_failure", "error": str(exc)})
            return
            
        if output_type == "final":
            await manager.broadcast("RUN_COMPLETED", {"status": "completed", "answer": output})
            return
            
        action = output
        assert isinstance(action, ParsedAction)
        
        await manager.broadcast("ACTION_PROPOSED", {"action": action.action, "tool": action.tool, "tool_input": action.tool_input})
        
        try:
            registry.resolve(action.tool)
            proposed = action.as_proposed_action()
        except Exception as exc:
            await manager.broadcast("RUNTIME_ERROR", {"error": str(exc)})
            await manager.broadcast("RUN_COMPLETED", {"status": "invalid_action", "error": str(exc)})
            return
            
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
            
        # Blocking call for detection, run in thread
        decision = await asyncio.to_thread(gate.evaluate, proposed, gate_prefix)
        decisions.append(decision)
        
        # Detector evaluated
        class_probs = decision.class_probabilities or {}
        await manager.broadcast("DETECTOR_EVALUATED", {
            "trajectory_step": decision.detection_step,
            "P_BENIGN": class_probs.get("BENIGN", 0.0),
            "P_INJECTION_RESISTED": class_probs.get("INJECTION_RESISTED", 0.0),
            "P_HIJACKED": decision.hijack_probability,
            "predicted_class": decision.predicted_class or "UNKNOWN",
            "threshold": decision.threshold,
            "pre_action": decision.pre_action
        })
        
        if decision.hijack_probability is not None and decision.hijack_probability >= decision.threshold:
            await manager.broadcast("THRESHOLD_CROSSED", {"p_hijacked": decision.hijack_probability, "threshold": decision.threshold})
            
        await manager.broadcast("GATE_DECISION", {
            "proposed_action": proposed.as_dict(),
            "decision": decision.decision,
            "p_hijacked": decision.hijack_probability,
            "threshold": decision.threshold,
            "detection_step": decision.detection_step,
            "pre_action": decision.pre_action
        })
        
        if decision.blocked:
            await manager.broadcast("ACTION_BLOCKED", {"action": proposed.as_dict()})
            await manager.broadcast("RUN_COMPLETED", {"status": "blocked", "blocked_action": proposed.as_dict()})
            return
            
        await manager.broadcast("TOOL_STARTED", {"tool": proposed.tool})
        
        try:
            result = await asyncio.to_thread(
                PreActionGate.execute_if_allowed,
                decision,
                lambda approved: registry.invoke(approved["tool"], approved["tool_input"])
            )
        except Exception as exc:
            await manager.broadcast("RUNTIME_ERROR", {"error": str(exc)})
            await manager.broadcast("RUN_COMPLETED", {"status": "tool_failure", "error": str(exc)})
            return
            
        tool_results.append(result)
        
        await manager.broadcast("TOOL_COMPLETED", {"tool": proposed.tool, "success": result.success})
        
        obs = result.as_observation()
        await manager.broadcast("OBSERVATION_RECEIVED", {"observation": obs})
        
        # Check for injection in observation for demo events
        if isinstance(obs, dict) and "Injected instruction" in str(obs):
            await manager.broadcast("INJECTION_OBSERVED", {"observation": obs})
            
        step_dict = {
            "step": trajectory.length() + 1,
            "action": proposed.action,
            "tool": proposed.tool,
            "tool_input": proposed.tool_input,
            "tool_observation": obs,
            "state": "Tool execution completed" if result.success else "Tool execution failed",
        }
        trajectory.add_step(step_dict)
        
        await manager.broadcast("TRAJECTORY_UPDATED", {"step": step_dict})
        
        if not result.success:
            await manager.broadcast("RUN_COMPLETED", {"status": "tool_failure", "error": result.error})
            return

    await manager.broadcast("RUN_COMPLETED", {"status": "max_steps_reached"})
