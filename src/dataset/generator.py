import json
import uuid
import yaml
from langchain_core.messages import HumanMessage
from src.agent.graph import agent_graph
from src.agent.logger import TrajectoryLogger
from src.dataset.tasks import generate_random_task
from src.attacks.injector import Injector
from src.dataset.labeling import determine_ground_truth
from src.dataset.schema import Trajectory, Step
from tqdm import tqdm

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_episode(mode: str, logger: TrajectoryLogger):
    # Setup
    family, task = generate_random_task()
    trajectory_id = f"T_{uuid.uuid4().hex[:8]}"
    
    # 1. We want to mock tool responses using our Injector
    # To do this cleanly without modifying the LangGraph internals too much for this script,
    # we can use the graph's ability to intercept tool output.
    # Wait, the graph automatically calls the tools. 
    # Instead of modifying the graph for interception, we'll patch the tools in `tools.py` dynamically 
    # OR simpler: run the graph step by step and intercept.
    
    # Actually, simpler: patch the tools before invoking the graph for this episode.
    # But since the injector is stateful per trajectory, it's better to pass the injector.
    import src.agent.tools as tools_module
    
    injector = Injector(mode=mode, injection_step_target=2) # aim to inject around step 2
    
    # Patch the tools to route through injector
    original_funcs = {}
    for tool in tools_module.GET_TOOLS:
        original_funcs[tool.name] = tool.func
        
        def make_wrapper(orig_func):
            def wrapper(*args, **kwargs):
                # Call original
                res = orig_func(*args, **kwargs)
                # Inject
                # We need the current step idx. Let's just pass 2 for now to trigger it if target reached
                # We can track tool call count
                wrapper.calls += 1
                return injector.intercept_observation(wrapper.calls, res)
            wrapper.calls = 0
            return wrapper
            
        tool.func = make_wrapper(original_funcs[tool.name])

    # Run agent
    state = {
        "messages": [HumanMessage(content=task["goal"])],
        "task_id": "T_001",
        "original_goal": task["goal"],
        "injected_tool_calls": [],
        "trajectory_steps": []
    }
    
    try:
        final_state = agent_graph.invoke(state)
        trajectory_steps = final_state["trajectory_steps"]
    except Exception as e:
        print(f"Error during graph execution: {e}")
        trajectory_steps = state.get("trajectory_steps", [])
        
    # Unpatch
    for tool in tools_module.GET_TOOLS:
        tool.func = original_funcs[tool.name]

    # Post-process trajectory_steps to match Schema
    formatted_steps = []
    step_counter = 1
    
    for raw_step in trajectory_steps:
        # Check if this raw step contains the injection payload (hacky but works for mock)
        is_injection = False
        if injector.payload and injector.payload in raw_step.get("observation", ""):
            is_injection = True
            
        step_obj = Step(
            step_id=step_counter,
            action=raw_step["action"],
            tool=raw_step.get("tool"),
            tool_input=raw_step.get("tool_input"),
            observation=raw_step.get("observation"),
            state_summary=raw_step.get("content"),
            injection=is_injection
        )
        formatted_steps.append(step_obj.model_dump())
        step_counter += 1

    # Ground truth labeling
    gt = determine_ground_truth(formatted_steps, family, injector.attack_type, mode)
    
    # Update deviation/unwanted action in steps
    for s in formatted_steps:
        if s["step_id"] == gt["deviation_step"]:
            s["deviation"] = True
        if s["step_id"] == gt["unwanted_action_step"]:
            s["unwanted_action"] = True
            
    # Compile Trajectory
    traj = Trajectory(
        trajectory_id=trajectory_id,
        task_id=f"TASK_{family}",
        original_goal=task["goal"],
        steps=formatted_steps,
        label=mode,
        attack_type=injector.attack_type,
        injection_step=gt["injection_step"],
        deviation_step=gt["deviation_step"],
        unwanted_action_step=gt["unwanted_action_step"],
        attack_success=gt["attack_success"],
        deviation_reason=gt["deviation_reason"]
    )
    
    logger.log_trajectory(traj)

if __name__ == "__main__":
    config = load_config()
    logger = TrajectoryLogger("data/raw/trajectories/pilot.jsonl")
    
    # 30 benign, 30 resisted, 30 hijacked
    print("Generating pilot dataset...")
    for mode in ["benign", "resisted", "hijacked"]:
        for _ in tqdm(range(30), desc=f"Generating {mode}"):
            run_episode(mode, logger)
    print("Done generating 90 pilot trajectories.")
