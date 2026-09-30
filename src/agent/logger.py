import json
import os
from src.dataset.schema import Trajectory, Step

class TrajectoryLogger:
    def __init__(self, output_file: str):
        self.output_file = output_file
        # Create dir if not exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def log_trajectory(self, trajectory: Trajectory):
        with open(self.output_file, 'a') as f:
            f.write(trajectory.model_dump_json() + '\n')

def replay_trajectory(trajectory_path: str, trajectory_id: str):
    """Utility to replay a trajectory for manual inspection."""
    with open(trajectory_path, 'r') as f:
        for line in f:
            traj_data = json.loads(line)
            if traj_data['trajectory_id'] == trajectory_id:
                print(f"==================================================")
                print(f"TRAJECTORY REPLAY: {trajectory_id}")
                print(f"Goal: {traj_data['original_goal']}")
                print(f"Label: {traj_data['label']}")
                if traj_data.get('attack_type'):
                    print(f"Attack: {traj_data['attack_type']}")
                print(f"==================================================")
                for step in traj_data['steps']:
                    print(f"\n[Step {step['step_id']}] Action: {step['action']}")
                    if step.get('tool'):
                        print(f"Tool: {step['tool']}")
                        print(f"Input: {step['tool_input']}")
                    if step.get('observation'):
                        print(f"Observation: {step['observation']}")
                    if step.get('state_summary'):
                        print(f"Agent Thought: {step['state_summary']}")
                    
                    flags = []
                    if step.get('injection'): flags.append("⚠ INJECTION")
                    if step.get('deviation'): flags.append("⚠ DEVIATION")
                    if step.get('unwanted_action'): flags.append("🚨 UNWANTED ACTION")
                    if flags:
                        print(" ".join(flags))
                print(f"==================================================")
                return
    print(f"Trajectory {trajectory_id} not found in {trajectory_path}")
