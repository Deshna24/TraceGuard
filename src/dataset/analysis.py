import json
import pandas as pd
from collections import Counter

def analyze_pilot(file_path):
    trajectories = []
    with open(file_path, 'r') as f:
        for line in f:
            trajectories.append(json.loads(line))
            
    print(f"Total trajectories: {len(trajectories)}")
    
    labels = [t['label'] for t in trajectories]
    print(f"\nClass Distribution:\n{pd.Series(labels).value_counts()}")
    
    lengths = [len(t['steps']) for t in trajectories]
    print(f"\nTrajectory Lengths:\nMin: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.2f}")
    
    attacks = [t['attack_type'] for t in trajectories if t['attack_type']]
    if attacks:
        print(f"\nAttack Types Distribution:\n{pd.Series(attacks).value_counts()}")
    
    injection_steps = [t['injection_step'] for t in trajectories if t['injection_step']]
    if injection_steps:
        print(f"\nInjection Step Distribution:\n{pd.Series(injection_steps).value_counts()}")
        
    deviation_steps = [t['deviation_step'] for t in trajectories if t['deviation_step']]
    if deviation_steps:
        print(f"\nDeviation Step Distribution:\n{pd.Series(deviation_steps).value_counts()}")
        
    print("\n--- SAMPLE TRAJECTORY SUMMARY ---")
    import random
    sample = random.choice(trajectories)
    print(f"ID: {sample['trajectory_id']}")
    print(f"Goal: {sample['original_goal']}")
    print(f"Label: {sample['label']}")
    print(f"Attack Success: {sample['attack_success']}")
    for s in sample['steps']:
        flags = []
        if s['injection']: flags.append("[INJECTION]")
        if s['deviation']: flags.append("[DEVIATION]")
        print(f"Step {s['step_id']}: {s['action']} - {s.get('tool', '')} {' '.join(flags)}")

if __name__ == "__main__":
    analyze_pilot("data/raw/trajectories/pilot.jsonl")
