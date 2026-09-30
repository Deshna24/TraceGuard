import json
import statistics
from collections import Counter

file_path = "data/raw/trajectories/traceguard_pilot_v2.jsonl"

total_trajectories = 0
class_dist = Counter()
attack_type_dist = Counter()
lengths = []

num_benign = 0
num_resisted = 0
num_hijacked = 0

has_injection_step = 0
has_deviation_step = 0
has_attack_success = 0

missing_fields = []
seen_ids = set()
duplicate_ids = set()
invalid_combinations = []

expected_fields = ["trajectory_id", "label", "steps", "injection", "attack_type", "injection_step", "deviation_step", "attack_success"]

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        total_trajectories += 1
        
        # Missing fields
        for field in expected_fields:
            if field not in data:
                missing_fields.append((data.get("trajectory_id", "UNKNOWN"), field))
        
        # Duplicate IDs
        traj_id = data.get("trajectory_id")
        if traj_id in seen_ids:
            duplicate_ids.add(traj_id)
        if traj_id:
            seen_ids.add(traj_id)
            
        label = data.get("label")
        class_dist[label] += 1
        
        if label == "BENIGN":
            num_benign += 1
        elif label == "INJECTION_RESISTED":
            num_resisted += 1
        elif label == "HIJACKED":
            num_hijacked += 1
            
        attack_type = data.get("attack_type")
        if attack_type:
            attack_type_dist[attack_type] += 1
            
        steps = data.get("steps", [])
        lengths.append(len(steps))
        
        injection = data.get("injection")
        injection_step = data.get("injection_step")
        deviation_step = data.get("deviation_step")
        attack_success = data.get("attack_success")
        
        if injection_step is not None:
            has_injection_step += 1
        if deviation_step is not None:
            has_deviation_step += 1
        if attack_success:
            has_attack_success += 1
            
        # Validate rules
        if label == "BENIGN":
            if not (injection is False and attack_type is None and injection_step is None and deviation_step is None and attack_success is False):
                invalid_combinations.append((traj_id, "BENIGN rules violated"))
        elif label == "INJECTION_RESISTED":
            if not (injection is True and injection_step is not None and deviation_step is None and attack_success is False):
                invalid_combinations.append((traj_id, "INJECTION_RESISTED rules violated"))
        elif label == "HIJACKED":
            if not (injection is True and injection_step is not None and deviation_step is not None and attack_success is True):
                invalid_combinations.append((traj_id, "HIJACKED rules violated"))
        else:
            invalid_combinations.append((traj_id, f"Unknown label: {label}"))

print("==================================================")
print("DATASET AUDIT REPORT")
print("==================================================")
print(f"Total Trajectories: {total_trajectories}")
print(f"Class Distribution: {dict(class_dist)}")
print(f"Attack-Family Distribution: {dict(attack_type_dist)}")
print(f"Minimum steps: {min(lengths) if lengths else 0}")
print(f"Maximum steps: {max(lengths) if lengths else 0}")
print(f"Mean steps: {statistics.mean(lengths) if lengths else 0:.2f}")
print(f"Number of BENIGN: {num_benign}")
print(f"Number of INJECTION_RESISTED: {num_resisted}")
print(f"Number of HIJACKED: {num_hijacked}")
print(f"Number of trajectories with injection_step: {has_injection_step}")
print(f"Number of trajectories with deviation_step: {has_deviation_step}")
print(f"Number of trajectories with attack_success=true: {has_attack_success}")

print(f"\nDuplicate Trajectory IDs: {len(duplicate_ids)}")
if duplicate_ids:
    print(f"Duplicates: {list(duplicate_ids)}")
    
print(f"Missing Fields: {len(missing_fields)}")
if missing_fields:
    print(missing_fields[:10])
    
print(f"Invalid Metadata Combinations: {len(invalid_combinations)}")
if invalid_combinations:
    for ic in invalid_combinations:
        print(f" - {ic[0]}: {ic[1]}")
