import sys
sys.path.insert(0, r"c:\Users\DESHNA\TraceGuard\traceguard")
from src.utils import set_seed, get_device
from src.data_loader import load_dataset
from src.split import create_group_aware_split, save_split
from src.preprocessing import build_trajectory_texts

set_seed(42)
device = get_device()
trajs = load_dataset()
train_t, val_t, test_t, split_info = create_group_aware_split(trajs, seed=42)
save_split(split_info, 42)

print("SPLIT OK:")
print("  train:", split_info["train_n_groups"], "groups,", split_info["train_n_trajs"], "trajs")
print("  val:  ", split_info["val_n_groups"],   "groups,", split_info["val_n_trajs"],   "trajs")
print("  test: ", split_info["test_n_groups"],  "groups,", split_info["test_n_trajs"],  "trajs")
print("  train dist:", split_info["train_label_dist"])
print("  val dist:  ", split_info["val_label_dist"])
print("  test dist: ", split_info["test_label_dist"])

traj = trajs[0]
texts = build_trajectory_texts(traj)
print("\nStep text preview (first 200 chars):")
print(texts[0][:200])
print("\nPhases 1-5 PASSED.")
