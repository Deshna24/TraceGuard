import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

LABEL_MAP = {"BENIGN": 0, "INJECTION_RESISTED": 1, "HIJACKED": 2}
INV_LABEL_MAP = {0: "BENIGN", 1: "INJECTION_RESISTED", 2: "HIJACKED"}

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, embeddings_dict):
        self.trajectories = trajectories
        self.embeddings_dict = embeddings_dict
        
    def __len__(self):
        return len(self.trajectories)
        
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_id = traj["trajectory_id"]
        emb = self.embeddings_dict[traj_id] # (seq_len, dim)
        label = LABEL_MAP[traj["label"]]
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long), traj_id

def collate_fn(batch):
    # batch is list of (emb, label, traj_id)
    embs, labels, traj_ids = zip(*batch)
    lengths = [len(e) for e in embs]
    max_len = max(lengths)
    dim = embs[0].shape[1]
    
    padded_embs = torch.zeros(len(batch), max_len, dim)
    for i, e in enumerate(embs):
        padded_embs[i, :len(e), :] = e
        
    return padded_embs, torch.stack(labels), lengths, traj_ids

def create_splits(trajectories, config):
    labels = [t["label"] for t in trajectories]
    
    # Train 70%, Val 15%, Test 15%
    val_test_ratio = config["splits"]["val"] + config["splits"]["test"]
    test_ratio_of_valtest = config["splits"]["test"] / val_test_ratio
    
    train_data, val_test_data, train_labels, val_test_labels = train_test_split(
        trajectories, labels, test_size=val_test_ratio, stratify=labels, random_state=config["seed"]
    )
    
    val_data, test_data, _, _ = train_test_split(
        val_test_data, val_test_labels, test_size=test_ratio_of_valtest, stratify=val_test_labels, random_state=config["seed"]
    )
    
    return train_data, val_data, test_data