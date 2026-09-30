"""
model_lstm.py - Unidirectional LSTM for TRACEGUARD trajectory classification.

ARCHITECTURE:
  Input: [batch, seq_len, embedding_dim]
  -> LSTM (unidirectional, causal)
  -> last hidden state
  -> Dropout
  -> Linear(hidden_size -> 3)
  -> logits

Unidirectional is mandatory: the detector must be causal (no future steps).
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TraceGuardLSTM(nn.Module):
    """
    Unidirectional LSTM for 3-class trajectory classification.

    Args:
        input_size:  Embedding dimension (e.g. 384 for all-MiniLM-L6-v2)
        hidden_size: LSTM hidden units (default 128)
        num_layers:  Number of LSTM layers (default 2)
        dropout:     Dropout probability (default 0.2)
        num_classes: Number of output classes (default 3)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int  = 2,
        dropout: float   = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout_p   = dropout
        self.num_classes = num_classes

        # Multi-layer LSTM; dropout applied between layers (not on last output)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # MUST be unidirectional (causal)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """
        Args:
            x:       FloatTensor [batch, seq_len, embedding_dim]
            lengths: list of actual sequence lengths per sample

        Returns:
            logits: FloatTensor [batch, num_classes]
        """
        # Pack for efficiency (handles variable length prefixes)
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        # hn: [num_layers, batch, hidden_size]  — take last layer
        last_hidden = hn[-1]              # [batch, hidden_size]
        out = self.dropout(last_hidden)
        logits = self.fc(out)             # [batch, num_classes]
        return logits

    def get_config(self) -> dict:
        return {
            "architecture":   "TraceGuardLSTM",
            "input_size":     self.input_size,
            "hidden_size":    self.hidden_size,
            "num_layers":     self.num_layers,
            "dropout":        self.dropout_p,
            "num_classes":    self.num_classes,
            "bidirectional":  False,
            "pooling":        "last_hidden_state",
        }


# ─────────────────────────────────────────────
# Dataset & DataLoader helpers
# ─────────────────────────────────────────────

import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.config import LABEL_MAP, INV_LABEL_MAP


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: list[dict], embeddings_dict: dict[str, np.ndarray]):
        self.trajectories    = trajectories
        self.embeddings_dict = embeddings_dict

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj    = self.trajectories[idx]
        traj_id = traj["trajectory_id"]
        emb     = self.embeddings_dict[traj_id]   # [6, D]
        label   = LABEL_MAP[traj["label"]]
        return (
            torch.tensor(emb, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            traj_id,
        )


def collate_fn(batch):
    """Pad to max length in batch (all trajectories are length 6, but
    prefix evaluation uses shorter sequences, so we keep this generic)."""
    embs, labels, traj_ids = zip(*batch)
    lengths = [e.shape[0] for e in embs]
    max_len = max(lengths)
    dim     = embs[0].shape[1]

    padded = torch.zeros(len(batch), max_len, dim, dtype=torch.float32)
    for i, e in enumerate(embs):
        padded[i, :e.shape[0], :] = e

    return padded, torch.stack(labels), lengths, traj_ids


def make_dataloader(
    trajectories: list[dict],
    embeddings_dict: dict,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    ds = TrajectoryDataset(trajectories, embeddings_dict)
    g  = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=g if shuffle else None,
    )
