"""Model B1: Gated Recurrent Unit (GRU) classifier for TRACEGUARD-X.

A unidirectional, causal GRU architecture matching the input/output contract
of TraceGuardLSTM. GRU uses fewer parameters per recurrent step (reset and
update gates vs. LSTM's input, forget, cell, output gates).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TraceGuardGRU(nn.Module):
    """Unidirectional GRU for 3-class trajectory classification.

    Args:
        input_size: Embedding dimension (canonical 384).
        hidden_size: GRU hidden units (default 128).
        num_layers: Number of recurrent layers (default 2).
        dropout: Dropout probability between recurrent layers and before FC.
        num_classes: Number of output classes (default 3).
    """

    def __init__(
        self,
        input_size: int = 384,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.num_classes = num_classes

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # strictly unidirectional (causal)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Forward pass on padded batch or single trajectory prefix.

        Args:
            x: Tensor of shape [batch, max_len, input_size]
            lengths: list of sequence lengths per item

        Returns:
            logits: Tensor of shape [batch, num_classes]
        """
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, hn = self.gru(packed)
        # hn: [num_layers, batch, hidden_size] -> last layer
        last_hidden = hn[-1]
        out = self.dropout(last_hidden)
        return self.fc(out)

    def get_config(self) -> dict:
        return {
            "architecture": "TraceGuardGRU",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_p,
            "num_classes": self.num_classes,
            "bidirectional": False,
            "pooling": "last_hidden_state",
        }

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        recurrent = sum(p.numel() for p in self.gru.parameters())
        fc = sum(p.numel() for p in self.fc.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "recurrent": recurrent,
            "classifier": fc,
        }
