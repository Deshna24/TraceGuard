"""Model B3: Multi-task early-warning model for TRACEGUARD-X.

Jointly predicts:
1. The 3-class trajectory state (BENIGN, INJECTION_RESISTED, HIJACKED).
2. An auxiliary early-warning score indicating whether a deviation/precursor
   is present or imminent within a prediction horizon.

CRITICAL SEPARATION NOTICE:
The auxiliary early-warning score is an experimental research indicator,
NEVER to be substituted for the official P(HIJACKED) safety gate probability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TraceGuardMultiTaskEarlyWarning(nn.Module):
    """Joint 3-class trajectory classifier and auxiliary early-warning detector.

    Args:
        input_size: Embedding dimension (canonical 384).
        hidden_size: Hidden representation size (default 128).
        num_layers: Recurrent layers (default 2, unidirectional GRU).
        dropout: Dropout probability.
        num_classes: Output classes for main task (default 3).
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

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(p=dropout)

        # Head 1: Official 3-class trajectory classification
        self.fc_main = nn.Linear(hidden_size, num_classes)

        # Head 2: Auxiliary binary early-warning indicator
        self.fc_warning = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, lengths: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing both main 3-class logits and auxiliary warning logit.

        Args:
            x: FloatTensor [batch, max_len, input_size]
            lengths: list of sequence lengths per sample

        Returns:
            (main_logits, warning_logit):
                main_logits: FloatTensor [batch, num_classes]
                warning_logit: FloatTensor [batch, 1]
        """
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, hn = self.encoder(packed)
        last_hidden = hn[-1]
        rep = self.dropout(last_hidden)

        main_logits = self.fc_main(rep)
        warning_logit = self.fc_warning(rep).squeeze(-1)
        return main_logits, warning_logit

    def predict_main(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Helper returning only the 3-class logits (for standard evaluators)."""
        logits, _ = self.forward(x, lengths)
        return logits

    def get_config(self) -> dict:
        return {
            "architecture": "TraceGuardMultiTaskEarlyWarning",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_p,
            "num_classes": self.num_classes,
            "bidirectional": False,
            "pooling": "last_hidden_state",
            "auxiliary_task": "deviation_imminent_indicator",
        }

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "main_head": sum(p.numel() for p in self.fc_main.parameters()),
            "warning_head": sum(p.numel() for p in self.fc_warning.parameters()),
        }
