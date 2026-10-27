"""Model B2: Temporal Attention classifier for TRACEGUARD-X.

Learns step-level attention weights over a causal recurrent trajectory
representation, pooling dynamically across observed steps while strictly
respecting sequence lengths and padding masks.

ATTENTION INTERPRETATION NOTICE:
Attention weights are diagnostic visualizations of representation pooling,
NOT causal explanations of model decisions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TraceGuardTemporalAttention(nn.Module):
    """Causal recurrent encoder with masked temporal attention pooling.

    Args:
        input_size: Embedding dimension (canonical 384).
        hidden_size: Hidden representation size (default 128).
        attention_dim: Internal attention projection dimension (default 64).
        num_layers: Recurrent encoder layers (default 2, unidirectional).
        dropout: Dropout rate applied to encoder outputs and pre-FC.
        num_classes: Output classes (default 3).
    """

    def __init__(
        self,
        input_size: int = 384,
        hidden_size: int = 128,
        attention_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.num_classes = num_classes

        # Causal unidirectional GRU encoder ensures chronological step representation
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Step-level attention scoring network
        self.attn_proj = nn.Linear(hidden_size, attention_dim)
        self.attn_query = nn.Linear(attention_dim, 1, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: list[int],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: FloatTensor [batch, max_len, input_size]
            lengths: list of sequence lengths per sample (1 <= length <= max_len)
            return_attention: whether to return attention weights tensor

        Returns:
            logits: FloatTensor [batch, num_classes] (if return_attention=False)
            (logits, attn_weights): tuple with attn_weights [batch, max_len]
        """
        batch_size, max_len, _ = x.shape

        # Causal encoding with pack/pad
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.encoder(packed)
        h, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=max_len
        )  # [batch, max_len, hidden_size]

        # Compute attention scores
        # u: [batch, max_len, attention_dim]
        u = torch.tanh(self.attn_proj(h))
        # scores: [batch, max_len]
        scores = self.attn_query(u).squeeze(-1)

        # Build mask for valid prefix steps (True for valid, False for padding)
        device = x.device
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        # Apply large negative value to masked positions before softmax
        scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)  # [batch, max_len]

        # Handle zero-length edge case safely (not present in our dataset)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Context vector: weighted sum of hidden states
        # [batch, 1, max_len] @ [batch, max_len, hidden_size] -> [batch, 1, hidden_size]
        context = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)

        out = self.dropout(context)
        logits = self.fc(out)

        if return_attention:
            return logits, attn_weights
        return logits

    def get_config(self) -> dict:
        return {
            "architecture": "TraceGuardTemporalAttention",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "attention_dim": self.attention_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_p,
            "num_classes": self.num_classes,
            "bidirectional": False,
            "pooling": "temporal_attention",
        }

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        attn = sum(p.numel() for p in self.attn_proj.parameters()) + sum(
            p.numel() for p in self.attn_query.parameters()
        )
        return {
            "total": total,
            "trainable": trainable,
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "attention": attn,
            "classifier": sum(p.numel() for p in self.fc.parameters()),
        }
