"""Model architectures for TRACEGUARD-X.

All models operate on canonical sentence-transformers embeddings (dim 384)
and output 3-class logits matching the canonical class ordering:
    0: BENIGN
    1: INJECTION_RESISTED
    2: HIJACKED
"""

from traceguard_x.models.gru import TraceGuardGRU
from traceguard_x.models.attention import TraceGuardTemporalAttention
from traceguard_x.models.multitask import TraceGuardMultiTaskEarlyWarning

__all__ = [
    "TraceGuardGRU",
    "TraceGuardTemporalAttention",
    "TraceGuardMultiTaskEarlyWarning",
]
