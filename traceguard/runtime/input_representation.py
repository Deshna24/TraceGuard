"""Runtime input construction that reuses the frozen TRACEGUARD path."""

from __future__ import annotations

import numpy as np
import torch

from src.config import EMBEDDING_MODEL_NAME
from src.model_lstm import collate_fn
from src.preprocessing import build_step_text


def build_runtime_step_text(
    step: dict,
    user_goal: str,
    step_number: int,
) -> str:
    """Delegate runtime text construction to the frozen preprocessing function."""
    return build_step_text(step, user_goal, step_number)


def build_runtime_trajectory_texts(
    steps: list[dict],
    user_goal: str,
) -> list[str]:
    """Build canonical texts while rejecting reordered or missing step numbers."""
    if not steps:
        raise ValueError("Runtime trajectory must contain at least one step")

    texts = []
    for expected_number, step in enumerate(steps, start=1):
        step_number = step.get("step")
        if step_number != expected_number:
            raise ValueError(
                "Runtime steps must be supplied in chronological order starting "
                f"at step 1; expected {expected_number}, got {step_number}"
            )
        texts.append(build_runtime_step_text(step, user_goal, step_number))
    return texts


def encode_runtime_steps(embedding_model, step_texts: list[str]) -> np.ndarray:
    """Mirror the frozen batch encoding call and return [steps, 384] embeddings."""
    if not step_texts:
        raise ValueError("At least one step text is required")
    return np.asarray(
        embedding_model.encode(
            step_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    )


def build_runtime_prefix_tensor(
    embeddings: np.ndarray,
    prefix_length: int,
) -> tuple[torch.Tensor, list[int]]:
    """Construct the unpadded [1, prefix, 384] tensor used by prefix inference."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected [steps, embedding_dim], got {embeddings.shape}")
    if embeddings.shape[1] != 384:
        raise ValueError(f"Expected embedding dimension 384, got {embeddings.shape[1]}")
    if prefix_length < 1 or prefix_length > embeddings.shape[0]:
        raise ValueError(f"Invalid prefix length {prefix_length}")

    prefix = embeddings[:prefix_length]
    tensor = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0)
    return tensor, [prefix_length]


def build_runtime_padded_batch(
    prefix_embeddings: list[np.ndarray],
) -> tuple[torch.Tensor, list[int]]:
    """Reuse the frozen collate function for right-zero-padded variable batches."""
    if not prefix_embeddings:
        raise ValueError("At least one prefix is required")

    batch = [
        (
            torch.tensor(embeddings, dtype=torch.float32),
            torch.tensor(0, dtype=torch.long),
            str(index),
        )
        for index, embeddings in enumerate(prefix_embeddings)
    ]
    padded, _labels, lengths, _ids = collate_fn(batch)
    return padded, lengths


__all__ = [
    "EMBEDDING_MODEL_NAME",
    "build_runtime_padded_batch",
    "build_runtime_prefix_tensor",
    "build_runtime_step_text",
    "build_runtime_trajectory_texts",
    "encode_runtime_steps",
]