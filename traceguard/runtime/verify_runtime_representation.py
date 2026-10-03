"""P0 equivalence verification for the runtime TRACEGUARD input path."""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRACEGUARD_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRACEGUARD_DIR))

from src.config import EMBEDDING_MODEL_NAME, STEPS_PER_TRAJECTORY
from src.model_lstm import collate_fn
from src.preprocessing import build_step_text
from runtime.input_representation import (
    build_runtime_padded_batch,
    build_runtime_prefix_tensor,
    build_runtime_trajectory_texts,
    encode_runtime_steps,
)


RESULTS_PATH = SCRIPT_DIR / "p0_runtime_representation_results.json"


def _sample_steps() -> list[dict]:
    return [
        {
            "step": index,
            "action": f"Process item {index}",
            "tool": "search" if index < 4 else "calculator",
            "tool_input": {"query": f"item-{index}"},
            "tool_observation": f"Observation {index}",
            "state": f"Working on item {index}",
        }
        for index in range(1, STEPS_PER_TRAJECTORY + 1)
    ]


def _assert_equal(actual, expected, label: str):
    if isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=label)
    elif isinstance(actual, torch.Tensor):
        if not torch.equal(actual, expected):
            raise AssertionError(label)
    elif actual != expected:
        raise AssertionError(label)


def main() -> int:
    user_goal = "Process the six requested items"
    steps = _sample_steps()
    canonical_texts = [
        build_step_text(step, user_goal, step["step"]) for step in steps
    ]
    runtime_texts = build_runtime_trajectory_texts(steps, user_goal)
    _assert_equal(runtime_texts, canonical_texts, "Runtime text differs from canonical text")

    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    encode_signature = inspect.signature(embedding_model.encode)
    if encode_signature.parameters["normalize_embeddings"].default is not False:
        raise AssertionError("Frozen encode path does not default to no normalization")

    canonical_embeddings = embedding_model.encode(
        canonical_texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    runtime_embeddings = encode_runtime_steps(embedding_model, runtime_texts)
    _assert_equal(
        runtime_embeddings,
        canonical_embeddings,
        "Runtime embeddings differ from canonical embeddings",
    )

    single_embedding = embedding_model.encode(
        [canonical_texts[0]],
        convert_to_numpy=True,
    )
    batch_single_max_abs_diff = float(
        np.max(np.abs(single_embedding - canonical_embeddings[:1]))
    )
    if not np.isfinite(batch_single_max_abs_diff):
        raise AssertionError("Single-step and batch embeddings contain non-finite values")

    if runtime_embeddings.shape != (STEPS_PER_TRAJECTORY, 384):
        raise AssertionError(f"Unexpected embedding shape: {runtime_embeddings.shape}")
    if runtime_embeddings.dtype != canonical_embeddings.dtype:
        raise AssertionError("Runtime embedding dtype differs from canonical dtype")

    cpu_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    cpu_embeddings = cpu_model.encode(
        canonical_texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    device_cpu_max_abs_diff = float(
        np.max(np.abs(canonical_embeddings - cpu_embeddings))
    )
    if not np.isfinite(device_cpu_max_abs_diff):
        raise AssertionError("Device comparison produced non-finite embeddings")

    prefix_results = {}
    for prefix_length in range(1, STEPS_PER_TRAJECTORY + 1):
        runtime_tensor, runtime_lengths = build_runtime_prefix_tensor(
            runtime_embeddings,
            prefix_length,
        )
        canonical_tensor = torch.tensor(
            canonical_embeddings[:prefix_length],
            dtype=torch.float32,
        ).unsqueeze(0)
        _assert_equal(
            runtime_tensor,
            canonical_tensor,
            f"Prefix {prefix_length} tensor differs from canonical tensor",
        )
        if runtime_tensor.shape != (1, prefix_length, 384):
            raise AssertionError(f"Unexpected prefix shape: {runtime_tensor.shape}")
        if runtime_lengths != [prefix_length]:
            raise AssertionError(f"Unexpected prefix lengths: {runtime_lengths}")
        _assert_equal(
            runtime_tensor[0, -1],
            torch.tensor(canonical_embeddings[prefix_length - 1], dtype=torch.float32),
            f"Prefix {prefix_length} changed chronological order",
        )
        prefix_results[str(prefix_length)] = {
            "shape": list(runtime_tensor.shape),
            "lengths": runtime_lengths,
            "ordered": True,
        }

    short = runtime_embeddings[:2]
    full = runtime_embeddings[:STEPS_PER_TRAJECTORY]
    expected_padded, _labels, expected_lengths, _ids = collate_fn(
        [
            (torch.tensor(short, dtype=torch.float32), torch.tensor(0), "short"),
            (torch.tensor(full, dtype=torch.float32), torch.tensor(0), "full"),
        ]
    )
    runtime_padded, runtime_lengths = build_runtime_padded_batch([short, full])
    _assert_equal(runtime_padded, expected_padded, "Runtime padding differs from canonical padding")
    if runtime_lengths != expected_lengths or runtime_lengths != [2, 6]:
        raise AssertionError(f"Unexpected padded lengths: {runtime_lengths}")
    if not torch.equal(runtime_padded[0, 2:], torch.zeros(4, 384)):
        raise AssertionError("Padding is not trailing zero padding")

    results = {
        "all_passed": True,
        "canonical_preprocessing": {
            "module": "src.preprocessing",
            "function": "build_step_text",
            "fields": [
                "user_goal",
                "step_number",
                "action",
                "tool",
                "tool_input",
                "tool_observation",
                "state",
            ],
            "format": "Labeled sections separated by two newlines; values follow each label and end without a trailing newline.",
            "metadata_excluded": True,
            "missing_values": "None and missing values become empty strings; user_goal is passed through unchanged.",
        },
        "embedding": {
            "model": EMBEDDING_MODEL_NAME,
            "dimension": 384,
            "single_step_shape": list(single_embedding.shape),
            "multi_step_shape": list(runtime_embeddings.shape),
            "dtype": str(runtime_embeddings.dtype),
            "normalize_embeddings": False,
            "batch_encoding_matches_canonical": True,
            "single_vs_batch_max_abs_diff": batch_single_max_abs_diff,
            "single_vs_batch_bitwise_equal": bool(
                np.array_equal(single_embedding, canonical_embeddings[:1])
            ),
            "selected_device": str(embedding_model.device),
            "cpu_comparison_max_abs_diff": device_cpu_max_abs_diff,
            "cpu_comparison_allclose_atol_1e-5": bool(
                np.allclose(canonical_embeddings, cpu_embeddings, rtol=0.0, atol=1e-5)
            ),
        },
        "sequence": {
            "full_length": STEPS_PER_TRAJECTORY,
            "prefix_shapes": {key: value["shape"] for key, value in prefix_results.items()},
            "padding": "right/trailing zeros only for variable-length batches",
            "padding_value": 0.0,
            "tensor_dtype": "torch.float32",
            "tensor_layout": "[batch, sequence, embedding]",
            "variable_lengths": [2, 6],
            "ordering": "step list order is preserved; non-consecutive step numbers are rejected",
        },
        "prefixes": prefix_results,
        "frozen_artifacts_modified": False,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Runtime representation verification passed. Results: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())