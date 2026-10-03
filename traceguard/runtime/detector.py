"""Inference-only runtime wrapper for the frozen TRACEGUARD LSTM."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import (
    CLASS_NAMES,
    DETECTION_THRESHOLD,
    EMBEDDING_MODEL_NAME,
    LSTM_DROPOUT,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    MODELS_DIR,
)
from src.model_lstm import TraceGuardLSTM
from src.utils import get_device
from runtime.input_representation import (
    build_runtime_prefix_tensor,
    build_runtime_trajectory_texts,
    encode_runtime_steps,
)


EXPECTED_MODEL_CONFIG = {
    "architecture": "TraceGuardLSTM",
    "input_size": 384,
    "hidden_size": LSTM_HIDDEN_SIZE,
    "num_layers": LSTM_NUM_LAYERS,
    "dropout": LSTM_DROPOUT,
    "num_classes": 3,
    "bidirectional": False,
    "pooling": "last_hidden_state",
}


def threshold_blocks(probability: float, threshold: float = DETECTION_THRESHOLD) -> bool:
    """Apply the fixed inclusive threshold without changing the probability."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    return probability >= threshold


@dataclass(frozen=True)
class DetectorPrediction:
    """Serializable information from one frozen-prefix inference."""

    probabilities: dict[str, float]
    predicted_class: str
    prefix_length: int


class TraceGuardDetector:
    """Stateful prefix holder around the frozen MiniLM + TRACEGUARD LSTM."""

    def __init__(
        self,
        user_goal: str,
        checkpoint_path: str | Path | None = None,
        device: torch.device | str | None = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        if not isinstance(user_goal, str):
            raise TypeError(f"user_goal must be a string, got {type(user_goal).__name__}")

        self.user_goal = user_goal
        self.checkpoint_path = Path(checkpoint_path or MODELS_DIR / "lstm_seed42_best.pth")
        self.device = torch.device(device) if device is not None else get_device()
        self.embedding_model_name = embedding_model_name
        self.steps: list[dict[str, Any]] = []
        self._input_tensor: torch.Tensor | None = None
        self._lengths: list[int] | None = None
        self._prediction: DetectorPrediction | None = None

        self.model = self._load_model()
        self.embedding_model = self._load_embedding_model()

    def _load_model(self) -> TraceGuardLSTM:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"TRACEGUARD checkpoint not found: {self.checkpoint_path}"
            )

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load TRACEGUARD checkpoint {self.checkpoint_path}: {exc}"
            ) from exc

        stored_config = checkpoint.get("model_config")
        if stored_config != EXPECTED_MODEL_CONFIG:
            raise ValueError(
                "Frozen checkpoint configuration mismatch: "
                f"expected {EXPECTED_MODEL_CONFIG}, got {stored_config}"
            )

        model = TraceGuardLSTM(
            input_size=384,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT,
            num_classes=3,
        )
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model weights from {self.checkpoint_path}: {exc}"
            ) from exc

        model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _load_embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                self.embedding_model_name,
                device=str(self.device),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model {self.embedding_model_name} "
                f"on {self.device}: {exc}"
            ) from exc

        dimension = model.get_sentence_embedding_dimension()
        if dimension != 384:
            raise ValueError(f"Expected MiniLM embedding dimension 384, got {dimension}")
        return model

    def add_step(self, step: dict[str, Any]) -> None:
        """Append one chronologically next step without mutating the caller's dict."""
        if not isinstance(step, dict):
            raise TypeError(f"step must be a dictionary, got {type(step).__name__}")

        expected_number = len(self.steps) + 1
        if step.get("step") != expected_number:
            raise ValueError(
                f"Expected runtime step number {expected_number}, got {step.get('step')}"
            )

        self.steps.append(copy.deepcopy(step))
        self._input_tensor = None
        self._lengths = None
        self._prediction = None

    def replace_steps(self, steps: list[dict[str, Any]]) -> None:
        """Replace the scored prefix with a validated completed trajectory prefix.

        A pre-action candidate has no tool observation.  Once its allowed tool
        completes, the runtime replaces that candidate with the authoritative
        completed step before evaluating the next action.
        """
        if not isinstance(steps, list):
            raise TypeError(f"steps must be a list, got {type(steps).__name__}")
        copied_steps: list[dict[str, Any]] = []
        for expected_number, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                raise TypeError(f"step must be a dictionary, got {type(step).__name__}")
            if step.get("step") != expected_number:
                raise ValueError(
                    f"Expected runtime step number {expected_number}, got {step.get('step')}"
                )
            copied_steps.append(copy.deepcopy(step))
        self.steps = copied_steps
        self._input_tensor = None
        self._lengths = None
        self._prediction = None

    def encode_prefix(self) -> torch.Tensor:
        """Encode the current ordered prefix into the frozen LSTM input tensor."""
        if not self.steps:
            raise ValueError("Cannot encode an empty TRACEGUARD prefix")

        try:
            texts = build_runtime_trajectory_texts(self.steps, self.user_goal)
            embeddings = encode_runtime_steps(self.embedding_model, texts)
            tensor, lengths = build_runtime_prefix_tensor(embeddings, len(self.steps))
            tensor = tensor.to(self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to encode TRACEGUARD prefix of length {len(self.steps)}: {exc}"
            ) from exc

        self._input_tensor = tensor
        self._lengths = lengths
        return tensor

    def predict(self) -> DetectorPrediction:
        """Run one inference-only prediction on the current prefix."""
        if self._prediction is not None:
            return self._prediction

        input_tensor = self.encode_prefix() if self._input_tensor is None else self._input_tensor
        lengths = self._lengths
        if lengths is None:
            raise RuntimeError("TRACEGUARD prefix lengths are missing after encoding")

        try:
            with torch.no_grad():
                logits = self.model(input_tensor, lengths)
                probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        except Exception as exc:
            raise RuntimeError(
                f"Frozen TRACEGUARD inference failed for prefix length {len(self.steps)}: {exc}"
            ) from exc

        if probabilities.shape != (3,) or not np.all(np.isfinite(probabilities)):
            raise RuntimeError(f"Invalid TRACEGUARD probability output: {probabilities}")
        if not np.all((probabilities >= 0.0) & (probabilities <= 1.0)):
            raise RuntimeError(f"TRACEGUARD probabilities out of range: {probabilities}")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
            raise RuntimeError(f"TRACEGUARD probabilities do not sum to 1: {probabilities}")

        probability_map = {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(CLASS_NAMES)
        }
        self._prediction = DetectorPrediction(
            probabilities=probability_map,
            predicted_class=CLASS_NAMES[int(np.argmax(probabilities))],
            prefix_length=len(self.steps),
        )
        return self._prediction

    def get_class_probabilities(self) -> dict[str, float]:
        """Return probabilities keyed by the verified class names."""
        return dict(self.predict().probabilities)

    def hijack_probability(self) -> float:
        """Return the frozen model's P(HIJACKED) output."""
        return self.get_class_probabilities()["HIJACKED"]

    def should_block(self, threshold: float = DETECTION_THRESHOLD) -> bool:
        """Return the inclusive threshold decision for the current prefix."""
        return threshold_blocks(self.hijack_probability(), threshold)


__all__ = [
    "DetectorPrediction",
    "EXPECTED_MODEL_CONFIG",
    "TraceGuardDetector",
    "threshold_blocks",
]
