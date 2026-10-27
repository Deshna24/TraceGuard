"""TRACEGUARD-X Unified Runtime Detector supporting baseline and experimental architectures.

Supported models:
- "lstm" (default): Frozen Baseline LSTM
- "gru": TraceGuardGRU (Track B1)
- "attention": TraceGuardTemporalAttention (Track B2)
- "multitask": TraceGuardMultiTaskEarlyWarning (Track B3)

Inference is strictly causal (prefix length k receives only steps 1..k),
models remain in evaluation mode with inference_mode(),
and embedding caching is keyed by (user_goal, step_index).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from runtime.detector import (
    DetectorPrediction,
    EXPECTED_MODEL_CONFIG,
    threshold_blocks,
)
from runtime.input_representation import (
    build_runtime_trajectory_texts,
    build_runtime_prefix_tensor,
    encode_runtime_steps,
)
import src.model_lstm as canonical_lstm
from src.utils import get_device

from traceguard_x.models.attention import TraceGuardTemporalAttention
from traceguard_x.models.gru import TraceGuardGRU
from traceguard_x.models.multitask import TraceGuardMultiTaskEarlyWarning
from traceguard_x.paths import experiment_dir, FROZEN_MODELS_DIR


ModelArchitecture = Literal["lstm", "gru", "attention", "multitask"]


@dataclass(frozen=True)
class TraceGuardXPrediction:
    """Extended prediction record containing both 3-class probabilities and optional research heads."""

    probabilities: dict[str, float]
    predicted_class: str
    prefix_length: int
    model_name: str
    warning_score: float | None = None
    attention_weights: list[float] | None = None


class TraceGuardXRuntimeDetector:
    """Stateful runtime detector for TRACEGUARD-X models."""

    def __init__(
        self,
        user_goal: str,
        model_name: ModelArchitecture | str = "lstm",
        checkpoint_path: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(user_goal, str):
            raise TypeError(f"user_goal must be a string, got {type(user_goal).__name__}")

        self.user_goal = user_goal
        self.model_name = str(model_name).lower()
        self.device = torch.device(device) if device is not None else get_device()
        self.steps: list[dict[str, Any]] = []

        # Model and weights
        self.model = self._load_model(checkpoint_path)
        self.last_prediction: TraceGuardXPrediction | None = None

        # Lazy-loaded embedding model from runtime cache
        from runtime.detector import TraceGuardDetector
        dummy_detector = TraceGuardDetector(user_goal="cache_loader")
        self.embedding_model = dummy_detector.embedding_model

    def _load_model(self, checkpoint_path: str | Path | None) -> nn.Module:
        if self.model_name == "lstm":
            path = Path(checkpoint_path or FROZEN_MODELS_DIR / "lstm_seed42_best.pth")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model = canonical_lstm.TraceGuardLSTM(
                input_size=384, hidden_size=128, num_layers=2, dropout=0.2, num_classes=3
            )
            model.load_state_dict(ckpt["model_state_dict"])

        elif self.model_name == "gru":
            path = Path(checkpoint_path or experiment_dir("track_b1_gru") / "gru_seed42_best.pth")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model = TraceGuardGRU(
                input_size=384, hidden_size=128, num_layers=2, dropout=0.2, num_classes=3
            )
            model.load_state_dict(ckpt["model_state_dict"])

        elif self.model_name == "attention":
            path = Path(checkpoint_path or experiment_dir("track_b2_attention") / "attention_seed42_best.pth")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model = TraceGuardTemporalAttention(
                input_size=384, hidden_size=128, num_layers=2, dropout=0.2, num_classes=3
            )
            model.load_state_dict(ckpt["model_state_dict"])

        elif self.model_name == "multitask":
            path = Path(checkpoint_path or experiment_dir("track_b3_earlywarn") / "multitask_seed42_best.pth")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model = TraceGuardMultiTaskEarlyWarning(
                input_size=384, hidden_size=128, num_layers=2, dropout=0.2, num_classes=3
            )
            model.load_state_dict(ckpt["model_state_dict"])

        else:
            raise ValueError(f"Unknown TRACEGUARD-X model architecture: '{self.model_name}'")

        model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def add_step(self, step: dict[str, Any]) -> None:
        self.steps.append(copy.deepcopy(step))
        self.last_prediction = None

    def replace_steps(self, steps: list[dict[str, Any]]) -> None:
        self.steps = [copy.deepcopy(s) for s in steps]
        self.last_prediction = None

    def predict(self) -> DetectorPrediction:
        """Run strictly prefix-causal inference on current steps."""
        if not self.steps:
            return DetectorPrediction(
                probabilities={"BENIGN": 1.0, "INJECTION_RESISTED": 0.0, "HIJACKED": 0.0},
                predicted_class="BENIGN",
                prefix_length=0,
            )

        # Encode steps with canonical text representation
        texts = build_runtime_trajectory_texts(self.steps, self.user_goal)
        embeddings = encode_runtime_steps(self.embedding_model, texts)
        x, lengths = build_runtime_prefix_tensor(embeddings, len(self.steps))
        x = x.to(self.device)

        warning_score = None
        attn_weights = None

        with torch.inference_mode():
            if self.model_name == "multitask":
                logits, warn_logit = self.model(x, lengths)
                warning_score = float(torch.sigmoid(warn_logit)[0].item())
            elif self.model_name == "attention":
                logits, weights = self.model(x, lengths, return_attention=True)
                attn_weights = weights[0].cpu().numpy().tolist()
            else:
                logits = self.model(x, lengths)

            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        class_names = ["BENIGN", "INJECTION_RESISTED", "HIJACKED"]
        prob_dict = {
            "BENIGN": float(probs[0]),
            "INJECTION_RESISTED": float(probs[1]),
            "HIJACKED": float(probs[2]),
        }
        pred_class = class_names[int(np.argmax(probs))]

        self.last_prediction = TraceGuardXPrediction(
            probabilities=prob_dict,
            predicted_class=pred_class,
            prefix_length=len(self.steps),
            model_name=self.model_name,
            warning_score=warning_score,
            attention_weights=attn_weights,
        )

        return DetectorPrediction(
            probabilities=prob_dict,
            predicted_class=pred_class,
            prefix_length=len(self.steps),
        )

    def should_block(self, threshold: float = 0.5) -> bool:
        pred = self.last_prediction or self.predict()
        return threshold_blocks(pred.probabilities["HIJACKED"], threshold=threshold)
