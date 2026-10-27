"""Model Registry for TRACEGUARD-X.

Provides centralized management, metadata, and lazy loading for:
- Frozen Baseline LSTM
- TRACEGUARD-X GRU (Track B1)
- Temporal Attention (Track B2)
- Multi-Task Early Warning (Track B3)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRACEGUARD_DIR = REPO_ROOT / "traceguard"
TRACEGUARD_X_DIR = REPO_ROOT / "traceguard_x"


@dataclass
class ModelMetadata:
    id: str
    name: str
    architecture: str
    parameters: int
    checkpoint_path: str
    checkpoint_available: bool
    supported_outputs: list[str]
    research_track: str
    macro_f1: float
    hijacked_f1: float
    description: str
    is_loaded: bool = False
    device: str = "cpu"


class ModelRegistry:
    """Central registry and lifecycle manager for TRACEGUARD-X models."""

    def __init__(self) -> None:
        self._detectors: dict[str, Any] = {}
        self._checkpoints = {
            "lstm": TRACEGUARD_DIR / "outputs" / "models" / "lstm_seed42_best.pth",
            "gru": TRACEGUARD_X_DIR / "experiments" / "track_b1_gru" / "gru_seed42_best.pth",
            "attention": TRACEGUARD_X_DIR / "experiments" / "track_b2_attention" / "attention_seed42_best.pth",
            "multitask": TRACEGUARD_X_DIR / "experiments" / "track_b3_earlywarn" / "multitask_seed42_best.pth",
        }

    def get_metadata(self, model_id: str) -> ModelMetadata:
        mid = model_id.lower()
        ckpt = self._checkpoints.get(mid)
        if not ckpt:
            raise KeyError(f"Unknown model identifier: '{model_id}'")

        ckpt_exists = ckpt.exists()
        is_loaded = mid in self._detectors
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if mid == "lstm":
            return ModelMetadata(
                id="lstm",
                name="Frozen Baseline LSTM",
                architecture="Unidirectional 2-Layer LSTM (384-D input, 128 hidden, 0.2 dropout)",
                parameters=395651,
                checkpoint_path=str(ckpt),
                checkpoint_available=ckpt_exists,
                supported_outputs=["class_probabilities", "predicted_class", "uncertainty_entropy"],
                research_track="Track A: Baseline Reproduction",
                macro_f1=0.8533,
                hijacked_f1=0.9032,
                description="Original frozen research baseline for 3-class trajectory prefix classification.",
                is_loaded=is_loaded,
                device=device,
            )
        elif mid == "gru":
            return ModelMetadata(
                id="gru",
                name="TraceGuard-X GRU",
                architecture="Gated Recurrent Unit (384-D input, 128 hidden, 2 layers, 0.2 dropout)",
                parameters=296835,
                checkpoint_path=str(ckpt),
                checkpoint_available=ckpt_exists,
                supported_outputs=["class_probabilities", "predicted_class", "uncertainty_entropy"],
                research_track="Track B1: Recurrent Efficiency",
                macro_f1=0.9339,
                hijacked_f1=0.9333,
                description="Lightweight recurrent architecture achieving 25.0% parameter reduction and improved confidence.",
                is_loaded=is_loaded,
                device=device,
            )
        elif mid == "attention":
            return ModelMetadata(
                id="attention",
                name="Temporal Attention",
                architecture="Temporal Attention Network with Step-Wise Softmax Pooling",
                parameters=305155,
                checkpoint_path=str(ckpt),
                checkpoint_available=ckpt_exists,
                supported_outputs=["class_probabilities", "predicted_class", "uncertainty_entropy", "attention_weights"],
                research_track="Track B2: Temporal Attention",
                macro_f1=0.8885,
                hijacked_f1=0.9333,
                description="Attention-weighted trajectory classifier producing interpretable step attention weights.",
                is_loaded=is_loaded,
                device=device,
            )
        elif mid == "multitask":
            return ModelMetadata(
                id="multitask",
                name="Multi-Task Early Warning",
                architecture="Multi-Task Network with Joint Trajectory & Imminent Deviation Heads",
                parameters=296964,
                checkpoint_path=str(ckpt),
                checkpoint_available=ckpt_exists,
                supported_outputs=["class_probabilities", "predicted_class", "uncertainty_entropy", "warning_score"],
                research_track="Track B3: Multi-Task Early Warning",
                macro_f1=0.9225,
                hijacked_f1=0.9032,
                description="Joint classification and auxiliary early warning head estimating imminent deviation risk.",
                is_loaded=is_loaded,
                device=device,
            )
        else:
            raise KeyError(f"Unknown model identifier: '{model_id}'")

    def list_models(self) -> list[ModelMetadata]:
        return [self.get_metadata(mid) for mid in ("lstm", "gru", "attention", "multitask")]

    def get_or_load_detector(self, model_id: str, user_goal: str = "API analysis task") -> Any:
        """Return or instantiate a TraceGuardXRuntimeDetector."""
        mid = model_id.lower()
        meta = self.get_metadata(mid)
        if not meta.checkpoint_available:
            raise FileNotFoundError(f"Checkpoint for '{mid}' not found at: {meta.checkpoint_path}")

        from traceguard_x.runtime.detector import TraceGuardXRuntimeDetector

        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = TraceGuardXRuntimeDetector(
            user_goal=user_goal,
            model_name=mid,
            checkpoint_path=meta.checkpoint_path,
            device=device,
        )
        self._detectors[mid] = detector
        return detector


# Global singleton registry
registry = ModelRegistry()
