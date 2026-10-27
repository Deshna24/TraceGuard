"""Inference and Policy Service for TRACEGUARD-X.

Provides clean, synchronous REST-callable inference and policy evaluation for:
- Arbitrary external agent trajectories
- UI Trajectory Analyzer
- Pre-action policy gating before dangerous tool execution
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Optional

import torch

from traceguard_x.gate.uncertainty_gate import compute_normalized_entropy
from traceguard_x.service.model_registry import registry


class InferenceService:
    """Service executing real PyTorch inference and policy evaluation."""

    def __init__(self) -> None:
        self.registry = registry

    def analyze(
        self,
        user_goal: str,
        steps: list[dict[str, Any]],
        model_id: str = "lstm",
        gate_type: str = "standard",
        candidate_action: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Perform full prefix preprocessing, neural inference, and policy evaluation."""
        if not isinstance(user_goal, str) or not user_goal.strip():
            raise ValueError("user_goal must be a non-empty string.")
        if not isinstance(steps, list):
            raise TypeError("steps must be a list of trajectory step dictionaries.")

        # Validate step indices
        for i, s in enumerate(steps, 1):
            if not isinstance(s, dict):
                raise TypeError(f"Step {i} must be a dictionary.")
            if "step" not in s:
                s["step"] = i

        start_time = time.perf_counter()

        # Load / retrieve detector
        detector = self.registry.get_or_load_detector(model_id, user_goal)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        # Replace steps and execute strictly causal prefix inference
        detector.replace_steps(steps)
        pred = detector.predict()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        probs = {
            "BENIGN": float(pred.probabilities["BENIGN"]),
            "INJECTION_RESISTED": float(pred.probabilities["INJECTION_RESISTED"]),
            "HIJACKED": float(pred.probabilities["HIJACKED"]),
        }
        p_hijacked = probs["HIJACKED"]
        entropy = float(compute_normalized_entropy(probs))

        # Auxiliary outputs
        attention_weights: list[float] | None = None
        warning_score: float | None = None

        if model_id == "attention" and hasattr(detector, "last_prediction") and detector.last_prediction:
            attention_weights = detector.last_prediction.attention_weights
        elif model_id == "multitask" and hasattr(detector, "last_prediction") and detector.last_prediction:
            warning_score = detector.last_prediction.warning_score

        # Policy Engine Evaluation
        gate_decision = self.evaluate_policy(
            p_hijacked=p_hijacked,
            entropy=entropy,
            gate_type=gate_type,
            candidate_action=candidate_action,
        )

        return {
            "model_id": model_id,
            "user_goal": user_goal,
            "trajectory_length": len(steps),
            "predicted_class": pred.predicted_class,
            "probabilities": probs,
            "uncertainty": {
                "normalized_entropy": round(entropy, 4),
                "level": "HIGH" if entropy >= 0.70 else ("MODERATE" if entropy >= 0.40 else "LOW"),
            },
            "attention_weights": attention_weights,
            "warning_score": round(warning_score, 4) if warning_score is not None else None,
            "policy": {
                "gate_type": gate_type,
                "decision": gate_decision["decision"],
                "reason": gate_decision["reason"],
                "action_permitted": gate_decision["action_permitted"],
                "requires_operator_review": gate_decision["requires_operator_review"],
                "threshold_block": gate_decision["threshold_block"],
                "threshold_abstain": gate_decision.get("threshold_abstain"),
            },
            "execution": {
                "device": device_str,
                "device_name": device_name,
                "latency_ms": round(elapsed_ms, 2),
            },
        }

    def evaluate_policy(
        self,
        p_hijacked: float,
        entropy: float,
        gate_type: str = "standard",
        candidate_action: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Independent policy engine mapping probabilities to safety gate action outcomes."""
        gate_mode = str(gate_type).lower()

        if gate_mode in ("uncertainty_aware", "uncertainty"):
            threshold_block = 0.50
            threshold_abstain = 0.25
            entropy_tau = 0.70

            if p_hijacked >= threshold_block:
                decision = "BLOCK"
                reason = f"Critical risk: P(HIJACKED)={p_hijacked:.4f} exceeds block threshold ({threshold_block:.2f}). Action denied."
                permitted = False
                operator_review = False
            elif p_hijacked >= threshold_abstain:
                decision = "ABSTAIN"
                reason = f"Precursor risk: P(HIJACKED)={p_hijacked:.4f} is in ambiguous abstain band [{threshold_abstain:.2f}, {threshold_block:.2f}). Escalated to operator."
                permitted = False
                operator_review = True
            elif entropy >= entropy_tau:
                decision = "ABSTAIN"
                reason = f"High epistemic uncertainty: Normalized entropy H={entropy:.4f} >= {entropy_tau:.2f}. Escalated to operator."
                permitted = False
                operator_review = True
            else:
                decision = "ALLOW"
                reason = f"Low risk: P(HIJACKED)={p_hijacked:.4f} and entropy H={entropy:.4f} are below safety limits. Action permitted."
                permitted = True
                operator_review = False

            return {
                "decision": decision,
                "reason": reason,
                "action_permitted": permitted,
                "requires_operator_review": operator_review,
                "threshold_block": threshold_block,
                "threshold_abstain": threshold_abstain,
                "entropy_tau": entropy_tau,
            }

        else:  # Standard binary gate
            threshold_block = 0.50
            if p_hijacked >= threshold_block:
                decision = "BLOCK"
                reason = f"P(HIJACKED)={p_hijacked:.4f} >= {threshold_block:.2f}. Binary gate denied tool execution."
                permitted = False
            else:
                decision = "ALLOW"
                reason = f"P(HIJACKED)={p_hijacked:.4f} < {threshold_block:.2f}. Binary gate permitted tool execution."
                permitted = True

            return {
                "decision": decision,
                "reason": reason,
                "action_permitted": permitted,
                "requires_operator_review": False,
                "threshold_block": threshold_block,
                "threshold_abstain": None,
            }


# Global singleton inference service
inference_service = InferenceService()
