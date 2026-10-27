"""Uncertainty-Aware 3-Way Pre-Action Safety Gate for TRACEGUARD-X.

Decisions:
- ALLOW: Safe to proceed with tool execution.
- ABSTAIN: Ambiguous risk or high epistemic/entropy uncertainty; requires human operator escalation.
- BLOCK: Confirmed behavioral hijacking; hard stop.

Invariants:
- BLOCK and ABSTAIN decisions NEVER execute the proposed tool.
- Detector errors FAIL CLOSED (defaults to BLOCK).
- Unknown tools are rejected before execution.
- Preserves full compatibility with canonical PreActionGate interface.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from runtime.action_gate import BlockedActionError, ProposedAction


UncertaintyDecision = Literal["ALLOW", "ABSTAIN", "BLOCK"]


class AbstainedActionError(RuntimeError):
    """Raised when an action is escalated/abstained and must not execute."""


@dataclass(frozen=True)
class UncertaintyGateDecision:
    """Structured result of uncertainty-aware pre-action gating."""

    decision: UncertaintyDecision
    hijack_probability: float | None
    uncertainty_score: float | None
    threshold_block: float
    threshold_abstain: float
    uncertainty_tau: float
    predicted_class: str | None
    class_probabilities: dict[str, float] | None
    detection_step: int
    pre_action: bool
    proposed_action: dict[str, Any]
    reason: str
    error: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == "ALLOW"

    @property
    def abstained(self) -> bool:
        return self.decision == "ABSTAIN"

    @property
    def blocked(self) -> bool:
        return self.decision == "BLOCK"

    @property
    def threshold(self) -> float:
        return self.threshold_block


def compute_normalized_entropy(probabilities: dict[str, float]) -> float:
    """Compute Shannon entropy normalized by log(num_classes) into [0, 1]."""
    probs = [max(1e-12, float(p)) for p in probabilities.values()]
    total = sum(probs)
    probs = [p / total for p in probs]
    k = len(probs)
    if k <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(k)
    return float(min(1.0, max(0.0, entropy / max_entropy)))


class UncertaintyActionGate:
    """Three-way uncertainty-aware gate enforcing strict tool execution boundaries."""

    def __init__(
        self,
        detector: Any,
        threshold_block: float = 0.50,
        threshold_abstain: float = 0.25,
        uncertainty_tau: float = 0.70,
        allowed_tools: Sequence[str] | None = None,
    ) -> None:
        if not (0.0 <= threshold_abstain <= threshold_block <= 1.0):
            raise ValueError(
                f"Threshold invariant violated: 0.0 <= threshold_abstain ({threshold_abstain}) "
                f"<= threshold_block ({threshold_block}) <= 1.0"
            )
        self.detector = detector
        self.threshold_block = threshold_block
        self.threshold_abstain = threshold_abstain
        self.uncertainty_tau = uncertainty_tau
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else {"search", "calculator", "database"}

    def _sync_prefix(self, steps: list[dict[str, Any]]) -> None:
        existing = copy.deepcopy(getattr(self.detector, "steps", []))
        if len(existing) > len(steps) or existing != steps[: len(existing)]:
            replace = getattr(self.detector, "replace_steps", None)
            if callable(replace):
                replace(steps)
                return
            setattr(self.detector, "steps", copy.deepcopy(steps))
            return
        for step in steps[len(existing) :]:
            self.detector.add_step(step)

    def evaluate(
        self,
        proposed_action: ProposedAction | Mapping[str, Any],
        trajectory_prefix: Sequence[Mapping[str, Any]] | Any,
    ) -> UncertaintyGateDecision:
        """Evaluate a proposed action and return an UncertaintyGateDecision without tool execution."""
        # Normalize action
        if isinstance(proposed_action, ProposedAction):
            action = proposed_action
        else:
            action = ProposedAction.from_mapping(proposed_action)

        # Enforce known-tool check before evaluation
        if action.tool not in self.allowed_tools:
            return UncertaintyGateDecision(
                decision="BLOCK",
                hijack_probability=None,
                uncertainty_score=None,
                threshold_block=self.threshold_block,
                threshold_abstain=self.threshold_abstain,
                uncertainty_tau=self.uncertainty_tau,
                predicted_class=None,
                class_probabilities=None,
                detection_step=0,
                pre_action=True,
                proposed_action=action.as_dict(),
                reason=f"Rejected: Tool '{action.tool}' is not in allowed sandbox tools.",
                error=f"Unknown tool: '{action.tool}'",
            )

        # Normalize prefix
        if hasattr(trajectory_prefix, "get_steps"):
            steps = trajectory_prefix.get_steps()
        elif isinstance(trajectory_prefix, Sequence):
            steps = [copy.deepcopy(s) for s in trajectory_prefix]
        else:
            raise TypeError(f"Invalid trajectory prefix type: {type(trajectory_prefix)}")

        detection_step = len(steps)

        # Evaluate through detector
        try:
            self._sync_prefix(steps)
            pred = self.detector.predict()
            probs = dict(pred.probabilities)
            p_hijacked = float(probs["HIJACKED"])
            entropy = compute_normalized_entropy(probs)

            # Decision policy:
            # 1. Hard BLOCK if P(HIJACKED) >= threshold_block
            if p_hijacked >= self.threshold_block:
                decision = "BLOCK"
                reason = f"High hijacking risk: P(HIJACKED)={p_hijacked:.4f} >= {self.threshold_block:.2f}."
            # 2. ABSTAIN if elevated risk or high uncertainty
            elif p_hijacked >= self.threshold_abstain:
                decision = "ABSTAIN"
                reason = f"Elevated hijacking risk: P(HIJACKED)={p_hijacked:.4f} in abstain band [{self.threshold_abstain:.2f}, {self.threshold_block:.2f})."
            elif entropy >= self.uncertainty_tau:
                decision = "ABSTAIN"
                reason = f"High prediction uncertainty: normalized entropy={entropy:.4f} >= {self.uncertainty_tau:.2f}."
            # 3. ALLOW
            else:
                decision = "ALLOW"
                reason = f"Action permitted: P(HIJACKED)={p_hijacked:.4f} < {self.threshold_abstain:.2f} and entropy={entropy:.4f} < {self.uncertainty_tau:.2f}."

            return UncertaintyGateDecision(
                decision=decision,
                hijack_probability=p_hijacked,
                uncertainty_score=entropy,
                threshold_block=self.threshold_block,
                threshold_abstain=self.threshold_abstain,
                uncertainty_tau=self.uncertainty_tau,
                predicted_class=pred.predicted_class,
                class_probabilities=probs,
                detection_step=detection_step,
                pre_action=True,
                proposed_action=action.as_dict(),
                reason=reason,
            )

        except Exception as exc:
            # FAIL CLOSED: any detector failure blocks the proposed action
            return UncertaintyGateDecision(
                decision="BLOCK",
                hijack_probability=None,
                uncertainty_score=None,
                threshold_block=self.threshold_block,
                threshold_abstain=self.threshold_abstain,
                uncertainty_tau=self.uncertainty_tau,
                predicted_class=None,
                class_probabilities=None,
                detection_step=detection_step,
                pre_action=True,
                proposed_action=action.as_dict(),
                reason=f"Gate evaluation failure; failing closed: {exc}",
                error=str(exc),
            )

    def execute_if_allowed(
        self,
        decision: UncertaintyGateDecision,
        executor_func: Callable[[Any], Any],
        tool_input: Any,
    ) -> Any:
        """Execute tool ONLY if decision is ALLOW; strictly blocks on BLOCK or ABSTAIN."""
        if decision.decision == "BLOCK":
            raise BlockedActionError(f"Action blocked by uncertainty gate: {decision.reason}")
        if decision.decision == "ABSTAIN":
            raise AbstainedActionError(f"Action abstained/escalated by uncertainty gate: {decision.reason}")
        if decision.decision != "ALLOW":
            raise RuntimeError(f"Unrecognized gate decision '{decision.decision}', failing closed.")
        return executor_func(tool_input)
