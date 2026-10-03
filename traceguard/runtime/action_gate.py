"""Fail-closed pre-action safety gate for TRACEGUARD."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from src.config import DETECTION_THRESHOLD
from runtime.detector import TraceGuardDetector
from runtime.trajectory import TrajectoryState


Decision = Literal["ALLOW", "BLOCK"]


@dataclass(frozen=True)
class ProposedAction:
    """The observable action proposed before its tool is executed."""

    action: str
    tool: str
    tool_input: Any

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or not self.action.strip():
            raise ValueError("proposed action must contain a non-empty action string")
        if not isinstance(self.tool, str) or not self.tool.strip():
            raise ValueError("proposed action must contain a non-empty tool string")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProposedAction":
        if not isinstance(value, Mapping):
            raise TypeError(
                f"proposed_action must be a mapping or ProposedAction, got {type(value).__name__}"
            )
        missing = [field for field in ("action", "tool", "tool_input") if field not in value]
        if missing:
            raise ValueError(f"proposed_action is missing required field(s): {', '.join(missing)}")
        return cls(
            action=value["action"],
            tool=value["tool"],
            tool_input=copy.deepcopy(value["tool_input"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "tool": self.tool,
            "tool_input": copy.deepcopy(self.tool_input),
        }


@dataclass(frozen=True)
class GateDecision:
    """Structured result of one pre-action detector evaluation."""

    decision: Decision
    hijack_probability: float | None
    threshold: float
    predicted_class: str | None
    class_probabilities: dict[str, float] | None
    detection_step: int
    pre_action: bool
    proposed_action: dict[str, Any]
    error: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == "ALLOW"

    @property
    def blocked(self) -> bool:
        return self.decision == "BLOCK"


class BlockedActionError(RuntimeError):
    """Raised when an execution helper receives a blocked gate decision."""


class PreActionGate:
    """Evaluate a trajectory prefix before allowing its proposed action."""

    def __init__(self, detector: TraceGuardDetector):
        if not self._is_detector_compatible(detector):
            raise TypeError(
                "detector must be a TraceGuardDetector or implement "
                "user_goal, steps, add_step(), predict(), and should_block()"
            )
        self.detector = detector
        self.threshold = DETECTION_THRESHOLD

    @staticmethod
    def _is_detector_compatible(detector: object) -> bool:
        return isinstance(detector, TraceGuardDetector) or all(
            callable(getattr(detector, method, None))
            for method in ("add_step", "predict", "should_block")
        ) and isinstance(getattr(detector, "steps", None), list)

    @staticmethod
    def _normalize_action(proposed_action: ProposedAction | Mapping[str, Any]) -> ProposedAction:
        if isinstance(proposed_action, ProposedAction):
            return proposed_action
        return ProposedAction.from_mapping(proposed_action)

    @staticmethod
    def _normalize_prefix(
        trajectory_prefix: TrajectoryState | Sequence[Mapping[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        if isinstance(trajectory_prefix, TrajectoryState):
            return trajectory_prefix.user_goal, trajectory_prefix.get_steps()
        if not isinstance(trajectory_prefix, Sequence) or isinstance(
            trajectory_prefix, (str, bytes)
        ):
            raise TypeError(
                "trajectory_prefix must be TrajectoryState or a sequence of step mappings"
            )
        steps = [copy.deepcopy(step) for step in trajectory_prefix]
        if not steps:
            raise ValueError("trajectory_prefix must contain at least one step")
        for expected, step in enumerate(steps, start=1):
            if not isinstance(step, Mapping):
                raise TypeError(f"trajectory step must be a mapping, got {type(step).__name__}")
            if step.get("step") != expected:
                raise ValueError(
                    f"trajectory prefix must contain chronological step {expected}, "
                    f"got {step.get('step')!r}"
                )
        return "", steps

    def _sync_detector_prefix(self, steps: list[dict[str, Any]]) -> None:
        existing_steps = copy.deepcopy(self.detector.steps)
        if len(existing_steps) > len(steps) or existing_steps != steps[: len(existing_steps)]:
            replace_steps = getattr(self.detector, "replace_steps", None)
            if not callable(replace_steps):
                raise ValueError("detector state does not match the supplied trajectory prefix")
            replace_steps(steps)
            return
        for step in steps[len(existing_steps) :]:
            self.detector.add_step(step)

    def evaluate(
        self,
        proposed_action: ProposedAction | Mapping[str, Any],
        trajectory_prefix: TrajectoryState | Sequence[Mapping[str, Any]],
    ) -> GateDecision:
        """Evaluate a proposed action without executing its tool."""
        action = self._normalize_action(proposed_action)
        _user_goal, steps = self._normalize_prefix(trajectory_prefix)
        detection_step = len(steps)

        try:
            self._sync_detector_prefix(steps)
            prediction = self.detector.predict()
            probability = float(prediction.probabilities["HIJACKED"])
            blocked = bool(self.detector.should_block(self.threshold))
            if blocked != (probability >= self.threshold):
                raise RuntimeError("detector threshold decision disagrees with P(HIJACKED)")
            return GateDecision(
                decision="BLOCK" if blocked else "ALLOW",
                hijack_probability=probability,
                threshold=self.threshold,
                predicted_class=prediction.predicted_class,
                class_probabilities=dict(prediction.probabilities),
                detection_step=detection_step,
                pre_action=True,
                proposed_action=action.as_dict(),
            )
        except Exception as exc:
            return GateDecision(
                decision="BLOCK",
                hijack_probability=None,
                threshold=self.threshold,
                predicted_class=None,
                class_probabilities=None,
                detection_step=detection_step,
                pre_action=True,
                proposed_action=action.as_dict(),
                error=f"TRACEGUARD evaluation failed; action blocked: {exc}",
            )

    def should_allow(
        self,
        proposed_action: ProposedAction | Mapping[str, Any],
        trajectory_prefix: TrajectoryState | Sequence[Mapping[str, Any]],
    ) -> bool:
        return self.evaluate(proposed_action, trajectory_prefix).allowed

    def should_block(
        self,
        proposed_action: ProposedAction | Mapping[str, Any],
        trajectory_prefix: TrajectoryState | Sequence[Mapping[str, Any]],
    ) -> bool:
        return self.evaluate(proposed_action, trajectory_prefix).blocked

    @staticmethod
    def execute_if_allowed(
        decision: GateDecision,
        executor: Callable[[dict[str, Any]], Any],
    ) -> Any:
        """Execute only an ALLOW decision; blocked decisions never call executor."""
        if not isinstance(decision, GateDecision):
            raise TypeError(f"decision must be GateDecision, got {type(decision).__name__}")
        if not callable(executor):
            raise TypeError("executor must be callable")
        if decision.blocked:
            raise BlockedActionError(decision.error or "TRACEGUARD blocked the proposed action")
        return executor(copy.deepcopy(decision.proposed_action))


__all__ = [
    "BlockedActionError",
    "Decision",
    "GateDecision",
    "PreActionGate",
    "ProposedAction",
]
