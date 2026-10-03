"""Ordered, copy-safe runtime trajectory state for TRACEGUARD."""

from __future__ import annotations

import copy
from typing import Any

from runtime.input_representation import build_runtime_trajectory_texts


CANONICAL_STEP_FIELDS = (
    "step",
    "action",
    "tool",
    "tool_input",
    "tool_observation",
    "state",
)


class TrajectoryState:
    """Own the user goal and ordered canonical runtime steps."""

    def __init__(self, user_goal: str):
        if not isinstance(user_goal, str):
            raise TypeError(f"user_goal must be a string, got {type(user_goal).__name__}")
        self._user_goal = user_goal
        self._steps: list[dict[str, Any]] = []

    @property
    def user_goal(self) -> str:
        """Return the immutable original user goal."""
        return self._user_goal

    def add_step(self, step: dict[str, Any]) -> None:
        """Append the next canonical step without retaining caller-owned objects."""
        if not isinstance(step, dict):
            raise TypeError(f"step must be a dictionary, got {type(step).__name__}")

        expected_number = len(self._steps) + 1
        step_number = step.get("step")
        if isinstance(step_number, bool) or not isinstance(step_number, int):
            raise ValueError(f"step must be integer {expected_number}, got {step_number!r}")
        if step_number != expected_number:
            raise ValueError(
                f"Expected chronological step {expected_number}, got {step_number}"
            )

        stored_step = {
            field: copy.deepcopy(step[field])
            for field in CANONICAL_STEP_FIELDS
            if field in step
        }
        stored_step["step"] = step_number
        self._steps.append(stored_step)

    def get_steps(self) -> list[dict[str, Any]]:
        """Return a deep copy of all ordered steps."""
        return copy.deepcopy(self._steps)

    def get_current_step(self) -> dict[str, Any] | None:
        """Return a deep copy of the latest step, or None when empty."""
        if not self._steps:
            return None
        return copy.deepcopy(self._steps[-1])

    def get_prefix(self, length: int) -> list[dict[str, Any]]:
        """Return a deep-copied prefix without changing trajectory state."""
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError(f"prefix length must be an integer, got {type(length).__name__}")
        if length < 1 or length > len(self._steps):
            raise ValueError(
                f"prefix length must be between 1 and {len(self._steps)}, got {length}"
            )
        return copy.deepcopy(self._steps[:length])

    def to_canonical_texts(self, length: int | None = None) -> list[str]:
        """Convert a full trajectory or prefix through canonical preprocessing."""
        steps = self.get_steps() if length is None else self.get_prefix(length)
        return build_runtime_trajectory_texts(steps, self.user_goal)

    def length(self) -> int:
        """Return the number of stored steps."""
        return len(self._steps)

    def clear(self) -> None:
        """Remove all steps while retaining the original user goal."""
        self._steps.clear()

    def reset(self) -> None:
        """Alias for clear, retaining the original user goal."""
        self.clear()


Trajectory = TrajectoryState


__all__ = ["CANONICAL_STEP_FIELDS", "Trajectory", "TrajectoryState"]