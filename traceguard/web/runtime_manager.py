"""Run lifecycle manager for the TRACEGUARD web application.

Tracks the current run state, prevents conflicting concurrent runs,
and stores the event history for the active run. Does NOT contain
detector logic, gate logic, or tool execution.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from web.schemas.api_models import EventEnvelope, EventType, RunStatus, ScenarioName

logger = logging.getLogger("traceguard.web.runtime_manager")


@dataclass
class RunState:
    """Mutable in-memory state for a single agent run."""
    run_id: str
    scenario: ScenarioName
    user_goal: str
    status: RunStatus = RunStatus.RUNNING
    current_step: int = 0
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    event_history: list[dict[str, Any]] = field(default_factory=list)
    final_result: Optional[dict[str, Any]] = None
    _task: Optional[asyncio.Task] = field(default=None, repr=False)

    def record_event(self, envelope: EventEnvelope) -> None:
        """Append a serialized envelope to the event history."""
        self.event_history.append(envelope.to_json_dict())

    def update_step(self, step: int) -> None:
        self.current_step = max(self.current_step, step)

    def add_trajectory_step(self, step_dict: dict[str, Any]) -> None:
        self.trajectory.append(copy.deepcopy(step_dict))

    def complete(self, status: RunStatus, result: dict[str, Any] | None = None) -> None:
        self.status = status
        self.final_result = copy.deepcopy(result) if result else None


class RuntimeManager:
    """Enforces single-run-at-a-time and exposes run state to the API layer."""

    def __init__(self) -> None:
        self._current_run: Optional[RunState] = None
        self._lock = asyncio.Lock()

    @property
    def current_run(self) -> Optional[RunState]:
        return self._current_run

    @property
    def is_running(self) -> bool:
        return (
            self._current_run is not None
            and self._current_run.status == RunStatus.RUNNING
        )

    async def start_run(
        self,
        scenario: ScenarioName,
        user_goal: str,
    ) -> RunState:
        """Create a new run, preventing concurrent conflicting runs."""
        async with self._lock:
            if self.is_running:
                raise RuntimeError(
                    f"A run is already in progress: {self._current_run.run_id}"
                )
            run_id = f"{scenario.value.lower()}-{uuid4().hex[:8]}"
            self._current_run = RunState(
                run_id=run_id,
                scenario=scenario,
                user_goal=user_goal,
            )
            logger.info("Run started: %s scenario=%s", run_id, scenario.value)
            return self._current_run

    async def stop_run(self) -> bool:
        """Cancel the current run if one is active."""
        async with self._lock:
            if not self.is_running:
                return False
            run = self._current_run
            if run._task and not run._task.done():
                run._task.cancel()
            run.complete(RunStatus.STOPPED)
            logger.info("Run stopped: %s", run.run_id)
            return True

    async def reset(self) -> bool:
        """Clear the current run state without touching frozen artifacts."""
        async with self._lock:
            if self.is_running:
                run = self._current_run
                if run._task and not run._task.done():
                    run._task.cancel()
            self._current_run = None
            logger.info("Runtime state reset.")
            return True

    def set_task(self, task: asyncio.Task) -> None:
        """Attach the asyncio task to the current run so it can be cancelled."""
        if self._current_run:
            self._current_run._task = task

    def get_status(self) -> dict[str, Any]:
        """Return a safe snapshot of the current run state for the API."""
        if self._current_run is None:
            return {
                "run_id": None,
                "scenario": None,
                "status": "idle",
                "step": 0,
                "trajectory_length": 0,
            }
        run = self._current_run
        return {
            "run_id": run.run_id,
            "scenario": run.scenario.value,
            "status": run.status.value,
            "step": run.current_step,
            "trajectory_length": len(run.trajectory),
        }


# Module-level singleton.
runtime_manager = RuntimeManager()


__all__ = ["RunState", "RuntimeManager", "runtime_manager"]
