"""Pydantic request/response models and the WebSocket event envelope.

These schemas define the contract between the backend and Person 2's frontend.
They NEVER contain detector logic, threshold decisions, or tool execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ScenarioName(str, Enum):
    """The three verified scenarios."""
    BENIGN = "BENIGN"
    INJECTION_RESISTED = "INJECTION_RESISTED"
    HIJACKED = "HIJACKED"


class EventType(str, Enum):
    """Every WebSocket event the backend can emit."""
    RUN_STARTED = "RUN_STARTED"
    USER_TASK_RECEIVED = "USER_TASK_RECEIVED"
    AGENT_STARTED = "AGENT_STARTED"
    ACTION_PROPOSED = "ACTION_PROPOSED"
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    OBSERVATION_RECEIVED = "OBSERVATION_RECEIVED"
    INJECTION_OBSERVED = "INJECTION_OBSERVED"
    TRAJECTORY_UPDATED = "TRAJECTORY_UPDATED"
    DETECTOR_EVALUATED = "DETECTOR_EVALUATED"
    THRESHOLD_CROSSED = "THRESHOLD_CROSSED"
    GATE_DECISION = "GATE_DECISION"
    ACTION_BLOCKED = "ACTION_BLOCKED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUNTIME_ERROR = "RUNTIME_ERROR"


class RunStatus(str, Enum):
    """Possible terminal statuses for a run."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    MODEL_FAILURE = "model_failure"
    TOOL_FAILURE = "tool_failure"
    INVALID_ACTION = "invalid_action"
    MAX_STEPS_REACHED = "max_steps_reached"
    STOPPED = "stopped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# WebSocket event envelope
# ---------------------------------------------------------------------------

class EventEnvelope(BaseModel):
    """Authoritative envelope for every WebSocket event."""
    run_id: str
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_type: EventType
    step: Optional[int] = None
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Serializable dict matching the contract."""
        return {
            "run_id": self.run_id,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "step": self.step,
            "payload": self.payload,
        }


# ---------------------------------------------------------------------------
# REST request/response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0"


class StatusResponse(BaseModel):
    run_id: Optional[str] = None
    scenario: Optional[str] = None
    status: str = "idle"
    step: int = 0
    trajectory_length: int = 0


class RunStartRequest(BaseModel):
    scenario: ScenarioName = Field(
        ..., description="One of BENIGN, INJECTION_RESISTED, HIJACKED"
    )


class RunStartResponse(BaseModel):
    run_id: str
    scenario: str
    status: str = "running"


class RunActionResponse(BaseModel):
    success: bool
    message: str
    run_id: Optional[str] = None


class ScenarioInfo(BaseModel):
    name: str
    user_goal: str
    description: str


class ScenariosResponse(BaseModel):
    scenarios: list[ScenarioInfo]


__all__ = [
    "EventEnvelope",
    "EventType",
    "HealthResponse",
    "RunActionResponse",
    "RunStartRequest",
    "RunStartResponse",
    "RunStatus",
    "ScenarioInfo",
    "ScenarioName",
    "ScenariosResponse",
    "StatusResponse",
]
