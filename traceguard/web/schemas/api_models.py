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
    """The verified scenarios plus an interactive custom mode."""
    BENIGN = "BENIGN"
    INJECTION_RESISTED = "INJECTION_RESISTED"
    HIJACKED = "HIJACKED"
    CUSTOM = "CUSTOM"


class ModelArchitectureName(str, Enum):
    """Supported detector architectures."""
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"
    MULTITASK = "multitask"


class GateTypeName(str, Enum):
    """Supported safety gate policies."""
    STANDARD = "standard"
    UNCERTAINTY_AWARE = "uncertainty_aware"
    UNCERTAINTY = "uncertainty"


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
        ..., description="One of BENIGN, INJECTION_RESISTED, HIJACKED, CUSTOM"
    )
    custom_goal: Optional[str] = Field(
        None, description="Interactive mode user goal"
    )
    custom_injection: Optional[str] = Field(
        None, description="Interactive mode injection payload"
    )
    injection_target: Optional[str] = Field(
        None, description="Interactive mode injection target (e.g. search, database, calculator)"
    )
    model_architecture: Optional[ModelArchitectureName] = Field(
        default=ModelArchitectureName.LSTM,
        description="Selected architecture: lstm, gru, attention, multitask",
    )
    gate_type: Optional[GateTypeName] = Field(
        default=GateTypeName.STANDARD,
        description="Selected safety gate: standard or uncertainty_aware",
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


# ---------------------------------------------------------------------------
# Direct Analysis & Productization API Models
# ---------------------------------------------------------------------------

class TrajectoryStepInput(BaseModel):
    step: Optional[int] = None
    action: Optional[str] = None
    tool: Optional[str] = None
    tool_input: Optional[dict[str, Any]] = None
    tool_observation: Optional[str] = None
    state: Optional[str] = None


class TrajectoryAnalysisRequest(BaseModel):
    user_goal: str = Field(..., description="High-level goal given by user to the agent")
    steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Sequential list of chronological steps in the trajectory prefix"
    )
    model_architecture: ModelArchitectureName = Field(
        default=ModelArchitectureName.LSTM,
        description="Model architecture to evaluate: lstm, gru, attention, multitask"
    )
    gate_type: GateTypeName = Field(
        default=GateTypeName.STANDARD,
        description="Safety gate policy mode: standard or uncertainty_aware"
    )
    candidate_action: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional proposed action to evaluate before tool invocation"
    )


class UncertaintyInfo(BaseModel):
    normalized_entropy: float
    level: str


class PolicyDecisionInfo(BaseModel):
    gate_type: str
    decision: str
    reason: str
    action_permitted: bool
    requires_operator_review: bool
    threshold_block: float
    threshold_abstain: Optional[float] = None
    entropy_tau: Optional[float] = None


class ExecutionInfo(BaseModel):
    device: str
    device_name: str
    latency_ms: float


class TrajectoryAnalysisResponse(BaseModel):
    model_id: str
    user_goal: str
    trajectory_length: int
    predicted_class: str
    probabilities: dict[str, float]
    uncertainty: UncertaintyInfo
    attention_weights: Optional[list[float]] = None
    warning_score: Optional[float] = None
    policy: PolicyDecisionInfo
    execution: ExecutionInfo


class PolicyEvaluateRequest(BaseModel):
    p_hijacked: float = Field(..., ge=0.0, le=1.0)
    entropy: float = Field(0.0, ge=0.0, le=1.0)
    gate_type: GateTypeName = Field(default=GateTypeName.STANDARD)
    candidate_action: Optional[dict[str, Any]] = None


class PolicyEvaluateResponse(BaseModel):
    decision: str
    reason: str
    action_permitted: bool
    requires_operator_review: bool
    threshold_block: float
    threshold_abstain: Optional[float] = None
    entropy_tau: Optional[float] = None


class ModelMetadataResponse(BaseModel):
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
    is_loaded: bool
    device: str


class SystemHealthResponse(BaseModel):
    status: str
    version: str
    device: str
    device_name: str
    cuda_available: bool
    vram_total_gb: Optional[float] = None
    vram_allocated_gb: Optional[float] = None
    torch_version: str
    embedding_cache_available: bool
    models_available: dict[str, bool]


__all__ = [
    "EventEnvelope",
    "EventType",
    "ExecutionInfo",
    "GateTypeName",
    "HealthResponse",
    "ModelArchitectureName",
    "ModelMetadataResponse",
    "PolicyDecisionInfo",
    "PolicyEvaluateRequest",
    "PolicyEvaluateResponse",
    "RunActionResponse",
    "RunStartRequest",
    "RunStartResponse",
    "RunStatus",
    "ScenarioInfo",
    "ScenarioName",
    "ScenariosResponse",
    "StatusResponse",
    "SystemHealthResponse",
    "TrajectoryAnalysisRequest",
    "TrajectoryAnalysisResponse",
    "TrajectoryStepInput",
    "UncertaintyInfo",
]

