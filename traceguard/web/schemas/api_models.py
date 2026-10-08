from pydantic import BaseModel, Field
from typing import Any, Optional, Dict, List

class RunScenarioRequest(BaseModel):
    scenario: str = Field(..., description="BENIGN, INJECTION_RESISTED, or HIJACKED")

class ScenarioResponse(BaseModel):
    name: str
    goal: str

class HealthResponse(BaseModel):
    status: str = "ok"

class EventPayload(BaseModel):
    event_type: str
    data: Dict[str, Any]
