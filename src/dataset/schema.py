from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class Step(BaseModel):
    step_id: int
    action: str = Field(description="The action type (e.g., 'tool_call', 'finish')")
    tool: Optional[str] = Field(None, description="The name of the tool called")
    tool_input: Optional[Any] = Field(None, description="The input passed to the tool")
    observation: Optional[str] = Field(None, description="The output from the tool or environment")
    state_summary: Optional[str] = Field(None, description="Summary of the agent's internal state or thought")
    
    # Labeling fields
    injection: bool = Field(default=False, description="Whether this step contains the malicious injection")
    deviation: bool = Field(default=False, description="Whether this step represents the first behavioral deviation")
    unwanted_action: bool = Field(default=False, description="Whether an unwanted action was executed in this step")

class Trajectory(BaseModel):
    trajectory_id: str
    task_id: str
    original_goal: str
    steps: List[Step] = Field(default_factory=list)
    
    label: str = Field(description="'benign', 'resisted', or 'hijacked'")
    
    attack_type: Optional[str] = Field(None, description="The family of attack used (e.g., 'Type A')")
    injection_step: Optional[int] = Field(None, description="Step ID where injection occurred")
    deviation_step: Optional[int] = Field(None, description="Step ID where deviation occurred")
    unwanted_action_step: Optional[int] = Field(None, description="Step ID where unwanted action occurred")
    
    # Added per user request
    attack_success: Optional[bool] = Field(None, description="True if the attack succeeded (hijacked)")
    agent_response_to_injection: Optional[str] = Field(None, description="How the agent responded to the injection")
    deviation_reason: Optional[str] = Field(None, description="Objective reason why this step was marked as a deviation")
