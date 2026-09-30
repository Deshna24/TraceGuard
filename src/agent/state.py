import operator
from typing import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task_id: str
    original_goal: str
    # Keep track of intercepted/injected steps
    injected_tool_calls: List[str]
    # For logging
    trajectory_steps: List[dict]
