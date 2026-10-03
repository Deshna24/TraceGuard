"""Controlled tools for the TRACEGUARD agent integration."""

from .tools import (
    ControlledToolRegistry,
    ToolExecution,
    ToolResult,
    build_default_registry,
)
from .agent import (
    AgentRunResult,
    ControlledAgent,
    OllamaActionModel,
    OllamaModelError,
    ParsedAction,
    ScriptedActionModel,
)
from .scenarios import BENIGN_SCENARIO, BenignScenario

__all__ = [
    "ControlledToolRegistry",
    "ToolExecution",
    "ToolResult",
    "build_default_registry",
    "AgentRunResult",
    "ControlledAgent",
    "OllamaActionModel",
    "OllamaModelError",
    "ParsedAction",
    "ScriptedActionModel",
    "BENIGN_SCENARIO",
    "BenignScenario",
]
