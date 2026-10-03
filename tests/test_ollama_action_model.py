import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "traceguard"))

from agent.agent import (
    ControlledAgent,
    OllamaActionModel,
    OllamaModelError,
    ParsedAction,
    parse_model_output,
    AgentOutputError,
)
from agent.tools import build_default_registry
from runtime.action_gate import PreActionGate
from runtime.detector import DetectorPrediction


class _AllowingDetector:
    def __init__(self):
        self.user_goal = "local Ollama integration test"
        self.steps = []

    def add_step(self, step):
        self.steps.append(copy.deepcopy(step))

    def predict(self):
        return DetectorPrediction(
            probabilities={"BENIGN": 0.9, "INJECTION_RESISTED": 0.05, "HIJACKED": 0.05},
            predicted_class="BENIGN",
            prefix_length=len(self.steps),
        )

    def should_block(self, threshold=0.5):
        return False


def _model():
    return OllamaActionModel(timeout=30)


def test_ollama_is_reachable():
    content = _model().propose(
        "Return a final answer saying reachable.",
        [],
        [],
    )
    assert content


def test_granite_produces_a_valid_structured_action():
    raw_output = _model().propose(
        "Use the calculator tool to calculate 2 + 2. Return an action, not a final answer.",
        [],
        [
            {
                "name": "calculator",
                "description": "Calculate a basic arithmetic expression.",
                "input_schema": {"expression": "string"},
            }
        ],
    )
    output_type, parsed = parse_model_output(raw_output)
    assert output_type == "action"
    assert isinstance(parsed, ParsedAction)
    assert parsed.tool == "calculator"
    assert parsed.tool_input == {"expression": "2 + 2"}


def test_ollama_action_uses_existing_agent_and_gate_path():
    registry, tools = build_default_registry()
    result = ControlledAgent(
        _model(),
        registry,
        PreActionGate(_AllowingDetector()),
        max_steps=2,
    ).run("Use the calculator tool to calculate 2 + 2, then report the result.")

    assert result.status == "completed", result
    assert result.gate_decisions
    assert result.gate_decisions[0].decision == "ALLOW"
    assert result.gate_decisions[0].pre_action is True
    assert tools["calculator"].execution_count == 1


def test_empty_ollama_content_fails_closed(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"message":{"content":"   "}}'

    monkeypatch.setattr("agent.agent.urlopen", lambda *_args, **_kwargs: _Response())
    with pytest.raises(OllamaModelError, match="non-empty message content"):
        _model().propose("Return an action.", [], [])


def test_malformed_ollama_content_is_rejected_by_agent_parser(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"message":{"content":"not json"}}'

    monkeypatch.setattr("agent.agent.urlopen", lambda *_args, **_kwargs: _Response())
    with pytest.raises(AgentOutputError, match="not valid JSON"):
        parse_model_output(_model().propose("Return an action.", [], []))
