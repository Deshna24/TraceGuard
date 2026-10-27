"""Unit tests for Track F UncertaintyActionGate and execution boundary safety."""

import pytest
from runtime.action_gate import BlockedActionError, ProposedAction
from runtime.detector import DetectorPrediction

from traceguard_x.gate.uncertainty_gate import (
    AbstainedActionError,
    UncertaintyActionGate,
    UncertaintyGateDecision,
    compute_normalized_entropy,
)


class MockDetector:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.steps = []
        self.user_goal = "Test Goal"

    def add_step(self, step):
        self.steps.append(step)

    def predict(self):
        predicted = max(self.probabilities, key=self.probabilities.get)
        return DetectorPrediction(
            probabilities=self.probabilities,
            predicted_class=predicted,
            prefix_length=len(self.steps),
        )

    def should_block(self, threshold=0.5):
        return self.probabilities["HIJACKED"] >= threshold


class FailingDetector:
    def __init__(self):
        self.steps = []
        self.user_goal = "Failing Goal"

    def add_step(self, step):
        self.steps.append(step)

    def predict(self):
        raise RuntimeError("GPU OOM / Detector failure")

    def should_block(self, threshold=0.5):
        return True


def test_normalized_entropy():
    # Maximum uncertainty for 3 classes: uniform [1/3, 1/3, 1/3] -> entropy = 1.0
    uniform = {"BENIGN": 1 / 3, "INJECTION_RESISTED": 1 / 3, "HIJACKED": 1 / 3}
    h_max = compute_normalized_entropy(uniform)
    assert h_max == pytest.approx(1.0, abs=1e-3)

    # Minimum uncertainty: one-hot [1, 0, 0] -> entropy = 0.0
    pure = {"BENIGN": 1.0, "INJECTION_RESISTED": 0.0, "HIJACKED": 0.0}
    h_min = compute_normalized_entropy(pure)
    assert h_min == pytest.approx(0.0, abs=1e-3)


def test_gate_decision_allow():
    detector = MockDetector({"BENIGN": 0.95, "INJECTION_RESISTED": 0.03, "HIJACKED": 0.02})
    gate = UncertaintyActionGate(detector, threshold_block=0.50, threshold_abstain=0.25, uncertainty_tau=0.70)

    action = ProposedAction(action="Search flight", tool="search", tool_input={"query": "flights"})
    decision = gate.evaluate(action, [{"step": 1, "action": "Search flight", "tool": "search"}])

    assert decision.decision == "ALLOW"
    assert decision.allowed
    assert not decision.blocked
    assert not decision.abstained

    executed = []
    result = gate.execute_if_allowed(decision, lambda inp: executed.append(inp) or "ok", action.tool_input)
    assert result == "ok"
    assert len(executed) == 1


def test_gate_decision_block():
    detector = MockDetector({"BENIGN": 0.05, "INJECTION_RESISTED": 0.05, "HIJACKED": 0.90})
    gate = UncertaintyActionGate(detector, threshold_block=0.50, threshold_abstain=0.25, uncertainty_tau=0.70)

    action = ProposedAction(action="Export database", tool="database", tool_input={"op": "export"})
    decision = gate.evaluate(action, [{"step": 1, "action": "Export", "tool": "database"}])

    assert decision.decision == "BLOCK"
    assert decision.blocked
    assert not decision.allowed

    # Tool MUST NOT execute
    executed = []
    with pytest.raises(BlockedActionError):
        gate.execute_if_allowed(decision, lambda inp: executed.append(inp), action.tool_input)
    assert len(executed) == 0


def test_gate_decision_abstain_by_risk():
    # P(HIJACKED) = 0.35 is in abstain band [0.25, 0.50)
    detector = MockDetector({"BENIGN": 0.55, "INJECTION_RESISTED": 0.10, "HIJACKED": 0.35})
    gate = UncertaintyActionGate(detector, threshold_block=0.50, threshold_abstain=0.25, uncertainty_tau=0.70)

    action = ProposedAction(action="Calculate", tool="calculator", tool_input={"expr": "1+1"})
    decision = gate.evaluate(action, [{"step": 1, "action": "Calc", "tool": "calculator"}])

    assert decision.decision == "ABSTAIN"
    assert decision.abstained
    assert not decision.allowed

    # Tool MUST NOT execute
    executed = []
    with pytest.raises(AbstainedActionError):
        gate.execute_if_allowed(decision, lambda inp: executed.append(inp), action.tool_input)
    assert len(executed) == 0


def test_gate_decision_abstain_by_entropy():
    # P(HIJACKED) is low (0.10), but high ambiguity between BENIGN (0.45) and RESISTED (0.45)
    detector = MockDetector({"BENIGN": 0.45, "INJECTION_RESISTED": 0.45, "HIJACKED": 0.10})
    gate = UncertaintyActionGate(detector, threshold_block=0.50, threshold_abstain=0.25, uncertainty_tau=0.70)

    action = ProposedAction(action="Search", tool="search", tool_input={"query": "test"})
    decision = gate.evaluate(action, [{"step": 1, "action": "Search", "tool": "search"}])

    assert decision.decision == "ABSTAIN"
    assert decision.abstained

    executed = []
    with pytest.raises(AbstainedActionError):
        gate.execute_if_allowed(decision, lambda inp: executed.append(inp), action.tool_input)
    assert len(executed) == 0


def test_gate_fails_closed_on_detector_error():
    detector = FailingDetector()
    gate = UncertaintyActionGate(detector)

    action = ProposedAction(action="Search", tool="search", tool_input={"query": "test"})
    decision = gate.evaluate(action, [{"step": 1, "action": "Search", "tool": "search"}])

    # Must fail closed: decision is BLOCK and execution raises BlockedActionError
    assert decision.decision == "BLOCK"
    assert decision.blocked
    assert decision.error is not None

    executed = []
    with pytest.raises(BlockedActionError):
        gate.execute_if_allowed(decision, lambda inp: executed.append(inp), action.tool_input)
    assert len(executed) == 0


def test_gate_rejects_unknown_tool():
    detector = MockDetector({"BENIGN": 0.99, "INJECTION_RESISTED": 0.01, "HIJACKED": 0.0})
    gate = UncertaintyActionGate(detector, allowed_tools={"search", "calculator", "database"})

    # Malicious or unknown tool outside sandbox
    action = ProposedAction(action="Run Shell", tool="bash_shell", tool_input={"cmd": "rm -rf"})
    decision = gate.evaluate(action, [{"step": 1, "action": "Shell", "tool": "bash_shell"}])

    assert decision.decision == "BLOCK"
    assert "not in allowed sandbox" in decision.reason
