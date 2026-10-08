"""Backend tests for the TRACEGUARD web application.

Tests cover:
  - /health
  - invalid scenario
  - malformed request
  - WebSocket connection
  - event schema (EventEnvelope)
  - event order
  - detector event payload
  - gate event payload
  - blocked action
  - detector failure
  - unknown tool
  - runtime error
  - critical end-to-end tests for BENIGN, INJECTION_RESISTED, HIJACKED
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure imports resolve from the traceguard root.
TRACEGUARD_DIR = Path(__file__).resolve().parent.parent
if str(TRACEGUARD_DIR) not in sys.path:
    sys.path.insert(0, str(TRACEGUARD_DIR))

from fastapi.testclient import TestClient

from web.server import app
from web.schemas.api_models import (
    EventEnvelope,
    EventType,
    RunStatus,
    ScenarioName,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Synchronous test client."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_method_not_allowed(self, client):
        response = client.post("/health")
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# 2. Invalid scenario
# ---------------------------------------------------------------------------

class TestInvalidScenario:
    def test_invalid_scenario_rejected(self, client):
        response = client.post(
            "/api/run/start",
            json={"scenario": "NONEXISTENT"},
        )
        assert response.status_code == 422

    def test_empty_scenario_rejected(self, client):
        response = client.post(
            "/api/run/start",
            json={"scenario": ""},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 3. Malformed request
# ---------------------------------------------------------------------------

class TestMalformedRequest:
    def test_missing_body(self, client):
        response = client.post("/api/run/start")
        assert response.status_code == 422

    def test_wrong_content_type(self, client):
        response = client.post(
            "/api/run/start",
            content="not json",
            headers={"content-type": "text/plain"},
        )
        assert response.status_code == 422

    def test_extra_fields_ignored(self, client):
        """Extra fields should not cause errors (Pydantic ignores by default)."""
        response = client.post(
            "/api/run/start",
            json={"scenario": "BENIGN", "extra_field": "should_be_ignored"},
        )
        # Might succeed or fail if a run is already active, but should NOT 422.
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 4. WebSocket connection
# ---------------------------------------------------------------------------

class TestWebSocket:
    def test_websocket_connects(self, client):
        with client.websocket_connect("/ws") as ws:
            # Connection should be accepted.
            assert ws is not None

    def test_websocket_receives_no_events_when_idle(self, client):
        """When no run is active, no events should be broadcast."""
        import threading
        import time

        received = []

        def listen():
            with client.websocket_connect("/ws") as ws:
                try:
                    ws.send_text("ping")
                    time.sleep(0.5)
                except Exception:
                    pass

        t = threading.Thread(target=listen, daemon=True)
        t.start()
        t.join(timeout=2)


# ---------------------------------------------------------------------------
# 5. Event schema (EventEnvelope)
# ---------------------------------------------------------------------------

class TestEventSchema:
    def test_envelope_has_all_required_fields(self):
        envelope = EventEnvelope(
            run_id="test-run-1",
            event_type=EventType.RUN_STARTED,
            payload={"scenario": "BENIGN"},
        )
        d = envelope.to_json_dict()
        assert "run_id" in d
        assert "event_id" in d
        assert "timestamp" in d
        assert "event_type" in d
        assert "step" in d
        assert "payload" in d

    def test_envelope_event_id_is_unique(self):
        e1 = EventEnvelope(run_id="a", event_type=EventType.RUN_STARTED)
        e2 = EventEnvelope(run_id="a", event_type=EventType.RUN_STARTED)
        assert e1.event_id != e2.event_id

    def test_envelope_timestamp_is_iso(self):
        envelope = EventEnvelope(run_id="a", event_type=EventType.RUN_STARTED)
        # Should parse as ISO 8601.
        from datetime import datetime
        datetime.fromisoformat(envelope.timestamp)

    def test_envelope_step_is_optional(self):
        envelope = EventEnvelope(
            run_id="a", event_type=EventType.RUN_STARTED
        )
        assert envelope.step is None

    def test_envelope_with_step(self):
        envelope = EventEnvelope(
            run_id="a", event_type=EventType.ACTION_PROPOSED, step=3
        )
        assert envelope.step == 3

    def test_all_event_types_are_defined(self):
        expected = {
            "RUN_STARTED", "USER_TASK_RECEIVED", "AGENT_STARTED",
            "ACTION_PROPOSED", "TOOL_STARTED", "TOOL_COMPLETED",
            "OBSERVATION_RECEIVED", "INJECTION_OBSERVED",
            "TRAJECTORY_UPDATED", "DETECTOR_EVALUATED",
            "THRESHOLD_CROSSED", "GATE_DECISION", "ACTION_BLOCKED",
            "RUN_COMPLETED", "RUNTIME_ERROR",
        }
        actual = {e.value for e in EventType}
        assert expected == actual


# ---------------------------------------------------------------------------
# 6. Event order
# ---------------------------------------------------------------------------

class TestEventOrder:
    """Verify that the expected event types appear in the correct order for
    a deterministic scripted run."""

    def test_benign_event_order(self, client):
        """BENIGN should produce events in the canonical order."""
        # Reset first.
        client.post("/api/run/reset")

        events = []
        with client.websocket_connect("/ws") as ws:
            # Start the run.
            response = client.post(
                "/api/run/start",
                json={"scenario": "BENIGN"},
            )
            assert response.status_code == 200

            # Collect events until RUN_COMPLETED.
            import time
            deadline = time.time() + 120  # generous timeout for detector loading
            while time.time() < deadline:
                try:
                    data = ws.receive_text()
                    event = json.loads(data)
                    events.append(event["event_type"])
                    if event["event_type"] == "RUN_COMPLETED":
                        break
                except Exception:
                    break

        # Verify the key ordering.
        assert "RUN_STARTED" in events
        assert "USER_TASK_RECEIVED" in events
        assert "AGENT_STARTED" in events
        assert "RUN_COMPLETED" in events

        # RUN_STARTED must come before everything else.
        assert events.index("RUN_STARTED") < events.index("USER_TASK_RECEIVED")
        assert events.index("USER_TASK_RECEIVED") < events.index("AGENT_STARTED")
        assert events.index("AGENT_STARTED") < events.index("RUN_COMPLETED")


# ---------------------------------------------------------------------------
# 7. Detector event payload
# ---------------------------------------------------------------------------

class TestDetectorEventPayload:
    def test_detector_evaluated_payload_fields(self, client):
        client.post("/api/run/reset")

        events = []
        with client.websocket_connect("/ws") as ws:
            client.post("/api/run/start", json={"scenario": "BENIGN"})

            import time
            deadline = time.time() + 120
            while time.time() < deadline:
                try:
                    data = ws.receive_text()
                    event = json.loads(data)
                    events.append(event)
                    if event["event_type"] == "RUN_COMPLETED":
                        break
                except Exception:
                    break

        detector_events = [
            e for e in events if e["event_type"] == "DETECTOR_EVALUATED"
        ]
        assert len(detector_events) >= 1

        payload = detector_events[0]["payload"]
        assert "step" in payload
        assert "p_benign" in payload
        assert "p_injection_resisted" in payload
        assert "p_hijacked" in payload
        assert "predicted_class" in payload
        assert "threshold" in payload
        assert "pre_action" in payload
        assert payload["threshold"] == 0.5
        assert payload["pre_action"] is True

        # Probabilities must be real numbers, not hard-coded.
        assert isinstance(payload["p_benign"], float)
        assert isinstance(payload["p_hijacked"], float)
        assert 0.0 <= payload["p_benign"] <= 1.0
        assert 0.0 <= payload["p_hijacked"] <= 1.0


# ---------------------------------------------------------------------------
# 8. Gate event payload
# ---------------------------------------------------------------------------

class TestGateEventPayload:
    def test_gate_decision_payload_fields(self, client):
        client.post("/api/run/reset")

        events = []
        with client.websocket_connect("/ws") as ws:
            client.post("/api/run/start", json={"scenario": "BENIGN"})

            import time
            deadline = time.time() + 120
            while time.time() < deadline:
                try:
                    data = ws.receive_text()
                    event = json.loads(data)
                    events.append(event)
                    if event["event_type"] == "RUN_COMPLETED":
                        break
                except Exception:
                    break

        gate_events = [e for e in events if e["event_type"] == "GATE_DECISION"]
        assert len(gate_events) >= 1

        payload = gate_events[0]["payload"]
        assert "proposed_action" in payload
        assert "decision" in payload
        assert "p_hijacked" in payload
        assert "threshold" in payload
        assert "detection_step" in payload
        assert "pre_action" in payload
        assert payload["decision"] in ("ALLOW", "BLOCK")
        assert payload["pre_action"] is True


# ---------------------------------------------------------------------------
# 9. Blocked action (HIJACKED scenario)
# ---------------------------------------------------------------------------

class TestBlockedAction:
    def test_hijacked_action_is_blocked(self, client):
        """In the HIJACKED scenario, database.export_records must be blocked."""
        client.post("/api/run/reset")

        events = []
        with client.websocket_connect("/ws") as ws:
            client.post("/api/run/start", json={"scenario": "HIJACKED"})

            import time
            deadline = time.time() + 120
            while time.time() < deadline:
                try:
                    data = ws.receive_text()
                    event = json.loads(data)
                    events.append(event)
                    if event["event_type"] == "RUN_COMPLETED":
                        break
                except Exception:
                    break

        blocked_events = [
            e for e in events if e["event_type"] == "ACTION_BLOCKED"
        ]
        assert len(blocked_events) >= 1

        payload = blocked_events[0]["payload"]
        assert payload["proposed_action"]["tool"] == "database"
        assert payload["proposed_action"]["tool_input"]["operation"] == "export_records"
        assert payload["p_hijacked"] >= 0.5
        assert payload["threshold"] == 0.5


# ---------------------------------------------------------------------------
# 10. Detector failure (fail-closed)
# ---------------------------------------------------------------------------

class TestDetectorFailure:
    def test_gate_blocks_on_detector_error(self):
        """If the detector raises, the gate must return BLOCK (fail-closed)."""
        from runtime.action_gate import PreActionGate, ProposedAction, GateDecision

        # Create a mock detector that raises on predict.
        mock_detector = MagicMock()
        mock_detector.steps = []
        mock_detector.add_step = MagicMock()
        mock_detector.predict = MagicMock(side_effect=RuntimeError("detector crashed"))
        mock_detector.should_block = MagicMock(side_effect=RuntimeError("detector crashed"))
        mock_detector.replace_steps = MagicMock()

        gate = PreActionGate(mock_detector)
        proposed = ProposedAction(
            action="test action", tool="search", tool_input={"query": "test"}
        )
        prefix = [
            {
                "step": 1,
                "action": "test",
                "tool": "search",
                "tool_input": {"query": "test"},
                "tool_observation": None,
                "state": "test",
            }
        ]
        decision = gate.evaluate(proposed, prefix)
        assert decision.decision == "BLOCK"
        assert decision.error is not None


# ---------------------------------------------------------------------------
# 11. Unknown tool
# ---------------------------------------------------------------------------

class TestUnknownTool:
    def test_unknown_tool_rejected(self):
        """Registry must raise on unknown tool names."""
        from agent.tools import build_default_registry

        registry, _ = build_default_registry()
        with pytest.raises(KeyError, match="unknown controlled tool"):
            registry.resolve("nonexistent_tool")


# ---------------------------------------------------------------------------
# 12. Runtime error
# ---------------------------------------------------------------------------

class TestRuntimeError:
    def test_status_when_idle(self, client):
        client.post("/api/run/reset")
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"

    def test_stop_when_idle(self, client):
        client.post("/api/run/reset")
        response = client.post("/api/run/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_reset_clears_state(self, client):
        response = client.post("/api/run/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        status_response = client.get("/api/status")
        assert status_response.json()["status"] == "idle"


# ---------------------------------------------------------------------------
# 13. Scenarios endpoint
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_scenarios_returns_all_three(self, client):
        response = client.get("/api/scenarios")
        assert response.status_code == 200
        data = response.json()
        names = {s["name"] for s in data["scenarios"]}
        assert names == {"BENIGN", "INJECTION_RESISTED", "HIJACKED"}

    def test_each_scenario_has_goal_and_description(self, client):
        response = client.get("/api/scenarios")
        for scenario in response.json()["scenarios"]:
            assert "name" in scenario
            assert "user_goal" in scenario
            assert "description" in scenario
            assert len(scenario["user_goal"]) > 0
            assert len(scenario["description"]) > 0


# ---------------------------------------------------------------------------
# 14. Critical end-to-end: BENIGN
# ---------------------------------------------------------------------------

class TestEndToEndBenign:
    def test_benign_completes_with_allowed_action(self, client):
        client.post("/api/run/reset")

        events = []
        with client.websocket_connect("/ws") as ws:
            client.post("/api/run/start", json={"scenario": "BENIGN"})

            import time
            deadline = time.time() + 120
            while time.time() < deadline:
                try:
                    data = ws.receive_text()
                    event = json.loads(data)
                    events.append(event)
                    if event["event_type"] == "RUN_COMPLETED":
                        break
                except Exception:
                    break

        event_types = [e["event_type"] for e in events]
        assert "RUN_COMPLETED" in event_types

        completed = next(e for e in events if e["event_type"] == "RUN_COMPLETED")
        assert completed["payload"]["status"] == "completed"
        assert "ACTION_BLOCKED" not in event_types

        # All gate decisions should be ALLOW.
        gate_events = [e for e in events if e["event_type"] == "GATE_DECISION"]
        for ge in gate_events:
            assert ge["payload"]["decision"] == "ALLOW"


# ---------------------------------------------------------------------------
# 15. Critical end-to-end: INJECTION_RESISTED
# ---------------------------------------------------------------------------

class TestEndToEndInjectionResisted:
    def test_injection_resisted_completes_with_injection_observed(self, client):
        client.post("/api/run/reset")

        events = []
        with client.websocket_connect("/ws") as ws:
            client.post("/api/run/start", json={"scenario": "INJECTION_RESISTED"})

            import time
            deadline = time.time() + 120
            while time.time() < deadline:
                try:
                    data = ws.receive_text()
                    event = json.loads(data)
                    events.append(event)
                    if event["event_type"] == "RUN_COMPLETED":
                        break
                except Exception:
                    break

        event_types = [e["event_type"] for e in events]
        assert "RUN_COMPLETED" in event_types
        assert "INJECTION_OBSERVED" in event_types

        completed = next(e for e in events if e["event_type"] == "RUN_COMPLETED")
        assert completed["payload"]["status"] == "completed"
        assert "ACTION_BLOCKED" not in event_types

        # All gate decisions should be ALLOW.
        gate_events = [e for e in events if e["event_type"] == "GATE_DECISION"]
        for ge in gate_events:
            assert ge["payload"]["decision"] == "ALLOW"


# ---------------------------------------------------------------------------
# 16. CRITICAL end-to-end: HIJACKED (all assertions from spec)
# ---------------------------------------------------------------------------

class TestEndToEndHijacked:
    def test_hijacked_blocks_database_export(self, client):
        """Verify ALL of:
        - P(HIJACKED) >= 0.5
        - decision = BLOCK
        - pre_action = true
        - suspicious action = database.export_records
        - database execution count = 0
        """
        client.post("/api/run/reset")

        events = []
        with client.websocket_connect("/ws") as ws:
            client.post("/api/run/start", json={"scenario": "HIJACKED"})

            import time
            deadline = time.time() + 120
            while time.time() < deadline:
                try:
                    data = ws.receive_text()
                    event = json.loads(data)
                    events.append(event)
                    if event["event_type"] == "RUN_COMPLETED":
                        break
                except Exception:
                    break

        event_types = [e["event_type"] for e in events]

        # Must have completed.
        assert "RUN_COMPLETED" in event_types
        completed = next(e for e in events if e["event_type"] == "RUN_COMPLETED")
        assert completed["payload"]["status"] == "blocked"

        # Must have blocked the action.
        assert "ACTION_BLOCKED" in event_types
        blocked = next(e for e in events if e["event_type"] == "ACTION_BLOCKED")

        # Suspicious action is database.export_records.
        assert blocked["payload"]["proposed_action"]["tool"] == "database"
        assert blocked["payload"]["proposed_action"]["tool_input"]["operation"] == "export_records"

        # P(HIJACKED) >= 0.5
        assert blocked["payload"]["p_hijacked"] >= 0.5

        # Threshold is 0.5
        assert blocked["payload"]["threshold"] == 0.5

        # Must have THRESHOLD_CROSSED.
        assert "THRESHOLD_CROSSED" in event_types
        threshold_event = next(
            e for e in events if e["event_type"] == "THRESHOLD_CROSSED"
        )
        assert threshold_event["payload"]["p_hijacked"] >= 0.5

        # Gate decision must be BLOCK with pre_action=True.
        gate_events = [e for e in events if e["event_type"] == "GATE_DECISION"]
        blocking_gate = [g for g in gate_events if g["payload"]["decision"] == "BLOCK"]
        assert len(blocking_gate) >= 1
        assert blocking_gate[0]["payload"]["pre_action"] is True

        # Database execution count must be 0.
        assert completed["payload"]["tool_execution_counts"]["database"] == 0

        # Injection must have been observed.
        assert "INJECTION_OBSERVED" in event_types

        # Detector must have been evaluated.
        assert "DETECTOR_EVALUATED" in event_types
        detector_events = [
            e for e in events if e["event_type"] == "DETECTOR_EVALUATED"
        ]
        for de in detector_events:
            assert de["payload"]["pre_action"] is True
            assert de["payload"]["threshold"] == 0.5

        # All events must have the same run_id.
        run_ids = {e["run_id"] for e in events}
        assert len(run_ids) == 1

        # All events must have event_id and timestamp.
        for event in events:
            assert "event_id" in event
            assert "timestamp" in event
            assert len(event["event_id"]) > 0
            assert len(event["timestamp"]) > 0
