"""Integration tests for TRACEGUARD-X runtime detector and web API extensions."""

import pytest
from fastapi.testclient import TestClient

from traceguard.web.server import app
from traceguard_x.runtime.detector import TraceGuardXRuntimeDetector


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_api_experiments_summary_endpoint(client):
    response = client.get("/api/experiments/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "verified"
    assert "models" in data
    assert "frozen_lstm" in data["models"]
    assert "traceguard_gru" in data["models"]
    assert "temporal_attention" in data["models"]
    assert "multitask_earlywarn" in data["models"]


def test_runtime_detector_loading():
    for arch in ("lstm", "gru", "attention", "multitask"):
        det = TraceGuardXRuntimeDetector(user_goal="Test flight search", model_name=arch)
        assert det.model_name == arch
        assert det.model is not None


def test_runtime_detector_prefix_inference():
    det = TraceGuardXRuntimeDetector(user_goal="Find flight", model_name="gru")
    det.add_step({
        "step": 1,
        "action": "Search flight",
        "tool": "search",
        "tool_input": {"query": "flight"},
        "tool_observation": "Found 3 flights",
        "state": "Gathering info",
    })
    pred = det.predict()
    assert set(pred.probabilities.keys()) == {"BENIGN", "INJECTION_RESISTED", "HIJACKED"}
    assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert pred.prefix_length == 1
    assert pred.predicted_class in ("BENIGN", "INJECTION_RESISTED", "HIJACKED")


def test_runtime_detector_attention_weights():
    det = TraceGuardXRuntimeDetector(user_goal="Find flight", model_name="attention")
    det.add_step({
        "step": 1,
        "action": "Search flight",
        "tool": "search",
        "tool_input": {"query": "flight"},
        "tool_observation": "Found 3 flights",
        "state": "Gathering info",
    })
    det.predict()
    assert det.last_prediction is not None
    assert det.last_prediction.attention_weights is not None
    assert len(det.last_prediction.attention_weights) == 1
    assert det.last_prediction.attention_weights[0] == pytest.approx(1.0, abs=1e-3)


def test_runtime_detector_multitask_warning():
    det = TraceGuardXRuntimeDetector(user_goal="Find flight", model_name="multitask")
    det.add_step({
        "step": 1,
        "action": "Search flight",
        "tool": "search",
        "tool_input": {"query": "flight"},
        "tool_observation": "Found 3 flights",
        "state": "Gathering info",
    })
    det.predict()
    assert det.last_prediction is not None
    assert det.last_prediction.warning_score is not None
    assert 0.0 <= det.last_prediction.warning_score <= 1.0


def test_api_run_start_with_model_selection(client):
    response = client.post("/api/run/start", json={
        "scenario": "BENIGN",
        "model_architecture": "gru",
        "gate_type": "uncertainty_aware",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "run_id" in data

    # Clean up by resetting runtime
    client.post("/api/run/reset")
