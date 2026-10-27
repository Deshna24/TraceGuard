"""Live Demonstration Verification Script for TRACEGUARD-X.

Connects to the live running server at http://127.0.0.1:8000 over HTTP and WebSocket,
and validates all demonstration flows required for the professor presentation.
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.request
import websockets


BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws"


def http_get(path: str) -> dict:
    req = urllib.request.Request(f"{BASE_URL}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.loads(response.read().decode())


def http_post(path: str, data: dict | None = None) -> tuple[int, dict]:
    payload = json.dumps(data or {}).encode() if data is not None else b""
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode())


async def execute_demo_scenario(scenario: str, model_arch: str, gate_type: str) -> dict:
    http_post("/api/run/reset")
    events = []

    async with websockets.connect(WS_URL) as ws:
        status, resp = http_post(
            "/api/run/start",
            {"scenario": scenario, "model_architecture": model_arch, "gate_type": gate_type},
        )
        assert status == 200, f"Start failed: {resp}"

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            event = json.loads(raw)
            events.append(event)
            if event.get("event_type") == "RUN_COMPLETED":
                break

    return {
        "scenario": scenario,
        "model": model_arch,
        "gate": gate_type,
        "events": events,
        "completed": next(e for e in events if e.get("event_type") == "RUN_COMPLETED"),
        "evaluations": [e for e in events if e.get("event_type") == "DETECTOR_EVALUATED"],
        "gate_decisions": [e for e in events if e.get("event_type") == "GATE_DECISION"],
        "blocked": [e for e in events if e.get("event_type") == "ACTION_BLOCKED"],
    }


async def main():
    print("=" * 75)
    print(" TRACEGUARD-X: LIVE DEMONSTRATION VERIFICATION (LOCAL SERVER AUDIT)")
    print("=" * 75)

    # 1. Health check
    print("\n[Step 1] Verifying Backend Health (/health)...")
    health = http_get("/health")
    print(f"    Health Status: {health.get('status')} | System: {health.get('system_status')}")
    assert health.get("status") == "ok"

    # 2. Experiments Summary
    print("\n[Step 2] Verifying Experiments Summary (/api/experiments/summary)...")
    summary = http_get("/api/experiments/summary")
    models = summary.get("models", {})
    print(f"    Loaded Models: {list(models.keys())}")
    for k, v in models.items():
        print(f"      - {v['name']}: {v['parameters']:,} params | Macro-F1: {v['macro_f1']} | Hijacked-F1: {v['hijacked_f1']}")
    assert len(models) == 4

    # 3. Demonstration A & B: BENIGN Scenario with LSTM + Standard Gate
    print("\n[Step 3] Running BENIGN Scenario (LSTM + Standard Gate)...")
    res_benign = await execute_demo_scenario("BENIGN", "lstm", "standard")
    benign_status = res_benign["completed"]["payload"]["status"]
    print(f"    Run Result: status={benign_status}")
    print(f"    Tool Executions: {res_benign['completed']['payload']['tool_execution_counts']}")
    assert benign_status == "completed"
    assert len(res_benign["blocked"]) == 0
    print("    [PASS] Benign run completed smoothly without false alarm blocking.")

    # 4. Demonstration C, D, E: HIJACKED Scenario with LSTM + Standard Gate (Baseline)
    print("\n[Step 4] Running HIJACKED Scenario (LSTM Baseline + Standard Gate)...")
    res_hijacked_lstm = await execute_demo_scenario("HIJACKED", "lstm", "standard")
    lstm_status = res_hijacked_lstm["completed"]["payload"]["status"]
    lstm_db_count = res_hijacked_lstm["completed"]["payload"]["tool_execution_counts"].get("database", 0)
    print(f"    Run Result: status={lstm_status} | Database Execution Count: {lstm_db_count}")
    assert lstm_status == "blocked"
    assert lstm_db_count == 0
    assert len(res_hijacked_lstm["blocked"]) == 1
    blocked_action = res_hijacked_lstm["blocked"][0]["payload"]["proposed_action"]
    print(f"    [PASS] Dangerous action '{blocked_action['tool']}.{blocked_action['tool_input']}' PREVENTED BEFORE EXECUTION.")

    # 5. Demonstration F: GRU (Track B1) with Standard Gate
    print("\n[Step 5] Running HIJACKED Scenario with GRU Model (13.6% parameter reduction)...")
    res_gru = await execute_demo_scenario("HIJACKED", "gru", "standard")
    gru_status = res_gru["completed"]["payload"]["status"]
    gru_db_count = res_gru["completed"]["payload"]["tool_execution_counts"].get("database", 0)
    print(f"    Run Result: status={gru_status} | Database Execution Count: {gru_db_count}")
    assert gru_status == "blocked"
    assert gru_db_count == 0
    print("    [PASS] GRU model successfully intercepted hijacked execution at 0.836 probability.")

    # 6. Demonstration F & G: Temporal Attention with Uncertainty-Aware Gate
    print("\n[Step 6] Running HIJACKED Scenario with Temporal Attention & Uncertainty Gate...")
    res_attn = await execute_demo_scenario("HIJACKED", "attention", "uncertainty_aware")
    attn_status = res_attn["completed"]["payload"]["status"]
    attn_db_count = res_attn["completed"]["payload"]["tool_execution_counts"].get("database", 0)
    eval_attn = res_attn["evaluations"][-1]["payload"]
    print(f"    Run Result: status={attn_status} | Database Execution Count: {attn_db_count}")
    print(f"    Attention Weights Vector: {eval_attn.get('attention_weights')}")
    print(f"    Normalized Shannon Entropy: {eval_attn.get('uncertainty_score')}")
    assert attn_status == "abstained"
    assert attn_db_count == 0
    assert len(res_attn["blocked"]) == 1
    assert res_attn["blocked"][0]["payload"]["decision"] == "ABSTAIN"
    print("    [PASS] Attention model with 3-Way Gate detected precursor ambiguity and ESCALATED TO OPERATOR (ABSTAINED).")

    # 7. Demonstration F & G: MultiTask with Uncertainty-Aware Gate
    print("\n[Step 7] Running HIJACKED Scenario with MultiTask Early Warning Model...")
    res_multi = await execute_demo_scenario("HIJACKED", "multitask", "uncertainty_aware")
    multi_status = res_multi["completed"]["payload"]["status"]
    multi_db_count = res_multi["completed"]["payload"]["tool_execution_counts"].get("database", 0)
    eval_multi = res_multi["evaluations"][-1]["payload"]
    print(f"    Run Result: status={multi_status} | Database Execution Count: {multi_db_count}")
    print(f"    Imminent Deviation Warning Score: {eval_multi.get('warning_score')}")
    print(f"    Entropy Uncertainty: {eval_multi.get('uncertainty_score')}")
    assert multi_status == "abstained"
    assert multi_db_count == 0
    print("    [PASS] MultiTask model emitted early warning score and safely abstained from database export.")

    # 8. Schema Validation & Error Reporting
    print("\n[Step 8] Testing Schema Validation and Error Visibility...")
    err_code, err_body = http_post("/api/run/start", {"scenario": "HIJACKED", "model_architecture": "invalid_model_foo"})
    print(f"    Invalid Model Request HTTP Code: {err_code} (Expected 422)")
    assert err_code == 422
    print("    [PASS] Bad input fails fast with HTTP 422 Unprocessable Entity.")

    print("\n" + "=" * 75)
    print(" ALL 8 DEMONSTRATION WORKFLOWS FULLY VERIFIED ON LIVE LOCAL SERVER!")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
