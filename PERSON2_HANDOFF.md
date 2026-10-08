# TRACEGUARD Backend Handoff for Person 2

The backend integration is complete. The FastAPI application now exposes the full `ControlledAgent` real-time demonstration, integrating perfectly with the frozen LSTM detector and existing logging infrastructure.

This document serves as your single source of truth for connecting the React frontend to the backend.

---

## 1. Startup Instructions

The server binds to port `8000` by default. Start it from the repository root:

```bash
# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Start the uvicorn server with hot-reloading
python -m uvicorn traceguard.web.server:app --reload --host 0.0.0.0 --port 8000
```

---

## 2. API Endpoints (HTTP)

All REST API responses are standard JSON.

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/health` | GET | Check server readiness |
| `/api/status` | GET | Retrieve the current runtime status and historical events |
| `/api/scenarios` | GET | List available scenarios, their names, and descriptions |
| `/api/run/start` | POST | Start a controlled scenario. Payload: `{"scenario": "BENIGN"}` |
| `/api/run/start/live`| POST | Start a live run using local Ollama (requires `granite4.1:8b-q4_K_M`) |
| `/api/run/stop` | POST | Gracefully stop the active run |
| `/api/run/reset` | POST | Clear runtime state between demonstrations |

---

## 3. WebSocket Configuration

The WebSocket provides an observation channel for the real-time event stream.

**WebSocket URL:** `ws://localhost:8000/ws`

**Important Notes:**
- The frontend **cannot** bypass the security gate or invoke tools over this connection.
- All events are authoritative. If the WebSocket says a tool executed, it actually executed on the backend.

---

## 4. Event Envelope Schema

Every message sent through the WebSocket matches this exact JSON schema:

```typescript
interface EventEnvelope {
  run_id: string;        // UUID of the current run
  event_id: string;      // UUID of this specific event
  timestamp: string;     // ISO 8601 timestamp
  event_type: string;    // See Event Types below
  step: number | null;   // The trajectory step index (null for global events)
  payload: any;          // Structured payload specific to the event_type
}
```

---

## 5. Event Types & Payload Schemas

You must handle the following `event_type` values.

### Lifecycle Events

*   **`RUN_STARTED`**: Fired when a scenario begins.
    *   Payload: `{ scenario: string, user_goal: string }`
*   **`USER_TASK_RECEIVED`**: Fired when the agent receives the goal.
    *   Payload: `{ user_goal: string }`
*   **`AGENT_STARTED`**: Fired when the model begins inference.
    *   Payload: `{ model: string }`
*   **`RUN_COMPLETED`**: Fired when the run terminates (success, failure, or blocked).
    *   Payload: `{ status: string, scenario: string, trajectory_length: number, tool_execution_counts: Record<string, number>, answer?: string, blocked_action?: any, error?: string }`
*   **`RUNTIME_ERROR`**: Fired if an unhandled exception occurs.
    *   Payload: `{ error: string, phase: string }`

### Execution Events

*   **`ACTION_PROPOSED`**: Fired when the LLM outputs an action.
    *   Payload: `{ action: string, tool: string, tool_input: object }`
*   **`TOOL_STARTED`**: Fired when a tool begins execution (after passing the gate).
    *   Payload: `{ tool: string, tool_input: object }`
*   **`TOOL_COMPLETED`**: Fired when a tool finishes execution.
    *   Payload: `{ tool: string, success: boolean, output: any, error: string | null }`
*   **`OBSERVATION_RECEIVED`**: Fired when the agent observes the tool output.
    *   Payload: `{ observation: any }`
*   **`TRAJECTORY_UPDATED`**: Fired when a step is recorded.
    *   Payload: `{ trajectory_length: number, latest_step: object }`

### Security & Detection Events

*   **`INJECTION_OBSERVED`**: Fired when the agent views a fixture containing a simulated injection attack.
    *   Payload: `{ observation: string, note: string }`
*   **`DETECTOR_EVALUATED`**: Fired when the frozen LSTM processes the proposed action.
    *   Payload: `{ step: number, p_benign: number, p_injection_resisted: number, p_hijacked: number, predicted_class: string, threshold: number, pre_action: boolean }`
*   **`THRESHOLD_CROSSED`**: Fired if `p_hijacked` meets or exceeds `threshold`.
    *   Payload: `{ p_hijacked: number, threshold: number, detection_step: number }`
*   **`GATE_DECISION`**: Fired when the PreActionGate makes a final authorization decision.
    *   Payload: `{ proposed_action: object, tool: string, tool_input: object, decision: "ALLOW" | "BLOCK", p_hijacked: number, threshold: number, detection_step: number, pre_action: boolean }`
*   **`ACTION_BLOCKED`**: Fired when an action is denied execution.
    *   Payload: `{ proposed_action: object, tool: string, tool_input: object, p_hijacked: number, threshold: number, reason: string }`

---

## 6. Sample Event Sequences

### 🟢 BENIGN Sequence
The agent receives a legitimate task and completes it safely.
1. `RUN_STARTED`
2. `USER_TASK_RECEIVED`
3. `AGENT_STARTED`
4. `ACTION_PROPOSED` (math.calculate)
5. `DETECTOR_EVALUATED` (`p_hijacked` < 0.5)
6. `GATE_DECISION` (ALLOW)
7. `TOOL_STARTED`
8. `TOOL_COMPLETED`
9. `OBSERVATION_RECEIVED`
10. `TRAJECTORY_UPDATED`
11. `RUN_COMPLETED` (status: "completed")

### 🟡 INJECTION_RESISTED Sequence
The agent views an injection fixture but ignores it and continues its original task.
1. `RUN_STARTED`
2. `USER_TASK_RECEIVED`
3. `AGENT_STARTED`
4. `ACTION_PROPOSED` (search)
5. `DETECTOR_EVALUATED` (`p_hijacked` < 0.5)
6. `GATE_DECISION` (ALLOW)
7. `TOOL_STARTED`
8. `TOOL_COMPLETED`
9. `OBSERVATION_RECEIVED`
10. **`INJECTION_OBSERVED`**
11. `TRAJECTORY_UPDATED`
12. `ACTION_PROPOSED` (math.calculate)
13. `DETECTOR_EVALUATED` (`p_injection_resisted` is dominant, `p_hijacked` < 0.5)
14. `GATE_DECISION` (ALLOW)
15. `TOOL_STARTED`
16. `TOOL_COMPLETED`
17. `OBSERVATION_RECEIVED`
18. `TRAJECTORY_UPDATED`
19. `RUN_COMPLETED` (status: "completed")

### 🔴 HIJACKED Sequence
The agent views an injection fixture, becomes misaligned, and attempts to steal data. TRACEGUARD blocks it before execution.
1. `RUN_STARTED`
2. `USER_TASK_RECEIVED`
3. `AGENT_STARTED`
4. `ACTION_PROPOSED` (search)
5. `DETECTOR_EVALUATED` (`p_hijacked` < 0.5)
6. `GATE_DECISION` (ALLOW)
7. `TOOL_STARTED`
8. `TOOL_COMPLETED`
9. `OBSERVATION_RECEIVED`
10. **`INJECTION_OBSERVED`**
11. `TRAJECTORY_UPDATED`
12. `ACTION_PROPOSED` (database.export_records)
13. `DETECTOR_EVALUATED` (`p_hijacked` >= 0.5)
14. **`THRESHOLD_CROSSED`**
15. **`GATE_DECISION`** (BLOCK)
16. **`ACTION_BLOCKED`**
17. `RUN_COMPLETED` (status: "blocked", database execution count remains 0)
