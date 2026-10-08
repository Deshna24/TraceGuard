"""FastAPI server for the TRACEGUARD web application.

Routes:
  GET  /health         – health check
  GET  /api/status     – current run status
  POST /api/run/start  – start a scenario run
  POST /api/run/stop   – stop the current run
  POST /api/run/reset  – reset runtime state
  GET  /api/scenarios  – list available scenarios
  WS   /ws             – real-time event stream

The server does NOT expose:
  - arbitrary tool/shell/Python execution endpoints
  - arbitrary filesystem endpoints
  - browser-to-Ollama direct connections

FastAPI owns the local runtime connection.
The frontend cannot bypass the gate or directly call tools.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Ensure the traceguard package root is importable.
SCRIPT_DIR = Path(__file__).resolve().parent
TRACEGUARD_DIR = SCRIPT_DIR.parent
if str(TRACEGUARD_DIR) not in sys.path:
    sys.path.insert(0, str(TRACEGUARD_DIR))

from web.agent_runner import get_scenario_info, run_scenario, run_scenario_live
from web.runtime_manager import runtime_manager
from web.schemas.api_models import (
    EventType,
    HealthResponse,
    RunActionResponse,
    RunStartRequest,
    RunStartResponse,
    ScenarioInfo,
    ScenarioName,
    ScenariosResponse,
    StatusResponse,
)
from web.websocket.manager import manager as ws_manager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("traceguard.web.server")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TRACEGUARD Web API",
    description="Backend for the TRACEGUARD real-time agent safety demonstration.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Returns OK when the server is reachable."""
    return HealthResponse(status="ok", version="1.0")


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------
@app.get("/api/status", response_model=StatusResponse, tags=["run"])
async def get_status():
    """Return the current run status snapshot."""
    info = runtime_manager.get_status()
    return StatusResponse(**info)


# ---------------------------------------------------------------------------
# POST /api/run/start
# ---------------------------------------------------------------------------
@app.post("/api/run/start", response_model=RunStartResponse, tags=["run"])
async def start_run(request: RunStartRequest):
    """Start a new scenario run.

    Only one run can be active at a time. Uses the deterministic
    ScriptedActionModel for reproducible scenario execution.
    """
    try:
        run = await runtime_manager.start_run(
            scenario=request.scenario,
            user_goal=get_scenario_info(request.scenario, request.custom_goal)["user_goal"],
            custom_injection=request.custom_injection,
        )
    except RuntimeError as exc:
        return RunStartResponse(
            run_id="", scenario=request.scenario.value, status=f"error: {exc}"
        )

    # Launch the scenario in a background task.
    if request.scenario == ScenarioName.CUSTOM:
        task = asyncio.create_task(run_scenario_live(run))
    else:
        task = asyncio.create_task(run_scenario(run))
    runtime_manager.set_task(task)

    return RunStartResponse(
        run_id=run.run_id,
        scenario=request.scenario.value,
        status="running",
    )


# ---------------------------------------------------------------------------
# POST /api/run/start/live  (optional Ollama live mode)
# ---------------------------------------------------------------------------
@app.post("/api/run/start/live", response_model=RunStartResponse, tags=["run"])
async def start_run_live(request: RunStartRequest):
    """Start a live scenario run using the local Ollama model.

    Requires Ollama to be running with granite4.1:8b-q4_K_M loaded.
    """
    try:
        run = await runtime_manager.start_run(
            scenario=request.scenario,
            user_goal=get_scenario_info(request.scenario, request.custom_goal)["user_goal"],
            custom_injection=request.custom_injection,
        )
    except RuntimeError as exc:
        return RunStartResponse(
            run_id="", scenario=request.scenario.value, status=f"error: {exc}"
        )

    task = asyncio.create_task(run_scenario_live(run))
    runtime_manager.set_task(task)

    return RunStartResponse(
        run_id=run.run_id,
        scenario=request.scenario.value,
        status="running",
    )


# ---------------------------------------------------------------------------
# POST /api/run/stop
# ---------------------------------------------------------------------------
@app.post("/api/run/stop", response_model=RunActionResponse, tags=["run"])
async def stop_run():
    """Stop the currently running scenario."""
    stopped = await runtime_manager.stop_run()
    if stopped:
        run = runtime_manager.current_run
        return RunActionResponse(
            success=True,
            message="Run stopped.",
            run_id=run.run_id if run else None,
        )
    return RunActionResponse(success=False, message="No active run to stop.")


# ---------------------------------------------------------------------------
# POST /api/run/reset
# ---------------------------------------------------------------------------
@app.post("/api/run/reset", response_model=RunActionResponse, tags=["run"])
async def reset_run():
    """Clear runtime state without modifying frozen artifacts."""
    await runtime_manager.reset()
    return RunActionResponse(success=True, message="Runtime state reset.")


# ---------------------------------------------------------------------------
# GET /api/scenarios
# ---------------------------------------------------------------------------
@app.get("/api/scenarios", response_model=ScenariosResponse, tags=["scenarios"])
async def list_scenarios():
    """Return metadata for all available scenarios."""
    scenarios = [
        ScenarioInfo(**get_scenario_info(name))
        for name in ScenarioName
    ]
    return ScenariosResponse(scenarios=scenarios)


# ---------------------------------------------------------------------------
# GET /api/history
# ---------------------------------------------------------------------------
@app.get("/api/history", tags=["history"])
async def get_history():
    """Return all recorded runtime evidence from the logs directory."""
    import json
    from runtime.logger import DEFAULT_LOG_DIRECTORY
    history = []
    if DEFAULT_LOG_DIRECTORY.exists():
        for log_file in DEFAULT_LOG_DIRECTORY.glob("*.json"):
            try:
                content = json.loads(log_file.read_text("utf-8"))
                history.append(content)
            except Exception:
                pass
    # Sort by timestamp descending if available, else by run_id
    history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return {"history": history}

# ---------------------------------------------------------------------------
# GET /api/database
# ---------------------------------------------------------------------------
@app.get("/api/database", tags=["database"])
async def get_database():
    """Expose the mock database records for UI presentation."""
    from agent.tools import DatabaseTool
    tool = DatabaseTool()
    # tool._records is what we want to expose
    return {"records": list(tool._records.values())}

# ---------------------------------------------------------------------------
# WebSocket /ws
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time event stream.

    This is an observation/control channel, NOT a security boundary replacement.
    The frontend cannot bypass the gate or directly call tools through this
    connection.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Accept and acknowledge client messages, but do not allow them
            # to trigger tool execution or bypass the gate.
            data = await websocket.receive_text()
            logger.debug("WebSocket received: %s", data[:200])
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
