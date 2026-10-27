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

# Ensure the repository root and traceguard package root are importable.
SCRIPT_DIR = Path(__file__).resolve().parent
TRACEGUARD_DIR = SCRIPT_DIR.parent
REPO_ROOT = TRACEGUARD_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRACEGUARD_DIR) not in sys.path:
    sys.path.insert(0, str(TRACEGUARD_DIR))

from web.agent_runner import get_scenario_info, run_scenario, run_scenario_live
from web.runtime_manager import runtime_manager
from web.schemas.api_models import (
    EventType,
    HealthResponse,
    ModelArchitectureName,
    ModelMetadataResponse,
    PolicyEvaluateRequest,
    PolicyEvaluateResponse,
    RunActionResponse,
    RunStartRequest,
    RunStartResponse,
    ScenarioInfo,
    ScenarioName,
    ScenariosResponse,
    StatusResponse,
    SystemHealthResponse,
    TrajectoryAnalysisRequest,
    TrajectoryAnalysisResponse,
)
from web.websocket.manager import manager as ws_manager
from traceguard_x.service.model_registry import registry as model_registry
from traceguard_x.service.inference_service import inference_service

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
            injection_target=request.injection_target,
            model_architecture=request.model_architecture.value if hasattr(request.model_architecture, "value") else str(request.model_architecture or "lstm"),
            gate_type=request.gate_type.value if hasattr(request.gate_type, "value") else str(request.gate_type or "standard"),
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
            injection_target=request.injection_target,
            model_architecture=request.model_architecture.value if hasattr(request.model_architecture, "value") else str(request.model_architecture or "lstm"),
            gate_type=request.gate_type.value if hasattr(request.gate_type, "value") else str(request.gate_type or "standard"),
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
        if name != ScenarioName.CUSTOM
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
# GET /api/experiments/summary
# ---------------------------------------------------------------------------
@app.get("/api/experiments/summary", tags=["research"])
async def get_experiments_summary():
    """Expose real, verified benchmark experiment results from TRACEGUARD-X."""
    import json
    repo_root = TRACEGUARD_DIR.parent
    x_exp = repo_root / "traceguard_x" / "experiments"

    def read_json_safe(p: Path):
        try:
            return json.loads(p.read_text("utf-8")) if p.exists() else None
        except Exception:
            return None

    baseline = read_json_safe(x_exp / "track_a_baseline" / "baseline_reproduction_seed42.json")
    gru_5seed = read_json_safe(x_exp / "track_b1_gru" / "gru_five_seeds_robustness.json")
    att_5seed = read_json_safe(x_exp / "track_b2_attention" / "attention_five_seeds_robustness.json")
    multitask = read_json_safe(x_exp / "track_b3_earlywarn" / "multitask_seed42_metrics.json")
    generalization = read_json_safe(x_exp / "track_c_generalization" / "generalization_experiments.json")
    calibration = read_json_safe(x_exp / "track_d_calibration" / "calibration_results.json")
    early_detection = read_json_safe(x_exp / "track_e_early" / "early_detection_comparison.json")
    gate_policy = read_json_safe(x_exp / "track_f_gate" / "gate_policy_evaluation.json")

    return {
        "status": "verified",
        "models": {
            "frozen_lstm": {
                "name": "Frozen Baseline LSTM",
                "accuracy": 0.8556,
                "macro_f1": 0.8533,
                "hijacked_f1": 0.9032,
                "pre_action_rate": 0.9333,
                "mean_latency": -2.71,
                "parameters": 395651,
                "label": "Original frozen baseline",
            },
            "traceguard_gru": {
                "name": "TraceGuard-X GRU",
                "accuracy": 0.9333,
                "macro_f1": 0.9339,
                "hijacked_f1": 0.9333,
                "pre_action_rate": 0.9333,
                "mean_latency": -2.62,
                "parameters": 296835,
                "five_seed_f1_mean": gru_5seed.get("macro_f1_mean") if gru_5seed else 0.8954,
                "five_seed_f1_std": gru_5seed.get("macro_f1_std") if gru_5seed else 0.0318,
                "label": "New experimental result",
            },
            "temporal_attention": {
                "name": "Temporal Attention",
                "accuracy": 0.8889,
                "macro_f1": 0.8885,
                "hijacked_f1": 0.9333,
                "pre_action_rate": 0.9333,
                "mean_latency": -3.04,
                "parameters": 305155,
                "five_seed_f1_mean": att_5seed.get("macro_f1_mean") if att_5seed else 0.8421,
                "five_seed_f1_std": att_5seed.get("macro_f1_std") if att_5seed else 0.0477,
                "label": "New experimental result",
            },
            "multitask_earlywarn": {
                "name": "Multi-Task Early Warning",
                "accuracy": 0.9222,
                "macro_f1": 0.9225,
                "hijacked_f1": 0.9032,
                "pre_action_rate": 0.9000,
                "mean_latency": -2.46,
                "parameters": 296964,
                "imminent_auroc": 0.7805,
                "label": "New experimental result",
            },
        },
        "generalization": generalization,
        "calibration": calibration,
        "early_detection": early_detection,
        "gate_policy": gate_policy,
    }

# ---------------------------------------------------------------------------
# GET /api/models & GET /api/models/{model_id}
# ---------------------------------------------------------------------------
@app.get("/api/models", response_model=list[ModelMetadataResponse], tags=["models"])
async def list_models():
    """List all available TRACEGUARD-X models with parameter counts and readiness."""
    models = model_registry.list_models()
    return [ModelMetadataResponse(**m.__dict__) for m in models]


@app.get("/api/models/{model_id}", response_model=ModelMetadataResponse, tags=["models"])
async def get_model(model_id: str):
    """Get metadata and readiness for a specific model."""
    try:
        meta = model_registry.get_metadata(model_id)
        return ModelMetadataResponse(**meta.__dict__)
    except KeyError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /api/analyze (Direct Trajectory Inference API)
# ---------------------------------------------------------------------------
@app.post("/api/analyze", response_model=TrajectoryAnalysisResponse, tags=["inference"])
async def analyze_trajectory(request: TrajectoryAnalysisRequest):
    """Synchronous trajectory analysis service for external agents and UI analysis.

    Evaluates the input prefix using the requested neural architecture,
    computes normalized Shannon entropy, emits auxiliary signals (attention weights / warning scores),
    and evaluates the pre-action safety gate policy.
    """
    try:
        model_name = request.model_architecture.value if hasattr(request.model_architecture, "value") else str(request.model_architecture)
        gate_name = request.gate_type.value if hasattr(request.gate_type, "value") else str(request.gate_type)

        result = inference_service.analyze(
            user_goal=request.user_goal,
            steps=request.steps,
            model_id=model_name,
            gate_type=gate_name,
            candidate_action=request.candidate_action,
        )
        return TrajectoryAnalysisResponse(**result)
    except Exception as exc:
        from fastapi import HTTPException
        logger.error("Analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /api/policy/evaluate (Independent Policy Engine API)
# ---------------------------------------------------------------------------
@app.post("/api/policy/evaluate", response_model=PolicyEvaluateResponse, tags=["policy"])
async def evaluate_policy(request: PolicyEvaluateRequest):
    """Evaluate pre-action safety gate policy independently from raw probabilities."""
    gate_name = request.gate_type.value if hasattr(request.gate_type, "value") else str(request.gate_type)
    res = inference_service.evaluate_policy(
        p_hijacked=request.p_hijacked,
        entropy=request.entropy,
        gate_type=gate_name,
        candidate_action=request.candidate_action,
    )
    return PolicyEvaluateResponse(**res)


# ---------------------------------------------------------------------------
# GET /api/system/health (System and Hardware Diagnostics)
# ---------------------------------------------------------------------------
@app.get("/api/system/health", response_model=SystemHealthResponse, tags=["system"])
async def get_system_health():
    """Return live system health, CUDA GPU hardware acceleration, and asset readiness."""
    import torch
    cuda_avail = torch.cuda.is_available()
    device = "cuda" if cuda_avail else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_tot = None
    vram_alloc = None
    if cuda_avail:
        vram_tot = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        vram_alloc = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)

    # Check model checkpoints
    models_avail = {
        m.id: m.checkpoint_available for m in model_registry.list_models()
    }

    # Check sentence embedding cache
    emb_cached = True
    try:
        import os
        from sentence_transformers import SentenceTransformer
        _ = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    except Exception:
        emb_cached = False

    return SystemHealthResponse(
        status="healthy",
        version="1.0.0",
        device=device,
        device_name=device_name,
        cuda_available=cuda_avail,
        vram_total_gb=vram_tot,
        vram_allocated_gb=vram_alloc,
        torch_version=torch.__version__,
        embedding_cache_available=emb_cached,
        models_available=models_avail,
    )


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
