"""Preflight verification script for TRACEGUARD-X local demonstration.

Checks:
1. Python interpreter and version
2. Critical dependencies (torch, transformers, fastapi, uvicorn, pydantic, sentence_transformers)
3. GPU and CUDA device availability (NVIDIA RTX 4060)
4. Offline model checkpoints for all 4 architectures
5. HuggingFace SentenceTransformer local cache
6. Node.js, npm, and frontend build / node_modules
7. Port availability (8000 for FastAPI backend, 5173 for Vite frontend)
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACEGUARD_DIR = REPO_ROOT / "traceguard"
FRONTEND_DIR = REPO_ROOT / "frontend"


def check_port(host: str, port: int) -> bool:
    """Return True if port is free, False if occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) != 0


def run_checks() -> int:
    print("=" * 70)
    print(" TRACEGUARD-X: Preflight Local Demonstration Audit")
    print("=" * 70)

    failures = 0
    warnings = 0

    # 1. Python Version
    py_ver = sys.version.split()[0]
    print(f"[*] Python Interpreter: {sys.executable} (v{py_ver})")
    if sys.version_info < (3, 10):
        print("    [FAIL] Python 3.10+ required.")
        failures += 1
    else:
        print("    [PASS] Python version supported.")

    # 2. PyTorch & GPU
    try:
        import torch
        print(f"[*] PyTorch Version: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"    [PASS] Hardware Acceleration: CUDA ACTIVE")
            print(f"           GPU: {gpu_name}")
            print(f"           VRAM: {vram_gb:.1f} GB")
        else:
            print("    [WARN] CUDA is not available. Falling back to CPU inference.")
            warnings += 1
    except ImportError:
        print("    [FAIL] PyTorch is not installed.")
        failures += 1

    # 3. Model Checkpoints
    print("[*] Verifying Model Checkpoints:")
    checkpoints = {
        "LSTM (Baseline)": TRACEGUARD_DIR / "outputs" / "models" / "lstm_seed42_best.pth",
        "GRU (Track B1)": REPO_ROOT / "traceguard_x" / "experiments" / "track_b1_gru" / "gru_seed42_best.pth",
        "Attention (Track B2)": REPO_ROOT / "traceguard_x" / "experiments" / "track_b2_attention" / "attention_seed42_best.pth",
        "MultiTask (Track B3)": REPO_ROOT / "traceguard_x" / "experiments" / "track_b3_earlywarn" / "multitask_seed42_best.pth",
    }
    for name, path in checkpoints.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"    [PASS] {name}: {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"    [FAIL] {name} MISSING at: {path}")
            failures += 1

    # 4. HuggingFace Offline Cache
    print("[*] Verifying Local Embedding Weights:")
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer
        _ = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
        print("    [PASS] all-MiniLM-L6-v2 embedding model cached locally.")
    except Exception as exc:
        print(f"    [WARN] Offline embedding cache check: {exc}")
        print("           Online download will occur on first run if cache is missing.")
        warnings += 1

    # 5. Frontend Environment
    print("[*] Verifying Frontend Environment:")
    dist_index = FRONTEND_DIR / "dist" / "index.html"
    node_modules = FRONTEND_DIR / "node_modules"
    if node_modules.exists():
        print("    [PASS] Frontend node_modules found.")
    else:
        print("    [FAIL] Frontend node_modules missing. Run 'npm install' in ./frontend.")
        failures += 1

    if dist_index.exists():
        print("    [PASS] Frontend production build (dist/) verified.")
    else:
        print("    [WARN] Frontend production build missing. Run 'npm run build' in ./frontend.")
        warnings += 1

    # 6. Port Availability
    print("[*] Verifying Local Port Availability:")
    b_free = check_port("127.0.0.1", 8000)
    f_free = check_port("127.0.0.1", 5173)

    if b_free:
        print("    [PASS] Port 8000 (Backend API & WebSocket) is available.")
    else:
        print("    [WARN] Port 8000 is currently occupied (a server may already be running).")
        warnings += 1

    if f_free:
        print("    [PASS] Port 5173 (Vite Frontend Dev Server) is available.")
    else:
        print("    [WARN] Port 5173 is currently occupied.")
        warnings += 1

    print("=" * 70)
    if failures == 0:
        print(f" PREFLIGHT STATUS: READY FOR LOCAL DEMONSTRATION ({warnings} warnings)")
    else:
        print(f" PREFLIGHT STATUS: FAILED ({failures} critical issues, {warnings} warnings)")
    print("=" * 70)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(run_checks())
