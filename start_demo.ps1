# start_demo.ps1 - TRACEGUARD-X Automated Demonstration Launcher
$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " TRACEGUARD-X: Live Demonstration Launcher" -ForegroundColor Cyan
Write-Host " Uncertainty-Aware LLM Agent Safety Gate Demonstration" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Determine Preferred Python Command
$PythonCmd = "python"
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pyCheck = py -3.12 -c "import torch; print('ok')" 2>$null
    if ($pyCheck -eq "ok") {
        $PythonCmd = "py -3.12"
    }
}

Write-Host "[1/4] Running Preflight Diagnostic Checks..." -ForegroundColor Yellow
if ($PythonCmd -eq "py -3.12") {
    py -3.12 "$PSScriptRoot\scripts\preflight_check.py"
} else {
    python "$PSScriptRoot\scripts\preflight_check.py"
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Preflight checks failed! Please resolve the above issues before continuing." -ForegroundColor Red
    exit 1
}

# 2. Launch Backend Server in a new window
Write-Host "[2/4] Starting FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Yellow
$BackendCmd = "cd '$PSScriptRoot'; $PythonCmd -m uvicorn traceguard.web.server:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCmd -Title "TRACEGUARD-X Backend (Port 8000)"

# 3. Wait for Backend Health Check
Write-Host "[3/4] Waiting for Backend Health Check (http://127.0.0.1:8000/health)..." -ForegroundColor Yellow
$Healthy = $false
$Deadline = (Get-Date).AddSeconds(15)

while ((Get-Date) -lt $Deadline) {
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($resp.status -eq "ok") {
            $Healthy = $true
            break
        }
    } catch {
        # Retry until timeout
    }
    Start-Sleep -Milliseconds 500
}

if (-not $Healthy) {
    Write-Host "[ERROR] Backend failed to report healthy within 15 seconds!" -ForegroundColor Red
    Write-Host "Check the 'TRACEGUARD-X Backend' window for tracebacks." -ForegroundColor Red
    exit 1
}
Write-Host "    [OK] Backend is healthy and ready for live WebSocket connections!" -ForegroundColor Green

# 4. Launch Frontend Dev Server in a new window
Write-Host "[4/4] Starting Vite Frontend on http://localhost:5173..." -ForegroundColor Yellow
$FrontendCmd = "cd '$PSScriptRoot\frontend'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FrontendCmd -Title "TRACEGUARD-X Frontend (Port 5173)"

Start-Sleep -Seconds 2

# 5. Open Browser
Write-Host "Opening demonstration dashboard at http://localhost:5173..." -ForegroundColor Green
Start-Process "http://localhost:5173"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " DEMONSTRATION IS READY!" -ForegroundColor Green
Write-Host "  - Frontend URL: http://localhost:5173" -ForegroundColor White
Write-Host "  - Backend URL:  http://127.0.0.1:8000" -ForegroundColor White
Write-Host "  - Health Check: http://127.0.0.1:8000/health" -ForegroundColor White
Write-Host "  - Evidence Tab: http://localhost:5173/evidence" -ForegroundColor White
Write-Host "  - To shutdown:  Run .\stop_demo.ps1 or stop_demo.bat" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan
