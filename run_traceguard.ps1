# run_traceguard.ps1
Write-Host "Starting TRACEGUARD Backend Server..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; .\.venv\Scripts\Activate.ps1; python -m uvicorn traceguard.web.server:app --host 0.0.0.0 --port 8000"

Write-Host "Starting TRACEGUARD Frontend Server..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend'; npm run dev"

Write-Host "Servers are launching in new windows! Wait a few seconds and then open http://localhost:5173 in your browser." -ForegroundColor Cyan
