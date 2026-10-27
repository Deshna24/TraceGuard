# stop_demo.ps1 - Gracefully shuts down TRACEGUARD-X demonstration servers
$ErrorActionPreference = "SilentlyContinue"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " TRACEGUARD-X: Shutting Down Demonstration Servers" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

function Stop-PortProcess([int]$Port) {
    Write-Host "Checking for active processes on port $Port..." -ForegroundColor Yellow
    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($connections) {
        $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($p in $pids) {
            if ($p -gt 0) {
                try {
                    $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
                    if ($proc) {
                        Write-Host "  Stopping PID $p ($($proc.ProcessName)) on port $Port..." -ForegroundColor Yellow
                        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
                    }
                } catch {
                    # Ignore errors if process already exited
                }
            }
        }
        Write-Host "  [OK] Port $Port is now cleared." -ForegroundColor Green
    } else {
        Write-Host "  [OK] No processes listening on port $Port." -ForegroundColor Green
    }
}

Stop-PortProcess -Port 8000
Stop-PortProcess -Port 5173

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " All TRACEGUARD-X demonstration processes stopped cleanly." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
