@echo off
title TRACEGUARD-X Demo Shutdown
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_demo.ps1"
pause
