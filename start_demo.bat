@echo off
title TRACEGUARD-X Demo Launcher
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_demo.ps1"
pause
