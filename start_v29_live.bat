@echo off
title V29 Universal Alpha Live Engine
color 0E

echo ===========================================================
echo   V29 UNIVERSAL ALPHA LIVE ENGINE STARTING...
echo   - Mode: Multi-Coin Universal (6 Coins)
echo   - Environment: Python 3.11 (venv311)
echo ===========================================================
echo.

:: Set working directory
cd /d "%~dp0"

:: 1. Specify direct path to venv python
set PYTHON_EXE=venv311\Scripts\python.exe

if not exist %PYTHON_EXE% (
    color 0C
    echo [Error] venv311 not found at: %PYTHON_EXE%
    pause
    exit /b
)

:loop
echo [System] Syncing time and launching V29 runner...
%PYTHON_EXE% v29_bybit_live.py

echo.
echo ===========================================================
echo [Warning] Bot process crashed or stopped. 
echo [System] Restarting in 15 seconds (Phoenix Protocol)...
timeout /t 15
goto loop
