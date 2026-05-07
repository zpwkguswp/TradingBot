@echo off
TITLE V35 Long/Short Sniper Live Bot
echo -------------------------------------------------------
echo   V35 Long/Short Sniper Live Engine Starting
echo   Model: V35 Rank 1 Champion (Score 240.7, Step 14.6M)
echo   Mode: Long + Short Dual Direction Sniper
echo -------------------------------------------------------
cd /d "%~dp0"
.\venv311\Scripts\python.exe v35_live.py
if errorlevel 1 (
    echo.
    echo ERROR: Bot stopped with error.
    pause
)
pause
