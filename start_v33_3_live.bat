@echo off
TITLE V34 Grand Finale Live Bot
echo -------------------------------------------------------
echo   V34 Grand Finale Live Engine Starting
echo   Model: V34 Rank 1 Champion
echo -------------------------------------------------------
cd /d "%~dp0"
.\venv311\Scripts\python.exe v33_3_live.py
if errorlevel 1 (
    echo.
    echo ERROR: Bot stopped with error.
    pause
)
pause
