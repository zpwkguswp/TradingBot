@echo off
setlocal

echo ============================================================
echo [V29 Trading Bot] New Computer Setup Script
echo ============================================================

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python is not installed or not in PATH.
    echo Please install Python 3.11 from https://www.python.org/
    pause
    exit /b
)

:: 2. Create Virtual Environment
echo [+] Creating Virtual Environment (venv311)...
python -m venv venv311
if %errorlevel% neq 0 (
    echo [!] Failed to create venv.
    pause
    exit /b
)

:: 3. Activate Venv and Upgrade Pip
echo [+] Activating environment and upgrading pip...
call .\venv311\Scripts\activate.bat
python -m pip install --upgrade pip

:: 4. Install PyTorch with CUDA 11.8 (Recommended)
echo [+] Installing PyTorch with CUDA 11.8 support...
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118

:: 5. Install Dependencies from requirements.txt
echo [+] Installing dependencies from requirements.txt...
if exist requirements.txt (
    python -m pip install -r requirements.txt
) else (
    echo [!] requirements.txt not found.
)

:: 6. Setup Directory Structure
echo [+] Ensuring data directories exist...
if not exist data_storage mkdir data_storage
if not exist elite_weights mkdir elite_weights
if not exist v29_logs mkdir v29_logs

echo.
echo ============================================================
echo [SUCCESS] Setup Complete!
echo.
echo 1. Edit config.py with your API Keys and Telegram ID.
echo 2. Place v29_best_model_2h.zip inside elite_weights/ folder.
echo 3. Run the bot using: start_v29_live.bat
echo ============================================================
pause
