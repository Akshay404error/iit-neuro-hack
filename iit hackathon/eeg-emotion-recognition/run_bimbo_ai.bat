@echo off
echo ========================================
echo BIMBO AI - EEG Emotion Recognition
echo Team Matsya N
echo ========================================
echo.

echo [1/4] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)
echo Python found!
echo.

echo [2/4] Installing required packages...
echo This may take a few minutes...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)
echo Requirements installed successfully!
echo.

echo [3/4] Creating necessary directories...
if not exist "data" mkdir data
if not exist "data\processed" mkdir data\processed
if not exist "data\features" mkdir data\features
if not exist "data\models" mkdir data\models
echo Directories created!
echo.

echo [4/4] Starting BIMBO AI Dashboard...
echo.
echo ========================================
echo Dashboard will open in your browser
echo Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run bimbo_ai_dashboard.py

pause
