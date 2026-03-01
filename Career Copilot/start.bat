@echo off
REM Quick Start Script for Career Copilot (Windows)

echo.
echo ============================================
echo   Career Copilot - Starting System
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Checking dependencies...
python -c "import fastapi, uvicorn, streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Dependencies not installed. Run: pip install -r requirements.txt
    pause
    exit /b 1
)

echo [2/3] Starting API Server (http://localhost:8000)...
start "Career Copilot API" cmd /k "python api_server.py"

timeout /t 5 /nobreak >nul

echo [3/3] Starting Streamlit UI (http://localhost:8501)...
timeout /t 2 /nobreak >nul
start "Career Copilot UI" cmd /k "streamlit run streamlit_app.py"

echo.
echo ============================================
echo   System Started Successfully!
echo ============================================
echo.
echo   API:  http://localhost:8000
echo   UI:   http://localhost:8501
echo.
echo   Press any key to open UI in browser...
pause >nul

start http://localhost:8501

echo.
echo System is running. Close terminal windows to stop.
echo.
pause
