@echo off
REM Windows batch script to start the Career Readiness API Server
REM This script loads the .env file and starts the FastAPI server

echo Starting Career Readiness API...
python start_api.py
pause
