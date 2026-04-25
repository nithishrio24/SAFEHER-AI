@echo off
REM SafeHer AI - One-Click Startup Script for Windows

echo ======================================================================
echo SAFEHER AI - One-Click Startup
echo ======================================================================

REM Check GPU
echo.
echo [GPU CHECK]
python -c "import torch; print('  GPU Available:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not Available - Using CPU')" 2>nul
if %errorlevel% neq 0 (
    echo   GPU check failed
)

REM Check models
echo.
echo [MODEL CHECK]
if exist "models\acoustic_model.pt" if exist "models\keyword_model\config.json" (
    echo   All model files present
) else (
    echo   Missing model files - Running training scripts...
    python src\train_acoustic.py
    python src\train_keyword.py
)

REM Start backend
echo.
echo [BACKEND]
echo   Starting FastAPI backend on port 8001...
start /B python -c "import sys; sys.path.insert(0, 'app'); from backend import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8001)"
timeout /t 3 /nobreak >nul
echo   Backend started

REM Start frontend
echo.
echo [FRONTEND]
echo   Starting frontend static server on port 3000...
cd app\flutter_app\build\web 2>nul || cd app\flutter_app
start /B python -m http.server 3000
cd ..\..\..
timeout /t 1 /nobreak >nul
echo   Frontend started

REM Start detector
echo.
echo [DETECTOR]
echo   Starting microphone listener...
start /B python src\main_listener.py
timeout /t 1 /nobreak >nul
echo   Detector started

REM Open browser
echo.
echo [BROWSER]
echo   Opening browser at http://localhost:3000...
start http://localhost:3000
echo   Browser opened

REM Show status
echo.
echo ======================================================================
echo SAFEHER AI - ALL SERVICES RUNNING
echo ======================================================================
echo Backend:  http://localhost:8001
echo Frontend: http://localhost:3000
echo Detector: Listening for distress...
echo ======================================================================
echo.
echo Press CTRL+C to stop all services
echo.

REM Wait for user to stop
pause
