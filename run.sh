#!/bin/bash

# SafeHer AI - One-Click Startup Script for Linux/Mac

echo "======================================================================"
echo "SAFEHER AI - One-Click Startup"
echo "======================================================================"

# Check GPU
echo ""
echo "[GPU CHECK]"
if python3 -c "import torch; print('✓ GPU Available:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not Available - Using CPU')" 2>/dev/null; then
    :
else
    echo "✗ GPU check failed"
fi

# Check models
echo ""
echo "[MODEL CHECK]"
if [ -f "models/acoustic_model.pt" ] && [ -f "models/keyword_model/config.json" ]; then
    echo "✓ All model files present"
else
    echo "✗ Missing model files - Running training scripts..."
    python3 src/train_acoustic.py
    python3 src/train_keyword.py
fi

# Start backend
echo ""
echo "[BACKEND]"
echo "Starting FastAPI backend on port 8001..."
python3 -c "import sys; sys.path.insert(0, 'app'); from backend import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8001)" &
BACKEND_PID=$!
sleep 3
echo "✓ Backend started (PID: $BACKEND_PID)"

# Start frontend
echo ""
echo "[FRONTEND]"
echo "Starting frontend static server on port 3000..."
cd app/flutter_app/build/web 2>/dev/null || cd app/flutter_app
python3 -m http.server 3000 &
FRONTEND_PID=$!
cd ../..
sleep 1
echo "✓ Frontend started (PID: $FRONTEND_PID)"

# Start detector
echo ""
echo "[DETECTOR]"
echo "Starting microphone listener..."
python3 src/main_listener.py &
DETECTOR_PID=$!
sleep 1
echo "✓ Detector started (PID: $DETECTOR_PID)"

# Open browser
echo ""
echo "[BROWSER]"
echo "Opening browser at http://localhost:3000..."
if command -v open >/dev/null 2>&1; then
    open http://localhost:3000
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://localhost:3000
fi
echo "✓ Browser opened"

# Show status
echo ""
echo "======================================================================"
echo "SAFEHER AI - ALL SERVICES RUNNING"
echo "======================================================================"
echo "Backend:  http://localhost:8001"
echo "Frontend: http://localhost:3000"
echo "Detector: Listening for distress..."
echo "======================================================================"
echo ""
echo "Press CTRL+C to stop all services"
echo ""

# Wait for user to stop
trap "echo ''; echo '[SHUTDOWN] Stopping all services...'; kill $BACKEND_PID $FRONTEND_PID $DETECTOR_PID 2>/dev/null; echo '✓ Services stopped'; exit 0" INT TERM

wait
