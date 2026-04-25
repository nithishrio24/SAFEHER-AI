#!/usr/bin/env python3
"""
SafeHer AI - One-Click Startup Script
Starts the entire project with a single command.
"""

import sys
import os
import time
import threading
import webbrowser
import subprocess
from pathlib import Path
import torch
import requests
from datetime import datetime

# ANSI color codes
BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_banner():
    """Print startup banner."""
    print("\n" + "="*70)
    print(BOLD + "SAFEHER AI - One-Click Startup" + RESET)
    print("="*70)


def check_gpu():
    """Check if GPU is available and print status."""
    print("\n" + BLUE + "[GPU CHECK]" + RESET)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  ✓ GPU Available: {gpu_name}")
        return True
    else:
        print("  ✗ GPU Not Available - Using CPU")
        return False


def check_models():
    """Check if all model files exist in /models folder."""
    print("\n" + BLUE + "[MODEL CHECK]" + RESET)
    
    models_dir = Path(__file__).parent / "models"
    required_files = [
        "acoustic_model.pt",
        "keyword_model/config.json",
        "keyword_model/model.safetensors"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = models_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"  ✗ Missing model files: {', '.join(missing_files)}")
        print(f"  → Running training scripts to generate models...")
        run_training_scripts()
    else:
        print("  ✓ All model files present")
        return True


def run_training_scripts():
    """Run training scripts to generate missing models."""
    print("\n" + YELLOW + "[TRAINING MODELS]" + RESET)
    
    # Run acoustic model training
    print("  Training acoustic model...")
    try:
        subprocess.run(
            [sys.executable, "src/train_acoustic.py"],
            cwd=Path(__file__).parent,
            timeout=300
        )
        print("  ✓ Acoustic model trained")
    except Exception as e:
        print(f"  ✗ Acoustic model training failed: {e}")
    
    # Run keyword model training
    print("  Training keyword model...")
    try:
        subprocess.run(
            [sys.executable, "src/train_keyword.py"],
            cwd=Path(__file__).parent,
            timeout=300
        )
        print("  ✓ Keyword model trained")
    except Exception as e:
        print(f"  ✗ Keyword model training failed: {e}")


def start_backend():
    """Start FastAPI backend on port 8001."""
    print("\n" + BLUE + "[BACKEND]" + RESET)
    
    # Check if backend is already running
    try:
        response = requests.get("http://localhost:8001/health", timeout=2)
        if response.status_code == 200:
            print("  ✓ Backend already running on port 8001")
            return None
    except:
        pass
    
    print("  Starting FastAPI backend on port 8001...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "app"))
        from backend import app
        import uvicorn
        
        # Run uvicorn in a separate thread
        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
        
        backend_thread = threading.Thread(target=run_server, daemon=True)
        backend_thread.start()
        
        # Wait for backend to be ready
        print("  Waiting for backend to start...", end="", flush=True)
        for _ in range(10):
            time.sleep(1)
            print(".", end="", flush=True)
        print(" ✓")
        
        # Verify backend is running
        try:
            response = requests.get("http://localhost:8001/health", timeout=2)
            if response.status_code == 200:
                print("  ✓ Backend is running")
            else:
                print("  ✗ Backend health check failed")
        except:
            print("  ⚠ Backend might not be fully ready yet")
        
        return backend_thread
    except Exception as e:
        print(f"  ✗ Failed to start backend: {e}")
        return None


def start_frontend():
    """Start frontend static server on port 3000."""
    print("\n" + BLUE + "[FRONTEND]" + RESET)
    
    # Check if frontend is already running
    try:
        response = requests.get("http://localhost:3000", timeout=2)
        if response.status_code in [200, 404]:
            print("  ✓ Frontend already running on port 3000")
            return None
    except:
        pass
    
    print("  Starting frontend static server on port 3000...")
    
    try:
        # Try to start a simple HTTP server for the frontend
        frontend_dir = Path(__file__).parent / "app" / "flutter_app" / "build" / "web"
        
        if not frontend_dir.exists():
            # If build doesn't exist, try the flutter_app directory
            frontend_dir = Path(__file__).parent / "app" / "flutter_app"
        
        def run_server():
            try:
                import http.server
                import socketserver
                os.chdir(frontend_dir)
                with socketserver.TCPServer(("", 3000), http.server.SimpleHTTPRequestHandler) as httpd:
                    httpd.serve_forever()
            except:
                pass
        
        frontend_thread = threading.Thread(target=run_server, daemon=True)
        frontend_thread.start()
        
        print("  ✓ Frontend server started")
        return frontend_thread
    except Exception as e:
        print(f"  ✗ Failed to start frontend: {e}")
        return None


def start_detector():
    """Start main microphone listener."""
    print("\n" + BLUE + "[DETECTOR]" + RESET)
    print("  Starting microphone listener...")
    
    try:
        from src.detector import DistressDetector
        import sounddevice as sd
        import numpy as np
        
        detector = DistressDetector()
        sample_rate = 16000
        duration = 3.0
        
        def run_detector():
            print(GREEN + "  Detector listening for distress..." + RESET)
            while True:
                try:
                    recording = sd.rec(
                        int(duration * sample_rate),
                        samplerate=sample_rate,
                        channels=1,
                        dtype=np.float32
                    )
                    sd.wait()
                    
                    audio = recording.flatten()
                    result = detector.process_audio_window(audio)
                    
                    if result.get('triggered', False):
                        print(RED + f"\n  ⚠️ DISTRESS DETECTED!" + RESET)
                        print(RED + f"  Confidence: {result['confidence']*100:.0f}%" + RESET)
                        print(RED + f"  Transcript: \"{result['transcript']}\"" + RESET)
                        
                        # Send alert to backend
                        try:
                            alert_data = {
                                "user_id": "main_listener",
                                "latitude": 12.9716,
                                "longitude": 77.5946,
                                "confidence": result['confidence'],
                                "transcript": result['transcript'],
                                "timestamp": datetime.utcnow().isoformat()
                            }
                            requests.post("http://localhost:8001/alert", json=alert_data, timeout=5)
                            print(GREEN + "  ✓ Alert sent to backend" + RESET)
                        except Exception as e:
                            print(RED + f"  ✗ Failed to send alert: {e}" + RESET)
                        
                        detector.consecutive_count = 0
                except Exception as e:
                    print(RED + f"  Detector error: {e}" + RESET)
                    time.sleep(1)
        
        detector_thread = threading.Thread(target=run_detector, daemon=True)
        detector_thread.start()
        
        print("  ✓ Detector started")
        return detector_thread
    except Exception as e:
        print(f"  ✗ Failed to start detector: {e}")
        return None


def open_browser():
    """Open browser at http://localhost:3000."""
    print("\n" + BLUE + "[BROWSER]" + RESET)
    print("  Opening browser at http://localhost:3000...")
    try:
        webbrowser.open("http://localhost:3000")
        print("  ✓ Browser opened")
    except Exception as e:
        print(f"  ✗ Failed to open browser: {e}")


def show_live_logs():
    """Show live logs from all services."""
    print("\n" + BOLD + "="*70)
    print("SAFEHER AI - ALL SERVICES RUNNING")
    print("="*70 + RESET)
    print(BLUE + "Backend:  http://localhost:8001" + RESET)
    print(GREEN + "Frontend: http://localhost:3000" + RESET)
    print(GREEN + "Detector: Listening for distress..." + RESET)
    print(BOLD + "="*70 + RESET)
    print("\n" + YELLOW + "Press CTRL+C to stop all services" + RESET + "\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + YELLOW + "[SHUTDOWN]" + RESET)
        print("  Stopping all services...")
        print("  ✓ Services stopped")
        print("\n" + BOLD + "SafeHer AI stopped." + RESET + "\n")


def main():
    """Main entry point."""
    print_banner()
    
    # Step 1: Check GPU
    gpu_available = check_gpu()
    
    # Step 2: Check models
    check_models()
    
    # Step 3: Start backend
    backend_thread = start_backend()
    
    # Step 4: Start frontend
    frontend_thread = start_frontend()
    
    # Step 5: Start detector
    detector_thread = start_detector()
    
    # Step 6: Open browser
    time.sleep(2)
    open_browser()
    
    # Step 7: Show live logs
    show_live_logs()


if __name__ == "__main__":
    main()
