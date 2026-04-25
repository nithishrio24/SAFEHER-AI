#!/usr/bin/env python3
"""
SafeHer AI Main Listener
Main entry point for the SAFEHER AI system - connects backend, detector, and frontend.
"""

import sys
import time
import threading
import requests
import webbrowser
import torch
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from pathlib import Path
from datetime import datetime
import logging
import uvicorn
from contextlib import redirect_stderr

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def start_backend_server():
    """Start FastAPI backend server in background thread."""
    try:
        # Import backend app
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from app.backend import app
        
        # Run uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    except Exception as e:
        logger.error(f"Error starting backend server: {e}")


def log_alert(timestamp, confidence, transcript):
    """Log alert event to logs/alerts.log."""
    try:
        logs_dir = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        alert_log_path = logs_dir / "alerts.log"
        
        with open(alert_log_path, 'a') as f:
            f.write(f"{timestamp} | Confidence: {confidence:.3f} | Transcript: \"{transcript}\"\n")
        
        logger.info(f"Alert logged to {alert_log_path}")
    except Exception as e:
        logger.error(f"Error logging alert: {e}")


def send_alert_to_backend(result):
    """Send alert to backend and log it."""
    try:
        alert_data = {
            "user_id": "main_listener",
            "latitude": 12.9716,
            "longitude": 77.5946,
            "confidence": result['confidence'],
            "transcript": result['transcript'],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = requests.post("http://localhost:8001/alert", json=alert_data, timeout=10)
        
        if response.status_code in [200, 201]:
            print(f"\n✓ Alert sent to backend successfully")
            print(f"  Alert ID: {response.json().get('alert_id', 'N/A')}")
            
            # Log the alert
            log_alert(result['timestamp'], result['confidence'], result['transcript'])
        else:
            print(f"\n✗ Failed to send alert: {response.status_code}")
    except Exception as e:
        print(f"\n✗ Error sending alert: {e}")
        logger.error(f"Error sending alert: {e}")


def run_live_detector():
    """Run live microphone detector with alert integration."""
    sample_rate = 16000
    duration = 3.0
    
    print("\nLive detector started - listening for distress...")
    
    try:
        while True:
            # Record 3 seconds of audio
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.int16
            )
            sd.wait()
            
            # Flatten to 1D
            audio = recording.flatten()
            
            # Convert to AudioData for speech recognition
            audio_data = sr.AudioData(
                audio.tobytes(),
                sample_rate,
                2  # 2 bytes per sample for int16
            )
            
            # Process through detector
            from detector import detect
            result = detect(audio_data)
            result['timestamp'] = datetime.now().isoformat()
            
            # Check if triggered
            if result.get('triggered', False):
                confidence = result.get('confidence', 0.0)
                transcript = result.get('transcript', '')
                
                # Print alert
                print(f"\n{'='*70}")
                print(f"⚠️  DISTRESS DETECTED!")
                print(f"Confidence: {confidence*100:.0f}%")
                print(f"Transcript: \"{transcript}\"")
                print(f"{'='*70}")
                
                # Send alert to backend
                send_alert_to_backend(result)
                
                # Reset consecutive count
                detector.consecutive_count = 0
            
            # Small delay
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Error in live detector: {e}")


def print_startup_banner():
    """Print startup banner."""
    # Check GPU
    gpu_available = torch.cuda.is_available()
    gpu_name = "Not Available"
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
    
    # Format GPU status
    if "3050" in gpu_name:
        gpu_status = "RTX 3050 ✓"
    elif gpu_available:
        gpu_status = "Available ✓"
    else:
        gpu_status = "Not Available ✗"
    
    print("\n╔══════════════════════════════════╗")
    print("║       SAFEHER AI - ACTIVE        ║")
    print("║  Backend  : http://localhost:8001 ║")
    print("║  Frontend : http://localhost:8001 ║")
    print(f"║  GPU      : {gpu_status:<27} ║")
    print("║  Status   : LISTENING            ║")
    print("╚══════════════════════════════════╝\n")


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("SAFEHER AI - Main Listener")
    print("="*70)
    
    # Start backend server in background thread
    print("\nStarting FastAPI backend server...")
    backend_thread = threading.Thread(target=start_backend_server, daemon=True)
    backend_thread.start()
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Open frontend in browser
    print("Opening frontend in browser...")
    webbrowser.open("http://localhost:8001")
    
    # Print startup banner
    print_startup_banner()
    
    # Start live detector
    detector_thread = threading.Thread(target=run_live_detector, daemon=True)
    detector_thread.start()
    
    print("Press CTRL+C to stop...\n")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down SafeHer AI...")
        print("Backend server stopping...")
        print("Detector stopping...")
        print("All services stopped.\n")


if __name__ == "__main__":
    main()
