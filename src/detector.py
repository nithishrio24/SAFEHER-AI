#!/usr/bin/env python3
"""
SafeHer AI Real-Time Distress Detector
Hybrid approach using speech_recognition and keyword matching.
"""

import speech_recognition as sr
import numpy as np
import sounddevice as sd
from datetime import datetime
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Distress words list
DISTRESS_WORDS = [
    "help", "help me", "leave me alone",
    "dont touch me", "call police", "somebody help",
    "please stop", "let me go", "get away",
    "bachao", "madad karo", "police bulao",
    "chodo mujhe", "stop it", "please help"
]

# Normal context words list
NORMAL_CONTEXT = [
    "find", "keys", "remote", "phone", "movie",
    "song", "music", "game", "food", "lunch"
]


def detect(audio_data):
    """
    Hybrid detection function using speech recognition and keyword matching.
    
    Args:
        audio_data: Audio data from speech_recognition
    
    Returns:
        Dict with triggered, confidence, transcript, and status
    """
    # Step 1: Convert voice to text
    r = sr.Recognizer()
    
    # Try Google Speech Recognition first
    try:
        transcript = r.recognize_google(audio_data)
        transcript = transcript.lower()
        logger.info(f"Transcript (Google): {transcript}")
    except sr.UnknownValueError:
        logger.warning("Google Speech could not understand audio, trying Sphinx...")
        # Fallback to offline Sphinx
        try:
            transcript = r.recognize_sphinx(audio_data)
            transcript = transcript.lower()
            logger.info(f"Transcript (Sphinx): {transcript}")
        except Exception as e:
            logger.error(f"Sphinx also failed: {type(e).__name__}: {e}")
            return {
                "triggered": False,
                "confidence": 0.0,
                "transcript": "",
                "status": "Could not understand audio"
            }
    except sr.RequestError as e:
        logger.error(f"Speech recognition service unavailable: {e}, trying Sphinx...")
        # Fallback to offline Sphinx
        try:
            transcript = r.recognize_sphinx(audio_data)
            transcript = transcript.lower()
            logger.info(f"Transcript (Sphinx): {transcript}")
        except Exception as e2:
            logger.error(f"Sphinx also failed: {type(e2).__name__}: {e2}")
            return {
                "triggered": False,
                "confidence": 0.0,
                "transcript": "",
                "status": "Service unavailable"
            }
    except Exception as e:
        logger.error(f"Speech recognition error: {type(e).__name__}: {e}")
        return {
            "triggered": False,
            "confidence": 0.0,
            "transcript": "",
            "status": "No speech detected"
        }
    
    # Step 2: Check for normal context first
    for word in NORMAL_CONTEXT:
        if word in transcript:
            logger.info(f"Normal context word detected: '{word}'")
            return {
                "triggered": False,
                "confidence": 0.0,
                "transcript": transcript,
                "status": "Normal speech"
            }
    
    # Step 3: Check for distress words
    distress_found = False
    for word in DISTRESS_WORDS:
        if word in transcript:
            distress_found = True
            logger.info(f"Distress word detected: '{word}'")
            break
    
    if not distress_found:
        return {
            "triggered": False,
            "confidence": 0.05,
            "transcript": transcript,
            "status": "Normal speech"
        }
    
    # Step 4: Analyze pitch/energy for confidence
    try:
        audio_array = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
        energy = np.sqrt(np.mean(audio_array**2))
        logger.info(f"Audio energy: {energy:.2f}")
        
        if energy < 500:
            confidence = 0.75
            logger.info("Low energy voice - calm distress")
        elif energy < 1500:
            confidence = 0.85
            logger.info("Medium energy voice - stressed")
        elif energy < 3000:
            confidence = 0.92
            logger.info("High energy voice - loud/panicked")
        else:
            confidence = 0.97
            logger.info("Very high energy voice - screaming")
    except Exception as e:
        logger.error(f"Error analyzing audio energy: {e}")
        confidence = 0.75
    
    return {
        "triggered": confidence >= 0.75,
        "confidence": confidence,
        "transcript": transcript,
        "status": "DISTRESS DETECTED"
    }


class DistressDetector:
    """Simple distress detector using speech recognition."""
    
    def __init__(self, sample_rate: int = 16000, window_duration: float = 3.0):
        """Initialize the distress detector."""
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.recognizer = sr.Recognizer()
        self.consecutive_count = 0
    
    def record_and_detect(self) -> Dict:
        """Record audio and detect distress."""
        logger.info(f"Recording {self.window_duration} seconds of audio...")
        
        # Record audio
        audio = sd.rec(
            int(self.window_duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        
        # Flatten to 1D
        audio = audio.flatten()
        
        # Convert to AudioData for speech recognition
        audio_data = sr.AudioData(
            audio.tobytes(),
            self.sample_rate,
            2  # 2 bytes per sample for int16
        )
        
        # Detect
        result = detect(audio_data)
        result['timestamp'] = datetime.now().isoformat()
        
        return result
    
    def process_audio_window(self, audio_array: np.ndarray) -> Dict:
        """Process pre-recorded audio array and detect distress."""
        try:
            # Convert float32 audio to int16 for speech recognition
            if audio_array.dtype == np.float32:
                audio_array = (audio_array * 32767).astype(np.int16)
            
            # Convert to AudioData for speech recognition
            audio_data = sr.AudioData(
                audio_array.tobytes(),
                self.sample_rate,
                2  # 2 bytes per sample for int16
            )
            
            # Detect
            result = detect(audio_data)
            result['timestamp'] = datetime.now().isoformat()
            
            return result
        except Exception as e:
            logger.error(f"Error processing audio window: {e}")
            return {
                "triggered": False,
                "confidence": 0.0,
                "transcript": "",
                "status": "Error processing audio",
                "timestamp": datetime.now().isoformat()
            }


def main():
    """Main function to run the detector."""
    detector = DistressDetector()
    
    print("Recording 3 seconds of audio...")
    result = detector.record_and_detect()
    
    print(f"\n[{result['timestamp']}]")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Transcript: {result['transcript']}")
    print(f"Status: {result['status']}")
    print(f"Triggered: {result['triggered']}")


if __name__ == "__main__":
    main()
