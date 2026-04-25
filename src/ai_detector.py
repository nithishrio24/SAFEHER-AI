import os
import json
import torch
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    print("[AI] ✅ Gemini AI loaded — free tier active")
else:
    model = None
    print("[AI] ⚠️ No GEMINI_API_KEY — using rule-based fallback")


SYSTEM_PROMPT = """
You are SafeHer AI — a women's safety distress detection system.

Analyze the spoken transcript and decide if the person 
is in GENUINE danger or distress.

You understand: English, Hindi, Tamil, Telugu, Kannada, 
Malayalam, Bengali, and mixed language sentences.

DISTRESS — score 0.80 to 0.99:
- Calls for help: help, bachao, udavi, sahayam, madad
- Physical threat: dont touch me, let me go, get away
- Fear/panic: i am scared, please no, stop it
- Emergency: call police, call ambulance, 911, 112
- Being hurt: he is hurting me, they attacked me
- Being followed, grabbed, trapped, kidnapped
- Any genuine cry for help in any language
- Repeated words showing panic: bachao bachao, no no no

SAFE — score 0.01 to 0.15:
- Everyday help requests: help me find keys, help me cook
- Media references: police movie, fire the employee
- Casual context: scared of spiders, save my seat
- Normal conversation: good morning, how are you
- Explaining words: what does bachao mean

IMPORTANT RULES:
- Context matters more than keywords alone
- "help me find keys" = SAFE even though it has "help"
- "bachao" alone from a panicked voice = DISTRESS
- Audio energy 0.8+ means loud/panicked voice
- Audio energy 0.2 means calm/safe voice
- When unsure, lean toward safety (higher score)

Respond ONLY with valid JSON — no extra text, no markdown:
{
  "confidence": 0.92,
  "triggered": true,
  "status": "DISTRESS",
  "reason": "Person screaming for help and asking to be released",
  "language_detected": "English",
  "distress_type": "Physical threat"
}

status must be exactly one of: SAFE, WARNING, DISTRESS
confidence must be a decimal between 0.0 and 1.0
"""


def analyze_with_ai(transcript: str,
                     audio_energy: float) -> dict:

    if not transcript or transcript.strip() == "":
        return _safe_result("Empty transcript")

    # Use Gemini if available
    if model:
        try:
            prompt = f"""
{SYSTEM_PROMPT}

Transcript: "{transcript}"
Audio Energy: {audio_energy}
(0.0=silent whisper, 0.5=normal, 0.8=panicked, 1.0=screaming)

Analyze and respond with JSON only.
            """

            response = model.generate_content(prompt)
            raw = response.text.strip()

            # Clean markdown if Gemini wraps in backticks
            raw = raw.replace("```json", "").replace("```", "").strip()

            print(f"[AI] Gemini raw: {raw}")

            result = json.loads(raw)

            # Validate and sanitize fields
            confidence = float(result.get("confidence", 0.01))
            confidence = max(0.01, min(0.99, confidence))

            return {
                "confidence":        confidence,
                "triggered":         confidence >= 0.80,
                "status":            result.get("status", "SAFE"),
                "reason":            result.get("reason", ""),
                "language_detected": result.get("language_detected", "Unknown"),
                "distress_type":     result.get("distress_type", "None")
            }

        except json.JSONDecodeError as e:
            print(f"[AI] JSON error: {e} — using fallback")
            return _fallback_detection(transcript, audio_energy)

        except Exception as e:
            print(f"[AI] Gemini error: {e} — using fallback")
            return _fallback_detection(transcript, audio_energy)

    # No API key — use rule-based
    return _fallback_detection(transcript, audio_energy)


def _safe_result(reason: str) -> dict:
    return {
        "confidence":        0.01,
        "triggered":         False,
        "status":            "SAFE",
        "reason":            reason,
        "language_detected": "Unknown",
        "distress_type":     "None"
    }


def _fallback_detection(transcript: str,
                         audio_energy: float) -> dict:
    """Smart rule-based fallback when API unavailable."""

    HIGH_DISTRESS = [
        "help me please", "please help me", "somebody help",
        "save me please", "dont touch me", "don't touch me",
        "stop touching me", "let me go", "let go of me",
        "leave me alone", "get away from me", "get off me",
        "call the police", "call police", "call 911", "call 112",
        "he is hurting me", "she is hurting me",
        "i am being attacked", "i am in danger",
        "no stop please", "please no please",
        "bachao bachao", "madad karo please",
        "kaapadu kaapadu", "sahayam cheyyandi",
        "mujhe chodo", "nanu vadalandi",
        "i cannot breathe", "i am trapped",
        "someone is following me", "he grabbed me",
    ]

    MEDIUM_DISTRESS = [
        "help", "help me", "save me", "stop it",
        "please stop", "leave me", "get away",
        "scared", "danger", "emergency", "attack",
        "hurting", "hurt me", "police", "ambulance",
        "bachao", "madad", "udavi", "kaapadu",
        "sahayam", "vaddu", "no no no", "please no",
        "rape", "assault", "kidnap", "fire", "thief",
    ]

    SAFE_CONTEXT = [
        "help me find", "help me with", "help me cook",
        "help me install", "help me understand",
        "help me write", "help me look", "help me search",
        "stop the music", "stop the video", "stop the alarm",
        "dont touch my", "don't touch my",
        "scared of spider", "scared of dog", "scared of exam",
        "scared of height", "police movie", "police story",
        "fire the", "save my seat", "save me some food",
        "let me go to", "let me go home",
        "emergency exit", "emergency meeting",
        "what is bachao", "bachao meaning",
        "leave me alone to", "leave me alone i am",
        "get away from my desk",
    ]

    text = transcript.lower().strip()

    for phrase in SAFE_CONTEXT:
        if phrase in text:
            return {
                "confidence":        0.05,
                "triggered":         False,
                "status":            "SAFE",
                "reason":            f"Safe context: {phrase}",
                "language_detected": "Unknown",
                "distress_type":     "None"
            }

    for phrase in HIGH_DISTRESS:
        if phrase in text:
            score = min(0.92 + audio_energy * 0.05, 0.99)
            return {
                "confidence":        round(score, 2),
                "triggered":         True,
                "status":            "DISTRESS",
                "reason":            f"High distress phrase detected: {phrase}",
                "language_detected": "Unknown",
                "distress_type":     "Physical threat"
            }

    matched = [w for w in MEDIUM_DISTRESS if w in text]
    if matched:
        base = 0.85 + (len(matched) - 1) * 0.03
        score = min(base + audio_energy * 0.05, 0.99)
        return {
            "confidence":        round(score, 2),
            "triggered":         score >= 0.80,
            "status":            "DISTRESS",
            "reason":            f"Distress words found: {matched}",
            "language_detected": "Unknown",
            "distress_type":     "General distress"
        }

    return {
        "confidence":        round(audio_energy * 0.10, 2),
        "triggered":         False,
        "status":            "SAFE",
        "reason":            "No distress signals detected",
        "language_detected": "Unknown",
        "distress_type":     "None"
    }


class AIDetector:
    """PyTorch-based acoustic model detector for distress detection."""
    
    def __init__(self, model_path: str = "models/acoustic_model.pt"):
        """Initialize the AIDetector with PyTorch acoustic model."""
        self._model_loaded = False
        self.model = None
        self.model_path = model_path
        
        try:
            # Load the acoustic model
            loaded = torch.load(model_path, map_location='cpu')
            
            # Handle different model formats
            if isinstance(loaded, dict):
                # If it's a state dict, we need to know the model architecture
                # For now, we'll store the state dict and use fallback
                print(f"[AIDetector] ⚠️ Model is a state dict, not a full model - using fallback")
                self.model_state = loaded
                self._model_loaded = False
            else:
                # It's a full model
                self.model = loaded
                self.model.eval()  # Set to evaluation mode
                self._model_loaded = True
                print(f"[AIDetector] ✅ Acoustic model loaded from {model_path}")
        except FileNotFoundError:
            print(f"[AIDetector] ⚠️ Model file not found: {model_path}")
        except Exception as e:
            print(f"[AIDetector] ❌ Failed to load model: {e}")
    
    @property
    def is_model_loaded(self) -> bool:
        """Return True if model loaded successfully."""
        return self._model_loaded
    
    def analyze(self, transcript: str) -> float:
        """
        Analyze transcript using the acoustic model and return confidence score.
        
        Args:
            transcript: Text transcript to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        if not self._model_loaded or self.model is None:
            # Fall back to simple keyword-based confidence
            return self._fallback_confidence(transcript)
        
        try:
            # Extract features from transcript for the acoustic model
            # Since acoustic models typically work with audio features,
            # we'll extract text-based features as a proxy
            features = self._extract_text_features(transcript)
            
            # Run through model
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                output = self.model(features_tensor)
                confidence = torch.sigmoid(output).item()
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            return confidence
            
        except Exception as e:
            print(f"[AIDetector] ❌ Error during analysis: {e}")
            return self._fallback_confidence(transcript)
    
    def _extract_text_features(self, transcript: str) -> list:
        """
        Extract text-based features as proxy for acoustic features.
        
        Args:
            transcript: Text transcript
            
        Returns:
            Feature vector for the model
        """
        # Simple feature extraction based on text characteristics
        text = transcript.lower()
        
        # Distress keywords
        distress_words = ["help", "save", "stop", "please", "bachao", "madad", 
                          "danger", "scared", "hurt", "attack", "police"]
        word_count = len(text.split())
        distress_count = sum(1 for word in distress_words if word in text)
        
        # Create a feature vector
        features = [
            word_count / 20.0,  # Normalize word count
            distress_count / 5.0,  # Normalize distress count
            len(text) / 100.0,  # Normalize length
            1.0 if any(word in text for word in distress_words) else 0.0,  # Has distress words
            text.count('!') / 5.0,  # Exclamation count (urgency)
            text.count('?') / 5.0,  # Question count
        ]
        
        # Pad to expected size if needed
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]  # Return first 10 features
    
    def _fallback_confidence(self, transcript: str) -> float:
        """
        Simple keyword-based confidence fallback when model unavailable.
        
        Args:
            transcript: Text transcript
            
        Returns:
            Confidence score between 0 and 1
        """
        text = transcript.lower()
        distress_words = ["help", "save", "stop", "please", "bachao", "madad",
                          "danger", "scared", "hurt", "attack", "police"]
        
        matched = sum(1 for word in distress_words if word in text)
        
        if matched == 0:
            return 0.05
        elif matched == 1:
            return 0.75
        elif matched == 2:
            return 0.85
        else:
            return 0.95


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AIDetector Test")
    print("="*60 + "\n")
    
    # Create AIDetector instance
    detector = AIDetector()
    
    # Print whether model loaded successfully
    print(f"Model loaded: {detector.is_model_loaded}")
    
    # Run analyze('help me') and print confidence score
    test_transcript = "help me"
    confidence = detector.analyze(test_transcript)
    print(f"Confidence for '{test_transcript}': {confidence:.4f}")
    
    print("\n" + "="*60 + "\n")
