#!/usr/bin/env python3
"""
SafeHer AI FastAPI Backend
Handles distress alerts, notifications, and emergency services integration.
"""

import os
import uuid
import math
import json
import torch
import numpy as np
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import httpx
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load environment variables
load_dotenv()

# Import SMS and Email services
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sms_service import send_all_contacts, send_cancel_sms
from email_alert import send_alert_email

# Import models
from app.models import (
    AlertRequest,
    CancelRequest,
    AlertResponse,
    CancelResponse,
    StatusResponse,
    ErrorResponse,
    AlertStatus,
    Location,
    NearbyPlace,
    NotifiedParty
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SafeHer AI API",
    description="Distress detection and emergency response API",
    version="1.0.0"
)

# Background thread pool for email sending
_pool = ThreadPoolExecutor(max_workers=2)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")
else:
    logger.warning(f"Frontend directory not found at {frontend_dir}")

# In-memory storage for alerts (in production, use a database)
alerts_storage: Dict[str, Dict] = {}
user_alerts: Dict[str, str] = {}  # user_id -> alert_id

# Contacts file path
CONTACTS_FILE = Path(__file__).parent.parent / "contacts.json"

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "models"
ACOUSTIC_MODEL_PATH = MODELS_DIR / "acoustic_model.pt"
KEYWORD_MODEL_PATH = MODELS_DIR / "keyword_model"
DISTRESS_MODEL_PATH = Path(__file__).parent.parent / "src" / "models" / "distress_model.pkl"

# ── DISTRESS WORD LISTS ──────────────────────────────

HIGH_DISTRESS = [
    # English — critical phrases
    "help me please", "please help me", "somebody help me",
    "someone help me", "help me now", "please save me",
    "dont touch me", "don't touch me", "stop touching me",
    "let me go", "let go of me", "leave me alone",
    "get away from me", "get off me", "get off",
    "call the police", "call police", "call 911", "call 112",
    "he is hurting me", "she is hurting me", "they are hurting me",
    "i am being attacked", "i am in danger",
    "stop it stop it", "please no please", "no stop no",
    "save me please", "anyone help", "please anyone",
    "i am scared help", "help me i am scared",

    # Hindi — critical
    "bachao bachao", "madad karo please", "police bulao",
    "mujhe chodo", "chhod do mujhe", "mujhe jane do",
    "bachao mujhe", "koi madad karo", "meri madad karo",

    # Tamil — critical
    "kaapadu kaapadu", "udavi pannu", "police ku sollunga",
    "vidunga vidunga", "kapadungga",

    # Telugu — critical
    "sahayam cheyyandi", "nanu vadalandi", "police ki cheppandi",
]

MEDIUM_DISTRESS = [
    # English — single strong words
    "help", "help me", "save me", "please help",
    "stop", "stop it", "please stop", "stop please",
    "leave me", "let me go", "get away",
    "scared", "i am scared", "im scared",
    "danger", "emergency", "attack",
    "no no no", "please no", "no please",
    "fire", "thief", "robbery", "rape", "assault",
    "hurting", "hurt me", "he hurt", "she hurt",
    "kidnap", "abduction", "trapped",

    # Hindi
    "bachao", "madad", "madad karo", "chodo",
    "rukko", "ruko", "nahi", "police",
    "dard", "darr", "chhod",

    # Tamil
    "udavi", "kaapadu", "vidunga", "police",
    "pavam", "vittudu", "aayutha",

    # Telugu
    "sahayam", "vaddu", "vadalandi",
    "cheyyandi", "help cheyyandi",
]

# Keep old DISTRESS_WORDS for backward compatibility with sklearn model
DISTRESS_WORDS = HIGH_DISTRESS + MEDIUM_DISTRESS

# Safe context words list
SAFE_CONTEXT = [
    "help me find", "help me look", "help me search",
    "help me with", "help me understand", "help me learn",
    "help me cook", "help me make", "help me write",
    "help me fix", "help me install", "help me open",
    "stop the music", "stop the video", "stop playing",
    "stop the song", "stop the alarm",
    "don't touch my food", "don't touch my phone",
    "don't touch my stuff", "don't touch that",
    "leave me alone to study", "leave me alone i am",
    "get away from my", "police movie", "police story",
    "fire the employee", "fire him", "fire her",
    "i am scared of", "scared of spider", "scared of dog",
    "scared of exam", "scared of heights",
    "bachao meaning", "what is bachao",
    "call police in movie",
]

# Global model variables
acoustic_model = None
keyword_model = None
keyword_tokenizer = None
distress_model = None
gpu_available = False
gpu_name = "Not Available"


def load_models():
    """Load ML models for distress detection."""
    global acoustic_model, keyword_model, keyword_tokenizer, distress_model, gpu_available, gpu_name
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name}")
    else:
        logger.info("No GPU detected, using CPU")
    
    # Load acoustic model
    try:
        if ACOUSTIC_MODEL_PATH.exists():
            acoustic_model = torch.load(ACOUSTIC_MODEL_PATH, map_location='cpu')
            # Handle if model is saved as a dict (state_dict)
            if isinstance(acoustic_model, dict):
                logger.info("Acoustic model is a state_dict, would need model architecture to load")
                acoustic_model = None  # Set to None since we can't use state_dict without architecture
            elif gpu_available:
                acoustic_model = acoustic_model.to('cuda')
            if acoustic_model is not None:
                logger.info("Acoustic model loaded successfully")
        else:
            logger.warning(f"Acoustic model not found at {ACOUSTIC_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading acoustic model: {e}")
        acoustic_model = None
    
    # Load keyword model (transformer)
    try:
        if KEYWORD_MODEL_PATH.exists():
            keyword_tokenizer = AutoTokenizer.from_pretrained(str(KEYWORD_MODEL_PATH))
            keyword_model = AutoModelForSequenceClassification.from_pretrained(str(KEYWORD_MODEL_PATH))
            if gpu_available:
                keyword_model = keyword_model.to('cuda')
            logger.info("Keyword model loaded successfully")
        else:
            logger.warning(f"Keyword model not found at {KEYWORD_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading keyword model: {e}")
    
    # Load sklearn distress model
    try:
        if DISTRESS_MODEL_PATH.exists():
            with open(DISTRESS_MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            # Only load the model, not the extract_features function
            distress_model = {"model": model_data["model"]}
            logger.info("✅ Sklearn distress model loaded successfully")
        else:
            logger.warning(f"Sklearn distress model not found at {DISTRESS_MODEL_PATH}")
            logger.info("Will use rule-based detection as fallback")
            distress_model = None
    except Exception as e:
        logger.error(f"Error loading sklearn distress model: {e}")
        distress_model = None


def calculate_confidence(transcript: str, audio_energy: float) -> dict:
    """Calculate distress confidence using ML model or rule-based fallback."""
    text = transcript.lower().strip()
    if not text:
        return {"confidence": 0.01, "triggered": False, "status": "SAFE"}

    # Use ML model if available
    if distress_model is not None:
        try:
            feat = extract_features(text, audio_energy)
            prob = distress_model["model"].predict_proba([feat])[0][1]
            score = round(prob, 2)
            print(f"[MODEL] ML score: {round(score*100)}%")
        except Exception as e:
            print(f"[MODEL] ML error: {e} — using rules")
            score = _rule_based_score(text, audio_energy)
    else:
        score = _rule_based_score(text, audio_energy)

    return {
        "confidence": score,
        "triggered":  score >= 0.80,
        "status":     "DISTRESS" if score >= 0.80 else
                      "WARNING"  if score >= 0.60 else "SAFE"
    }


def _rule_based_score(text: str, audio_energy: float) -> float:
    """Rule-based confidence scoring as fallback."""
    # Check safe context first
    for phrase in SAFE_CONTEXT:
        if phrase in text:
            return 0.05  # Very low confidence for safe context

    # Check high distress phrases
    for phrase in HIGH_DISTRESS:
        if phrase in text:
            base = 0.92
            if audio_energy > 0.7:
                return min(0.98, base + (audio_energy - 0.7) * 0.1)
            return base

    # Check medium distress words
    distress_count = sum(1 for word in MEDIUM_DISTRESS if word in text)
    if distress_count > 0:
        base = 0.85 + (distress_count * 0.03)
        if audio_energy > 0.6:
            return min(0.94, base + (audio_energy - 0.6) * 0.1)
        return min(0.92, base)

    # Default safe
    return 0.05


# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


def load_contacts() -> Dict:
    """Load emergency contacts from contacts.json."""
    if CONTACTS_FILE.exists():
        try:
            with open(CONTACTS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading contacts: {e}")
    # Return default contacts if file doesn't exist
    return {
        "user_name": "User",
        "contacts": []
    }


def save_contacts(data: Dict) -> bool:
    """Save emergency contacts to contacts.json."""
    try:
        with open(CONTACTS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Contacts saved to {CONTACTS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving contacts: {e}")
        return False


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in kilometers using Haversine formula."""
    R = 6371  # Earth's radius in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance


def extract_features(transcript: str, audio_energy: float) -> List[float]:
    """Extract features from transcript and audio energy for sklearn model."""
    transcript_lower = transcript.lower()
    
    # Feature 1: Has distress word (1/0)
    has_distress_word = 0
    for word in DISTRESS_WORDS:
        if word in transcript_lower:
            has_distress_word = 1
            break
    
    # Feature 2: Distress word count
    distress_word_count = 0
    for word in DISTRESS_WORDS:
        if word in transcript_lower:
            distress_word_count += 1
    
    # Feature 3: Audio energy
    audio_energy_val = audio_energy
    
    # Feature 4: Transcript length
    transcript_length = len(transcript)
    
    # Feature 5: Has safe context (1/0)
    has_safe_context = 0
    for phrase in SAFE_CONTEXT:
        if phrase in transcript_lower:
            has_safe_context = 1
            break
    
    return [
        has_distress_word,
        distress_word_count,
        audio_energy_val,
        transcript_length,
        has_safe_context
    ]


async def send_twilio_sms(phone: str, message: str) -> bool:
    """Send SMS using Twilio API."""
    try:
        twilio_sid = os.getenv("TWILIO_SID")
        twilio_token = os.getenv("TWILIO_TOKEN")
        twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not all([twilio_sid, twilio_token, twilio_phone]):
            logger.warning("Twilio credentials not configured, using mock SMS")
            logger.info(f"[MOCK SMS] To: {phone} | Message: {message[:50]}...")
            return True
        
        url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                auth=(twilio_sid, twilio_token),
                data={
                    "From": twilio_phone,
                    "To": phone,
                    "Body": message
                },
                timeout=10.0
            )
            
            if response.status_code == 201:
                logger.info(f"SMS sent successfully to {phone}")
                return True
            else:
                logger.error(f"Failed to send SMS: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Error sending SMS via Twilio: {e}")
        return False


async def send_firebase_push_notification(device_token: str, title: str, body: str) -> bool:
    """Send push notification using Firebase Admin SDK."""
    try:
        firebase_key = os.getenv("FIREBASE_KEY")
        
        if not firebase_key:
            logger.warning("Firebase key not configured, using mock push notification")
            logger.info(f"[MOCK PUSH] To: {device_token} | Title: {title} | Body: {body[:50]}...")
            return True
        
        # Firebase Cloud Messaging endpoint
        url = "https://fcm.googleapis.com/fcm/send"
        
        headers = {
            "Authorization": f"key={firebase_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "to": device_token,
            "notification": {
                "title": title,
                "body": body
            },
            "data": {
                "alert_type": "distress",
                "timestamp": datetime.utcnow().isoformat()
            },
            "priority": "high"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=10.0
            )
            
            if response.status_code == 200:
                logger.info(f"Push notification sent successfully to {device_token}")
                return True
            else:
                logger.error(f"Failed to send push notification: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"Error sending push notification via Firebase: {e}")
        return False


async def find_nearby_places(
    latitude: float,
    longitude: float,
    place_type: Optional[str] = None,
    keyword: Optional[str] = None,
    radius: int = 5000
) -> List[NearbyPlace]:
    """Find nearby places using Google Maps Places API."""
    try:
        google_maps_key = os.getenv("GOOGLE_MAPS_KEY")
        
        if not google_maps_key:
            logger.warning("Google Maps API key not configured, using mock nearby places")
            # Return mock nearby places for testing
            mock_places = []
            
            if place_type == "police" or keyword == "police":
                mock_places.append(NearbyPlace(
                    name="Central Police Station",
                    address="123 Main Street",
                    distance_km=0.5,
                    place_id="mock_police_1",
                    rating=4.5,
                    phone="+1234567890"
                ))
            
            if keyword == "women shelter":
                mock_places.append(NearbyPlace(
                    name="Women's Safe Haven",
                    address="456 Oak Avenue",
                    distance_km=1.2,
                    place_id="mock_shelter_1",
                    rating=4.8,
                    phone="+0987654321"
                ))
            
            logger.info(f"[MOCK PLACES] Found {len(mock_places)} mock nearby places")
            return mock_places
        
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        
        params = {
            "location": f"{latitude},{longitude}",
            "radius": radius,
            "key": google_maps_key
        }
        
        if place_type:
            params["type"] = place_type
        if keyword:
            params["keyword"] = keyword
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                
                places = []
                for place in data.get("results", []):
                    location = place.get("geometry", {}).get("location", {})
                    place_lat = location.get("lat")
                    place_lng = location.get("lng")
                    
                    if place_lat and place_lng:
                        distance = haversine_distance(latitude, longitude, place_lat, place_lng)
                        
                        nearby_place = NearbyPlace(
                            name=place.get("name", "Unknown"),
                            address=place.get("vicinity", "Address not available"),
                            distance_km=round(distance, 2),
                            place_id=place.get("place_id", ""),
                            rating=place.get("rating"),
                            phone=place.get("formatted_phone_number")
                        )
                        places.append(nearby_place)
                
                # Sort by distance
                places.sort(key=lambda x: x.distance_km)
                return places[:5]  # Return top 5 nearest places
            else:
                logger.error(f"Google Maps API error: {response.text}")
                return []
                
    except Exception as e:
        logger.error(f"Error finding nearby places: {e}")
        return []


@app.post("/alert")
async def send_alert(request: Request):
    """Send distress alert with email in background."""
    body = await request.json()
    transcript = body.get("transcript", "")
    confidence = float(body.get("confidence", 0.0))
    latitude = body.get("latitude", None)
    longitude = body.get("longitude", None)

    print(f"[/alert] RECEIVED — '{transcript}' {round(confidence*100)}%")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_pool, lambda: _bg_email(
        transcript, confidence, latitude, longitude
    ))

    # Return instantly — email sends in background
    return {"status": "alert_received", "email_sent": True}


def _bg_email(transcript, confidence, lat, lng):
    """Background email sender."""
    from src.email_alert import send_alert_email
    try:
        ok = send_alert_email(transcript, confidence, lat, lng)
        print(f"[EMAIL] Background result: {ok}")
    except Exception as e:
        print(f"[EMAIL] Background error: {e}")


@app.post("/cancel", response_model=CancelResponse)
async def cancel_alert(request: CancelRequest):
    """
    Cancel a pending alert within 3 seconds of trigger.
    
    This endpoint allows users to cancel false alarms within a 3-second window.
    """
    try:
        # Check if user has a recent alert
        alert_id = user_alerts.get(request.user_id)
        
        if not alert_id:
            return CancelResponse(
                success=False,
                message="No pending alert found for this user"
            )
        
        alert_data = alerts_storage.get(alert_id)
        
        if not alert_data:
            return CancelResponse(
                success=False,
                message="Alert not found"
            )
        
        # Check if alert is within 3-second window
        alert_time = alert_data["created_at"]
        time_diff = (datetime.utcnow() - alert_time).total_seconds()
        
        if time_diff > 3:
            alert_data["status"] = AlertStatus.EXPIRED
            return CancelResponse(
                success=False,
                message=f"Alert cancellation window expired ({time_diff:.1f}s > 3s)",
                alert_id=alert_id
            )
        
        # Cancel the alert
        alert_data["status"] = AlertStatus.CANCELLED
        alert_data["cancelled_at"] = datetime.utcnow()
        
        logger.info(f"Alert cancelled: {alert_id} by user {request.user_id}")
        
        # Send safe message to all contacts
        contacts_data = load_contacts()
        user_name = contacts_data.get("user_name", "User")
        contacts_list = contacts_data.get("contacts", [])
        
        # If no contacts saved, use default
        if not contacts_list:
            contacts_list = EMERGENCY_CONTACTS
        
        # Send cancel SMS to all contacts
        for contact in contacts_list:
            send_cancel_sms(contact['phone'], user_name)
        
        logger.info(f"Cancel SMS sent to {len(contacts_list)} contacts")
        
        # Send cancel email
        # Cancel email not implemented in new email module
        # alert_email = os.getenv("ALERT_EMAIL")
        # alert_email_password = os.getenv("ALERT_EMAIL_PASSWORD")
        # if alert_email and alert_email_password and alert_email != "your_gmail@gmail.com":
        #     try:
        #         send_cancel_email(alert_data.get("transcript", ""))
        #         logger.info("Cancel email sent")
        #     except Exception as e:
        #         logger.error(f"Error sending cancel email: {e}")
        
        return CancelResponse(
            success=True,
            message="Alert cancelled successfully",
            alert_id=alert_id
        )
        
    except Exception as e:
        logger.error(f"Error cancelling alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel alert: {str(e)}"
        )


@app.get("/status/{user_id}", response_model=StatusResponse)
async def get_alert_status(user_id: str):
    """
    Get the last alert status for a user.
    
    Returns information about the most recent alert, including:
    - Alert status (pending, sent, cancelled, expired)
    - Confidence score
    - Transcript
    - Location
    - Notified parties
    - Nearby emergency services
    """
    try:
        alert_id = user_alerts.get(user_id)
        
        if not alert_id:
            return StatusResponse(
                user_id=user_id,
                status=None,
                notified_parties=[]
            )
        
        alert_data = alerts_storage.get(alert_id)
        
        if not alert_data:
            return StatusResponse(
                user_id=user_id,
                alert_id=alert_id,
                status=None,
                notified_parties=[]
            )
        
        # Reconstruct response from stored data
        location = Location(
            latitude=alert_data["latitude"],
            longitude=alert_data["longitude"]
        )
        
        notified_parties = [
            NotifiedParty(**p) for p in alert_data.get("notified_parties", [])
        ]
        
        nearest_police = None
        if alert_data.get("nearest_police"):
            nearest_police = NearbyPlace(**alert_data["nearest_police"])
        
        nearest_shelter = None
        if alert_data.get("nearest_shelter"):
            nearest_shelter = NearbyPlace(**alert_data["nearest_shelter"])
        
        return StatusResponse(
            user_id=user_id,
            alert_id=alert_id,
            status=alert_data["status"],
            timestamp=alert_data["timestamp"],
            confidence=alert_data["confidence"],
            transcript=alert_data["transcript"],
            location=location,
            notified_parties=notified_parties,
            nearest_police=nearest_police,
            nearest_shelter=nearest_shelter
        )
        
    except Exception as e:
        logger.error(f"Error getting alert status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alert status: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "SafeHer AI Backend"
    }


@app.get("/test-email")
async def test_email():
    """Test email sending endpoint for debugging."""
    try:
        from src.email_alert import send_alert_email
        result = send_alert_email(
            transcript="help me please",
            confidence=0.92,
            latitude=12.9716,
            longitude=77.5946
        )
        print(f"[TEST EMAIL] Result: {result}")
        return {"email_sent": True, "result": str(result)}
    except Exception as e:
        print(f"[TEST EMAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"email_sent": False, "error": str(e)}


@app.get("/model-status")
async def get_model_status():
    """Get model loading status and configuration."""
    return {
        "acoustic_model": "loaded" if acoustic_model is not None else "not_loaded",
        "keyword_model": "loaded" if keyword_model is not None else "not_loaded",
        "gpu_enabled": gpu_available,
        "gpu_name": gpu_name,
        "threshold": 0.85
    }


@app.post("/detect")
async def detect(request: Request):
    """Detect distress using Gemini AI with rule-based fallback."""
    from src.ai_detector import analyze_with_ai
    
    body = await request.json()
    transcript = body.get("transcript", "").strip()
    audio_energy = float(body.get("audio_energy", 0.6))

    print(f"[/detect] '{transcript}' | energy={audio_energy}")

    result = analyze_with_ai(transcript, audio_energy)

    pct = round(result['confidence'] * 100)
    print(f"[/detect] {pct}% | {result['status']} | {result['reason']}")

    return result


@app.get("/contacts")
async def get_contacts():
    """Get saved emergency contacts."""
    try:
        contacts_data = load_contacts()
        return contacts_data
    except Exception as e:
        logger.error(f"Error getting contacts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get contacts: {str(e)}"
        )


@app.post("/contacts")
async def update_contacts(data: Dict):
    """
    Update emergency contacts.
    
    Expected format:
    {
        "user_name": "User Name",
        "contacts": [
            {"name": "Contact 1", "phone": "+91xxxxxxxxxx"},
            {"name": "Contact 2", "phone": "+91xxxxxxxxxx"}
        ]
    }
    """
    try:
        success = save_contacts(data)
        if success:
            return {"message": "Contacts saved successfully", "data": data}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save contacts"
            )
    except Exception as e:
        logger.error(f"Error updating contacts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update contacts: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint that serves the frontend index.html."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        return {
            "name": "SafeHer AI API",
            "version": "1.0.0",
            "endpoints": {
                "POST /alert": "Create and process distress alert",
                "POST /cancel": "Cancel pending alert",
                "GET /status/{user_id}": "Get alert status",
                "GET /health": "Health check"
            },
            "message": "Frontend not found. Please ensure frontend/index.html exists."
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
