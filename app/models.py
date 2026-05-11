"""
Pydantic models for SafeHer AI API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class AlertStatus(str, Enum):
    """Alert status enumeration."""
    PENDING = "pending"
    SENT = "sent"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class Location(BaseModel):
    """Location coordinates."""
    latitude: float
    longitude: float


class NearbyPlace(BaseModel):
    """Nearby place information."""
    name: str
    address: str
    distance_km: float
    place_id: str
    rating: Optional[float] = None
    phone: Optional[str] = None


class NotifiedParty(BaseModel):
    """Information about a party that was notified."""
    name: str
    contact: str
    method: str  # "sms", "email", "push"
    sent_at: datetime
    status: str = "sent"


class AlertRequest(BaseModel):
    """Request model for creating a distress alert."""
    user_id: str
    transcript: str
    confidence: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: Optional[datetime] = None


class AlertResponse(BaseModel):
    """Response model for alert creation."""
    alert_id: str
    status: AlertStatus
    message: str
    timestamp: datetime
    notified_parties: List[NotifiedParty] = []
    nearest_police: Optional[NearbyPlace] = None
    nearest_shelter: Optional[NearbyPlace] = None


class CancelRequest(BaseModel):
    """Request model for cancelling an alert."""
    user_id: str
    alert_id: Optional[str] = None


class CancelResponse(BaseModel):
    """Response model for alert cancellation."""
    success: bool
    message: str
    alert_id: Optional[str] = None


class StatusResponse(BaseModel):
    """Response model for alert status."""
    user_id: str
    alert_id: Optional[str] = None
    status: Optional[AlertStatus] = None
    timestamp: Optional[datetime] = None
    confidence: Optional[float] = None
    transcript: Optional[str] = None
    location: Optional[Location] = None
    notified_parties: List[NotifiedParty] = []
    nearest_police: Optional[NearbyPlace] = None
    nearest_shelter: Optional[NearbyPlace] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
