"""
CharacterBox Data Models

Pydantic models for CharacterBox API requests and responses.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class CharacterInfo(BaseModel):
    """Information about a CharacterBox character"""
    
    character_id: str = Field(..., description="Unique character identifier")
    name: str = Field(..., description="Character display name")
    description: Optional[str] = Field(None, description="Character description")
    personality_traits: List[str] = Field(default_factory=list, description="List of personality traits")
    voice_settings: Dict[str, Any] = Field(default_factory=dict, description="Voice configuration")
    supported_emotions: List[str] = Field(default_factory=list, description="Supported emotions")
    language_codes: List[str] = Field(default_factory=list, description="Supported languages")
    avatar_url: Optional[str] = Field(None, description="Character avatar image URL")
    is_premium: bool = Field(False, description="Whether character requires premium access")
    usage_count: int = Field(0, description="Number of times character has been used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CharacterRequest(BaseModel):
    """Request for character speech generation"""
    
    character_id: str = Field(..., description="Character to use for speech")
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    emotion: str = Field("neutral", description="Emotion to express")
    voice_settings: Dict[str, Any] = Field(default_factory=dict, description="Voice customization")
    language_code: str = Field("en-US", description="Language for synthesis")
    speaking_rate: float = Field(1.0, ge=0.25, le=4.0, description="Speaking rate multiplier")
    pitch: float = Field(0.0, ge=-20.0, le=20.0, description="Pitch adjustment in semitones")
    output_format: str = Field("mp3", description="Audio output format")


class CharacterResponse(BaseModel):
    """Response from character speech generation"""
    
    audio_url: str = Field(..., description="URL to generated audio file")
    audio_content: Optional[bytes] = Field(None, description="Raw audio content")
    character_info: CharacterInfo = Field(..., description="Character information")
    duration: float = Field(..., description="Audio duration in seconds")
    emotion_score: float = Field(0.0, description="Emotion confidence score")
    quality_score: float = Field(0.0, description="Audio quality score")
    generation_time: float = Field(0.0, description="Time taken to generate in seconds")
    
    class Config:
        arbitrary_types_allowed = True


class ConversationRequest(BaseModel):
    """Request for multi-character conversation"""
    
    characters: List[str] = Field(..., min_items=2, description="Character IDs for conversation")
    script: str = Field(..., min_length=1, description="Conversation script")
    conversation_type: str = Field("dialogue", description="Type of conversation")
    scene_setting: Optional[str] = Field(None, description="Scene or context setting")
    emotion_flow: Dict[str, str] = Field(default_factory=dict, description="Emotion mapping for characters")
    voice_settings: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Per-character voice settings")


class ConversationResponse(BaseModel):
    """Response from conversation generation"""
    
    conversation_id: str = Field(..., description="Unique conversation identifier")
    audio_segments: List[Dict[str, Any]] = Field(..., description="List of audio segments")
    total_duration: float = Field(..., description="Total conversation duration")
    character_parts: Dict[str, List[Dict]] = Field(..., description="Parts spoken by each character")
    combined_audio_url: Optional[str] = Field(None, description="URL to combined audio")
    transcript: List[Dict[str, Any]] = Field(..., description="Conversation transcript with timing")


class VoiceSettings(BaseModel):
    """Voice customization settings"""
    
    pitch: float = Field(0.0, ge=-20.0, le=20.0, description="Pitch adjustment")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speech speed")
    volume: float = Field(1.0, ge=0.1, le=2.0, description="Volume level")
    tone: str = Field("neutral", description="Voice tone/style")
    accent: Optional[str] = Field(None, description="Accent or regional variant")


class EmotionMapping(BaseModel):
    """Emotion configuration for character speech"""
    
    primary_emotion: str = Field("neutral", description="Primary emotion")
    intensity: float = Field(0.5, ge=0.0, le=1.0, description="Emotion intensity")
    secondary_emotions: List[str] = Field(default_factory=list, description="Secondary emotions")
    emotion_transitions: Dict[str, float] = Field(default_factory=dict, description="Emotion timing")


class CharacterUsageStats(BaseModel):
    """Usage statistics for characters"""
    
    character_id: str = Field(..., description="Character identifier")
    total_requests: int = Field(0, description="Total synthesis requests")
    successful_requests: int = Field(0, description="Successful requests")
    average_duration: float = Field(0.0, description="Average audio duration")
    total_characters_processed: int = Field(0, description="Total text characters processed")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    popular_emotions: List[str] = Field(default_factory=list, description="Most used emotions")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
