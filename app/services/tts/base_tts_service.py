"""
Base TTS Service Interface

Abstract base class defining the common interface for all TTS service implementations.
Ensures consistency across different providers while allowing for provider-specific features.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TTSServiceError(Exception):
    """Base exception for TTS service errors"""
    
    def __init__(self, message: str, provider: str = None, error_code: str = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code
        self.timestamp = datetime.utcnow()


class VoiceInfo:
    """Information about a TTS voice"""
    
    def __init__(self, name: str, language: str, gender: str = None, 
                 natural_sample_rate: int = None, character=None):
        self.name = name
        self.language = language
        self.gender = gender
        self.natural_sample_rate = natural_sample_rate
        self.character = character  # For CharacterBox voices
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "language": self.language,
            "gender": self.gender,
            "natural_sample_rate": self.natural_sample_rate,
            "character": self.character.__dict__ if self.character else None
        }


class TTSRequest:
    """TTS synthesis request"""
    
    def __init__(self, text: str, voice_name: str, language_code: str,
                 provider: str = "google", speaking_rate: float = 1.0,
                 pitch: float = 0.0, volume_gain: float = 0.0,
                 gender: str = None, character_id: str = None,
                 emotion: str = "neutral", voice_settings: Dict = None):
        self.text = text
        self.voice_name = voice_name
        self.language_code = language_code
        self.provider = provider
        self.speaking_rate = speaking_rate
        self.pitch = pitch
        self.volume_gain = volume_gain
        self.gender = gender
        self.character_id = character_id
        self.emotion = emotion
        self.voice_settings = voice_settings or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "voice_name": self.voice_name,
            "language_code": self.language_code,
            "provider": self.provider,
            "speaking_rate": self.speaking_rate,
            "pitch": self.pitch,
            "volume_gain": self.volume_gain,
            "gender": self.gender,
            "character_id": self.character_id,
            "emotion": self.emotion,
            "voice_settings": self.voice_settings
        }


class TTSResponse:
    """TTS synthesis response"""
    
    def __init__(self, audio_content: bytes, audio_format: str, duration: float,
                 voice_info: VoiceInfo, audio_file_path: str = None,
                 subtitle_data: List[Dict] = None, quality_score: float = None,
                 emotion_score: float = None, cache_hit: bool = False):
        self.audio_content = audio_content
        self.audio_format = audio_format
        self.duration = duration
        self.voice_info = voice_info
        self.audio_file_path = audio_file_path
        self.subtitle_data = subtitle_data or []
        self.quality_score = quality_score
        self.emotion_score = emotion_score
        self.cache_hit = cache_hit
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_format": self.audio_format,
            "duration": self.duration,
            "voice_info": self.voice_info.to_dict(),
            "audio_file_path": self.audio_file_path,
            "subtitle_data": self.subtitle_data,
            "quality_score": self.quality_score,
            "emotion_score": self.emotion_score,
            "cache_hit": self.cache_hit
        }


class BaseTTSService(ABC):
    """Abstract base class for TTS services"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__.lower().replace("ttsservice", "")
        logger.info(f"Initializing {self.provider_name} TTS service")
    
    @abstractmethod
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech from text
        
        Args:
            request: TTSRequest containing text and voice parameters
            
        Returns:
            TTSResponse with audio content and metadata
            
        Raises:
            TTSServiceError: If synthesis fails
        """
        pass
    
    @abstractmethod
    def get_voices(self) -> List[VoiceInfo]:
        """
        Get available voices for this provider
        
        Returns:
            List of VoiceInfo objects describing available voices
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate service configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
    
    def _calculate_duration(self, audio_content: bytes) -> float:
        """
        Calculate audio duration from content
        
        Args:
            audio_content: Raw audio bytes
            
        Returns:
            Duration in seconds (estimated)
        """
        # Simple estimation: assume average speaking rate
        # More accurate calculation would require audio analysis
        if not audio_content:
            return 0.0
        
        # Rough estimation: 1 byte per sample at 22kHz = ~45ms per 1000 bytes
        estimated_duration = len(audio_content) / 1000 * 0.045
        return max(0.1, estimated_duration)  # Minimum 0.1 seconds
    
    def _extract_subtitle_data(self, text: str, duration: float) -> List[Dict]:
        """
        Extract subtitle timing data from text
        
        Args:
            text: Input text
            duration: Total audio duration
            
        Returns:
            List of subtitle segments with timing
        """
        if not text or duration <= 0:
            return []
        
        # Simple word-based segmentation
        words = text.split()
        if not words:
            return []
        
        # Calculate timing per word
        words_per_second = len(words) / duration
        
        subtitles = []
        current_time = 0.0
        
        for i, word in enumerate(words):
            word_duration = 1.0 / words_per_second if words_per_second > 0 else 1.0
            
            subtitles.append({
                "start": round(current_time, 2),
                "end": round(current_time + word_duration, 2),
                "text": word,
                "word_index": i
            })
            
            current_time += word_duration
        
        return subtitles
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information and capabilities"""
        return {
            "name": self.provider_name,
            "class": self.__class__.__name__,
            "config_keys": list(self.config.keys()) if self.config else [],
            "supports_ssml": getattr(self, 'supports_ssml', False),
            "supports_emotions": getattr(self, 'supports_emotions', False),
            "supports_characters": getattr(self, 'supports_characters', False),
            "max_text_length": getattr(self, 'max_text_length', 5000)
        }
