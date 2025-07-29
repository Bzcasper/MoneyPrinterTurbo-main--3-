"""
Google Cloud Text-to-Speech Service Implementation

Provides high-quality neural voice synthesis through Google Cloud TTS API.
Features include SSML support, voice customization, and intelligent caching.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import hashlib
import json

try:
    from google.cloud import texttospeech
    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_TTS_AVAILABLE = False
    texttospeech = None

from .base_tts_service import BaseTTSService, TTSRequest, TTSResponse, VoiceInfo, TTSServiceError

logger = logging.getLogger(__name__)


class GoogleTTSService(BaseTTSService):
    """Google Cloud Text-to-Speech service implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not GOOGLE_TTS_AVAILABLE:
            raise TTSServiceError(
                "Google Cloud TTS library not available. Install with: pip install google-cloud-texttospeech",
                provider="google"
            )
        
        # Initialize Google TTS client
        try:
            if "credentials_path" in config:
                # Use service account file
                import os
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config["credentials_path"]
            
            self.client = texttospeech.TextToSpeechClient()
            logger.info("Google TTS client initialized successfully")
        except Exception as e:
            raise TTSServiceError(f"Failed to initialize Google TTS client: {e}", provider="google")
        
        # Service capabilities
        self.supports_ssml = True
        self.supports_emotions = False  # Google TTS doesn't have built-in emotions
        self.supports_characters = False
        self.max_text_length = 5000
        
        # Voice caching
        self._voices_cache = None
        self._cache_expiry = None
        self._cache_ttl_hours = config.get("voice_cache_ttl_hours", 24)
        
        # Request settings
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech using Google TTS
        
        Args:
            request: TTSRequest with text and voice parameters
            
        Returns:
            TTSResponse with synthesized audio
        """
        try:
            logger.info(f"Starting Google TTS synthesis for {len(request.text)} characters")
            
            # Validate request
            self._validate_request(request)
            
            # Prepare synthesis input
            if request.text.strip().startswith('<speak>'):
                # SSML input
                synthesis_input = texttospeech.SynthesisInput(ssml=request.text)
            else:
                # Plain text input
                synthesis_input = texttospeech.SynthesisInput(text=request.text)
            
            # Configure voice
            voice = texttospeech.VoiceSelectionParams(
                language_code=request.language_code,
                name=request.voice_name
            )
            
            # Add gender if specified
            if request.gender:
                gender_map = {
                    "male": texttospeech.SsmlVoiceGender.MALE,
                    "female": texttospeech.SsmlVoiceGender.FEMALE,
                    "neutral": texttospeech.SsmlVoiceGender.NEUTRAL
                }
                voice.ssml_gender = gender_map.get(request.gender.lower(), texttospeech.SsmlVoiceGender.NEUTRAL)
            
            # Configure audio output
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=request.speaking_rate,
                pitch=request.pitch,
                volume_gain_db=request.volume_gain,
                sample_rate_hertz=22050  # Standard quality
            )
            
            # Perform synthesis with retry logic
            response = await self._synthesize_with_retry(synthesis_input, voice, audio_config)
            
            # Calculate duration and generate subtitles
            duration = self._calculate_duration(response.audio_content)
            subtitle_data = self._extract_subtitle_data(request.text, duration)
            
            # Create voice info
            voice_info = VoiceInfo(
                name=request.voice_name,
                language=request.language_code,
                gender=request.gender,
                natural_sample_rate=22050
            )
            
            # Calculate quality score based on voice type and settings
            quality_score = self._calculate_quality_score(request, response)
            
            logger.info(f"Google TTS synthesis completed successfully in {duration:.2f}s")
            
            return TTSResponse(
                audio_content=response.audio_content,
                audio_format="mp3",
                duration=duration,
                voice_info=voice_info,
                subtitle_data=subtitle_data,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Google TTS synthesis failed: {e}")
            raise TTSServiceError(f"Google TTS synthesis failed: {e}", provider="google")
    
    def get_voices(self) -> List[VoiceInfo]:
        """
        Get available Google TTS voices with intelligent caching
        
        Returns:
            List of available VoiceInfo objects
        """
        # Check cache validity
        if self._voices_cache and self._cache_valid():
            logger.debug("Returning cached Google TTS voices")
            return self._voices_cache
        
        try:
            logger.info("Fetching Google TTS voices from API")
            voices_response = self.client.list_voices()
            
            self._voices_cache = []
            for voice in voices_response.voices:
                for language_code in voice.language_codes:
                    voice_info = VoiceInfo(
                        name=voice.name,
                        language=language_code,
                        gender=voice.ssml_gender.name.lower() if voice.ssml_gender else None,
                        natural_sample_rate=voice.natural_sample_rate_hertz
                    )
                    self._voices_cache.append(voice_info)
            
            # Update cache expiry
            self._cache_expiry = datetime.now() + timedelta(hours=self._cache_ttl_hours)
            
            logger.info(f"Fetched {len(self._voices_cache)} Google TTS voices")
            return self._voices_cache
            
        except Exception as e:
            logger.error(f"Failed to fetch Google TTS voices: {e}")
            return []
    
    def validate_config(self) -> bool:
        """
        Validate Google TTS configuration
        
        Returns:
            True if configuration is valid
        """
        try:
            # Test basic client functionality
            self.client.list_voices()
            return True
        except Exception as e:
            logger.error(f"Google TTS configuration validation failed: {e}")
            return False
    
    async def _synthesize_with_retry(self, synthesis_input, voice, audio_config) -> Any:
        """
        Perform synthesis with retry logic
        
        Args:
            synthesis_input: Google TTS synthesis input
            voice: Voice selection parameters
            audio_config: Audio configuration
            
        Returns:
            Google TTS response
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Run synthesis in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.synthesize_speech(
                        input=synthesis_input,
                        voice=voice,
                        audio_config=audio_config
                    )
                )
                return response
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Google TTS attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Google TTS failed after {self.max_retries} attempts")
        
        raise last_error
    
    def _validate_request(self, request: TTSRequest):
        """
        Validate TTS request parameters
        
        Args:
            request: TTSRequest to validate
            
        Raises:
            TTSServiceError: If request is invalid
        """
        if not request.text or not request.text.strip():
            raise TTSServiceError("Text cannot be empty", provider="google")
        
        if len(request.text) > self.max_text_length:
            raise TTSServiceError(
                f"Text length {len(request.text)} exceeds maximum {self.max_text_length}",
                provider="google"
            )
        
        if not request.voice_name:
            raise TTSServiceError("Voice name is required", provider="google")
        
        if not request.language_code:
            raise TTSServiceError("Language code is required", provider="google")
        
        # Validate rate limits
        if not 0.25 <= request.speaking_rate <= 4.0:
            raise TTSServiceError("Speaking rate must be between 0.25 and 4.0", provider="google")
        
        if not -20.0 <= request.pitch <= 20.0:
            raise TTSServiceError("Pitch must be between -20.0 and 20.0", provider="google")
    
    def _calculate_quality_score(self, request: TTSRequest, response: Any) -> float:
        """
        Calculate quality score for the synthesis
        
        Args:
            request: Original TTS request
            response: Google TTS response
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.8  # Base score for Google TTS
        
        # Boost for neural voices
        if "wavenet" in request.voice_name.lower() or "neural" in request.voice_name.lower():
            score += 0.15
        
        # Boost for standard speaking rate
        if 0.9 <= request.speaking_rate <= 1.1:
            score += 0.05
        
        # Small penalty for extreme pitch
        if abs(request.pitch) > 10.0:
            score -= 0.05
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
    
    def _cache_valid(self) -> bool:
        """Check if voice cache is still valid"""
        return (
            self._cache_expiry is not None and
            datetime.now() < self._cache_expiry
        )
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get Google TTS provider information"""
        info = super().get_provider_info()
        info.update({
            "display_name": "Google Cloud Text-to-Speech",
            "neural_voices": True,
            "premium_voices": True,
            "languages_supported": 100+,  # Approximate
            "pricing_model": "pay_per_character",
            "api_docs": "https://cloud.google.com/text-to-speech/docs"
        })
        return info
