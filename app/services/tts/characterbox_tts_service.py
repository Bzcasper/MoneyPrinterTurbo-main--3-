"""
CharacterBox TTS Service

TTS service implementation that combines CharacterBox character personalities
with traditional TTS synthesis, providing character-based voice generation
with fallback to standard TTS providers.
"""

import asyncio
import logging
from typing import Dict, Any, List
import aiohttp

from ..tts.base_tts_service import BaseTTSService, TTSRequest, TTSResponse, VoiceInfo, TTSServiceError
from ..tts.tts_factory import TTSServiceFactory
from .characterbox_service import CharacterBoxService, CharacterBoxError
from .character_models import CharacterRequest, CharacterInfo

logger = logging.getLogger(__name__)


class CharacterBoxTTSService(BaseTTSService):
    """TTS service with CharacterBox character personalities"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize CharacterBox service
        characterbox_config = config.get("characterbox", {})
        self.characterbox = CharacterBoxService(characterbox_config)
        
        # Initialize fallback TTS service
        fallback_provider = config.get("fallback_provider", "google")
        fallback_config = config.get("fallback_config", {})
        
        try:
            self.fallback_tts = TTSServiceFactory.create_service(fallback_provider, fallback_config)
            logger.info(f"CharacterBox TTS initialized with {fallback_provider} fallback")
        except Exception as e:
            logger.warning(f"Failed to initialize fallback TTS: {e}")
            self.fallback_tts = None
        
        # Service capabilities
        self.supports_ssml = False  # CharacterBox typically doesn't support SSML
        self.supports_emotions = True
        self.supports_characters = True
        self.max_text_length = 5000
        
        # Character cache
        self._character_voices_cache = None
        self._voices_cache_valid = False
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech with character personality or fallback to standard TTS
        
        Args:
            request: TTSRequest containing text and character/voice parameters
            
        Returns:
            TTSResponse with synthesized audio
        """
        # Check if this is a character-based request
        if hasattr(request, 'character_id') and request.character_id:
            try:
                return await self._synthesize_with_character(request)
            except CharacterBoxError as e:
                logger.warning(f"CharacterBox synthesis failed, falling back to standard TTS: {e}")
                if self.fallback_tts:
                    return await self.fallback_tts.synthesize(request)
                else:
                    raise TTSServiceError(f"CharacterBox synthesis failed and no fallback available: {e}")
        
        # Standard TTS request - use fallback
        if self.fallback_tts:
            return await self.fallback_tts.synthesize(request)
        else:
            raise TTSServiceError("No fallback TTS service available for non-character requests")
    
    async def _synthesize_with_character(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech using CharacterBox character
        
        Args:
            request: TTSRequest with character_id
            
        Returns:
            TTSResponse with character voice
        """
        # Convert TTS request to CharacterBox request
        character_request = CharacterRequest(
            character_id=request.character_id,
            text=request.text,
            emotion=getattr(request, 'emotion', 'neutral'),
            voice_settings=self._convert_tts_to_character_settings(request),
            language_code=request.language_code,
            speaking_rate=request.speaking_rate,
            pitch=request.pitch,
            output_format="mp3"
        )
        
        # Generate character speech
        async with self.characterbox:
            char_response = await self.characterbox.generate_character_speech(character_request)
        
        # Create TTS response from CharacterBox response
        voice_info = VoiceInfo(
            name=char_response.character_info.name,
            language=request.language_code,
            character=char_response.character_info
        )
        
        # Extract subtitle data from the text and duration
        subtitle_data = self._extract_subtitle_data(request.text, char_response.duration)
        
        return TTSResponse(
            audio_content=char_response.audio_content or b"",
            audio_format="mp3",
            duration=char_response.duration,
            voice_info=voice_info,
            subtitle_data=subtitle_data,
            quality_score=char_response.quality_score,
            emotion_score=char_response.emotion_score
        )
    
    def get_voices(self) -> List[VoiceInfo]:
        """
        Get available character voices and fallback voices
        
        Returns:
            List of VoiceInfo objects including characters and standard voices
        """
        voices = []
        
        # Get CharacterBox characters as voices
        try:
            # This would need to be called in an async context
            # For now, return cached character voices if available
            if self._character_voices_cache and self._voices_cache_valid:
                voices.extend(self._character_voices_cache)
        except Exception as e:
            logger.warning(f"Failed to fetch CharacterBox voices: {e}")
        
        # Get fallback TTS voices
        if self.fallback_tts:
            try:
                fallback_voices = self.fallback_tts.get_voices()
                voices.extend(fallback_voices)
            except Exception as e:
                logger.warning(f"Failed to fetch fallback voices: {e}")
        
        return voices
    
    async def get_voices_async(self) -> List[VoiceInfo]:
        """
        Async version of get_voices that can fetch CharacterBox characters
        
        Returns:
            List of VoiceInfo objects including characters and standard voices
        """
        voices = []
        
        # Get CharacterBox characters as voices
        try:
            async with self.characterbox:
                characters = await self.characterbox.get_characters()
                
                character_voices = []
                for character in characters:
                    voice_info = VoiceInfo(
                        name=character.name,
                        language=character.language_codes[0] if character.language_codes else "en-US",
                        character=character
                    )
                    character_voices.append(voice_info)
                
                voices.extend(character_voices)
                
                # Cache character voices
                self._character_voices_cache = character_voices
                self._voices_cache_valid = True
                
        except Exception as e:
            logger.warning(f"Failed to fetch CharacterBox characters: {e}")
        
        # Get fallback TTS voices
        if self.fallback_tts:
            try:
                fallback_voices = self.fallback_tts.get_voices()
                voices.extend(fallback_voices)
            except Exception as e:
                logger.warning(f"Failed to fetch fallback voices: {e}")
        
        return voices
    
    def validate_config(self) -> bool:
        """
        Validate CharacterBox and fallback TTS configuration
        
        Returns:
            True if at least one service is properly configured
        """
        characterbox_valid = False
        fallback_valid = False
        
        # Test CharacterBox configuration
        try:
            if self.characterbox.api_key:
                characterbox_valid = True
        except Exception as e:
            logger.warning(f"CharacterBox configuration invalid: {e}")
        
        # Test fallback TTS configuration
        if self.fallback_tts:
            try:
                fallback_valid = self.fallback_tts.validate_config()
            except Exception as e:
                logger.warning(f"Fallback TTS configuration invalid: {e}")
        
        return characterbox_valid or fallback_valid
    
    def _convert_tts_to_character_settings(self, request: TTSRequest) -> Dict[str, Any]:
        """
        Convert TTS request parameters to CharacterBox voice settings
        
        Args:
            request: TTSRequest with voice parameters
            
        Returns:
            Dictionary of CharacterBox voice settings
        """
        settings = getattr(request, 'voice_settings', {}).copy()
        
        # Map TTS parameters to CharacterBox settings
        settings.update({
            "speaking_rate": request.speaking_rate,
            "pitch": request.pitch,
            "volume_gain": request.volume_gain
        })
        
        # Add emotion mapping if present
        if hasattr(request, 'emotion'):
            settings["emotion"] = request.emotion
        
        return settings
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get CharacterBox TTS provider information"""
        info = super().get_provider_info()
        info.update({
            "display_name": "CharacterBox Character Voices",
            "neural_voices": True,
            "character_voices": True,
            "emotion_support": True,
            "conversation_support": True,
            "fallback_provider": self.fallback_tts.provider_name if self.fallback_tts else None,
            "api_docs": "https://docs.characterbox.ai"
        })
        return info
    
    async def get_character_info(self, character_id: str) -> CharacterInfo:
        """
        Get detailed information about a specific character
        
        Args:
            character_id: Character identifier
            
        Returns:
            CharacterInfo object
        """
        async with self.characterbox:
            character = await self.characterbox.get_character_by_id(character_id)
            if not character:
                raise TTSServiceError(f"Character not found: {character_id}")
            return character
    
    async def get_supported_emotions(self, character_id: str) -> List[str]:
        """
        Get emotions supported by a specific character
        
        Args:
            character_id: Character identifier
            
        Returns:
            List of supported emotion names
        """
        character = await self.get_character_info(character_id)
        return character.supported_emotions
    
    async def close(self):
        """Close the service and cleanup resources"""
        if self.characterbox:
            await self.characterbox.close()
        
        # Note: We don't close fallback_tts as it might be shared
