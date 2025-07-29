"""
CharacterBox Service Implementation

Service for integrating with CharacterBox API to provide character-based voice synthesis,
personality-driven speech generation, and multi-character conversations.
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .character_models import (
    CharacterInfo, CharacterRequest, CharacterResponse,
    ConversationRequest, ConversationResponse, CharacterUsageStats
)

logger = logging.getLogger(__name__)


class CharacterBoxError(Exception):
    """Exception for CharacterBox service errors"""
    
    def __init__(self, message: str, error_code: str = None, character_id: str = None):
        super().__init__(message)
        self.error_code = error_code
        self.character_id = character_id
        self.timestamp = datetime.utcnow()


class CharacterBoxService:
    """Service for CharacterBox character interactions"""
    
    def __init__(self, api_config: Dict[str, Any]):
        self.api_key = api_config.get("api_key")
        self.base_url = api_config.get("base_url", "https://api.characterbox.ai")
        self.timeout = api_config.get("timeout", 30)
        self.max_retries = api_config.get("max_retries", 3)
        self.rate_limit_per_minute = api_config.get("rate_limit_per_minute", 60)
        
        # Caching
        self._characters_cache = None
        self._cache_expiry = None
        self._cache_ttl_hours = api_config.get("cache_ttl_hours", 6)
        
        # Rate limiting
        self._request_times = []
        
        # Session for connection pooling
        self._session = None
        
        if not self.api_key:
            raise CharacterBoxError("CharacterBox API key is required")
        
        logger.info("CharacterBox service initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        if not self._session or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "MoneyPrinterTurbo/1.0"
                }
            )
    
    async def get_characters(self) -> List[CharacterInfo]:
        """
        Retrieve available characters with caching
        
        Returns:
            List of available CharacterInfo objects
        """
        # Check cache
        if self._characters_cache and self._cache_valid():
            logger.debug("Returning cached characters")
            return self._characters_cache
        
        try:
            logger.info("Fetching characters from CharacterBox API")
            response = await self._make_request("GET", "/v1/characters")
            
            characters_data = response.get("characters", [])
            self._characters_cache = [
                CharacterInfo(**char_data) for char_data in characters_data
            ]
            
            # Update cache expiry
            self._cache_expiry = datetime.now() + timedelta(hours=self._cache_ttl_hours)
            
            logger.info(f"Fetched {len(self._characters_cache)} characters")
            return self._characters_cache
            
        except Exception as e:
            logger.error(f"Failed to fetch characters: {e}")
            raise CharacterBoxError(f"Character fetch failed: {e}")
    
    async def get_character_by_id(self, character_id: str) -> Optional[CharacterInfo]:
        """
        Get specific character by ID
        
        Args:
            character_id: Character identifier
            
        Returns:
            CharacterInfo object or None if not found
        """
        characters = await self.get_characters()
        for character in characters:
            if character.character_id == character_id:
                return character
        return None
    
    async def generate_character_speech(self, request: CharacterRequest) -> CharacterResponse:
        """
        Generate speech with character personality
        
        Args:
            request: CharacterRequest with text and character settings
            
        Returns:
            CharacterResponse with audio and metadata
        """
        start_time = time.time()
        
        try:
            # Validate character exists
            character = await self.get_character_by_id(request.character_id)
            if not character:
                raise CharacterBoxError(f"Character not found: {request.character_id}", character_id=request.character_id)
            
            # Validate emotion is supported
            if request.emotion not in character.supported_emotions:
                logger.warning(f"Emotion '{request.emotion}' not supported by character {request.character_id}, using 'neutral'")
                request.emotion = "neutral"
            
            # Prepare request payload
            payload = {
                "character_id": request.character_id,
                "text": request.text,
                "emotion": request.emotion,
                "voice_settings": {
                    **character.voice_settings,
                    **request.voice_settings,
                    "speaking_rate": request.speaking_rate,
                    "pitch": request.pitch
                },
                "language_code": request.language_code,
                "output_format": request.output_format
            }
            
            logger.info(f"Generating speech for character {request.character_id} with emotion {request.emotion}")
            response = await self._make_request("POST", "/v1/synthesize", json=payload)
            
            # Download audio content if URL provided
            audio_content = None
            if response.get("audio_url"):
                audio_content = await self._download_audio(response["audio_url"])
            
            generation_time = time.time() - start_time
            
            # Create response
            character_response = CharacterResponse(
                audio_url=response["audio_url"],
                audio_content=audio_content,
                character_info=character,
                duration=response.get("duration", 0.0),
                emotion_score=response.get("emotion_score", 0.0),
                quality_score=response.get("quality_score", 0.8),
                generation_time=generation_time
            )
            
            logger.info(f"Character speech generated successfully in {generation_time:.2f}s")
            return character_response
            
        except Exception as e:
            logger.error(f"Character speech generation failed: {e}")
            raise CharacterBoxError(f"Speech generation failed: {e}", character_id=request.character_id)
    
    async def create_conversation(self, request: ConversationRequest) -> ConversationResponse:
        """
        Create multi-character conversation
        
        Args:
            request: ConversationRequest with characters and script
            
        Returns:
            ConversationResponse with conversation audio and metadata
        """
        try:
            # Validate all characters exist
            characters = await self.get_characters()
            character_map = {char.character_id: char for char in characters}
            
            for char_id in request.characters:
                if char_id not in character_map:
                    raise CharacterBoxError(f"Character not found: {char_id}")
            
            # Prepare conversation payload
            payload = {
                "characters": request.characters,
                "script": request.script,
                "conversation_type": request.conversation_type,
                "scene_setting": request.scene_setting,
                "emotion_flow": request.emotion_flow,
                "voice_settings": request.voice_settings
            }
            
            logger.info(f"Creating conversation with {len(request.characters)} characters")
            response = await self._make_request("POST", "/v1/conversations", json=payload)
            
            # Process response
            conversation_response = ConversationResponse(**response)
            
            logger.info(f"Conversation created successfully: {conversation_response.conversation_id}")
            return conversation_response
            
        except Exception as e:
            logger.error(f"Conversation creation failed: {e}")
            raise CharacterBoxError(f"Conversation creation failed: {e}")
    
    async def get_character_usage_stats(self, character_id: str) -> CharacterUsageStats:
        """
        Get usage statistics for a character
        
        Args:
            character_id: Character identifier
            
        Returns:
            CharacterUsageStats object
        """
        try:
            response = await self._make_request("GET", f"/v1/characters/{character_id}/stats")
            return CharacterUsageStats(**response)
        except Exception as e:
            logger.error(f"Failed to get usage stats for character {character_id}: {e}")
            # Return empty stats on error
            return CharacterUsageStats(character_id=character_id)
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request to CharacterBox API with rate limiting and retry logic
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            Response data as dictionary
        """
        await self._ensure_session()
        await self._enforce_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                async with self._session.request(method, url, **kwargs) as response:
                    if response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    if response.status >= 400:
                        error_text = await response.text()
                        raise CharacterBoxError(
                            f"API request failed: {response.status} - {error_text}",
                            error_code=str(response.status)
                        )
                    
                    return await response.json()
                    
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request timeout, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise CharacterBoxError(f"Request timed out after {self.max_retries} attempts")
            
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    break
        
        raise CharacterBoxError(f"Request failed after {self.max_retries} attempts: {last_error}")
    
    async def _download_audio(self, audio_url: str) -> bytes:
        """
        Download audio content from URL
        
        Args:
            audio_url: URL to audio file
            
        Returns:
            Raw audio bytes
        """
        try:
            await self._ensure_session()
            async with self._session.get(audio_url) as response:
                if response.status != 200:
                    raise CharacterBoxError(f"Failed to download audio: {response.status}")
                return await response.read()
        except Exception as e:
            logger.error(f"Audio download failed: {e}")
            raise CharacterBoxError(f"Audio download failed: {e}")
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting for API requests"""
        now = time.time()
        
        # Remove old timestamps
        cutoff = now - 60  # 1 minute ago
        self._request_times = [t for t in self._request_times if t > cutoff]
        
        # Check if we're at the limit
        if len(self._request_times) >= self.rate_limit_per_minute:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self._request_times.append(now)
    
    def _cache_valid(self) -> bool:
        """Check if character cache is still valid"""
        return (
            self._cache_expiry is not None and
            datetime.now() < self._cache_expiry
        )
    
    async def close(self):
        """Close the service and cleanup resources"""
        if self._session:
            await self._session.close()
        logger.info("CharacterBox service closed")
