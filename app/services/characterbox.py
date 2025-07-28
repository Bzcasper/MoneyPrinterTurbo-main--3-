"""
CharacterBox Service

This module provides integration with CharacterBox API for generating
character-based text-to-speech with various personality voices.
"""

import os
import json
import asyncio
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pathlib import Path

import requests
import aiohttp
from loguru import logger

from app.config import config
from app.utils import utils


class CharacterBoxService:
    """CharacterBox API service implementation for character-based TTS."""

    def __init__(self):
        """Initialize CharacterBox service with API configuration."""
        self.api_key = None
        self.base_url = None
        self.session = None
        self._initialize_service()

    def _initialize_service(self) -> None:
        """Initialize CharacterBox service with API credentials."""
        try:
            # Get CharacterBox configuration
            characterbox_config = config.app.get("characterbox", {})
            self.api_key = characterbox_config.get("api_key")
            self.base_url = characterbox_config.get("base_url", "https://api.characterbox.com/v1")
            
            # Optional configuration
            self.timeout = characterbox_config.get("timeout", 30)
            self.max_retries = characterbox_config.get("max_retries", 3)

            if not self.api_key:
                logger.warning("CharacterBox API key not configured")
                return

            logger.info("CharacterBox service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize CharacterBox service: {str(e)}")

    def is_available(self) -> bool:
        """Check if CharacterBox service is available and properly configured."""
        return self.api_key is not None and self.base_url is not None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper headers."""
        if self.session is None or self.session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "MoneyPrinterTurbo/2.0.0"
            }
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session

    async def get_available_characters(self) -> List[Dict[str, Any]]:
        """
        Get list of available characters from CharacterBox API.

        Returns:
            List of character configurations with metadata
        """
        if not self.is_available():
            logger.error("CharacterBox service not available")
            return []

        try:
            session = await self._get_session()
            url = f"{self.base_url}/characters"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    characters = data.get("characters", [])
                    
                    logger.info(f"Retrieved {len(characters)} CharacterBox characters")
                    return characters
                else:
                    error_msg = await response.text()
                    logger.error(f"Failed to get characters: {response.status} - {error_msg}")
                    return []

        except Exception as e:
            logger.error(f"Error getting CharacterBox characters: {str(e)}")
            return []

    async def get_character_info(self, character_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific character.

        Args:
            character_id: Unique identifier for the character

        Returns:
            Character information dictionary or None if not found
        """
        if not self.is_available():
            logger.error("CharacterBox service not available")
            return None

        try:
            session = await self._get_session()
            url = f"{self.base_url}/characters/{character_id}"

            async with session.get(url) as response:
                if response.status == 200:
                    character_info = await response.json()
                    logger.info(f"Retrieved character info for: {character_id}")
                    return character_info
                elif response.status == 404:
                    logger.warning(f"Character not found: {character_id}")
                    return None
                else:
                    error_msg = await response.text()
                    logger.error(f"Failed to get character info: {response.status} - {error_msg}")
                    return None

        except Exception as e:
            logger.error(f"Error getting character info: {str(e)}")
            return None

    async def synthesize_character_speech(
        self,
        text: str,
        character_id: str,
        voice_settings: Optional[Dict[str, Any]] = None,
        output_file: Optional[str] = None,
        audio_format: str = "mp3",
        quality: str = "high",
    ) -> Optional[str]:
        """
        Generate speech using a specific character voice.

        Args:
            text: Text to convert to speech
            character_id: Unique identifier for the character
            voice_settings: Optional voice customization settings
            output_file: Optional output file path
            audio_format: Audio format (mp3, wav, ogg)
            quality: Audio quality (low, medium, high)

        Returns:
            Path to generated audio file or None if failed
        """
        if not self.is_available():
            logger.error("CharacterBox service not available")
            return None

        try:
            session = await self._get_session()
            url = f"{self.base_url}/synthesize"

            # Prepare request payload
            payload = {
                "text": text,
                "character_id": character_id,
                "audio_format": audio_format,
                "quality": quality,
            }

            # Add voice settings if provided
            if voice_settings:
                payload["voice_settings"] = voice_settings

            logger.info(f"Synthesizing speech with character: {character_id}")

            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    # Check if response contains audio data or download URL
                    content_type = response.headers.get("content-type", "")
                    
                    if "audio" in content_type:
                        # Direct audio response
                        audio_data = await response.read()
                        
                        # Generate output file path if not provided
                        if not output_file:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_file = utils.temp_dir() / f"characterbox_{character_id}_{timestamp}.{audio_format}"
                        
                        output_file = Path(output_file)
                        output_file.parent.mkdir(parents=True, exist_ok=True)

                        # Save audio data
                        with open(output_file, "wb") as f:
                            f.write(audio_data)

                        logger.success(f"CharacterBox synthesis completed: {output_file}")
                        return str(output_file)
                    
                    else:
                        # JSON response with download URL
                        result = await response.json()
                        download_url = result.get("audio_url")
                        
                        if download_url:
                            return await self._download_audio(download_url, output_file, audio_format)
                        else:
                            logger.error("No audio URL in response")
                            return None

                else:
                    error_msg = await response.text()
                    logger.error(f"CharacterBox synthesis failed: {response.status} - {error_msg}")
                    return None

        except Exception as e:
            logger.error(f"CharacterBox synthesis error: {str(e)}")
            return None

    async def _download_audio(
        self, 
        download_url: str, 
        output_file: Optional[str], 
        audio_format: str
    ) -> Optional[str]:
        """
        Download audio file from provided URL.

        Args:
            download_url: URL to download audio from
            output_file: Optional output file path
            audio_format: Audio format extension

        Returns:
            Path to downloaded audio file or None if failed
        """
        try:
            session = await self._get_session()
            
            async with session.get(download_url) as response:
                if response.status == 200:
                    # Generate output file path if not provided
                    if not output_file:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = utils.temp_dir() / f"characterbox_download_{timestamp}.{audio_format}"
                    
                    output_file = Path(output_file)
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    # Download and save audio
                    audio_data = await response.read()
                    with open(output_file, "wb") as f:
                        f.write(audio_data)

                    logger.success(f"Audio downloaded: {output_file}")
                    return str(output_file)
                else:
                    logger.error(f"Failed to download audio: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Audio download error: {str(e)}")
            return None

    async def create_custom_character(
        self,
        name: str,
        description: str,
        voice_sample_file: str,
        personality_traits: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Create a custom character with voice sample.

        Args:
            name: Character name
            description: Character description
            voice_sample_file: Path to voice sample audio file
            personality_traits: Optional personality configuration

        Returns:
            Character ID if successful, None if failed
        """
        if not self.is_available():
            logger.error("CharacterBox service not available")
            return None

        try:
            if not os.path.exists(voice_sample_file):
                logger.error(f"Voice sample file not found: {voice_sample_file}")
                return None

            session = await self._get_session()
            url = f"{self.base_url}/characters/create"

            # Prepare form data
            data = aiohttp.FormData()
            data.add_field("name", name)
            data.add_field("description", description)
            
            if personality_traits:
                data.add_field("personality_traits", json.dumps(personality_traits))

            # Add voice sample file
            with open(voice_sample_file, "rb") as f:
                data.add_field(
                    "voice_sample",
                    f,
                    filename=os.path.basename(voice_sample_file),
                    content_type="audio/wav"
                )

                logger.info(f"Creating custom character: {name}")

                async with session.post(url, data=data) as response:
                    if response.status == 201:
                        result = await response.json()
                        character_id = result.get("character_id")
                        logger.success(f"Custom character created: {character_id}")
                        return character_id
                    else:
                        error_msg = await response.text()
                        logger.error(f"Failed to create character: {response.status} - {error_msg}")
                        return None

        except Exception as e:
            logger.error(f"Error creating custom character: {str(e)}")
            return None

    async def get_synthesis_history(
        self, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get synthesis history for the current API key.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of synthesis history records
        """
        if not self.is_available():
            logger.error("CharacterBox service not available")
            return []

        try:
            session = await self._get_session()
            url = f"{self.base_url}/history"
            params = {"limit": limit, "offset": offset}

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    history = data.get("history", [])
                    logger.info(f"Retrieved {len(history)} synthesis history records")
                    return history
                else:
                    error_msg = await response.text()
                    logger.error(f"Failed to get history: {response.status} - {error_msg}")
                    return []

        except Exception as e:
            logger.error(f"Error getting synthesis history: {str(e)}")
            return []

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the CharacterBox service status and configuration.

        Returns:
            Dictionary containing service information
        """
        service_info = {
            "service": "CharacterBox",
            "available": self.is_available(),
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
        }

        if self.is_available():
            service_info.update({
                "status": "configured",
                "api_key_configured": bool(self.api_key),
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            })
        else:
            service_info.update({
                "status": "not_configured",
                "error": "API key or base URL not configured"
            })

        return service_info

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()


# Global service instance
_characterbox_service = None


def get_characterbox_service() -> CharacterBoxService:
    """Get the global CharacterBox service instance."""
    global _characterbox_service
    if _characterbox_service is None:
        _characterbox_service = CharacterBoxService()
    return _characterbox_service


# Helper functions for backward compatibility
async def get_characterbox_characters() -> List[Dict[str, Any]]:
    """Get available CharacterBox characters."""
    service = get_characterbox_service()
    return await service.get_available_characters()


async def characterbox_synthesis(
    text: str,
    character_id: str,
    output_file: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """Synthesize speech using CharacterBox."""
    service = get_characterbox_service()
    return await service.synthesize_character_speech(
        text=text,
        character_id=character_id,
        output_file=output_file,
        **kwargs
    )


def characterbox_service_status() -> Dict[str, Any]:
    """Get CharacterBox service status."""
    service = get_characterbox_service()
    return service.get_service_info()