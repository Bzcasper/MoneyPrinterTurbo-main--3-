"""
Google Text-to-Speech Service

This module provides integration with Google Cloud Text-to-Speech API
for generating high-quality speech audio from text.
"""

import os
import json
import asyncio
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger
from google.cloud import texttospeech
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions

from app.config import config
from app.utils import utils


class GoogleTTSService:
    """Google Cloud Text-to-Speech service implementation."""

    def __init__(self):
        """Initialize Google TTS service with authentication and configuration."""
        self.client = None
        self.credentials = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Google TTS client with proper authentication."""
        try:
            # Get service account configuration
            service_account_config = config.app.get("google_tts", {})
            credentials_path = service_account_config.get("credentials_path")
            credentials_json = service_account_config.get("credentials_json")
            project_id = service_account_config.get("project_id")

            if credentials_path and os.path.exists(credentials_path):
                # Use service account file
                self.credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
                logger.info(f"Google TTS initialized with credentials file: {credentials_path}")
            elif credentials_json:
                # Use inline JSON credentials
                if isinstance(credentials_json, str):
                    credentials_info = json.loads(credentials_json)
                else:
                    credentials_info = credentials_json
                
                self.credentials = service_account.Credentials.from_service_account_info(
                    credentials_info
                )
                logger.info("Google TTS initialized with inline credentials")
            elif project_id:
                # Use default credentials with explicit project
                os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
                self.credentials = None
                logger.info(f"Google TTS initialized with default credentials for project: {project_id}")
            else:
                logger.warning("Google TTS not configured - missing credentials or project ID")
                return

            # Initialize the client
            if self.credentials:
                self.client = texttospeech.TextToSpeechClient(credentials=self.credentials)
            else:
                self.client = texttospeech.TextToSpeechClient()

            logger.success("Google TTS client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Google TTS client: {str(e)}")
            self.client = None

    def is_available(self) -> bool:
        """Check if Google TTS service is available and properly configured."""
        return self.client is not None

    def get_available_voices(self, language_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available voices from Google TTS.

        Args:
            language_code: Optional language code filter (e.g., 'en-US', 'zh-CN')

        Returns:
            List of voice configurations
        """
        if not self.is_available():
            logger.error("Google TTS client not available")
            return []

        try:
            # Perform the list voices request
            voices = self.client.list_voices()
            voice_list = []

            for voice in voices.voices:
                # Filter by language if specified
                if language_code and not any(
                    lang.startswith(language_code) for lang in voice.language_codes
                ):
                    continue

                voice_info = {
                    "name": voice.name,
                    "language_codes": list(voice.language_codes),
                    "ssml_gender": voice.ssml_gender.name,
                    "natural_sample_rate_hertz": voice.natural_sample_rate_hertz,
                }
                voice_list.append(voice_info)

            logger.info(f"Retrieved {len(voice_list)} Google TTS voices")
            return voice_list

        except Exception as e:
            logger.error(f"Failed to get Google TTS voices: {str(e)}")
            return []

    async def synthesize_speech(
        self,
        text: str,
        voice_name: str = "en-US-Neural2-C",
        language_code: str = "en-US",
        audio_encoding: str = "MP3",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        volume_gain_db: float = 0.0,
        output_file: Optional[str] = None,
    ) -> Optional[str]:
        """
        Convert text to speech using Google TTS.

        Args:
            text: Text to convert to speech
            voice_name: Voice name (e.g., 'en-US-Neural2-C')
            language_code: Language code (e.g., 'en-US')
            audio_encoding: Audio format (MP3, WAV, OGG_OPUS)
            speaking_rate: Speaking rate (0.25 to 4.0)
            pitch: Voice pitch (-20.0 to 20.0)
            volume_gain_db: Volume adjustment (-96.0 to 16.0 dB)
            output_file: Optional output file path

        Returns:
            Path to generated audio file or None if failed
        """
        if not self.is_available():
            logger.error("Google TTS client not available")
            return None

        try:
            # Validate parameters
            speaking_rate = max(0.25, min(4.0, speaking_rate))
            pitch = max(-20.0, min(20.0, pitch))
            volume_gain_db = max(-96.0, min(16.0, volume_gain_db))

            # Set up the input text
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Configure voice parameters
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )

            # Map audio encoding string to enum
            encoding_map = {
                "MP3": texttospeech.AudioEncoding.MP3,
                "WAV": texttospeech.AudioEncoding.LINEAR16,
                "OGG_OPUS": texttospeech.AudioEncoding.OGG_OPUS,
            }
            audio_encoding_enum = encoding_map.get(audio_encoding, texttospeech.AudioEncoding.MP3)

            # Configure audio output
            audio_config = texttospeech.AudioConfig(
                audio_encoding=audio_encoding_enum,
                speaking_rate=speaking_rate,
                pitch=pitch,
                volume_gain_db=volume_gain_db,
            )

            # Perform the text-to-speech request
            logger.info(f"Synthesizing speech: voice={voice_name}, rate={speaking_rate}")
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )

            # Generate output file path if not provided
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                extension = audio_encoding.lower().replace("_", ".")
                if extension == "linear16":
                    extension = "wav"
                output_file = utils.temp_dir() / f"google_tts_{timestamp}.{extension}"
            
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save audio content to file
            with open(output_file, "wb") as audio_file:
                audio_file.write(response.audio_content)

            logger.success(f"Google TTS synthesis completed: {output_file}")
            return str(output_file)

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Google TTS API error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Google TTS synthesis failed: {str(e)}")
            return None

    def create_ssml_text(
        self,
        text: str,
        emphasis_words: Optional[List[str]] = None,
        break_seconds: Optional[float] = None,
        prosody_rate: Optional[str] = None,
        prosody_pitch: Optional[str] = None,
    ) -> str:
        """
        Create SSML (Speech Synthesis Markup Language) formatted text.

        Args:
            text: Base text content
            emphasis_words: Words to emphasize
            break_seconds: Pause duration in seconds
            prosody_rate: Speaking rate ('x-slow', 'slow', 'medium', 'fast', 'x-fast')
            prosody_pitch: Voice pitch ('x-low', 'low', 'medium', 'high', 'x-high')

        Returns:
            SSML formatted text
        """
        ssml_text = text

        # Add emphasis to specific words
        if emphasis_words:
            for word in emphasis_words:
                ssml_text = ssml_text.replace(
                    word, f'<emphasis level="strong">{word}</emphasis>'
                )

        # Add prosody changes
        if prosody_rate or prosody_pitch:
            prosody_attrs = []
            if prosody_rate:
                prosody_attrs.append(f'rate="{prosody_rate}"')
            if prosody_pitch:
                prosody_attrs.append(f'pitch="{prosody_pitch}"')
            
            prosody_tag = f'<prosody {" ".join(prosody_attrs)}>'
            ssml_text = f"{prosody_tag}{ssml_text}</prosody>"

        # Add breaks
        if break_seconds:
            break_tag = f'<break time="{break_seconds}s"/>'
            ssml_text = f"{break_tag}{ssml_text}"

        # Wrap in SSML speak tag
        ssml_text = f'<speak>{ssml_text}</speak>'

        return ssml_text

    async def synthesize_ssml_speech(
        self,
        ssml_text: str,
        voice_name: str = "en-US-Neural2-C",
        language_code: str = "en-US",
        audio_encoding: str = "MP3",
        output_file: Optional[str] = None,
    ) -> Optional[str]:
        """
        Convert SSML text to speech using Google TTS.

        Args:
            ssml_text: SSML formatted text
            voice_name: Voice name
            language_code: Language code
            audio_encoding: Audio format
            output_file: Optional output file path

        Returns:
            Path to generated audio file or None if failed
        """
        if not self.is_available():
            logger.error("Google TTS client not available")
            return None

        try:
            # Set up the SSML input
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

            # Configure voice parameters
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )

            # Map audio encoding
            encoding_map = {
                "MP3": texttospeech.AudioEncoding.MP3,
                "WAV": texttospeech.AudioEncoding.LINEAR16,
                "OGG_OPUS": texttospeech.AudioEncoding.OGG_OPUS,
            }
            audio_encoding_enum = encoding_map.get(audio_encoding, texttospeech.AudioEncoding.MP3)

            # Configure audio output
            audio_config = texttospeech.AudioConfig(audio_encoding=audio_encoding_enum)

            # Perform the text-to-speech request
            logger.info(f"Synthesizing SSML speech: voice={voice_name}")
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )

            # Generate output file path if not provided
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                extension = audio_encoding.lower().replace("_", ".")
                if extension == "linear16":
                    extension = "wav"
                output_file = utils.temp_dir() / f"google_ssml_tts_{timestamp}.{extension}"
            
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save audio content to file
            with open(output_file, "wb") as audio_file:
                audio_file.write(response.audio_content)

            logger.success(f"Google SSML TTS synthesis completed: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Google SSML TTS synthesis failed: {str(e)}")
            return None

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the Google TTS service status and configuration.

        Returns:
            Dictionary containing service information
        """
        service_info = {
            "service": "Google Cloud Text-to-Speech",
            "available": self.is_available(),
            "timestamp": datetime.now().isoformat(),
        }

        if self.is_available():
            try:
                # Test connection by getting a small list of voices
                voices = self.client.list_voices()
                service_info.update({
                    "status": "connected",
                    "total_voices": len(voices.voices),
                    "languages": list(set(
                        lang for voice in voices.voices for lang in voice.language_codes
                    ))[:10],  # First 10 languages
                })
            except Exception as e:
                service_info.update({
                    "status": "error",
                    "error": str(e)
                })
        else:
            service_info.update({
                "status": "not_configured",
                "error": "Google TTS client not initialized"
            })

        return service_info


# Global service instance
_google_tts_service = None


def get_google_tts_service() -> GoogleTTSService:
    """Get the global Google TTS service instance."""
    global _google_tts_service
    if _google_tts_service is None:
        _google_tts_service = GoogleTTSService()
    return _google_tts_service


# Helper functions for backward compatibility
def get_google_tts_voices(language_code: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get available Google TTS voices."""
    service = get_google_tts_service()
    return service.get_available_voices(language_code)


async def google_tts_synthesis(
    text: str,
    voice_name: str = "en-US-Neural2-C",
    language_code: str = "en-US",
    output_file: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """Synthesize speech using Google TTS."""
    service = get_google_tts_service()
    return await service.synthesize_speech(
        text=text,
        voice_name=voice_name,
        language_code=language_code,
        output_file=output_file,
        **kwargs
    )


def google_tts_service_status() -> Dict[str, Any]:
    """Get Google TTS service status."""
    service = get_google_tts_service()
    return service.get_service_info()