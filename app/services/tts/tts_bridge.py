"""
TTS Service Migration Bridge

Provides backward compatibility bridge between the existing video generation
pipeline and the new TTS service architecture. This ensures existing code
can seamlessly use the new TTS services without breaking changes.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import os

from .tts_factory import TTSServiceFactory
from .base_tts_service import TTSRequest, TTSResponse, TTSServiceError
from app.config.config import load_config

logger = logging.getLogger(__name__)


class TTSServiceBridge:
    """
    Migration bridge for TTS services
    
    Provides compatibility layer between existing voice.py functions
    and new TTS service architecture.
    """
    
    def __init__(self):
        self.config = load_config()
        self._service_cache = {}
        self._default_provider = "edge"  # Default to Edge TTS for compatibility
        
        logger.info("TTS Service Bridge initialized")
    
    def get_tts_service(self, provider: str = None):
        """
        Get a TTS service instance with caching
        
        Args:
            provider: TTS provider name (edge, google, siliconflow, gpt_sovits)
            
        Returns:
            TTS service instance
        """
        if provider is None:
            provider = self._default_provider
        
        if provider in self._service_cache:
            return self._service_cache[provider]
        
        try:
            # Get provider-specific configuration
            tts_config = self.config.get("tts", {})
            provider_config = tts_config.get(provider, {})
            
            # Create service instance
            service = TTSServiceFactory.create_service(provider, provider_config)
            
            # Cache the service
            self._service_cache[provider] = service
            
            logger.info(f"Created and cached TTS service: {provider}")
            return service
            
        except Exception as e:
            logger.error(f"Failed to create TTS service {provider}: {e}")
            
            # Fallback to Edge TTS if other providers fail
            if provider != "edge":
                logger.info("Falling back to Edge TTS")
                return self.get_tts_service("edge")
            
            raise TTSServiceError(f"Failed to initialize TTS service: {e}")
    
    async def synthesize_async(
        self,
        text: str,
        voice: str,
        provider: str = None,
        speed: float = 1.0,
        volume_gain: float = 0.0,
        language: str = "en-US",
        output_file: str = None
    ) -> TTSResponse:
        """
        Async TTS synthesis with new service architecture
        
        Args:
            text: Text to synthesize
            voice: Voice name/identifier
            provider: TTS provider (auto-detected from voice if not specified)
            speed: Speaking rate (0.25 to 4.0)
            volume_gain: Volume adjustment in dB
            language: Language code
            output_file: Optional output file path
            
        Returns:
            TTSResponse with synthesized audio
        """
        try:
            # Auto-detect provider from voice name if not specified
            if provider is None:
                provider = self._detect_provider_from_voice(voice)
            
            # Get TTS service
            service = self.get_tts_service(provider)
            
            # Create TTS request
            request = TTSRequest(
                text=text,
                voice_name=voice,
                language_code=language,
                speaking_rate=speed,
                volume_gain=volume_gain
            )
            
            # Synthesize speech
            response = await service.synthesize(request)
            
            # Save to file if specified
            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'wb') as f:
                    f.write(response.audio_content)
                logger.info(f"Saved TTS audio to: {output_file}")
            
            return response
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise TTSServiceError(f"TTS synthesis failed: {e}")
    
    def synthesize(
        self,
        text: str,
        voice: str,
        provider: str = None,
        speed: float = 1.0,
        volume_gain: float = 0.0,
        language: str = "en-US",
        output_file: str = None
    ) -> TTSResponse:
        """
        Synchronous TTS synthesis wrapper
        
        Args:
            text: Text to synthesize
            voice: Voice name/identifier
            provider: TTS provider
            speed: Speaking rate
            volume_gain: Volume adjustment in dB
            language: Language code
            output_file: Optional output file path
            
        Returns:
            TTSResponse with synthesized audio
        """
        try:
            # Run async synthesis in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.synthesize_async(text, voice, provider, speed, volume_gain, language, output_file)
                    )
                    return future.result()
            else:
                # Run directly
                return loop.run_until_complete(
                    self.synthesize_async(text, voice, provider, speed, volume_gain, language, output_file)
                )
        except Exception as e:
            logger.error(f"Synchronous TTS synthesis failed: {e}")
            raise TTSServiceError(f"TTS synthesis failed: {e}")
    
    def get_available_voices(self, provider: str = None) -> list:
        """
        Get available voices for a provider
        
        Args:
            provider: TTS provider name
            
        Returns:
            List of available voices
        """
        if provider is None:
            # Return voices from all providers
            all_voices = []
            for provider_name in TTSServiceFactory.get_available_providers():
                try:
                    service = self.get_tts_service(provider_name)
                    voices = service.get_voices()
                    all_voices.extend(voices)
                except Exception as e:
                    logger.warning(f"Failed to get voices from {provider_name}: {e}")
            return all_voices
        else:
            try:
                service = self.get_tts_service(provider)
                return service.get_voices()
            except Exception as e:
                logger.error(f"Failed to get voices from {provider}: {e}")
                return []
    
    def _detect_provider_from_voice(self, voice: str) -> str:
        """
        Auto-detect TTS provider from voice name
        
        Args:
            voice: Voice name/identifier
            
        Returns:
            Provider name
        """
        voice_lower = voice.lower()
        
        # Check voice name patterns
        if voice_lower.startswith("edge:") or voice_lower.startswith("azure:"):
            return "edge"
        elif voice_lower.startswith("siliconflow:"):
            return "siliconflow"
        elif voice_lower.startswith("gpt_sovits:"):
            return "gpt_sovits"
        elif voice_lower.startswith("google:") or any(lang in voice_lower for lang in ["en-US", "zh-CN", "ja-JP"]):
            return "google"
        
        # Default fallback
        return self._default_provider
    
    def is_provider_available(self, provider: str) -> bool:
        """
        Check if a TTS provider is available and properly configured
        
        Args:
            provider: Provider name to check
            
        Returns:
            True if provider is available
        """
        try:
            service = self.get_tts_service(provider)
            return service.validate_config()
        except Exception as e:
            logger.warning(f"Provider {provider} is not available: {e}")
            return False


# Global bridge instance
_tts_bridge = None


def get_tts_bridge() -> TTSServiceBridge:
    """
    Get the global TTS bridge instance
    
    Returns:
        TTSServiceBridge instance
    """
    global _tts_bridge
    if _tts_bridge is None:
        _tts_bridge = TTSServiceBridge()
    return _tts_bridge


# Compatibility functions for existing code
def tts_synthesize(text: str, voice: str, output_file: str, **kwargs) -> bool:
    """
    Compatibility function for existing TTS synthesis calls
    
    Args:
        text: Text to synthesize
        voice: Voice identifier
        output_file: Output audio file path
        **kwargs: Additional parameters (speed, volume, etc.)
        
    Returns:
        True if synthesis succeeded
    """
    try:
        bridge = get_tts_bridge()
        
        # Extract parameters
        speed = kwargs.get("speed", 1.0)
        volume_gain = kwargs.get("volume_gain", 0.0)
        language = kwargs.get("language", "en-US")
        provider = kwargs.get("provider", None)
        
        # Synthesize
        response = bridge.synthesize(
            text=text,
            voice=voice,
            provider=provider,
            speed=speed,
            volume_gain=volume_gain,
            language=language,
            output_file=output_file
        )
        
        return True
        
    except Exception as e:
        logger.error(f"TTS synthesis compatibility function failed: {e}")
        return False


def get_available_tts_voices(provider: str = None) -> list:
    """
    Compatibility function for getting available voices
    
    Args:
        provider: Optional provider filter
        
    Returns:
        List of available voice names
    """
    try:
        bridge = get_tts_bridge()
        voices = bridge.get_available_voices(provider)
        return [voice.name for voice in voices]
    except Exception as e:
        logger.error(f"Failed to get TTS voices: {e}")
        return []
