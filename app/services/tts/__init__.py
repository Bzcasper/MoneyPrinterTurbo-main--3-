"""
TTS (Text-to-Speech) Service Package

This package provides a comprehensive TTS service architecture supporting multiple providers
including Google TTS, Azure, CharacterBox, and existing edge TTS services.

Key Components:
- BaseTTSService: Abstract interface for all TTS implementations
- TTSServiceFactory: Factory pattern for creating TTS service instances
- Provider-specific implementations (Google, Azure, CharacterBox, etc.)
- Caching and performance optimization layers
- Circuit breaker and fallback strategies
"""

from .base_tts_service import BaseTTSService, TTSRequest, TTSResponse, VoiceInfo, TTSServiceError
from .tts_factory import TTSServiceFactory
from .tts_bridge import TTSServiceBridge, get_tts_bridge, tts_synthesize, get_available_tts_voices

# Service implementations
from .edge_tts_service import EdgeTTSService
from .google_tts_service import GoogleTTSService
from .siliconflow_tts_service import SiliconFlowTTSService
from .gpt_sovits_tts_service import GPTSoVITSTTSService

__all__ = [
    # Core classes
    "BaseTTSService",
    "TTSRequest", 
    "TTSResponse",
    "VoiceInfo",
    "TTSServiceError",
    
    # Factory and bridge
    "TTSServiceFactory",
    "TTSServiceBridge",
    "get_tts_bridge",
    
    # Service implementations
    "EdgeTTSService",
    "GoogleTTSService", 
    "SiliconFlowTTSService",
    "GPTSoVITSTTSService",
    
    # Compatibility functions
    "tts_synthesize",
    "get_available_tts_voices"
]
