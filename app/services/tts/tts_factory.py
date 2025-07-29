"""
TTS Service Factory

Factory pattern implementation for creating TTS service instances.
Provides centralized service registration and instantiation.
"""

import logging
from typing import Dict, Any, Type
from .base_tts_service import BaseTTSService
from .google_tts_service import GoogleTTSService
from .edge_tts_service import EdgeTTSService
from .siliconflow_tts_service import SiliconFlowTTSService
from .gpt_sovits_tts_service import GPTSoVITSTTSService

logger = logging.getLogger(__name__)


class TTSServiceFactory:
    """Factory for creating TTS service instances"""
    
    # Registry of available TTS services
    _services: Dict[str, Type[BaseTTSService]] = {
        "google": GoogleTTSService,
        "edge": EdgeTTSService,
        "siliconflow": SiliconFlowTTSService,
        "gpt_sovits": GPTSoVITSTTSService,
    }
    
    @classmethod
    def register_service(cls, name: str, service_class: Type[BaseTTSService]):
        """
        Register a new TTS service
        
        Args:
            name: Service identifier
            service_class: Service implementation class
        """
        cls._services[name] = service_class
        logger.info(f"Registered TTS service: {name}")
    
    @classmethod
    def create_service(cls, provider: str, config: Dict[str, Any]) -> BaseTTSService:
        """
        Create a TTS service instance
        
        Args:
            provider: TTS provider name
            config: Service configuration
            
        Returns:
            Configured TTS service instance
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider not in cls._services:
            available = ", ".join(cls._services.keys())
            raise ValueError(f"Unsupported TTS provider: {provider}. Available: {available}")
        
        service_class = cls._services[provider]
        logger.info(f"Creating TTS service: {provider}")
        
        try:
            service = service_class(config)
            logger.info(f"Successfully created {provider} TTS service")
            return service
        except Exception as e:
            logger.error(f"Failed to create {provider} TTS service: {e}")
            raise
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """
        Get list of available TTS providers
        
        Returns:
            List of provider names
        """
        return list(cls._services.keys())
    
    @classmethod
    def is_provider_supported(cls, provider: str) -> bool:
        """
        Check if a TTS provider is supported
        
        Args:
            provider: Provider name to check
            
        Returns:
            True if provider is supported
        """
        return provider in cls._services

import logging
from typing import Dict, Type, List, Any, Optional
from .base_tts_service import BaseTTSService, TTSServiceError

logger = logging.getLogger(__name__)


class TTSServiceFactory:
    """Factory for creating TTS service instances"""
    
    # Registry of available TTS services
    _services: Dict[str, Type[BaseTTSService]] = {}
    
    # Service metadata and capabilities
    _service_metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_service(cls, provider: str, service_class: Type[BaseTTSService], 
                        metadata: Dict[str, Any] = None):
        """
        Register a TTS service provider
        
        Args:
            provider: Provider name (e.g., 'google', 'azure')
            service_class: TTS service class
            metadata: Optional metadata about the service
        """
        cls._services[provider] = service_class
        cls._service_metadata[provider] = metadata or {}
        logger.info(f"Registered TTS service: {provider}")
    
    @classmethod
    def create_service(cls, provider: str, config: Dict[str, Any]) -> BaseTTSService:
        """
        Create TTS service instance
        
        Args:
            provider: Provider name
            config: Provider configuration
            
        Returns:
            Configured TTS service instance
            
        Raises:
            TTSServiceError: If provider is not supported or creation fails
        """
        if provider not in cls._services:
            available = list(cls._services.keys())
            raise TTSServiceError(
                f"Unsupported TTS provider: {provider}. Available: {available}",
                provider=provider
            )
        
        try:
            service_class = cls._services[provider]
            service = service_class(config)
            
            # Validate configuration
            if not service.validate_config():
                raise TTSServiceError(f"Invalid configuration for provider: {provider}", provider=provider)
            
            logger.info(f"Created TTS service instance for provider: {provider}")
            return service
            
        except Exception as e:
            logger.error(f"Failed to create TTS service for provider {provider}: {e}")
            raise TTSServiceError(f"Service creation failed: {e}", provider=provider)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get list of available TTS providers
        
        Returns:
            List of provider names
        """
        return list(cls._services.keys())
    
    @classmethod
    def get_provider_metadata(cls, provider: str) -> Dict[str, Any]:
        """
        Get metadata for a specific provider
        
        Args:
            provider: Provider name
            
        Returns:
            Provider metadata dictionary
        """
        return cls._service_metadata.get(provider, {})
    
    @classmethod
    def get_providers_by_capability(cls, capability: str) -> List[str]:
        """
        Get providers that support a specific capability
        
        Args:
            capability: Capability name (e.g., 'neural_voices', 'emotions')
            
        Returns:
            List of provider names supporting the capability
        """
        providers = []
        for provider, metadata in cls._service_metadata.items():
            capabilities = metadata.get('capabilities', [])
            if capability in capabilities:
                providers.append(provider)
        
        return providers
    
    @classmethod
    def create_service_with_fallback(cls, providers: List[str], configs: Dict[str, Dict[str, Any]]) -> BaseTTSService:
        """
        Create TTS service with fallback chain
        
        Args:
            providers: List of providers in order of preference
            configs: Configuration for each provider
            
        Returns:
            First successfully created TTS service
            
        Raises:
            TTSServiceError: If all providers fail
        """
        errors = []
        
        for provider in providers:
            try:
                if provider not in configs:
                    logger.warning(f"No configuration found for provider: {provider}")
                    continue
                
                service = cls.create_service(provider, configs[provider])
                logger.info(f"Successfully created TTS service with provider: {provider}")
                return service
                
            except Exception as e:
                error_msg = f"Provider {provider} failed: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        # All providers failed
        error_summary = "; ".join(errors)
        raise TTSServiceError(f"All TTS providers failed. Errors: {error_summary}")
    
    @classmethod
    def get_recommended_provider(cls, requirements: Dict[str, Any]) -> Optional[str]:
        """
        Get recommended provider based on requirements
        
        Args:
            requirements: Dictionary of requirements (e.g., {'quality': 'high', 'cost': 'low'})
            
        Returns:
            Recommended provider name or None
        """
        # Provider scoring based on requirements
        scores = {}
        
        for provider, metadata in cls._service_metadata.items():
            score = 0
            
            # Quality preference
            quality_pref = requirements.get('quality', 'medium')
            if quality_pref == 'high' and metadata.get('neural_voices', False):
                score += 3
            elif quality_pref == 'medium':
                score += 2
            else:  # low quality
                score += 1
            
            # Cost preference
            cost_pref = requirements.get('cost', 'medium')
            pricing = metadata.get('pricing_model', 'unknown')
            if cost_pref == 'low':
                if pricing == 'free':
                    score += 3
                elif pricing == 'pay_per_use':
                    score += 1
            elif cost_pref == 'high':
                if pricing == 'premium':
                    score += 3
            
            # Language support
            languages_needed = requirements.get('languages', [])
            languages_supported = metadata.get('languages_supported', 0)
            if languages_needed and languages_supported >= len(languages_needed):
                score += 2
            
            # Special features
            if requirements.get('emotions', False) and 'emotions' in metadata.get('capabilities', []):
                score += 2
            if requirements.get('characters', False) and 'characters' in metadata.get('capabilities', []):
                score += 2
            
            scores[provider] = score
        
        # Return provider with highest score
        if scores:
            return max(scores, key=scores.get)
        
        return None


# Auto-register built-in services
def _register_builtin_services():
    """Register built-in TTS services"""
    
    # Register Google TTS
    try:
        from .google_tts_service import GoogleTTSService
        TTSServiceFactory.register_service(
            "google",
            GoogleTTSService,
            {
                "display_name": "Google Cloud Text-to-Speech",
                "neural_voices": True,
                "premium_voices": True,
                "capabilities": ["neural_voices", "ssml", "multi_language"],
                "pricing_model": "pay_per_character",
                "languages_supported": 100,
                "quality": "high",
                "priority": 10
            }
        )
    except ImportError:
        logger.info("Google TTS service not available (missing dependencies)")
    
    # Register Azure TTS (placeholder for future implementation)
    # try:
    #     from .azure_tts_service import AzureTTSService
    #     TTSServiceFactory.register_service(
    #         "azure",
    #         AzureTTSService,
    #         {
    #             "display_name": "Azure Cognitive Services Speech",
    #             "neural_voices": True,
    #             "capabilities": ["neural_voices", "ssml", "custom_voices"],
    #             "pricing_model": "pay_per_character",
    #             "languages_supported": 75,
    #             "quality": "high",
    #             "priority": 9
    #         }
    #     )
    # except ImportError:
    #     logger.info("Azure TTS service not available")
    
    # Register CharacterBox TTS
    try:
        from .characterbox_tts_service import CharacterBoxTTSService
        TTSServiceFactory.register_service(
            "characterbox",
            CharacterBoxTTSService,
            {
                "display_name": "CharacterBox Character Voices",
                "neural_voices": True,
                "capabilities": ["character_voices", "emotions", "conversations"],
                "pricing_model": "pay_per_use",
                "languages_supported": 25,
                "quality": "high",
                "priority": 8
            }
        )
    except ImportError:
        logger.info("CharacterBox TTS service not available")
    
    # Register existing Edge TTS as fallback
    try:
        # Import existing edge TTS service if available
        # This would need to be adapted from existing voice.py implementation
        pass
    except ImportError:
        logger.info("Edge TTS service not available")


# Auto-register services on module import
_register_builtin_services()
