"""
TTS Providers API Endpoints

Endpoints for managing TTS providers, getting available voices,
and provider health monitoring.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from app.models.schema import (
    TTSProvider, VoiceListResponse, VoiceInfo, 
    ProviderHealthStatus, TTSAnalyticsResponse
)
from app.services.tts import TTSServiceFactory
from app.config.config import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/providers", tags=["TTS Providers"])


def get_tts_config() -> Dict[str, Any]:
    """Get TTS configuration from app config"""
    return getattr(config, 'tts', {})


@router.get("/", response_model=List[str])
async def get_available_providers():
    """
    Get list of available TTS providers
    
    Returns:
        List of provider names
    """
    try:
        providers = TTSServiceFactory.get_available_providers()
        logger.info(f"Retrieved {len(providers)} available TTS providers")
        return providers
    except Exception as e:
        logger.error(f"Failed to get TTS providers: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve providers")


@router.get("/metadata", response_model=List[TTSProvider])
async def get_providers_metadata():
    """
    Get detailed metadata for all available providers
    
    Returns:
        List of provider metadata
    """
    try:
        providers = TTSServiceFactory.get_available_providers()
        provider_list = []
        
        for provider in providers:
            metadata = TTSServiceFactory.get_provider_metadata(provider)
            provider_info = TTSProvider(
                name=provider,
                display_name=metadata.get("display_name", provider.title()),
                is_active=True,  # Assume active if registered
                capabilities=metadata.get("capabilities", []),
                priority=metadata.get("priority", 0)
            )
            provider_list.append(provider_info)
        
        # Sort by priority (descending)
        provider_list.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Retrieved metadata for {len(provider_list)} providers")
        return provider_list
        
    except Exception as e:
        logger.error(f"Failed to get provider metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve provider metadata")


@router.get("/{provider}/voices", response_model=VoiceListResponse)
async def get_provider_voices(
    provider: str,
    language: str = Query(None, description="Filter by language code"),
    gender: str = Query(None, description="Filter by gender"),
    neural_only: bool = Query(False, description="Show only neural voices")
):
    """
    Get available voices for a specific provider
    
    Args:
        provider: Provider name
        language: Optional language filter
        gender: Optional gender filter  
        neural_only: Show only neural voices
        
    Returns:
        VoiceListResponse with filtered voices
    """
    try:
        # Get provider configuration
        tts_config = get_tts_config()
        provider_config = tts_config.get("providers", {}).get(provider, {})
        
        if not provider_config:
            raise HTTPException(
                status_code=404, 
                detail=f"Configuration not found for provider: {provider}"
            )
        
        # Create service instance
        try:
            service = TTSServiceFactory.create_service(provider, provider_config)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to initialize provider {provider}: {str(e)}"
            )
        
        # Get voices
        voices = service.get_voices()
        
        # Apply filters
        filtered_voices = voices
        
        if language:
            filtered_voices = [v for v in filtered_voices if v.language.startswith(language)]
        
        if gender:
            filtered_voices = [v for v in filtered_voices if v.gender == gender.lower()]
        
        if neural_only:
            filtered_voices = [v for v in filtered_voices if v.is_neural]
        
        logger.info(f"Retrieved {len(filtered_voices)} voices for provider {provider}")
        
        return VoiceListResponse(
            provider=provider,
            voices=filtered_voices,
            total_count=len(filtered_voices)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get voices for provider {provider}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve voices: {str(e)}")


@router.get("/{provider}/health", response_model=ProviderHealthStatus)
async def check_provider_health(provider: str):
    """
    Check health status of a specific provider
    
    Args:
        provider: Provider name
        
    Returns:
        ProviderHealthStatus with current status
    """
    try:
        # Get provider configuration
        tts_config = get_tts_config()
        provider_config = tts_config.get("providers", {}).get(provider, {})
        
        if not provider_config:
            raise HTTPException(
                status_code=404,
                detail=f"Provider not found: {provider}"
            )
        
        # Test provider health
        start_time = time.time()
        is_healthy = False
        error_message = None
        
        try:
            service = TTSServiceFactory.create_service(provider, provider_config)
            is_healthy = service.validate_config()
            if not is_healthy:
                error_message = "Configuration validation failed"
        except Exception as e:
            error_message = str(e)
        
        response_time = time.time() - start_time
        
        status = ProviderHealthStatus(
            provider=provider,
            is_healthy=is_healthy,
            last_check=datetime.utcnow(),
            response_time=response_time,
            error_rate=0.0,  # Would need historical data to calculate
            message=error_message if not is_healthy else "Provider is healthy"
        )
        
        logger.info(f"Health check for {provider}: {'healthy' if is_healthy else 'unhealthy'}")
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed for provider {provider}: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/capabilities/{capability}")
async def get_providers_by_capability(capability: str) -> List[str]:
    """
    Get providers that support a specific capability
    
    Args:
        capability: Capability name (e.g., 'neural_voices', 'emotions')
        
    Returns:
        List of provider names supporting the capability
    """
    try:
        providers = TTSServiceFactory.get_providers_by_capability(capability)
        logger.info(f"Found {len(providers)} providers supporting '{capability}'")
        return providers
    except Exception as e:
        logger.error(f"Failed to get providers by capability '{capability}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query capability: {str(e)}")


@router.get("/recommend")
async def recommend_provider(
    quality: str = Query("medium", description="Quality preference: low, medium, high"),
    cost: str = Query("medium", description="Cost preference: low, medium, high"),
    languages: List[str] = Query([], description="Required languages"),
    emotions: bool = Query(False, description="Emotion support required"),
    characters: bool = Query(False, description="Character voices required")
) -> Dict[str, Any]:
    """
    Get recommended provider based on requirements
    
    Args:
        quality: Quality preference
        cost: Cost preference
        languages: Required languages
        emotions: Emotion support required
        characters: Character voices required
        
    Returns:
        Recommended provider with reasoning
    """
    try:
        requirements = {
            "quality": quality,
            "cost": cost,
            "languages": languages,
            "emotions": emotions,
            "characters": characters
        }
        
        recommended = TTSServiceFactory.get_recommended_provider(requirements)
        
        if not recommended:
            return {
                "provider": None,
                "message": "No provider found matching requirements",
                "alternatives": TTSServiceFactory.get_available_providers()
            }
        
        metadata = TTSServiceFactory.get_provider_metadata(recommended)
        
        return {
            "provider": recommended,
            "display_name": metadata.get("display_name", recommended.title()),
            "reasoning": "Best match for specified requirements",
            "capabilities": metadata.get("capabilities", []),
            "priority": metadata.get("priority", 0)
        }
        
    except Exception as e:
        logger.error(f"Provider recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@router.get("/analytics", response_model=TTSAnalyticsResponse)
async def get_tts_analytics():
    """
    Get TTS usage analytics and performance metrics
    
    Returns:
        TTSAnalyticsResponse with comprehensive analytics
    """
    try:
        # This would typically query a metrics database
        # For now, return placeholder data
        
        providers = TTSServiceFactory.get_available_providers()
        usage_stats = []
        provider_health = []
        
        for provider in providers:
            # Placeholder usage stats
            usage_stats.append({
                "provider": provider,
                "total_requests": 0,
                "successful_requests": 0,
                "average_duration": 0.0,
                "average_synthesis_time": 0.0,
                "cache_hit_rate": 0.0
            })
            
            # Placeholder health status
            provider_health.append({
                "provider": provider,
                "is_healthy": True,
                "last_check": datetime.utcnow(),
                "response_time": 1.0,
                "error_rate": 0.0,
                "message": "No data available"
            })
        
        analytics = TTSAnalyticsResponse(
            usage_stats=usage_stats,
            provider_health=provider_health,
            popular_voices=[],
            cost_analysis={},
            recommendations=[
                "Consider implementing usage tracking for better analytics",
                "Set up monitoring for provider health checks"
            ]
        )
        
        logger.info("Retrieved TTS analytics (placeholder data)")
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get TTS analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")
