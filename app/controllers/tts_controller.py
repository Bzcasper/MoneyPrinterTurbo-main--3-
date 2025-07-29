"""
TTS Controller

FastAPI controller for TTS (Text-to-Speech) services.
Provides REST API endpoints for speech synthesis and voice management.
"""

import asyncio
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import Response
import tempfile
import os
import uuid

from app.models.schema import TTSProviderResponse, TTSSynthesisRequest, TTSSynthesisResponse, TTSBatchRequest
from app.services.tts.tts_factory import TTSServiceFactory
from app.services.tts.tts_bridge import get_tts_bridge
from app.services.tts.base_tts_service import TTSRequest, TTSServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tts", tags=["TTS"])


@router.get("/providers", response_model=List[TTSProviderResponse])
async def get_tts_providers():
    """
    Get available TTS providers and their capabilities
    
    Returns:
        List of TTS provider information
    """
    try:
        bridge = get_tts_bridge()
        providers = []
        
        for provider_name in TTSServiceFactory.get_available_providers():
            try:
                service = bridge.get_tts_service(provider_name)
                provider_info = service.get_provider_info()
                
                # Check if provider is available
                is_available = bridge.is_provider_available(provider_name)
                
                provider_response = TTSProviderResponse(
                    name=provider_name,
                    display_name=provider_info.get("display_name", provider_name.title()),
                    available=is_available,
                    neural_voices=provider_info.get("neural_voices", False),
                    supports_ssml=provider_info.get("supports_ssml", False),
                    supports_emotions=provider_info.get("supports_emotions", False),
                    max_text_length=provider_info.get("max_text_length", 5000),
                    languages_supported=provider_info.get("languages_supported", 1),
                    pricing_model=provider_info.get("pricing_model", "unknown"),
                    capabilities=provider_info.get("capabilities", [])
                )
                
                providers.append(provider_response)
                
            except Exception as e:
                logger.warning(f"Failed to get info for provider {provider_name}: {e}")
                # Add basic info even if service fails
                providers.append(TTSProviderResponse(
                    name=provider_name,
                    display_name=provider_name.title(),
                    available=False,
                    neural_voices=False,
                    supports_ssml=False,
                    supports_emotions=False,
                    max_text_length=1000,
                    languages_supported=1,
                    pricing_model="unknown",
                    capabilities=[]
                ))
        
        logger.info(f"Retrieved {len(providers)} TTS providers")
        return providers
        
    except Exception as e:
        logger.error(f"Failed to get TTS providers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get TTS providers: {e}")


@router.get("/providers/{provider}/voices")
async def get_provider_voices(provider: str):
    """
    Get available voices for a specific TTS provider
    
    Args:
        provider: TTS provider name
        
    Returns:
        List of available voices
    """
    try:
        if not TTSServiceFactory.is_provider_supported(provider):
            raise HTTPException(status_code=404, detail=f"TTS provider '{provider}' not found")
        
        bridge = get_tts_bridge()
        voices = bridge.get_available_voices(provider)
        
        # Convert VoiceInfo objects to dictionaries
        voice_list = []
        for voice in voices:
            voice_dict = {
                "name": voice.name,
                "language": voice.language,
                "gender": voice.gender,
                "natural_sample_rate": voice.natural_sample_rate,
                "is_neural": voice.is_neural,
                "supports_emotions": getattr(voice, 'supports_emotions', False)
            }
            voice_list.append(voice_dict)
        
        logger.info(f"Retrieved {len(voice_list)} voices for provider {provider}")
        return voice_list
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get voices for provider {provider}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get voices: {e}")


@router.post("/synthesize", response_model=TTSSynthesisResponse)
async def synthesize_speech(request: TTSSynthesisRequest):
    """
    Synthesize speech from text
    
    Args:
        request: TTS synthesis request
        
    Returns:
        Synthesis response with audio data
    """
    try:
        bridge = get_tts_bridge()
        
        # Create TTS request
        tts_request = TTSRequest(
            text=request.text,
            voice_name=request.voice,
            language_code=request.language,
            speaking_rate=request.speed,
            volume_gain=request.volume_gain
        )
        
        # Synthesize speech
        logger.info(f"Starting TTS synthesis for provider: {request.provider}")
        response = await bridge.synthesize_async(
            text=request.text,
            voice=request.voice,
            provider=request.provider,
            speed=request.speed,
            volume_gain=request.volume_gain,
            language=request.language
        )
        
        # Return synthesis response
        synthesis_response = TTSSynthesisResponse(
            audio_data=response.audio_content,
            audio_format=response.audio_format,
            duration=response.duration,
            voice_info={
                "name": response.voice_info.name,
                "language": response.voice_info.language,
                "gender": response.voice_info.gender
            },
            subtitle_data=response.subtitle_data,
            quality_score=response.quality_score
        )
        
        logger.info(f"TTS synthesis completed successfully, duration: {response.duration}s")
        return synthesis_response
        
    except TTSServiceError as e:
        logger.error(f"TTS service error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")


@router.post("/synthesize/file")
async def synthesize_to_file(request: TTSSynthesisRequest):
    """
    Synthesize speech and return as audio file
    
    Args:
        request: TTS synthesis request
        
    Returns:
        Audio file response
    """
    try:
        bridge = get_tts_bridge()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{request.output_format or 'mp3'}"
        )
        temp_file.close()
        
        # Synthesize to file
        response = await bridge.synthesize_async(
            text=request.text,
            voice=request.voice,
            provider=request.provider,
            speed=request.speed,
            volume_gain=request.volume_gain,
            language=request.language,
            output_file=temp_file.name
        )
        
        # Read file content
        with open(temp_file.name, 'rb') as f:
            audio_content = f.read()
        
        # Clean up temp file
        os.unlink(temp_file.name)
        
        # Determine MIME type
        mime_type = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'ogg': 'audio/ogg',
            'flac': 'audio/flac'
        }.get(response.audio_format, 'audio/mpeg')
        
        # Return audio file
        return Response(
            content=audio_content,
            media_type=mime_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{response.audio_format}",
                "X-Duration": str(response.duration),
                "X-Quality-Score": str(response.quality_score)
            }
        )
        
    except TTSServiceError as e:
        logger.error(f"TTS service error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS file synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"File synthesis failed: {e}")


@router.post("/batch", response_model=List[TTSSynthesisResponse])
async def batch_synthesize(request: TTSBatchRequest):
    """
    Batch synthesize multiple texts
    
    Args:
        request: Batch TTS request
        
    Returns:
        List of synthesis responses
    """
    try:
        bridge = get_tts_bridge()
        
        # Limit batch size
        max_batch_size = 10
        if len(request.texts) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.texts)} exceeds maximum {max_batch_size}"
            )
        
        # Process batch
        results = []
        tasks = []
        
        for i, text in enumerate(request.texts):
            task = bridge.synthesize_async(
                text=text,
                voice=request.voice,
                provider=request.provider,
                speed=request.speed,
                volume_gain=request.volume_gain,
                language=request.language
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Batch item {i} failed: {response}")
                # Add error response
                results.append(TTSSynthesisResponse(
                    audio_data=b"",
                    audio_format="mp3",
                    duration=0.0,
                    voice_info={"name": "", "language": "", "gender": ""},
                    subtitle_data=[],
                    quality_score=0.0,
                    error=str(response)
                ))
            else:
                # Add successful response
                results.append(TTSSynthesisResponse(
                    audio_data=response.audio_content,
                    audio_format=response.audio_format,
                    duration=response.duration,
                    voice_info={
                        "name": response.voice_info.name,
                        "language": response.voice_info.language,
                        "gender": response.voice_info.gender
                    },
                    subtitle_data=response.subtitle_data,
                    quality_score=response.quality_score
                ))
        
        logger.info(f"Batch synthesis completed: {len(results)} items")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch synthesis failed: {e}")


@router.get("/health")
async def health_check():
    """
    Health check for TTS services
    
    Returns:
        Service status
    """
    try:
        bridge = get_tts_bridge()
        providers = TTSServiceFactory.get_available_providers()
        
        status = {
            "status": "healthy",
            "providers": {},
            "total_providers": len(providers),
            "available_providers": 0
        }
        
        for provider in providers:
            is_available = bridge.is_provider_available(provider)
            status["providers"][provider] = {
                "available": is_available,
                "status": "healthy" if is_available else "unavailable"
            }
            
            if is_available:
                status["available_providers"] += 1
        
        logger.info("TTS health check completed")
        return status
        
    except Exception as e:
        logger.error(f"TTS health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "providers": {},
            "total_providers": 0,
            "available_providers": 0
        }
