"""
TTS Synthesis API Endpoints

Endpoints for text-to-speech synthesis including single requests,
batch processing, and caching management.
"""

import asyncio
import logging
import os
import uuid
import hashlib
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

from app.models.schema import (
    TTSRequest, TTSResponse, BatchTTSRequest, BatchTTSResponse,
    VoiceInfo
)
from app.services.tts import TTSServiceFactory, TTSServiceError
from app.config.config import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/synthesis", tags=["TTS Synthesis"])


def get_tts_config() -> Dict[str, Any]:
    """Get TTS configuration from app config"""
    return getattr(config, 'tts', {})


def get_audio_storage_path() -> str:
    """Get path for storing generated audio files"""
    storage_path = getattr(config, 'storage_path', './storage')
    audio_path = os.path.join(storage_path, 'audio', 'tts')
    os.makedirs(audio_path, exist_ok=True)
    return audio_path


async def save_audio_file(audio_content: bytes, format: str = "mp3", task_id: str = None) -> str:
    """
    Save audio content to file
    
    Args:
        audio_content: Raw audio bytes
        format: Audio format (mp3, wav, etc.)
        task_id: Optional task ID for filename
        
    Returns:
        Path to saved audio file
    """
    storage_path = get_audio_storage_path()
    
    if task_id:
        filename = f"tts_{task_id}.{format}"
    else:
        filename = f"tts_{uuid.uuid4().hex}.{format}"
    
    file_path = os.path.join(storage_path, filename)
    
    with open(file_path, 'wb') as f:
        f.write(audio_content)
    
    logger.info(f"Saved TTS audio to: {file_path}")
    return file_path


@router.post("/", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Synthesize speech from text using specified TTS provider
    
    Args:
        request: TTSRequest with text and voice parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        TTSResponse with synthesized audio
    """
    try:
        logger.info(f"Starting TTS synthesis with provider: {request.provider}")
        
        # Get provider configuration
        tts_config = get_tts_config()
        provider_config = tts_config.get("providers", {}).get(request.provider, {})
        
        if not provider_config:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration not found for provider: {request.provider}"
            )
        
        # Create service instance
        try:
            service = TTSServiceFactory.create_service(request.provider, provider_config)
        except TTSServiceError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Perform synthesis
        start_time = datetime.utcnow()
        response = await service.synthesize(request)
        synthesis_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Save audio file
        if response.audio_content:
            file_path = await save_audio_file(
                response.audio_content,
                response.audio_format,
                str(uuid.uuid4())
            )
            response.audio_file_path = file_path
        
        # Add synthesis timing
        response.synthesis_time = synthesis_time
        
        logger.info(f"TTS synthesis completed in {synthesis_time:.2f}s")
        
        # Schedule cleanup of temporary file after 24 hours
        background_tasks.add_task(cleanup_temp_file, response.audio_file_path, delay_hours=24)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@router.post("/batch", response_model=BatchTTSResponse)
async def batch_synthesize_speech(
    batch_request: BatchTTSRequest,
    background_tasks: BackgroundTasks
):
    """
    Batch synthesize multiple TTS requests
    
    Args:
        batch_request: BatchTTSRequest with multiple synthesis requests
        background_tasks: FastAPI background tasks
        
    Returns:
        BatchTTSResponse with results for all requests
    """
    try:
        logger.info(f"Starting batch TTS synthesis for {len(batch_request.requests)} requests")
        start_time = datetime.utcnow()
        
        results = []
        successful_count = 0
        failed_count = 0
        
        if batch_request.parallel_processing:
            # Process requests in parallel with concurrency limit
            semaphore = asyncio.Semaphore(batch_request.max_concurrent)
            tasks = [
                _process_single_request(req, semaphore, background_tasks)
                for req in batch_request.requests
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process requests sequentially
            for req in batch_request.requests:
                result = await _process_single_request(req, None, background_tasks)
                results.append(result)
        
        # Count successes and failures
        for result in results:
            if isinstance(result, TTSResponse):
                successful_count += 1
            else:
                failed_count += 1
        
        total_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Batch synthesis completed: {successful_count} success, {failed_count} failed")
        
        return BatchTTSResponse(
            results=results,
            successful_count=successful_count,
            failed_count=failed_count,
            total_time=total_time
        )
        
    except Exception as e:
        logger.error(f"Batch TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch synthesis failed: {str(e)}")


async def _process_single_request(
    request: TTSRequest,
    semaphore: Optional[asyncio.Semaphore],
    background_tasks: BackgroundTasks
) -> Union[TTSResponse, Dict[str, str]]:
    """
    Process a single TTS request with optional concurrency control
    
    Args:
        request: TTSRequest to process
        semaphore: Optional semaphore for concurrency control
        background_tasks: FastAPI background tasks
        
    Returns:
        TTSResponse on success or error dict on failure
    """
    async def _do_synthesis():
        try:
            # Get provider configuration
            tts_config = get_tts_config()
            provider_config = tts_config.get("providers", {}).get(request.provider, {})
            
            if not provider_config:
                return {"error": f"Configuration not found for provider: {request.provider}"}
            
            # Create service and synthesize
            service = TTSServiceFactory.create_service(request.provider, provider_config)
            response = await service.synthesize(request)
            
            # Save audio file
            if response.audio_content:
                file_path = await save_audio_file(
                    response.audio_content,
                    response.audio_format,
                    str(uuid.uuid4())
                )
                response.audio_file_path = file_path
                
                # Schedule cleanup
                background_tasks.add_task(cleanup_temp_file, file_path, delay_hours=24)
            
            return response
            
        except Exception as e:
            logger.error(f"Single TTS request failed: {e}")
            return {"error": str(e)}
    
    if semaphore:
        async with semaphore:
            return await _do_synthesis()
    else:
        return await _do_synthesis()


@router.get("/audio/{file_id}")
async def get_audio_file(file_id: str):
    """
    Download generated audio file
    
    Args:
        file_id: Audio file identifier
        
    Returns:
        FileResponse with audio content
    """
    try:
        storage_path = get_audio_storage_path()
        
        # Find file with matching ID
        for filename in os.listdir(storage_path):
            if file_id in filename:
                file_path = os.path.join(storage_path, filename)
                if os.path.exists(file_path):
                    return FileResponse(
                        file_path,
                        media_type="audio/mpeg",
                        filename=filename
                    )
        
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve audio file {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve audio file")


@router.post("/validate")
async def validate_tts_request(request: TTSRequest) -> Dict[str, Any]:
    """
    Validate TTS request without performing synthesis
    
    Args:
        request: TTSRequest to validate
        
    Returns:
        Validation result with details
    """
    try:
        validation_result = {
            "valid": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check provider availability
        available_providers = TTSServiceFactory.get_available_providers()
        if request.provider not in available_providers:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Provider '{request.provider}' not available")
            validation_result["recommendations"].append(f"Available providers: {available_providers}")
        
        # Check text length
        if len(request.text) > 10000:
            validation_result["valid"] = False
            validation_result["issues"].append("Text exceeds maximum length of 10,000 characters")
        
        if len(request.text.strip()) == 0:
            validation_result["valid"] = False
            validation_result["issues"].append("Text cannot be empty")
        
        # Check parameter ranges
        if not 0.25 <= request.speaking_rate <= 4.0:
            validation_result["valid"] = False
            validation_result["issues"].append("Speaking rate must be between 0.25 and 4.0")
        
        if not -20.0 <= request.pitch <= 20.0:
            validation_result["valid"] = False
            validation_result["issues"].append("Pitch must be between -20.0 and 20.0")
        
        # Provider-specific validation
        if validation_result["valid"] and request.provider in available_providers:
            try:
                tts_config = get_tts_config()
                provider_config = tts_config.get("providers", {}).get(request.provider, {})
                
                if provider_config:
                    service = TTSServiceFactory.create_service(request.provider, provider_config)
                    
                    # Validate voice exists
                    voices = service.get_voices()
                    voice_names = [v.name for v in voices]
                    
                    if request.voice_name and request.voice_name not in voice_names:
                        validation_result["recommendations"].append(
                            f"Voice '{request.voice_name}' not found. Available voices: {voice_names[:5]}..."
                        )
                        
            except Exception as e:
                validation_result["recommendations"].append(f"Provider validation warning: {e}")
        
        logger.info(f"TTS request validation: {'valid' if validation_result['valid'] else 'invalid'}")
        return validation_result
        
    except Exception as e:
        logger.error(f"TTS validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.delete("/cache")
async def clear_tts_cache() -> Dict[str, str]:
    """
    Clear TTS response cache
    
    Returns:
        Cache clearing result
    """
    try:
        # This would typically clear Redis cache or similar
        # For now, just clear temporary audio files older than 1 day
        
        storage_path = get_audio_storage_path()
        cleared_count = 0
        
        for filename in os.listdir(storage_path):
            file_path = os.path.join(storage_path, filename)
            
            # Check file age
            file_age = datetime.now().timestamp() - os.path.getctime(file_path)
            if file_age > 86400:  # 24 hours
                try:
                    os.remove(file_path)
                    cleared_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove file {file_path}: {e}")
        
        logger.info(f"Cleared {cleared_count} old TTS audio files")
        
        return {
            "message": f"Cache cleared successfully. Removed {cleared_count} old files.",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear TTS cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clearing failed: {str(e)}")


async def cleanup_temp_file(file_path: str, delay_hours: int = 24):
    """
    Background task to cleanup temporary audio files
    
    Args:
        file_path: Path to file to cleanup
        delay_hours: Hours to wait before cleanup
    """
    try:
        # Wait for specified delay
        await asyncio.sleep(delay_hours * 3600)
        
        # Remove file if it still exists
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
            
    except Exception as e:
        logger.warning(f"Failed to cleanup file {file_path}: {e}")
