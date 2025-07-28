"""
Text-to-Speech API Controllers

This module provides FastAPI endpoints for managing TTS services including
Google TTS and CharacterBox integration.
"""

import asyncio
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from fastapi import BackgroundTasks, Depends, Request, HTTPException, Query
from fastapi.responses import FileResponse
from loguru import logger

from app.controllers import base
from app.controllers.v1.base import new_router
from app.models.schema import BaseResponse
from app.models.exception import HttpException
from app.services.google_tts import get_google_tts_service, google_tts_service_status
from app.services.characterbox import get_characterbox_service, characterbox_service_status
from app.utils import utils

# Create router for TTS endpoints
router = new_router()


# Request/Response Models
class TTSVoicesResponse(BaseModel):
    """Response model for TTS voices list."""
    service: str
    voices: List[Dict[str, Any]]
    total_count: int


class TTSServiceStatusResponse(BaseModel):
    """Response model for TTS service status."""
    google_tts: Dict[str, Any]
    characterbox: Dict[str, Any]


class GoogleTTSRequest(BaseModel):
    """Request model for Google TTS synthesis."""
    text: str
    voice_name: str = "en-US-Neural2-C"
    language_code: str = "en-US"
    audio_encoding: str = "MP3"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    volume_gain_db: float = 0.0


class GoogleSSMLTTSRequest(BaseModel):
    """Request model for Google SSML TTS synthesis."""
    ssml_text: str
    voice_name: str = "en-US-Neural2-C"
    language_code: str = "en-US"
    audio_encoding: str = "MP3"


class CharacterBoxTTSRequest(BaseModel):
    """Request model for CharacterBox TTS synthesis."""
    text: str
    character_id: str
    voice_settings: Optional[Dict[str, Any]] = None
    audio_format: str = "mp3"
    quality: str = "high"


class CustomCharacterRequest(BaseModel):
    """Request model for creating custom CharacterBox character."""
    name: str
    description: str
    personality_traits: Optional[Dict[str, Any]] = None


class TTSSynthesisResponse(BaseModel):
    """Response model for TTS synthesis."""
    task_id: str
    audio_file: str
    service: str
    duration: Optional[float] = None
    file_size: Optional[int] = None


# TTS Service Status Endpoint
@router.get(
    "/tts/status",
    response_model=BaseResponse,
    summary="Get TTS services status",
    description="Check the availability and configuration of all TTS services"
)
async def get_tts_services_status(request: Request):
    """Get status of all TTS services."""
    request_id = base.get_task_id(request)
    
    try:
        google_status = google_tts_service_status()
        characterbox_status = characterbox_service_status()
        
        status_data = {
            "google_tts": google_status,
            "characterbox": characterbox_status,
            "overall_status": "healthy" if (
                google_status.get("available", False) or 
                characterbox_status.get("available", False)
            ) else "unavailable"
        }
        
        logger.info(f"TTS services status retrieved: {request_id}")
        return utils.get_response(200, status_data)
        
    except Exception as e:
        logger.error(f"Failed to get TTS services status: {str(e)}")
        raise HttpException(
            task_id=request_id,
            status_code=500,
            message=f"Failed to get TTS services status: {str(e)}"
        )


# Google TTS Endpoints
@router.get(
    "/tts/google/voices",
    response_model=BaseResponse,
    summary="Get Google TTS voices",
    description="Retrieve list of available Google TTS voices"
)
async def get_google_tts_voices(
    request: Request,
    language_code: Optional[str] = Query(None, description="Filter by language code (e.g., 'en-US')")
):
    """Get available Google TTS voices."""
    request_id = base.get_task_id(request)
    
    try:
        google_service = get_google_tts_service()
        
        if not google_service.is_available():
            raise HttpException(
                task_id=request_id,
                status_code=503,
                message="Google TTS service not available"
            )
        
        voices = google_service.get_available_voices(language_code)
        
        response_data = {
            "service": "google_tts",
            "voices": voices,
            "total_count": len(voices),
            "language_filter": language_code
        }
        
        logger.info(f"Google TTS voices retrieved: {len(voices)} voices")
        return utils.get_response(200, response_data)
        
    except Exception as e:
        logger.error(f"Failed to get Google TTS voices: {str(e)}")
        raise HttpException(
            task_id=request_id,
            status_code=500,
            message=f"Failed to get Google TTS voices: {str(e)}"
        )


@router.post(
    "/tts/google/synthesize",
    response_model=BaseResponse,
    summary="Synthesize speech with Google TTS",
    description="Convert text to speech using Google Cloud Text-to-Speech"
)
async def google_tts_synthesize(
    request: Request,
    background_tasks: BackgroundTasks,
    body: GoogleTTSRequest
):
    """Synthesize speech using Google TTS."""
    request_id = base.get_task_id(request)
    task_id = utils.get_uuid()
    
    try:
        google_service = get_google_tts_service()
        
        if not google_service.is_available():
            raise HttpException(
                task_id=task_id,
                status_code=503,
                message="Google TTS service not available"
            )
        
        # Validate input text
        if not body.text.strip():
            raise HttpException(
                task_id=task_id,
                status_code=400,
                message="Text content cannot be empty"
            )
        
        logger.info(f"Starting Google TTS synthesis: {task_id}")
        
        # Synthesize speech
        audio_file = await google_service.synthesize_speech(
            text=body.text,
            voice_name=body.voice_name,
            language_code=body.language_code,
            audio_encoding=body.audio_encoding,
            speaking_rate=body.speaking_rate,
            pitch=body.pitch,
            volume_gain_db=body.volume_gain_db
        )
        
        if not audio_file:
            raise HttpException(
                task_id=task_id,
                status_code=500,
                message="Google TTS synthesis failed"
            )
        
        # Get file info
        import os
        file_size = os.path.getsize(audio_file) if os.path.exists(audio_file) else 0
        
        response_data = {
            "task_id": task_id,
            "audio_file": audio_file,
            "service": "google_tts",
            "voice_name": body.voice_name,
            "language_code": body.language_code,
            "file_size": file_size,
            "audio_encoding": body.audio_encoding
        }
        
        logger.success(f"Google TTS synthesis completed: {task_id}")
        return utils.get_response(200, response_data)
        
    except HttpException:
        raise
    except Exception as e:
        logger.error(f"Google TTS synthesis error: {str(e)}")
        raise HttpException(
            task_id=task_id,
            status_code=500,
            message=f"Google TTS synthesis failed: {str(e)}"
        )


@router.post(
    "/tts/google/synthesize-ssml",
    response_model=BaseResponse,
    summary="Synthesize SSML speech with Google TTS",
    description="Convert SSML text to speech using Google Cloud Text-to-Speech"
)
async def google_ssml_tts_synthesize(
    request: Request,
    background_tasks: BackgroundTasks,
    body: GoogleSSMLTTSRequest
):
    """Synthesize speech using Google TTS with SSML input."""
    request_id = base.get_task_id(request)
    task_id = utils.get_uuid()
    
    try:
        google_service = get_google_tts_service()
        
        if not google_service.is_available():
            raise HttpException(
                task_id=task_id,
                status_code=503,
                message="Google TTS service not available"
            )
        
        # Validate SSML input
        if not body.ssml_text.strip():
            raise HttpException(
                task_id=task_id,
                status_code=400,
                message="SSML content cannot be empty"
            )
        
        logger.info(f"Starting Google SSML TTS synthesis: {task_id}")
        
        # Synthesize SSML speech
        audio_file = await google_service.synthesize_ssml_speech(
            ssml_text=body.ssml_text,
            voice_name=body.voice_name,
            language_code=body.language_code,
            audio_encoding=body.audio_encoding
        )
        
        if not audio_file:
            raise HttpException(
                task_id=task_id,
                status_code=500,
                message="Google SSML TTS synthesis failed"
            )
        
        # Get file info
        import os
        file_size = os.path.getsize(audio_file) if os.path.exists(audio_file) else 0
        
        response_data = {
            "task_id": task_id,
            "audio_file": audio_file,
            "service": "google_ssml_tts",
            "voice_name": body.voice_name,
            "language_code": body.language_code,
            "file_size": file_size,
            "audio_encoding": body.audio_encoding
        }
        
        logger.success(f"Google SSML TTS synthesis completed: {task_id}")
        return utils.get_response(200, response_data)
        
    except HttpException:
        raise
    except Exception as e:
        logger.error(f"Google SSML TTS synthesis error: {str(e)}")
        raise HttpException(
            task_id=task_id,
            status_code=500,
            message=f"Google SSML TTS synthesis failed: {str(e)}"
        )


# CharacterBox Endpoints
@router.get(
    "/tts/characterbox/characters",
    response_model=BaseResponse,
    summary="Get CharacterBox characters",
    description="Retrieve list of available CharacterBox characters"
)
async def get_characterbox_characters(request: Request):
    """Get available CharacterBox characters."""
    request_id = base.get_task_id(request)
    
    try:
        characterbox_service = get_characterbox_service()
        
        if not characterbox_service.is_available():
            raise HttpException(
                task_id=request_id,
                status_code=503,
                message="CharacterBox service not available"
            )
        
        characters = await characterbox_service.get_available_characters()
        
        response_data = {
            "service": "characterbox",
            "characters": characters,
            "total_count": len(characters)
        }
        
        logger.info(f"CharacterBox characters retrieved: {len(characters)} characters")
        return utils.get_response(200, response_data)
        
    except Exception as e:
        logger.error(f"Failed to get CharacterBox characters: {str(e)}")
        raise HttpException(
            task_id=request_id,
            status_code=500,
            message=f"Failed to get CharacterBox characters: {str(e)}"
        )


@router.get(
    "/tts/characterbox/characters/{character_id}",
    response_model=BaseResponse,
    summary="Get CharacterBox character details",
    description="Get detailed information about a specific CharacterBox character"
)
async def get_characterbox_character_info(request: Request, character_id: str):
    """Get CharacterBox character information."""
    request_id = base.get_task_id(request)
    
    try:
        characterbox_service = get_characterbox_service()
        
        if not characterbox_service.is_available():
            raise HttpException(
                task_id=request_id,
                status_code=503,
                message="CharacterBox service not available"
            )
        
        character_info = await characterbox_service.get_character_info(character_id)
        
        if not character_info:
            raise HttpException(
                task_id=request_id,
                status_code=404,
                message=f"Character not found: {character_id}"
            )
        
        logger.info(f"CharacterBox character info retrieved: {character_id}")
        return utils.get_response(200, character_info)
        
    except HttpException:
        raise
    except Exception as e:
        logger.error(f"Failed to get CharacterBox character info: {str(e)}")
        raise HttpException(
            task_id=request_id,
            status_code=500,
            message=f"Failed to get CharacterBox character info: {str(e)}"
        )


@router.post(
    "/tts/characterbox/synthesize",
    response_model=BaseResponse,
    summary="Synthesize speech with CharacterBox",
    description="Convert text to speech using CharacterBox character voices"
)
async def characterbox_tts_synthesize(
    request: Request,
    background_tasks: BackgroundTasks,
    body: CharacterBoxTTSRequest
):
    """Synthesize speech using CharacterBox."""
    request_id = base.get_task_id(request)
    task_id = utils.get_uuid()
    
    try:
        characterbox_service = get_characterbox_service()
        
        if not characterbox_service.is_available():
            raise HttpException(
                task_id=task_id,
                status_code=503,
                message="CharacterBox service not available"
            )
        
        # Validate input text
        if not body.text.strip():
            raise HttpException(
                task_id=task_id,
                status_code=400,
                message="Text content cannot be empty"
            )
        
        logger.info(f"Starting CharacterBox TTS synthesis: {task_id}")
        
        # Synthesize speech
        audio_file = await characterbox_service.synthesize_character_speech(
            text=body.text,
            character_id=body.character_id,
            voice_settings=body.voice_settings,
            audio_format=body.audio_format,
            quality=body.quality
        )
        
        if not audio_file:
            raise HttpException(
                task_id=task_id,
                status_code=500,
                message="CharacterBox TTS synthesis failed"
            )
        
        # Get file info
        import os
        file_size = os.path.getsize(audio_file) if os.path.exists(audio_file) else 0
        
        response_data = {
            "task_id": task_id,
            "audio_file": audio_file,
            "service": "characterbox",
            "character_id": body.character_id,
            "file_size": file_size,
            "audio_format": body.audio_format,
            "quality": body.quality
        }
        
        logger.success(f"CharacterBox TTS synthesis completed: {task_id}")
        return utils.get_response(200, response_data)
        
    except HttpException:
        raise
    except Exception as e:
        logger.error(f"CharacterBox TTS synthesis error: {str(e)}")
        raise HttpException(
            task_id=task_id,
            status_code=500,
            message=f"CharacterBox TTS synthesis failed: {str(e)}"
        )


@router.get(
    "/tts/characterbox/history",
    response_model=BaseResponse,
    summary="Get CharacterBox synthesis history",
    description="Retrieve synthesis history for CharacterBox API"
)
async def get_characterbox_history(
    request: Request,
    limit: int = Query(50, ge=1, le=100, description="Maximum number of records"),
    offset: int = Query(0, ge=0, description="Number of records to skip")
):
    """Get CharacterBox synthesis history."""
    request_id = base.get_task_id(request)
    
    try:
        characterbox_service = get_characterbox_service()
        
        if not characterbox_service.is_available():
            raise HttpException(
                task_id=request_id,
                status_code=503,
                message="CharacterBox service not available"
            )
        
        history = await characterbox_service.get_synthesis_history(limit, offset)
        
        response_data = {
            "service": "characterbox",
            "history": history,
            "count": len(history),
            "limit": limit,
            "offset": offset
        }
        
        logger.info(f"CharacterBox history retrieved: {len(history)} records")
        return utils.get_response(200, response_data)
        
    except Exception as e:
        logger.error(f"Failed to get CharacterBox history: {str(e)}")
        raise HttpException(
            task_id=request_id,
            status_code=500,
            message=f"Failed to get CharacterBox history: {str(e)}"
        )


# Audio file download endpoint
@router.get(
    "/tts/download/{file_path:path}",
    summary="Download TTS audio file",
    description="Download generated TTS audio file"
)
async def download_tts_audio(request: Request, file_path: str):
    """Download TTS audio file."""
    request_id = base.get_task_id(request)
    
    try:
        import os
        from pathlib import Path
        
        # Validate file path and ensure it's within allowed directories
        full_path = Path(file_path).resolve()
        
        if not full_path.exists():
            raise HttpException(
                task_id=request_id,
                status_code=404,
                message="Audio file not found"
            )
        
        # Check if file is an audio file
        allowed_extensions = {'.mp3', '.wav', '.ogg', '.m4a'}
        if full_path.suffix.lower() not in allowed_extensions:
            raise HttpException(
                task_id=request_id,
                status_code=400,
                message="Invalid audio file format"
            )
        
        logger.info(f"Downloading TTS audio file: {file_path}")
        
        return FileResponse(
            path=str(full_path),
            filename=full_path.name,
            media_type=f"audio/{full_path.suffix[1:]}",
            headers={"Content-Disposition": f"attachment; filename={full_path.name}"}
        )
        
    except HttpException:
        raise
    except Exception as e:
        logger.error(f"Failed to download audio file: {str(e)}")
        raise HttpException(
            task_id=request_id,
            status_code=500,
            message=f"Failed to download audio file: {str(e)}"
        )