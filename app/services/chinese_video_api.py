"""
Chinese Video Generation API Integration
=======================================

FastAPI endpoints for Chinese video generation with GPT-SoVITS integration.
Provides REST API for creating Chinese motivational videos optimized for YouTube Shorts.

Author: VideoEngineer Agent  
Version: 2.0.0
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import os
import time
from loguru import logger

from app.services.chinese_video_workflow import (
    ChineseVideoWorkflow, 
    ChineseVideoRequest, 
    VideoGenerationResult,
    create_chinese_workflow
)
from app.services.video_generator_enhanced import VideoFormat


# Request/Response Models
class ChineseVideoGenerationRequest(BaseModel):
    """Request model for Chinese video generation"""
    script: str = Field(..., description="Chinese script text", min_length=10, max_length=2000)
    title: str = Field(..., description="Video title", min_length=1, max_length=200)
    keywords: Optional[List[str]] = Field(default=[], description="Search keywords for video materials")
    voice: str = Field(default="professional", description="GPT-SoVITS voice name")
    format: str = Field(default="youtube_shorts", description="Video format")
    duration: int = Field(default=30, description="Video duration in seconds", ge=10, le=60)
    background_music: bool = Field(default=True, description="Include background music")
    auto_upload: bool = Field(default=False, description="Auto-upload to YouTube")
    
    class Config:
        json_schema_extra = {
            "example": {
                "script": "成功不是偶然，而是坚持不懈的努力。每一天都是新的开始，每一步都是向前的力量。",
                "title": "每日励志 - 坚持的力量", 
                "keywords": ["励志", "成功", "坚持"],
                "voice": "professional",
                "format": "youtube_shorts",
                "duration": 30,
                "background_music": True,
                "auto_upload": False
            }
        }


class BatchVideoRequest(BaseModel):
    """Request model for batch video generation"""
    videos: List[ChineseVideoGenerationRequest] = Field(..., description="List of video requests")
    max_concurrent: int = Field(default=2, description="Maximum concurrent generations", ge=1, le=5)
    
    class Config:
        json_schema_extra = {
            "example": {
                "videos": [
                    {
                        "script": "成功从小事开始，坚持每一天的努力。",
                        "title": "励志短片1",
                        "keywords": ["成功", "坚持"]
                    },
                    {
                        "script": "梦想需要勇气去追求，永不放弃就是胜利。",
                        "title": "励志短片2", 
                        "keywords": ["梦想", "勇气"]
                    }
                ],
                "max_concurrent": 2
            }
        }


class VideoGenerationResponse(BaseModel):
    """Response model for video generation"""
    success: bool
    video_id: Optional[str] = None
    video_path: Optional[str] = None
    download_url: Optional[str] = None
    duration: Optional[float] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class VideoStatusResponse(BaseModel):
    """Response model for video generation status"""
    video_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float = 0.0
    video_path: Optional[str] = None
    download_url: Optional[str] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status"""
    initialized: bool
    background_music_tracks: int
    supported_formats: List[str]
    available_voices: List[str]
    default_duration: int
    max_duration: int


# Global workflow instance and job tracking
_workflow: Optional[ChineseVideoWorkflow] = None
_active_jobs: Dict[str, Dict[str, Any]] = {}


def get_workflow() -> ChineseVideoWorkflow:
    """Get or create workflow instance"""
    global _workflow
    if _workflow is None:
        _workflow = create_chinese_workflow()
    return _workflow


def generate_video_id() -> str:
    """Generate unique video ID"""
    return f"video_{int(time.time())}_{os.urandom(4).hex()}"


# API Router
router = APIRouter(prefix="/api/v1/chinese-video", tags=["Chinese Video Generation"])


@router.post("/generate", response_model=VideoGenerationResponse)
async def generate_chinese_video(
    request: ChineseVideoGenerationRequest,
    background_tasks: BackgroundTasks
) -> VideoGenerationResponse:
    """
    Generate a Chinese video with GPT-SoVITS voice synthesis
    
    This endpoint creates a complete Chinese video with:
    - GPT-SoVITS voice synthesis
    - 9:16 vertical video format optimization
    - Automated Chinese subtitles
    - Background music integration
    """
    try:
        video_id = generate_video_id()
        
        # Convert request to internal format
        video_request = ChineseVideoRequest(
            script=request.script,
            title=request.title,
            keywords=request.keywords,
            voice=request.voice,
            format=request.format,
            duration=request.duration,
            background_music=request.background_music,
            auto_upload=request.auto_upload,
            output_dir=f"storage/output/{video_id}"
        )
        
        # Initialize job tracking
        _active_jobs[video_id] = {
            "status": "processing",
            "progress": 0.0,
            "created_at": time.time(),
            "request": video_request
        }
        
        logger.info(f"🎬 Starting video generation: {video_id}")
        
        # Start background generation
        background_tasks.add_task(
            _process_video_generation,
            video_id,
            video_request
        )
        
        return VideoGenerationResponse(
            success=True,
            video_id=video_id,
            metadata={
                "status": "processing",
                "format": request.format,
                "duration": request.duration,
                "voice": request.voice
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{video_id}", response_model=VideoStatusResponse)
async def get_video_status(video_id: str) -> VideoStatusResponse:
    """
    Get the status of a video generation job
    
    Returns current status, progress, and download URL when complete.
    """
    if video_id not in _active_jobs:
        raise HTTPException(status_code=404, detail="Video not found")
    
    job = _active_jobs[video_id]
    
    download_url = None
    if job["status"] == "completed" and job.get("video_path"):
        download_url = f"/api/v1/chinese-video/download/{video_id}"
    
    return VideoStatusResponse(
        video_id=video_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        video_path=job.get("video_path"),
        download_url=download_url,
        error=job.get("error"),
        created_at=job["created_at"],
        completed_at=job.get("completed_at")
    )


@router.get("/download/{video_id}")
async def download_video(video_id: str) -> FileResponse:
    """
    Download the generated video file
    
    Returns the video file as a downloadable response.
    """
    if video_id not in _active_jobs:
        raise HTTPException(status_code=404, detail="Video not found")
    
    job = _active_jobs[video_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video not ready for download")
    
    video_path = job.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    filename = os.path.basename(video_path)
    return FileResponse(
        path=video_path,
        filename=filename,
        media_type="video/mp4"
    )


@router.post("/batch", response_model=List[VideoGenerationResponse])
async def batch_generate_videos(
    request: BatchVideoRequest,
    background_tasks: BackgroundTasks
) -> List[VideoGenerationResponse]:
    """
    Generate multiple Chinese videos in batch
    
    Processes multiple video requests concurrently with configurable limits.
    """
    try:
        responses = []
        
        for video_request in request.videos:
            video_id = generate_video_id()
            
            # Convert to internal format
            internal_request = ChineseVideoRequest(
                script=video_request.script,
                title=video_request.title,
                keywords=video_request.keywords,
                voice=video_request.voice,
                format=video_request.format,
                duration=video_request.duration,
                background_music=video_request.background_music,
                auto_upload=video_request.auto_upload,
                output_dir=f"storage/output/{video_id}"
            )
            
            # Initialize job tracking
            _active_jobs[video_id] = {
                "status": "pending",
                "progress": 0.0,
                "created_at": time.time(),
                "request": internal_request
            }
            
            responses.append(VideoGenerationResponse(
                success=True,
                video_id=video_id,
                metadata={
                    "status": "pending",
                    "format": video_request.format,
                    "title": video_request.title
                }
            ))
        
        # Start batch processing
        background_tasks.add_task(
            _process_batch_generation,
            [job["request"] for job in _active_jobs.values() if job["status"] == "pending"],
            request.max_concurrent
        )
        
        logger.info(f"🎬 Started batch generation of {len(request.videos)} videos")
        
        return responses
        
    except Exception as e:
        logger.error(f"Failed to start batch generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/formats")
async def get_supported_formats() -> Dict[str, Any]:
    """
    Get supported video formats and their specifications
    """
    formats = {
        "youtube_shorts": {
            "name": "YouTube Shorts",
            "aspect_ratio": "9:16",
            "resolution": "1080x1920",
            "max_duration": 60,
            "recommended_duration": 30
        },
        "instagram_reels": {
            "name": "Instagram Reels", 
            "aspect_ratio": "9:16",
            "resolution": "1080x1920",
            "max_duration": 60,
            "recommended_duration": 30
        },
        "tiktok": {
            "name": "TikTok",
            "aspect_ratio": "9:16", 
            "resolution": "1080x1920",
            "max_duration": 60,
            "recommended_duration": 15
        },
        "landscape": {
            "name": "Landscape",
            "aspect_ratio": "16:9",
            "resolution": "1920x1080", 
            "max_duration": 300,
            "recommended_duration": 60
        }
    }
    
    return {
        "formats": formats,
        "default_format": "youtube_shorts"
    }


@router.get("/voices")
async def get_available_voices() -> Dict[str, Any]:
    """
    Get available GPT-SoVITS voices
    """
    try:
        workflow = get_workflow()
        voices = workflow.get_workflow_status()["available_voices"]
        
        voice_details = {
            "professional": {
                "name": "专业播报",
                "gender": "male",
                "language": "zh-CN",
                "description": "清晰专业的男声播报"
            },
            "warm_female": {
                "name": "温和女声",
                "gender": "female", 
                "language": "zh-CN",
                "description": "温和甜美的女声"
            },
            "energetic": {
                "name": "活力青春",
                "gender": "male",
                "language": "zh-CN", 
                "description": "充满活力的青年男声"
            }
        }
        
        return {
            "voices": voice_details,
            "available": voices,
            "default_voice": "professional"
        }
        
    except Exception as e:
        logger.error(f"Failed to get voices: {str(e)}")
        return {
            "voices": {},
            "available": [],
            "default_voice": "professional",
            "error": str(e)
        }


@router.get("/status")
async def get_workflow_status() -> WorkflowStatusResponse:
    """
    Get the current status of the video generation workflow
    """
    try:
        workflow = get_workflow()
        status = workflow.get_workflow_status()
        
        return WorkflowStatusResponse(
            initialized=status["initialized"],
            background_music_tracks=status["background_music_tracks"],
            supported_formats=status["supported_formats"],
            available_voices=status["available_voices"],
            default_duration=status["default_duration"],
            max_duration=status["max_duration"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{video_id}")
async def cancel_video_job(video_id: str) -> Dict[str, str]:
    """
    Cancel a video generation job
    """
    if video_id not in _active_jobs:
        raise HTTPException(status_code=404, detail="Video not found")
    
    job = _active_jobs[video_id]
    
    if job["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel {job['status']} job")
    
    job["status"] = "cancelled"
    job["completed_at"] = time.time()
    
    logger.info(f"🚫 Video job cancelled: {video_id}")
    
    return {"message": f"Video job {video_id} cancelled"}


# Background task functions
async def _process_video_generation(video_id: str, request: ChineseVideoRequest):
    """Background task for processing video generation"""
    try:
        job = _active_jobs[video_id]
        job["status"] = "processing"
        job["progress"] = 10.0
        
        workflow = get_workflow()
        
        # Update progress
        job["progress"] = 30.0
        
        # Generate video
        result = await workflow.generate_chinese_video(request)
        
        if result.success:
            job["status"] = "completed"
            job["progress"] = 100.0
            job["video_path"] = result.video_path
            job["duration"] = result.duration
            job["processing_time"] = result.processing_time
            job["completed_at"] = time.time()
            
            logger.success(f"✅ Video generation completed: {video_id}")
        else:
            job["status"] = "failed"
            job["error"] = result.error
            job["completed_at"] = time.time()
            
            logger.error(f"❌ Video generation failed: {video_id} - {result.error}")
            
    except Exception as e:
        job = _active_jobs.get(video_id, {})
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = time.time()
        
        logger.error(f"❌ Video generation exception: {video_id} - {str(e)}")


async def _process_batch_generation(requests: List[ChineseVideoRequest], max_concurrent: int):
    """Background task for batch video generation"""
    try:
        workflow = get_workflow()
        
        # Process in batches with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(request):
            async with semaphore:
                return await workflow.generate_chinese_video(request)
        
        # Execute batch
        tasks = [process_single(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"🎉 Batch processing completed: {len(results)} videos")
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {str(e)}")


# Export router
__all__ = ["router"]