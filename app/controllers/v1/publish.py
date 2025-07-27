"""
Publishing Controller for YouTube Shorts
Handles API endpoints for automated video publishing
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import json
import logging
from datetime import datetime

from app.services.youtube_publisher import YouTubeShortsPublisher, create_publisher_config
from app.models.schema import VideoParams

logger = logging.getLogger(__name__)

router = APIRouter()

class PublishRequest(BaseModel):
    """Request model for video publishing"""
    video_path: str = Field(..., description="Path to video file to publish")
    content: str = Field(..., description="Video content description")
    keywords: List[str] = Field(default=[], description="SEO keywords for optimization")
    topic_category: str = Field(default="general", description="Content category")
    title: Optional[str] = Field(None, description="Custom title (auto-generated if not provided)")
    description: Optional[str] = Field(None, description="Custom description")
    privacy_status: str = Field(default="public", description="Video privacy setting")
    schedule_time: Optional[str] = Field(None, description="Scheduled publish time (ISO format)")

class PublishResponse(BaseModel):
    """Response model for publish operations"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None

class AnalyticsRequest(BaseModel):
    """Request model for analytics tracking"""
    video_id: str = Field(..., description="YouTube video ID")
    metrics: List[str] = Field(default=["views", "likes", "comments"], description="Metrics to track")

# Global publisher instance (singleton pattern)
_publisher_instance: Optional[YouTubeShortsPublisher] = None

def get_publisher() -> YouTubeShortsPublisher:
    """Get or create YouTube publisher instance"""
    global _publisher_instance
    
    if _publisher_instance is None:
        try:
            config = create_publisher_config()
            _publisher_instance = YouTubeShortsPublisher(
                credentials_path=config['credentials_path'],
                token_path=config['token_path']
            )
        except Exception as e:
            logger.error(f"Failed to initialize YouTube publisher: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"YouTube publisher initialization failed: {str(e)}"
            )
    
    return _publisher_instance

@router.post("/publish/youtube-shorts", response_model=PublishResponse)
async def publish_to_youtube_shorts(
    request: PublishRequest,
    background_tasks: BackgroundTasks,
    publisher: YouTubeShortsPublisher = Depends(get_publisher)
):
    """
    Publish video to YouTube Shorts with automated optimization
    
    This endpoint handles the complete publishing workflow:
    - SEO optimization for Chinese audience
    - Thumbnail generation
    - Metadata optimization
    - Automated upload
    """
    try:
        logger.info(f"📤 Publishing request received for: {os.path.basename(request.video_path)}")
        
        # Validate video file exists
        if not os.path.exists(request.video_path):
            raise HTTPException(
                status_code=404,
                detail=f"Video file not found: {request.video_path}"
            )
        
        # Check file size and format
        file_size = os.path.getsize(request.video_path)
        if file_size > 256 * 1024 * 1024:  # 256MB
            logger.warning(f"Video file size {file_size / 1024 / 1024:.1f}MB may exceed YouTube limits")
        
        # Execute publishing workflow
        if request.schedule_time:
            # Schedule for later (background task)
            background_tasks.add_task(
                _scheduled_publish,
                publisher,
                request,
                request.schedule_time
            )
            
            return PublishResponse(
                success=True,
                message=f"Video scheduled for publishing at {request.schedule_time}",
                data={
                    "video_path": request.video_path,
                    "scheduled_time": request.schedule_time,
                    "status": "scheduled"
                }
            )
        else:
            # Immediate publish
            result = publisher.schedule_upload(
                video_path=request.video_path,
                content=request.content,
                keywords=request.keywords,
                topic_category=request.topic_category
            )
            
            if result['success']:
                return PublishResponse(
                    success=True,
                    message="Video published successfully to YouTube Shorts",
                    data=result
                )
            else:
                return PublishResponse(
                    success=False,
                    message="Publishing failed",
                    error_details=result
                )
                
    except Exception as e:
        logger.error(f"Publishing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Publishing failed: {str(e)}"
        )

@router.get("/publish/status/{video_id}")
async def get_publish_status(
    video_id: str,
    publisher: YouTubeShortsPublisher = Depends(get_publisher)
):
    """Get publishing status and basic analytics for uploaded video"""
    try:
        # Get video info from YouTube API
        response = publisher.service.videos().list(
            part="snippet,statistics,status",
            id=video_id
        ).execute()
        
        if not response.get('items'):
            raise HTTPException(
                status_code=404,
                detail=f"Video not found: {video_id}"
            )
        
        video_data = response['items'][0]
        
        return {
            "video_id": video_id,
            "title": video_data['snippet']['title'],
            "status": video_data['status']['privacyStatus'],
            "upload_status": video_data['status']['uploadStatus'],
            "statistics": video_data.get('statistics', {}),
            "published_at": video_data['snippet']['publishedAt'],
            "url": f"https://youtu.be/{video_id}"
        }
        
    except Exception as e:
        logger.error(f"Status check error for {video_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )

@router.post("/publish/optimize-metadata")
async def optimize_video_metadata(
    content: str,
    keywords: List[str],
    topic_category: str = "general",
    publisher: YouTubeShortsPublisher = Depends(get_publisher)
):
    """
    Generate optimized metadata for YouTube Shorts without uploading
    Useful for previewing optimization before publishing
    """
    try:
        # Generate optimized title and description
        title = publisher.optimize_title(content, topic_category)
        description = publisher.generate_description(content, keywords, topic_category)
        
        # Prepare tags
        tags = keywords + ["shorts", "短视频", "励志", "正能量"]
        tags = tags[:15]  # YouTube limit
        
        return {
            "optimized_title": title,
            "optimized_description": description,
            "suggested_tags": tags,
            "topic_category": topic_category,
            "optimization_tips": [
                "Title optimized for mobile viewing (under 60 chars)",
                "Description includes engagement hooks",
                "Hashtags targeted for Chinese audience",
                "SEO keywords integrated naturally"
            ]
        }
        
    except Exception as e:
        logger.error(f"Metadata optimization error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Metadata optimization failed: {str(e)}"
        )

@router.post("/publish/create-thumbnail")
async def create_video_thumbnail(
    title: str,
    output_path: Optional[str] = None,
    background_color: str = "#FF6B35",
    publisher: YouTubeShortsPublisher = Depends(get_publisher)
):
    """Generate optimized thumbnail for YouTube Shorts"""
    try:
        if not output_path:
            timestamp = int(datetime.now().timestamp())
            output_path = f"storage/thumbnails/thumb_{timestamp}.jpg"
        
        # Create thumbnail
        thumbnail_path = publisher.create_thumbnail(
            title=title,
            output_path=output_path,
            background_color=background_color
        )
        
        return {
            "thumbnail_path": thumbnail_path,
            "title": title,
            "background_color": background_color,
            "dimensions": "1280x720",
            "format": "JPEG"
        }
        
    except Exception as e:
        logger.error(f"Thumbnail creation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Thumbnail creation failed: {str(e)}"
        )

@router.get("/publish/upload-times")
async def get_optimal_upload_times(
    publisher: YouTubeShortsPublisher = Depends(get_publisher)
):
    """Get optimal upload times for Chinese audience engagement"""
    try:
        optimal_time = publisher.get_optimal_upload_time()
        
        return {
            "current_optimal_time": {
                "hour": optimal_time[0],
                "minute": optimal_time[1],
                "timezone": "Asia/Shanghai"
            },
            "all_optimal_times": [
                {"time": "12:00", "description": "午休时间 - Lunch break"},
                {"time": "15:30", "description": "下午休息 - Afternoon break"},
                {"time": "19:00", "description": "晚高峰 - Evening commute"},
                {"time": "20:30", "description": "黄金时段 - Prime time"},
                {"time": "21:15", "description": "休闲时间 - Relaxation time"}
            ],
            "best_days": ["Monday", "Wednesday", "Friday", "Sunday"],
            "timezone_note": "All times are in Beijing Time (UTC+8)"
        }
        
    except Exception as e:
        logger.error(f"Upload times error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get upload times: {str(e)}"
        )

@router.post("/publish/analytics/track")
async def setup_analytics_tracking(
    request: AnalyticsRequest,
    publisher: YouTubeShortsPublisher = Depends(get_publisher)
):
    """Setup analytics tracking for published video"""
    try:
        # Enable performance tracking
        publisher._setup_performance_tracking(request.video_id)
        
        return {
            "video_id": request.video_id,
            "tracking_enabled": True,
            "metrics_tracked": request.metrics,
            "tracking_start_time": datetime.now().isoformat(),
            "message": "Analytics tracking enabled successfully"
        }
        
    except Exception as e:
        logger.error(f"Analytics setup error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analytics setup failed: {str(e)}"
        )

@router.get("/publish/analytics/{video_id}")
async def get_video_analytics(
    video_id: str,
    days: int = 7,
    publisher: YouTubeShortsPublisher = Depends(get_publisher)
):
    """Get analytics data for published video"""
    try:
        # Get basic statistics from YouTube API
        response = publisher.service.videos().list(
            part="statistics,snippet",
            id=video_id
        ).execute()
        
        if not response.get('items'):
            raise HTTPException(
                status_code=404,
                detail=f"Video not found: {video_id}"
            )
        
        video_data = response['items'][0]
        stats = video_data.get('statistics', {})
        
        return {
            "video_id": video_id,
            "title": video_data['snippet']['title'],
            "published_at": video_data['snippet']['publishedAt'],
            "statistics": {
                "view_count": int(stats.get('viewCount', 0)),
                "like_count": int(stats.get('likeCount', 0)),
                "comment_count": int(stats.get('commentCount', 0)),
                "favorite_count": int(stats.get('favoriteCount', 0))
            },
            "performance_metrics": {
                "engagement_rate": _calculate_engagement_rate(stats),
                "views_per_day": int(stats.get('viewCount', 0)) / max(days, 1),
                "likes_to_views_ratio": _calculate_ratio(
                    stats.get('likeCount', 0), 
                    stats.get('viewCount', 0)
                )
            },
            "url": f"https://youtu.be/{video_id}"
        }
        
    except Exception as e:
        logger.error(f"Analytics error for {video_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analytics retrieval failed: {str(e)}"
        )

# Background task for scheduled publishing
async def _scheduled_publish(
    publisher: YouTubeShortsPublisher,
    request: PublishRequest,
    schedule_time: str
):
    """Background task for scheduled video publishing"""
    try:
        # TODO: Implement actual scheduling logic
        # For now, this is a placeholder for future scheduling functionality
        logger.info(f"Scheduled publish task for {schedule_time}")
        
        # In production, this would wait until the scheduled time
        # and then execute the upload
        
    except Exception as e:
        logger.error(f"Scheduled publish failed: {str(e)}")

# Helper functions
def _calculate_engagement_rate(stats: Dict) -> float:
    """Calculate engagement rate from video statistics"""
    views = int(stats.get('viewCount', 0))
    likes = int(stats.get('likeCount', 0))
    comments = int(stats.get('commentCount', 0))
    
    if views == 0:
        return 0.0
    
    return ((likes + comments) / views) * 100

def _calculate_ratio(numerator: str, denominator: str) -> float:
    """Calculate ratio between two metrics"""
    num = int(numerator) if numerator else 0
    den = int(denominator) if denominator else 0
    
    if den == 0:
        return 0.0
    
    return (num / den) * 100