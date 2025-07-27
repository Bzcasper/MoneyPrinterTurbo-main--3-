"""
Chinese Video Generation Workflow
================================

Automated workflow for creating Chinese motivational videos with GPT-SoVITS voice synthesis,
optimized for YouTube Shorts and other vertical video platforms.

Features:
- GPT-SoVITS Chinese voice synthesis
- 9:16 vertical video optimization  
- Automated Chinese subtitle generation
- Background music integration
- YouTube Shorts workflow
- Batch processing capabilities

Author: VideoEngineer Agent
Version: 2.0.0
"""

import asyncio
import os
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from loguru import logger

from app.services.video_generator_enhanced import (
    EnhancedVideoGenerator, 
    ChineseVideoConfig, 
    VideoFormat, 
    create_chinese_video_generator
)
from app.services.material import search_and_download_videos
from app.config import config


@dataclass
class ChineseVideoRequest:
    """Chinese video generation request"""
    script: str
    title: str
    keywords: List[str] = field(default_factory=list)
    voice: str = "professional"
    format: str = "youtube_shorts"  
    duration: int = 30
    background_music: bool = True
    auto_upload: bool = False
    output_dir: str = "storage/output"
    

@dataclass  
class VideoGenerationResult:
    """Video generation result"""
    success: bool
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    subtitle_path: Optional[str] = None
    duration: float = 0.0
    processing_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChineseVideoWorkflow:
    """Complete workflow for Chinese video generation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = config
        self.video_generator = None
        self.background_music_library = self._discover_background_music()
        
        logger.info("Chinese video workflow initialized")
        
    def _discover_background_music(self) -> List[str]:
        """Discover available background music files"""
        music_dirs = [
            "resource/songs",
            "resource/music", 
            "storage/music"
        ]
        
        music_files = []
        for music_dir in music_dirs:
            if os.path.exists(music_dir):
                for file in os.listdir(music_dir):
                    if file.lower().endswith(('.mp3', '.wav', '.aac', '.m4a')):
                        music_files.append(os.path.join(music_dir, file))
        
        logger.info(f"Discovered {len(music_files)} background music tracks")
        return music_files
    
    async def generate_chinese_video(self, request: ChineseVideoRequest) -> VideoGenerationResult:
        """
        Generate a complete Chinese video from request
        
        Args:
            request: Video generation request
            
        Returns:
            Generation result with paths and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎬 Starting Chinese video workflow")
            logger.info(f"   📝 Title: {request.title}")
            logger.info(f"   🎯 Format: {request.format}")
            logger.info(f"   ⏰ Duration: {request.duration}s")
            
            # Step 1: Initialize video generator with request config
            self.video_generator = create_chinese_video_generator(
                format=request.format,
                duration=request.duration,
                voice=request.voice
            )
            
            # Step 2: Search and download video materials
            logger.info("🔍 Searching for video materials...")
            video_materials = await self._get_video_materials(request)
            
            if not video_materials:
                raise Exception("No suitable video materials found")
            
            # Step 3: Select background music
            background_music = None
            if request.background_music and self.background_music_library:
                background_music = random.choice(self.background_music_library)
                logger.info(f"🎵 Selected background music: {os.path.basename(background_music)}")
            
            # Step 4: Generate output paths
            output_paths = self._prepare_output_paths(request)
            
            # Step 5: Generate the video
            logger.info("🎥 Generating video...")
            result = await self.video_generator.generate_chinese_video(
                script_text=request.script,
                video_materials=video_materials,
                output_path=output_paths['video'],
                background_music=background_music
            )
            
            # Step 6: Create result object
            processing_time = time.time() - start_time
            
            video_result = VideoGenerationResult(
                success=True,
                video_path=result['output_path'],
                duration=result['audio_duration'],
                processing_time=processing_time,
                metadata={
                    'title': request.title,
                    'format': request.format,
                    'resolution': result['resolution'],
                    'voice': request.voice,
                    'materials_count': len(video_materials),
                    'background_music': background_music is not None,
                    'subtitles': True,
                    'gpu_acceleration': result['metrics']['gpu_acceleration']
                }
            )
            
            # Step 7: Store workflow results
            await self._store_workflow_results(request, video_result)
            
            logger.success("🎉 Chinese video workflow completed!")
            logger.success(f"   ⏱️  Total time: {processing_time:.1f}s")
            logger.success(f"   📁 Output: {video_result.video_path}")
            
            return video_result
            
        except Exception as e:
            error_msg = f"Video workflow failed: {str(e)}"
            logger.error(error_msg)
            
            return VideoGenerationResult(
                success=False,
                error=error_msg,
                processing_time=time.time() - start_time
            )
    
    async def _get_video_materials(self, request: ChineseVideoRequest) -> List[str]:
        """Get video materials based on request keywords"""
        
        # Use keywords from request, or generate from script
        search_terms = request.keywords if request.keywords else self._extract_keywords(request.script)
        
        if not search_terms:
            search_terms = ["motivation", "success", "business"]  # Default keywords
        
        logger.info(f"🔍 Searching materials for: {', '.join(search_terms[:3])}")
        
        # Search and download videos
        video_count = max(3, min(8, request.duration // 5))  # 3-8 videos based on duration
        
        try:
            # Use existing material search system
            downloaded_videos = []
            for term in search_terms[:3]:  # Use top 3 terms
                videos = await search_and_download_videos(
                    query=term,
                    count=video_count // len(search_terms[:3]) + 1,
                    duration_range=(5, 30),
                    quality="hd"
                )
                downloaded_videos.extend(videos)
            
            # Remove duplicates and limit count
            unique_videos = list(set(downloaded_videos))[:video_count]
            
            logger.info(f"📥 Downloaded {len(unique_videos)} video materials")
            return unique_videos
            
        except Exception as e:
            logger.error(f"Material search failed: {str(e)}")
            
            # Fallback: use local test materials if available
            test_materials = self._get_test_materials()
            if test_materials:
                logger.warning("Using test materials as fallback")
                return test_materials[:video_count]
            
            raise Exception("No video materials available")
    
    def _extract_keywords(self, script: str) -> List[str]:
        """Extract keywords from Chinese script"""
        # Simple keyword extraction for Chinese text
        # In production, use proper Chinese NLP
        
        common_keywords = [
            "成功", "动机", "励志", "目标", "梦想", "坚持", "努力",
            "创业", "商业", "财富", "投资", "学习", "成长", "改变"
        ]
        
        found_keywords = []
        for keyword in common_keywords:
            if keyword in script:
                found_keywords.append(keyword)
        
        # Add English equivalents for video search
        keyword_mapping = {
            "成功": "success",
            "动机": "motivation", 
            "励志": "inspiration",
            "目标": "goals",
            "梦想": "dreams",
            "坚持": "persistence",
            "努力": "hard work",
            "创业": "entrepreneur",
            "商业": "business",
            "财富": "wealth",
            "投资": "investment",
            "学习": "learning",
            "成长": "growth",
            "改变": "change"
        }
        
        english_keywords = [keyword_mapping.get(kw, kw) for kw in found_keywords]
        return english_keywords[:5]  # Return top 5
    
    def _get_test_materials(self) -> List[str]:
        """Get test video materials for fallback"""
        test_dirs = [
            "test/resources",
            "storage/test_videos",
            "resource/test"
        ]
        
        test_videos = []
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for file in os.listdir(test_dir):
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        test_videos.append(os.path.join(test_dir, file))
        
        return test_videos
    
    def _prepare_output_paths(self, request: ChineseVideoRequest) -> Dict[str, str]:
        """Prepare output file paths"""
        timestamp = int(time.time())
        safe_title = "".join(c for c in request.title if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
        
        base_name = f"{safe_title}_{timestamp}"
        
        # Ensure output directory exists
        os.makedirs(request.output_dir, exist_ok=True)
        
        return {
            'video': os.path.join(request.output_dir, f"{base_name}.mp4"),
            'audio': os.path.join(request.output_dir, f"{base_name}.wav"),
            'subtitle': os.path.join(request.output_dir, f"{base_name}.srt"),
            'metadata': os.path.join(request.output_dir, f"{base_name}_metadata.json")
        }
    
    async def _store_workflow_results(self, request: ChineseVideoRequest, result: VideoGenerationResult):
        """Store workflow results in coordination memory"""
        try:
            import json
            from app.services.hive_memory import store_memory
            
            workflow_data = {
                'request': {
                    'title': request.title,
                    'script_length': len(request.script),
                    'keywords': request.keywords,
                    'format': request.format,
                    'duration': request.duration,
                    'voice': request.voice
                },
                'result': {
                    'success': result.success,
                    'video_path': result.video_path,
                    'duration': result.duration,
                    'processing_time': result.processing_time,
                    'error': result.error
                },
                'metadata': result.metadata
            }
            
            await store_memory(
                key=f"chinese_workflow/{int(time.time())}",
                data=workflow_data,
                metadata={
                    'agent': 'VideoEngineer',
                    'type': 'workflow_result',
                    'timestamp': time.time()
                }
            )
            
            logger.debug("Workflow results stored in coordination memory")
            
        except Exception as e:
            logger.warning(f"Failed to store workflow results: {str(e)}")
    
    async def batch_generate_videos(self, requests: List[ChineseVideoRequest]) -> List[VideoGenerationResult]:
        """Generate multiple videos in batch"""
        logger.info(f"🎬 Starting batch generation of {len(requests)} videos")
        
        results = []
        for i, request in enumerate(requests, 1):
            logger.info(f"📹 Processing video {i}/{len(requests)}: {request.title}")
            
            try:
                result = await self.generate_chinese_video(request)
                results.append(result)
                
                if result.success:
                    logger.success(f"✅ Video {i} completed: {result.video_path}")
                else:
                    logger.error(f"❌ Video {i} failed: {result.error}")
                
                # Brief pause between videos
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Video {i} failed with exception: {str(e)}")
                results.append(VideoGenerationResult(
                    success=False,
                    error=str(e)
                ))
        
        success_count = sum(1 for r in results if r.success)
        logger.info(f"🎉 Batch generation completed: {success_count}/{len(requests)} successful")
        
        return results
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            'initialized': self.video_generator is not None,
            'background_music_tracks': len(self.background_music_library),
            'supported_formats': ['youtube_shorts', 'instagram_reels', 'tiktok'],
            'available_voices': self.video_generator.get_available_voices() if self.video_generator else [],
            'default_duration': 30,
            'max_duration': 60
        }


# Factory function
def create_chinese_workflow() -> ChineseVideoWorkflow:
    """Create Chinese video workflow instance"""
    return ChineseVideoWorkflow()


# Example usage function
async def generate_sample_chinese_video():
    """Generate a sample Chinese motivational video"""
    
    workflow = create_chinese_workflow()
    
    sample_request = ChineseVideoRequest(
        script="""
        成功不是偶然，而是坚持不懈的努力。
        每一天都是新的开始，每一步都是向前的力量。
        相信自己，永不放弃，梦想就在前方等着你。
        今天就开始行动，让改变从现在开始！
        """,
        title="每日励志 - 坚持的力量",
        keywords=["励志", "成功", "坚持", "梦想"],
        voice="professional",
        format="youtube_shorts",
        duration=30,
        background_music=True
    )
    
    result = await workflow.generate_chinese_video(sample_request)
    
    if result.success:
        logger.success(f"Sample video generated: {result.video_path}")
    else:
        logger.error(f"Sample video failed: {result.error}")
    
    return result


# Export main classes
__all__ = [
    'ChineseVideoWorkflow',
    'ChineseVideoRequest', 
    'VideoGenerationResult',
    'create_chinese_workflow',
    'generate_sample_chinese_video'
]