"""
Enhanced Video Generation System with GPT-SoVITS Integration
=========================================================

Advanced video generation system optimized for Chinese content creation with:
- GPT-SoVITS voice synthesis integration  
- 9:16 vertical video format optimization
- Automated Chinese subtitle generation
- YouTube Shorts workflow optimization
- GPU-accelerated processing

Author: VideoEngineer Agent
Version: 2.0.0
"""

import asyncio
import os
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from loguru import logger

from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, ImageClip
from moviepy.video.fx import Resize, Crop
from moviepy.config import change_settings

from app.services.video_pipeline import EnhancedVideoPipeline, PipelineConfig, ProcessingStrategy
from app.services.gpu_manager import get_gpu_manager, GPUVendor
from app.services.voice import gpt_sovits_tts, create_subtitle, get_audio_duration
from app.config import config


class VideoFormat(Enum):
    """Video format specifications"""
    YOUTUBE_SHORTS = "youtube_shorts"  # 9:16 vertical
    INSTAGRAM_REELS = "instagram_reels"  # 9:16 vertical  
    TIKTOK = "tiktok"  # 9:16 vertical
    LANDSCAPE = "landscape"  # 16:9 horizontal
    SQUARE = "square"  # 1:1 square


@dataclass
class VideoSpecs:
    """Video format specifications"""
    width: int
    height: int
    aspect_ratio: str
    bitrate: str = "4M"
    fps: int = 30
    duration_limit: int = 60  # seconds
    
    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass 
class ChineseVideoConfig:
    """Configuration for Chinese video generation"""
    format: VideoFormat = VideoFormat.YOUTUBE_SHORTS
    duration: int = 30  # seconds
    language: str = "zh-CN"
    voice_model: str = "gpt-sovits-v2"
    voice_name: str = "professional"
    subtitle_style: Dict[str, Any] = field(default_factory=lambda: {
        'fontsize': 48,
        'color': 'white',
        'font': 'resource/fonts/MicrosoftYaHeiBold.ttc',
        'stroke_color': 'black',
        'stroke_width': 2,
        'method': 'caption'
    })
    background_music: bool = True
    auto_captions: bool = True


class EnhancedGPTSoVITSClient:
    """Enhanced GPT-SoVITS client with advanced features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_base_url = config.get("api_base_url", "http://localhost:9880")
        self.api_key = config.get("api_key", "")
        self.timeout = config.get("timeout", 60)
        
        # Voice models and configurations
        self.voice_models = {
            "professional": {
                "model_path": "models/professional_cn.pth",
                "config_path": "configs/professional_cn.json",
                "description": "专业播报声音",
                "gender": "male",
                "language": "zh-CN"
            },
            "warm_female": {
                "model_path": "models/warm_female_cn.pth", 
                "config_path": "configs/warm_female_cn.json",
                "description": "温和女声",
                "gender": "female",
                "language": "zh-CN"
            },
            "energetic": {
                "model_path": "models/energetic_cn.pth",
                "config_path": "configs/energetic_cn.json", 
                "description": "活力青春声音",
                "gender": "male",
                "language": "zh-CN"
            }
        }
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voice models"""
        return list(self.voice_models.keys())
    
    def generate_speech(
        self,
        text: str,
        voice: str = "professional",
        speed: float = 1.0,
        emotion: str = "neutral",
        output_path: str = None
    ) -> Tuple[str, float]:
        """
        Generate speech using GPT-SoVITS
        
        Returns:
            Tuple of (audio_file_path, duration_seconds)
        """
        if voice not in self.voice_models:
            raise ValueError(f"Voice '{voice}' not available. Available: {list(self.voice_models.keys())}")
        
        voice_config = self.voice_models[voice]
        
        # Prepare request payload
        payload = {
            "text": text,
            "text_lang": "auto",
            "ref_audio_path": voice_config["model_path"],
            "aux_ref_audio_paths": [],
            "prompt_lang": "auto", 
            "prompt_text": "",
            "top_k": 5,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut5",
            "batch_size": 1,
            "speed_factor": speed,
            "fragment_interval": 0.3,
            "seed": -1,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
            "repetition_penalty": 1.35
        }
        
        # Add emotion control if supported
        if emotion != "neutral":
            payload["emotion_control"] = {
                "emotion": emotion,
                "intensity": 0.8
            }
        
        try:
            import requests
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            url = f"{self.api_base_url.rstrip('/')}/tts"
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                # Save audio file
                if not output_path:
                    output_path = f"temp_gpt_sovits_{int(time.time())}.wav"
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                # Get audio duration
                audio_clip = AudioFileClip(output_path)
                duration = audio_clip.duration
                audio_clip.close()
                
                logger.success(f"GPT-SoVITS speech generated: {output_path} ({duration:.2f}s)")
                return output_path, duration
            
            else:
                raise Exception(f"GPT-SoVITS API error {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"GPT-SoVITS generation failed: {str(e)}")
            raise


class VerticalVideoProcessor:
    """Processor for 9:16 vertical video optimization"""
    
    def __init__(self):
        self.format_specs = {
            VideoFormat.YOUTUBE_SHORTS: VideoSpecs(1080, 1920, "9:16"),
            VideoFormat.INSTAGRAM_REELS: VideoSpecs(1080, 1920, "9:16"),
            VideoFormat.TIKTOK: VideoSpecs(1080, 1920, "9:16"),
            VideoFormat.LANDSCAPE: VideoSpecs(1920, 1080, "16:9"),
            VideoFormat.SQUARE: VideoSpecs(1080, 1080, "1:1")
        }
    
    def get_specs(self, format: VideoFormat) -> VideoSpecs:
        """Get video specifications for format"""
        return self.format_specs[format]
    
    def optimize_for_vertical(self, video_clip: VideoFileClip, target_format: VideoFormat) -> VideoFileClip:
        """Optimize video clip for vertical format"""
        specs = self.get_specs(target_format)
        
        # Get current dimensions
        current_w, current_h = video_clip.size
        current_ratio = current_w / current_h
        target_ratio = specs.width / specs.height
        
        logger.info(f"Optimizing video: {current_w}x{current_h} -> {specs.width}x{specs.height}")
        
        if current_ratio > target_ratio:
            # Video is too wide, crop sides
            new_width = int(current_h * target_ratio)
            x_center = current_w // 2
            x1 = x_center - new_width // 2
            x2 = x_center + new_width // 2
            
            video_clip = video_clip.subclipped().crop(x1=x1, x2=x2)
            logger.debug(f"Cropped width: {new_width} (removed {current_w - new_width} pixels)")
            
        elif current_ratio < target_ratio:
            # Video is too tall, crop top/bottom
            new_height = int(current_w / target_ratio)
            y_center = current_h // 2
            y1 = y_center - new_height // 2
            y2 = y_center + new_height // 2
            
            video_clip = video_clip.subclipped().crop(y1=y1, y2=y2)
            logger.debug(f"Cropped height: {new_height} (removed {current_h - new_height} pixels)")
        
        # Resize to target dimensions
        video_clip = video_clip.resized((specs.width, specs.height))
        
        return video_clip
    
    def add_vertical_background(self, video_clip: VideoFileClip, specs: VideoSpecs) -> VideoFileClip:
        """Add blurred background for vertical content"""
        # Create blurred background version
        background = (video_clip.resized((specs.width, specs.height))
                     .with_effects([lambda c: c.blur(sigma=10)])
                     .with_opacity(0.8))
        
        # Scale main video to fit nicely
        main_height = int(specs.height * 0.85)
        main_width = int(main_height * (video_clip.w / video_clip.h))
        
        main_video = video_clip.resized((main_width, main_height))
        
        # Center the main video
        main_video = main_video.with_position('center')
        
        # Composite background + main video
        final_video = CompositeVideoClip([background, main_video], size=(specs.width, specs.height))
        
        return final_video


class ChineseSubtitleGenerator:
    """Advanced Chinese subtitle generator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.style = config
        self.font_path = self._get_chinese_font()
    
    def _get_chinese_font(self) -> str:
        """Get suitable Chinese font"""
        chinese_fonts = [
            "resource/fonts/MicrosoftYaHeiBold.ttc",
            "resource/fonts/MicrosoftYaHeiNormal.ttc",
            "resource/fonts/STHeitiMedium.ttc",
            "resource/fonts/STHeitiLight.ttc"
        ]
        
        for font in chinese_fonts:
            if os.path.exists(font):
                return font
        
        logger.warning("No Chinese font found, using default")
        return None
    
    def create_subtitle_clips(
        self, 
        subtitle_data: List[Dict], 
        video_duration: float,
        specs: VideoSpecs
    ) -> List[TextClip]:
        """Create subtitle clips optimized for vertical video"""
        
        subtitle_clips = []
        
        for sub_data in subtitle_data:
            start_time = sub_data['start_time']
            end_time = sub_data['end_time'] 
            text = sub_data['text']
            
            # Create text clip with Chinese optimization
            txt_clip = TextClip(
                text,
                font=self.font_path,
                fontsize=self.style.get('fontsize', 48),
                color=self.style.get('color', 'white'),
                stroke_color=self.style.get('stroke_color', 'black'),
                stroke_width=self.style.get('stroke_width', 2),
                method=self.style.get('method', 'caption'),
                size=(specs.width * 0.9, None),  # 90% of video width
                align='center'
            ).with_start(start_time).with_duration(end_time - start_time)
            
            # Position for vertical video (lower third)
            txt_clip = txt_clip.with_position(('center', specs.height * 0.75))
            
            subtitle_clips.append(txt_clip)
        
        return subtitle_clips
    
    def generate_from_audio(
        self, 
        audio_path: str, 
        text: str, 
        specs: VideoSpecs
    ) -> List[TextClip]:
        """Generate subtitles from audio file"""
        from app.utils import utils
        
        # Split text into sentences
        sentences = utils.split_string_by_punctuations(text)
        
        # Get audio duration
        audio_clip = AudioFileClip(audio_path)
        total_duration = audio_clip.duration
        audio_clip.close()
        
        # Calculate timing for each sentence
        subtitle_data = []
        current_time = 0.0
        
        total_chars = sum(len(s) for s in sentences)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Estimate duration based on character count
            char_ratio = len(sentence) / total_chars if total_chars > 0 else 0
            duration = total_duration * char_ratio
            
            subtitle_data.append({
                'start_time': current_time,
                'end_time': current_time + duration,
                'text': sentence.strip()
            })
            
            current_time += duration
        
        return self.create_subtitle_clips(subtitle_data, total_duration, specs)


class EnhancedVideoGenerator:
    """Enhanced video generation system for Chinese content"""
    
    def __init__(self, config: ChineseVideoConfig = None):
        self.config = config or ChineseVideoConfig()
        self.gpu_manager = get_gpu_manager()
        self.vertical_processor = VerticalVideoProcessor()
        self.subtitle_generator = ChineseSubtitleGenerator(self.config.subtitle_style)
        
        # Initialize GPT-SoVITS client
        gpt_sovits_config = getattr(config, 'gpt_sovits', {
            "api_base_url": "http://localhost:9880",
            "api_key": "",
            "timeout": 60
        })
        self.gpt_sovits = EnhancedGPTSoVITSClient(gpt_sovits_config)
        
        # Initialize enhanced video pipeline
        pipeline_config = PipelineConfig(
            strategy=ProcessingStrategy.GPU_ACCELERATED,
            hardware_acceleration=True,
            target_quality='balanced',
            enable_telemetry=True
        )
        self.video_pipeline = EnhancedVideoPipeline(pipeline_config)
        
        logger.info("Enhanced video generator initialized")
    
    async def generate_chinese_video(
        self,
        script_text: str,
        video_materials: List[str],
        output_path: str,
        background_music: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate complete Chinese video with GPT-SoVITS voice
        
        Args:
            script_text: Chinese script text
            video_materials: List of video file paths
            output_path: Output video file path  
            background_music: Optional background music file
            
        Returns:
            Generation result with metrics
        """
        
        start_time = time.time()
        specs = self.vertical_processor.get_specs(self.config.format)
        
        logger.info(f"🎬 Starting Chinese video generation")
        logger.info(f"   📝 Script: {len(script_text)} characters")
        logger.info(f"   🎥 Format: {self.config.format.value} ({specs.resolution})")
        logger.info(f"   📁 Materials: {len(video_materials)} videos")
        
        try:
            # Step 1: Generate voice using GPT-SoVITS
            logger.info("🗣️ Generating voice with GPT-SoVITS...")
            temp_audio_path = f"temp_audio_{int(time.time())}.wav"
            
            audio_path, audio_duration = self.gpt_sovits.generate_speech(
                text=script_text,
                voice=self.config.voice_name,
                speed=1.0,
                output_path=temp_audio_path
            )
            
            # Ensure duration matches target
            if audio_duration > self.config.duration:
                logger.warning(f"Audio too long ({audio_duration:.1f}s), will be trimmed to {self.config.duration}s")
                audio_clip = AudioFileClip(audio_path).subclipped(0, self.config.duration)
                audio_clip.write_audiofile(temp_audio_path + "_trimmed.wav")
                audio_clip.close()
                audio_path = temp_audio_path + "_trimmed.wav"
                audio_duration = self.config.duration
            
            # Step 2: Process video materials for vertical format
            logger.info("🎥 Processing video materials...")
            processed_videos = []
            
            for video_file in video_materials:
                try:
                    video_clip = VideoFileClip(video_file)
                    
                    # Optimize for vertical format
                    vertical_clip = self.vertical_processor.optimize_for_vertical(
                        video_clip, self.config.format
                    )
                    
                    # Trim to fit audio duration proportionally  
                    clip_duration = audio_duration / len(video_materials)
                    if vertical_clip.duration > clip_duration:
                        vertical_clip = vertical_clip.subclipped(0, clip_duration)
                    
                    processed_videos.append(vertical_clip)
                    video_clip.close()
                    
                except Exception as e:
                    logger.error(f"Failed to process video {video_file}: {str(e)}")
                    continue
            
            if not processed_videos:
                raise Exception("No videos could be processed")
            
            # Step 3: Concatenate video clips
            logger.info("🔗 Concatenating video clips...")
            from moviepy.editor import concatenate_videoclips
            
            # Ensure total duration matches audio
            total_video_duration = sum(clip.duration for clip in processed_videos)
            if total_video_duration < audio_duration:
                # Loop videos to match audio duration
                scale_factor = audio_duration / total_video_duration
                processed_videos = [clip.resized_duration(clip.duration * scale_factor) 
                                  for clip in processed_videos]
            
            final_video = concatenate_videoclips(processed_videos, method="compose")
            final_video = final_video.subclipped(0, audio_duration)
            
            # Step 4: Generate Chinese subtitles
            if self.config.auto_captions:
                logger.info("📝 Generating Chinese subtitles...")
                subtitle_clips = self.subtitle_generator.generate_from_audio(
                    audio_path, script_text, specs
                )
                
                if subtitle_clips:
                    final_video = CompositeVideoClip([final_video] + subtitle_clips)
            
            # Step 5: Add background music if provided
            if background_music and os.path.exists(background_music):
                logger.info("🎵 Adding background music...")
                music_clip = AudioFileClip(background_music).subclipped(0, audio_duration)
                music_clip = music_clip.volumex(0.3)  # Lower volume for background
                
                # Mix voice and music
                voice_clip = AudioFileClip(audio_path)
                mixed_audio = CompositeAudioClip([voice_clip, music_clip])
                final_video = final_video.with_audio(mixed_audio)
                
                music_clip.close()
                voice_clip.close()
            else:
                # Just add the voice
                voice_clip = AudioFileClip(audio_path)
                final_video = final_video.with_audio(voice_clip)
                voice_clip.close()
            
            # Step 6: Export final video
            logger.info(f"💾 Exporting final video to {output_path}...")
            
            # Optimize export settings for vertical video
            export_params = {
                'fps': specs.fps,
                'bitrate': specs.bitrate,
                'audio_bitrate': '192k',
                'codec': 'libx264',
                'audio_codec': 'aac'
            }
            
            # Use GPU acceleration if available
            gpu = self.gpu_manager.get_best_gpu_for_task(
                required_memory_mb=1024,
                preferred_vendor=GPUVendor.NVIDIA
            )
            
            if gpu and gpu.vendor == GPUVendor.NVIDIA:
                export_params['codec'] = 'h264_nvenc'
                export_params['preset'] = 'p4'  # Balanced preset
                logger.info(f"Using GPU acceleration: {gpu.name}")
            
            final_video.write_videofile(output_path, **export_params)
            
            # Clean up
            final_video.close()
            for clip in processed_videos:
                clip.close()
            
            # Calculate metrics
            total_time = time.time() - start_time
            
            # Store results in memory
            await self._store_generation_results({
                'output_path': output_path,
                'audio_duration': audio_duration,
                'processing_time': total_time,
                'format': self.config.format.value,
                'resolution': specs.resolution,
                'success': True
            })
            
            logger.success("🎉 Chinese video generation completed!")
            logger.success(f"   ⏱️  Processing time: {total_time:.1f}s")
            logger.success(f"   🎥 Output: {output_path}")
            logger.success(f"   📏 Resolution: {specs.resolution}")
            logger.success(f"   ⏰ Duration: {audio_duration:.1f}s")
            
            return {
                'success': True,
                'output_path': output_path,
                'audio_duration': audio_duration,
                'processing_time': total_time,
                'format': self.config.format.value,
                'resolution': specs.resolution,
                'metrics': {
                    'materials_processed': len(processed_videos),
                    'gpu_acceleration': gpu is not None,
                    'subtitle_generated': self.config.auto_captions,
                    'background_music': background_music is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            await self._store_generation_results({
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            })
            raise
    
    async def _store_generation_results(self, results: Dict[str, Any]):
        """Store generation results in coordination memory"""
        try:
            from app.services.hive_memory import store_memory
            await store_memory(
                key=f"video_generation/{int(time.time())}",
                data=results,
                metadata={
                    'agent': 'VideoEngineer',
                    'type': 'generation_result',
                    'timestamp': time.time()
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store results: {str(e)}")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported video formats"""
        return [format.value for format in VideoFormat]
    
    def get_available_voices(self) -> List[str]:
        """Get list of available GPT-SoVITS voices"""
        return self.gpt_sovits.get_available_voices()


# Factory function
def create_chinese_video_generator(
    format: str = "youtube_shorts",
    duration: int = 30,
    voice: str = "professional"
) -> EnhancedVideoGenerator:
    """
    Factory function to create Chinese video generator
    
    Args:
        format: Video format (youtube_shorts, instagram_reels, tiktok)
        duration: Video duration in seconds
        voice: GPT-SoVITS voice name
    
    Returns:
        Configured EnhancedVideoGenerator
    """
    config = ChineseVideoConfig(
        format=VideoFormat(format),
        duration=duration,
        voice_name=voice,
        language="zh-CN",
        auto_captions=True,
        background_music=True
    )
    
    return EnhancedVideoGenerator(config)


# Export main classes
__all__ = [
    'EnhancedVideoGenerator',
    'ChineseVideoConfig', 
    'VideoFormat',
    'VideoSpecs',
    'create_chinese_video_generator'
]