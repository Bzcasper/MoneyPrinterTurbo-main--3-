"""
Advanced Video Quality Enhancement System for MoneyPrinterTurbo

This module provides comprehensive video quality improvements including:
- Noise reduction and denoising
- Color correction and grading
- Contrast optimization
- Artifact removal
- Upscaling and sharpening
- Audio enhancement for voice clarity
- Subtitle quality optimization

Optimized for en-US-JennyNeural voice and money-themed content.
"""

import os
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from loguru import logger
from moviepy import VideoFileClip, AudioFileClip
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import ffmpeg

from app.models.schema import VideoParams
from app.services.video import MemoryMonitor, close_clip, codec_optimizer

# Neural training imports
try:
    from app.services.neural_training.model_integration import get_neural_processor
    NEURAL_MODELS_AVAILABLE = True
except ImportError:
    NEURAL_MODELS_AVAILABLE = False
    logger.warning("Neural models not available - falling back to traditional enhancement")


class QualityEnhancementConfig:
    """Configuration for quality enhancement operations"""
    
    def __init__(self):
        self.noise_reduction_strength = 0.3  # 0.0 to 1.0
        self.color_enhancement_factor = 1.2   # 1.0 to 2.0
        self.contrast_boost = 1.15           # 1.0 to 2.0
        self.sharpening_strength = 0.4       # 0.0 to 1.0
        self.upscale_factor = 1.0           # 1.0 to 2.0 (1.0 = no upscaling)
        self.audio_noise_reduction = True
        self.voice_enhancement = True
        self.subtitle_clarity_boost = True
        self.money_content_optimization = True  # Specific for financial content
        
        # Advanced filters
        self.enable_temporal_denoising = True
        self.enable_chromatic_aberration_fix = True
        self.enable_dynamic_range_compression = True
        self.enable_ai_upscaling = False  # Requires additional dependencies
        
        # Voice-specific enhancements for en-US-JennyNeural
        self.voice_frequency_boost = [100, 3000]  # Hz range for voice clarity
        self.voice_compression_ratio = 2.5
        self.voice_eq_preset = "female_vocal"
        
        # Money content specific
        self.currency_detail_enhancement = True
        self.metallic_surface_optimization = True
        self.text_legibility_boost = True
        
        # Neural enhancement settings
        self.use_neural_models = NEURAL_MODELS_AVAILABLE
        self.neural_upscaling = True
        self.neural_quality_enhancement = True
        self.neural_fallback_enabled = True  # Fallback to traditional methods if neural fails


class VideoQualityEnhancer:
    """Main class for video quality enhancement operations"""
    
    def __init__(self, config: Optional[QualityEnhancementConfig] = None):
        self.config = config or QualityEnhancementConfig()
        self.temp_dir = None
        self.processing_stats = {
            'total_time': 0,
            'operations_applied': [],
            'quality_improvements': {},
            'memory_usage': []
        }
        
        # Initialize neural processor if available
        self.neural_processor = None
        if self.config.use_neural_models and NEURAL_MODELS_AVAILABLE:
            try:
                self.neural_processor = get_neural_processor()
                logger.info("Neural video processor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize neural processor: {e}")
                self.neural_processor = None
    
    async def enhance_video(self, input_path: str, output_path: str, params: VideoParams) -> Dict:
        """
        Main entry point for video quality enhancement
        
        Args:
            input_path: Path to input video file
            output_path: Path for enhanced output video
            params: Video processing parameters
            
        Returns:
            Dictionary with enhancement results and statistics
        """
        start_time = time.time()
        logger.info(f"🎯 Starting quality enhancement for: {input_path}")
        logger.info(f"📋 Voice: {params.voice_name}, Subtitles: {params.subtitle_enabled}")
        
        # Initialize memory monitoring
        initial_memory = MemoryMonitor.get_memory_usage_mb()
        self.processing_stats['memory_usage'].append(('start', initial_memory))
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.temp_dir = temp_dir
                
                # Step 1: Analyze video content
                analysis = self._analyze_video_quality(input_path)
                logger.info(f"📊 Video analysis: {analysis['issues_detected']} issues detected")
                
                # Step 2: Apply enhancements based on analysis
                enhanced_path = await self._apply_video_enhancements(
                    input_path, analysis, params
                )
                
                # Step 3: Enhance audio for en-US-JennyNeural voice
                if params.voice_name == "en-US-JennyNeural":
                    enhanced_path = self._enhance_jenny_voice_audio(enhanced_path)
                
                # Step 4: Optimize for money-themed content
                if self.config.money_content_optimization:
                    enhanced_path = self._optimize_money_content(enhanced_path)
                
                # Step 5: Final quality validation and optimization
                final_path = self._finalize_quality_enhancement(
                    enhanced_path, output_path, params
                )
                
                # Calculate processing statistics
                end_time = time.time()
                self.processing_stats['total_time'] = end_time - start_time
                final_memory = MemoryMonitor.get_memory_usage_mb()
                self.processing_stats['memory_usage'].append(('end', final_memory))
                
                logger.success(f"✅ Quality enhancement completed in {self.processing_stats['total_time']:.2f}s")
                logger.info(f"💾 Memory efficiency: {initial_memory:.1f}MB → {final_memory:.1f}MB")
                
                return {
                    'success': True,
                    'output_path': final_path,
                    'processing_time': self.processing_stats['total_time'],
                    'operations_applied': self.processing_stats['operations_applied'],
                    'quality_improvements': self.processing_stats['quality_improvements'],
                    'memory_efficiency': (initial_memory - final_memory) / initial_memory * 100
                }
                
        except Exception as e:
            logger.error(f"❌ Quality enhancement failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _analyze_video_quality(self, video_path: str) -> Dict:
        """
        Analyze video quality and detect issues that need enhancement
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("🔍 Analyzing video quality...")
        
        analysis = {
            'resolution': None,
            'fps': None,
            'duration': None,
            'issues_detected': [],
            'noise_level': 0.0,
            'contrast_level': 0.0,
            'color_quality': 0.0,
            'sharpness_score': 0.0,
            'audio_quality': {}
        }
        
        try:
            # Basic video info using FFprobe
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                
                for stream in probe_data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        analysis['resolution'] = (stream.get('width'), stream.get('height'))
                        analysis['fps'] = eval(stream.get('r_frame_rate', '30/1'))
                        
                        # Detect low resolution
                        if stream.get('width', 0) < 720 or stream.get('height', 0) < 720:
                            analysis['issues_detected'].append('low_resolution')
                        
                        # Detect low bitrate
                        if int(stream.get('bit_rate', 0)) < 1000000:  # < 1Mbps
                            analysis['issues_detected'].append('low_bitrate')
                    
                    elif stream.get('codec_type') == 'audio':
                        analysis['audio_quality'] = {
                            'sample_rate': stream.get('sample_rate'),
                            'channels': stream.get('channels'),
                            'bit_rate': stream.get('bit_rate')
                        }
                        
                        # Detect audio quality issues
                        if int(stream.get('bit_rate', 0)) < 128000:  # < 128kbps
                            analysis['issues_detected'].append('low_audio_quality')
            
            # Advanced analysis using OpenCV for frame-based metrics
            analysis.update(self._analyze_frame_quality(video_path))
            
            # Store analysis results
            self.processing_stats['quality_improvements']['initial_analysis'] = analysis
            
            logger.info(f"📈 Quality analysis complete: {len(analysis['issues_detected'])} issues found")
            
        except Exception as e:
            logger.warning(f"⚠️ Video analysis had issues: {str(e)}")
            analysis['issues_detected'].append('analysis_error')
        
        return analysis
    
    def _analyze_frame_quality(self, video_path: str) -> Dict:
        """
        Analyze frame-level quality metrics using OpenCV
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with frame analysis results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'noise_level': 0.5, 'contrast_level': 0.5, 'sharpness_score': 0.5}
            
            # Sample frames for analysis (every 30th frame, max 10 frames)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_interval = max(30, frame_count // 10)
            
            noise_levels = []
            contrast_levels = []
            sharpness_scores = []
            
            for i in range(0, frame_count, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Noise level estimation (using Laplacian variance)
                noise = cv2.Laplacian(gray, cv2.CV_64F).var()
                noise_levels.append(noise)
                
                # Contrast level (standard deviation of pixel intensities)
                contrast = gray.std()
                contrast_levels.append(contrast)
                
                # Sharpness score (high frequency content)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_scores.append(sharpness)
                
                if len(noise_levels) >= 10:  # Limit analysis to 10 frames
                    break
            
            cap.release()
            
            # Calculate average metrics
            avg_noise = np.mean(noise_levels) if noise_levels else 0
            avg_contrast = np.mean(contrast_levels) if contrast_levels else 0
            avg_sharpness = np.mean(sharpness_scores) if sharpness_scores else 0
            
            # Normalize scores (0-1 scale)
            return {
                'noise_level': min(avg_noise / 1000, 1.0),  # Normalize noise
                'contrast_level': min(avg_contrast / 128, 1.0),  # Normalize contrast
                'sharpness_score': min(avg_sharpness / 1000, 1.0)  # Normalize sharpness
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Frame analysis failed: {str(e)}")
            return {'noise_level': 0.5, 'contrast_level': 0.5, 'sharpness_score': 0.5}
    
    async def _apply_video_enhancements(self, input_path: str, analysis: Dict, params: VideoParams) -> str:
        """
        Apply video enhancements based on quality analysis
        
        Args:
            input_path: Path to input video
            analysis: Quality analysis results
            params: Video processing parameters
            
        Returns:
            Path to enhanced video file
        """
        logger.info("🎨 Applying video enhancements...")
        
        # Try neural enhancement first if available
        if self.neural_processor and self.config.neural_quality_enhancement:
            try:
                neural_enhanced_path = await self._apply_neural_enhancements(
                    input_path, analysis, params
                )
                if neural_enhanced_path:
                    logger.success("✅ Neural enhancement applied")
                    return neural_enhanced_path
            except Exception as e:
                logger.warning(f"⚠️ Neural enhancement failed: {e}")
                if not self.config.neural_fallback_enabled:
                    raise
        
        # Fallback to traditional enhancement
        return self._apply_traditional_enhancements(input_path, analysis, params)
    
    async def _apply_neural_enhancements(self, input_path: str, analysis: Dict, params: VideoParams) -> Optional[str]:
        """Apply neural model-based enhancements"""
        try:
            logger.info("🧠 Applying neural enhancements...")
            
            # Load video frames
            cap = cv2.VideoCapture(input_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                # Limit frames for memory efficiency
                if len(frames) >= 50:
                    break
            
            cap.release()
            
            if not frames:
                return None
            
            # Apply neural quality enhancement
            enhanced_frames = await self.neural_processor.enhance_video_quality(frames)
            
            # Apply neural upscaling if needed
            if 'low_resolution' in analysis['issues_detected'] and self.config.neural_upscaling:
                enhanced_frames = await self.neural_processor.upscale_video(enhanced_frames)
                self.processing_stats['operations_applied'].append('neural_upscaling')
            
            # Save enhanced video
            enhanced_path = os.path.join(self.temp_dir, "neural_enhanced.mp4")
            
            # Get video properties
            original_cap = cv2.VideoCapture(input_path)
            fps = original_cap.get(cv2.CAP_PROP_FPS)
            original_cap.release()
            
            # Write enhanced video
            height, width = enhanced_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(enhanced_path, fourcc, fps, (width, height))
            
            for frame in enhanced_frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            self.processing_stats['operations_applied'].append('neural_quality_enhancement')
            logger.success("✅ Neural enhancement completed")
            
            return enhanced_path
            
        except Exception as e:
            logger.error(f"❌ Neural enhancement error: {str(e)}")
            return None
    
    def _apply_traditional_enhancements(self, input_path: str, analysis: Dict, params: VideoParams) -> str:
        """Apply traditional FFmpeg-based enhancements"""
        logger.info("🔧 Applying traditional enhancements...")
        
        enhanced_path = os.path.join(self.temp_dir, "video_enhanced.mp4")
        filter_chain = []
        
        # Build FFmpeg filter chain based on detected issues and configuration
        
        # 1. Noise reduction (always apply some level)
        if 'low_bitrate' in analysis['issues_detected'] or analysis['noise_level'] > 0.3:
            strength = min(self.config.noise_reduction_strength * 2, 1.0)  # Boost for noisy content
            filter_chain.append(f"hqdn3d={strength}:{strength}:0.3:0.3")
            self.processing_stats['operations_applied'].append('noise_reduction')
        
        # 2. Temporal denoising for motion artifacts
        if self.config.enable_temporal_denoising:
            filter_chain.append("bm3d=sigma=3.0")
            self.processing_stats['operations_applied'].append('temporal_denoising')
        
        # 3. Color correction and enhancement
        if analysis['color_quality'] < 0.7:
            # Enhance colors for better vibrancy
            color_factor = self.config.color_enhancement_factor
            filter_chain.append(f"eq=saturation={color_factor}:gamma=1.1")
            self.processing_stats['operations_applied'].append('color_enhancement')
        
        # 4. Contrast optimization
        if analysis['contrast_level'] < 0.6:
            contrast_boost = self.config.contrast_boost
            filter_chain.append(f"eq=contrast={contrast_boost}:brightness=0.05")
            self.processing_stats['operations_applied'].append('contrast_optimization')
        
        # 5. Sharpening for crisp details
        if analysis['sharpness_score'] < 0.5:
            sharpness = self.config.sharpening_strength
            filter_chain.append(f"unsharp=5:5:{sharpness}:5:5:0.0")
            self.processing_stats['operations_applied'].append('sharpening')
        
        # 6. Chromatic aberration correction
        if self.config.enable_chromatic_aberration_fix:
            filter_chain.append("lenscorrection=cx=0.5:cy=0.5:k1=-0.1:k2=0.0")
            self.processing_stats['operations_applied'].append('chromatic_correction')
        
        # 7. Dynamic range compression for better shadows/highlights
        if self.config.enable_dynamic_range_compression:
            filter_chain.append("histeq=strength=0.3:intensity=0.2")
            self.processing_stats['operations_applied'].append('dynamic_range')
        
        # 8. Upscaling if needed
        if 'low_resolution' in analysis['issues_detected'] and self.config.upscale_factor > 1.0:
            scale_factor = self.config.upscale_factor
            new_width = int(analysis['resolution'][0] * scale_factor)
            new_height = int(analysis['resolution'][1] * scale_factor)
            filter_chain.append(f"scale={new_width}:{new_height}:flags=lanczos")
            self.processing_stats['operations_applied'].append('upscaling')
        
        # Get optimal codec settings
        codec_settings = codec_optimizer.get_optimal_codec_settings(
            content_type='text_heavy' if params.subtitle_enabled else 'general',
            target_quality='quality'
        )
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'warning',
            '-i', input_path
        ]
        
        # Add video filters
        if filter_chain:
            cmd.extend(['-vf', ','.join(filter_chain)])
        
        # Add codec settings
        cmd.extend([
            '-c:v', codec_settings['codec'],
            '-c:a', 'aac',
            '-b:a', '192k'
        ])
        
        # Add codec-specific parameters
        if codec_settings['encoder_type'] == 'software':
            cmd.extend(['-preset', 'slow', '-crf', '18'])  # High quality
        elif codec_settings['encoder_type'] == 'qsv':
            cmd.extend(['-preset', 'veryslow', '-global_quality', '18'])
        elif codec_settings['encoder_type'] == 'nvenc':
            cmd.extend(['-preset', 'p7', '-cq', '18'])  # Highest quality preset
        
        cmd.extend(['-movflags', '+faststart', '-y', enhanced_path])
        
        # Execute enhancement
        logger.debug(f"🔧 Executing: {' '.join(cmd[:10])}... (filter count: {len(filter_chain)})")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                logger.error(f"❌ Video enhancement failed: {result.stderr}")
                return input_path  # Fallback to original
            
            logger.success(f"✅ Video enhancement applied ({len(filter_chain)} filters)")
            return enhanced_path
            
        except subprocess.TimeoutExpired:
            logger.error("⏰ Video enhancement timed out")
            return input_path
        except Exception as e:
            logger.error(f"❌ Video enhancement error: {str(e)}")
            return input_path
    
    def _enhance_jenny_voice_audio(self, video_path: str) -> str:
        """
        Apply specific audio enhancements for en-US-JennyNeural voice
        
        Args:
            video_path: Path to video with audio
            
        Returns:
            Path to video with enhanced audio
        """
        logger.info("🎤 Enhancing en-US-JennyNeural voice audio...")
        
        enhanced_path = os.path.join(self.temp_dir, "jenny_voice_enhanced.mp4")
        
        # Audio enhancement chain optimized for Jenny's voice characteristics
        audio_filters = []
        
        # 1. Noise reduction specifically tuned for female voice
        if self.config.audio_noise_reduction:
            audio_filters.append("afftdn=nr=12:nf=-25:tn=1")
            
        # 2. EQ for female vocal enhancement
        if self.config.voice_enhancement:
            # Boost presence (2-4kHz) and clarity (5-8kHz) for Jenny's voice
            eq_settings = [
                "equalizer=f=100:width_type=h:width=50:g=2",    # Warmth
                "equalizer=f=2500:width_type=h:width=1000:g=3", # Presence
                "equalizer=f=6000:width_type=h:width=2000:g=2", # Clarity
                "equalizer=f=12000:width_type=h:width=4000:g=1" # Air
            ]
            audio_filters.extend(eq_settings)
        
        # 3. Dynamic range compression for consistent levels
        compression_ratio = self.config.voice_compression_ratio
        audio_filters.append(f"acompressor=threshold=-18dB:ratio={compression_ratio}:attack=5:release=50")
        
        # 4. De-essing to reduce harsh sibilants
        audio_filters.append("deesser=i=0.1:m=0.5:f=0.5:s=o")
        
        # 5. Final limiter to prevent clipping
        audio_filters.append("alimiter=level_in=1:level_out=0.9:limit=0.95")
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'warning',
            '-i', video_path,
            '-c:v', 'copy',  # Don't re-encode video
            '-af', ','.join(audio_filters),
            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
            '-y', enhanced_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.success("✅ Jenny voice audio enhanced")
                self.processing_stats['operations_applied'].append('jenny_voice_enhancement')
                return enhanced_path
            else:
                logger.warning(f"⚠️ Audio enhancement failed: {result.stderr}")
                return video_path
        except Exception as e:
            logger.warning(f"⚠️ Audio enhancement error: {str(e)}")
            return video_path
    
    def _optimize_money_content(self, video_path: str) -> str:
        """
        Apply optimizations specific to money-themed content
        
        Args:
            video_path: Path to input video
            
        Returns:
            Path to optimized video
        """
        logger.info("💰 Optimizing for money-themed content...")
        
        optimized_path = os.path.join(self.temp_dir, "money_optimized.mp4")
        filter_chain = []
        
        # 1. Enhance currency details and text legibility
        if self.config.currency_detail_enhancement:
            # Increase local contrast for fine details
            filter_chain.append("cas=0.3")  # Contrast Adaptive Sharpening
            self.processing_stats['operations_applied'].append('currency_detail_enhancement')
        
        # 2. Optimize metallic surfaces (coins, etc.)
        if self.config.metallic_surface_optimization:
            # Enhance reflections and metallic shine
            filter_chain.append("eq=gamma=1.1:saturation=1.2:contrast=1.1")
            self.processing_stats['operations_applied'].append('metallic_optimization')
        
        # 3. Text legibility boost for financial information
        if self.config.text_legibility_boost:
            # Enhance edge definition for text
            filter_chain.append("unsharp=5:5:0.3:5:5:0.0")
            self.processing_stats['operations_applied'].append('text_legibility')
        
        # 4. Color grading for professional financial look
        # Slight desaturation with emphasis on greens (money) and blues (trust)
        filter_chain.append("curves=m='0/0 0.25/0.23 0.5/0.48 0.75/0.73 1/1':r='0/0 0.5/0.48 1/1':g='0/0 0.5/0.52 1/1':b='0/0 0.5/0.52 1/1'")
        self.processing_stats['operations_applied'].append('financial_color_grading')
        
        if not filter_chain:
            return video_path  # No optimization needed
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'warning',
            '-i', video_path,
            '-vf', ','.join(filter_chain),
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
            '-c:a', 'copy',  # Keep enhanced audio
            '-movflags', '+faststart',
            '-y', optimized_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.success("✅ Money content optimization applied")
                return optimized_path
            else:
                logger.warning(f"⚠️ Money optimization failed: {result.stderr}")
                return video_path
        except Exception as e:
            logger.warning(f"⚠️ Money optimization error: {str(e)}")
            return video_path
    
    def _finalize_quality_enhancement(self, input_path: str, output_path: str, params: VideoParams) -> str:
        """
        Final quality validation and output optimization
        
        Args:
            input_path: Path to enhanced video
            output_path: Final output path
            params: Video processing parameters
            
        Returns:
            Path to final optimized video
        """
        logger.info("🏁 Finalizing quality enhancement...")
        
        # Get optimal codec settings for final output
        codec_settings = codec_optimizer.get_optimal_codec_settings(
            content_type='text_heavy' if params.subtitle_enabled else 'general',
            target_quality='quality'
        )
        
        # Build final optimization command
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'warning',
            '-i', input_path,
            '-c:v', codec_settings['codec'],
            '-c:a', 'aac', '-b:a', '192k'
        ]
        
        # Add codec-specific high-quality settings
        if codec_settings['encoder_type'] == 'software':
            cmd.extend(['-preset', 'slow', '-crf', '18'])
        elif codec_settings['encoder_type'] == 'qsv':
            cmd.extend(['-preset', 'veryslow', '-global_quality', '18'])
        elif codec_settings['encoder_type'] == 'nvenc':
            cmd.extend(['-preset', 'p7', '-cq', '18'])
        
        cmd.extend([
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-y', output_path
        ])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                logger.success("✅ Quality enhancement finalized")
                self.processing_stats['operations_applied'].append('final_optimization')
                return output_path
            else:
                logger.error(f"❌ Final optimization failed: {result.stderr}")
                # Fallback: just copy the enhanced file
                import shutil
                shutil.copy2(input_path, output_path)
                return output_path
        except Exception as e:
            logger.error(f"❌ Final optimization error: {str(e)}")
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path


async def enhance_video_quality(input_path: str, output_path: str, params: VideoParams, 
                         config: Optional[QualityEnhancementConfig] = None) -> Dict:
    """
    Main function to enhance video quality for MoneyPrinterTurbo content
    
    Args:
        input_path: Path to input video file
        output_path: Path for enhanced output video
        params: Video processing parameters
        config: Optional enhancement configuration
        
    Returns:
        Dictionary with enhancement results
    """
    enhancer = VideoQualityEnhancer(config)
    return await enhancer.enhance_video(input_path, output_path, params)


def create_quality_enhancement_config(
    noise_reduction: float = 0.3,
    color_enhancement: float = 1.2,
    contrast_boost: float = 1.15,
    enable_jenny_voice_optimization: bool = True,
    enable_money_content_optimization: bool = True
) -> QualityEnhancementConfig:
    """
    Create a customized quality enhancement configuration
    
    Args:
        noise_reduction: Noise reduction strength (0.0-1.0)
        color_enhancement: Color enhancement factor (1.0-2.0)
        contrast_boost: Contrast boost factor (1.0-2.0)
        enable_jenny_voice_optimization: Enable Jenny voice specific optimizations
        enable_money_content_optimization: Enable money content optimizations
        
    Returns:
        Configured QualityEnhancementConfig instance
    """
    config = QualityEnhancementConfig()
    config.noise_reduction_strength = noise_reduction
    config.color_enhancement_factor = color_enhancement
    config.contrast_boost = contrast_boost
    config.voice_enhancement = enable_jenny_voice_optimization
    config.money_content_optimization = enable_money_content_optimization
    
    return config