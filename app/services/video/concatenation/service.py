"""
Concatenation Service - Efficient video concatenation

This module provides optimized video concatenation with multiple strategies,
progressive processing, and format compatibility handling.

Author: MoneyPrinterTurbo Enhanced System
Version: 1.0.0
"""

import os
import subprocess
import tempfile
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from loguru import logger


class ConcatMode(Enum):
    """Video concatenation modes"""
    PROGRESSIVE = "progressive"
    FILTER_COMPLEX = "filter_complex"
    DEMUXER = "demuxer"
    COPY_CONCAT = "copy_concat"


@dataclass
class ConcatResult:
    """Result of concatenation operation"""
    success: bool
    output_path: Optional[str] = None
    processing_time: float = 0.0
    error: Optional[str] = None
    method_used: Optional[str] = None
    file_size: int = 0


@dataclass
class VideoInfo:
    """Video file information for concatenation"""
    path: str
    duration: float
    codec: str
    resolution: str
    frame_rate: float
    audio_codec: str = "unknown"


class FormatAnalyzer:
    """Analyze video formats for optimal concatenation strategy"""
    
    def __init__(self):
        """Initialize format analyzer"""
        pass
    
    def analyze_clips(self, clip_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze clips to determine optimal concatenation strategy
        
        Args:
            clip_paths: List of video file paths
            
        Returns:
            Analysis result with recommended strategy
        """
        if not clip_paths:
            return {"compatible": False, "reason": "No clips provided"}
        
        clip_infos = []
        for path in clip_paths:
            info = self._get_video_info(path)
            if info:
                clip_infos.append(info)
        
        if not clip_infos:
            return {"compatible": False, "reason": "No valid clips found"}
        
        # Check format compatibility
        compatibility = self._check_compatibility(clip_infos)
        
        return {
            "compatible": compatibility["compatible"],
            "reason": compatibility.get("reason", ""),
            "recommended_mode": compatibility.get("recommended_mode", ConcatMode.PROGRESSIVE),
            "clips_info": clip_infos,
            "total_duration": sum(info.duration for info in clip_infos)
        }
    
    def _get_video_info(self, path: str) -> Optional[VideoInfo]:
        """Get video file information using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
            
            import json
            data = json.loads(result.stdout)
            
            # Find video and audio streams
            video_stream = None
            audio_stream = None
            
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video' and not video_stream:
                    video_stream = stream
                elif stream.get('codec_type') == 'audio' and not audio_stream:
                    audio_stream = stream
            
            if not video_stream:
                return None
            
            return VideoInfo(
                path=path,
                duration=float(video_stream.get('duration', 0)),
                codec=video_stream.get('codec_name', 'unknown'),
                resolution=f"{video_stream.get('width', 0)}x{video_stream.get('height', 0)}",
                frame_rate=self._parse_frame_rate(video_stream.get('r_frame_rate', '0/1')),
                audio_codec=audio_stream.get('codec_name', 'none') if audio_stream else 'none'
            )
            
        except Exception as e:
            logger.warning(f"Failed to analyze video {path}: {str(e)}")
            return None
    
    def _parse_frame_rate(self, fps_string: str) -> float:
        """Parse frame rate string"""
        try:
            if '/' in fps_string:
                num, den = fps_string.split('/')
                return float(num) / float(den) if float(den) != 0 else 0.0
            return float(fps_string)
        except Exception:
            return 0.0
    
    def _check_compatibility(self, clip_infos: List[VideoInfo]) -> Dict[str, Any]:
        """Check clip compatibility and recommend strategy"""
        if len(clip_infos) <= 1:
            return {
                "compatible": True,
                "recommended_mode": ConcatMode.COPY_CONCAT,
                "reason": "Single or no clips"
            }
        
        # Check codec consistency
        video_codecs = set(info.codec for info in clip_infos)
        audio_codecs = set(info.audio_codec for info in clip_infos)
        resolutions = set(info.resolution for info in clip_infos)
        
        # If everything matches, use fastest method
        if len(video_codecs) == 1 and len(audio_codecs) == 1 and len(resolutions) == 1:
            return {
                "compatible": True,
                "recommended_mode": ConcatMode.DEMUXER,
                "reason": "All clips have identical formats"
            }
        
        # If resolutions differ, need re-encoding
        if len(resolutions) > 1:
            return {
                "compatible": True,
                "recommended_mode": ConcatMode.FILTER_COMPLEX,
                "reason": "Different resolutions require scaling"
            }
        
        # If codecs differ but resolutions match
        if len(video_codecs) > 1 or len(audio_codecs) > 1:
            return {
                "compatible": True,
                "recommended_mode": ConcatMode.PROGRESSIVE,
                "reason": "Different codecs require re-encoding"
            }
        
        return {
            "compatible": True,
            "recommended_mode": ConcatMode.PROGRESSIVE,
            "reason": "Default progressive mode"
        }


class ConcatenationService:
    """
    High-performance video concatenation service
    
    Provides multiple concatenation strategies with automatic
    format detection and optimization.
    """
    
    def __init__(self):
        """Initialize concatenation service"""
        self.format_analyzer = FormatAnalyzer()
        self.temp_dir = Path(tempfile.gettempdir()) / "video_concatenation"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Performance settings
        self.max_retries = 3
        self.timeout_per_gb = 300  # 5 minutes per GB
        
        logger.info("ConcatenationService initialized")
    
    async def concatenate(
        self,
        clips: List[str],
        output_path: str,
        concat_mode: str = "auto"
    ) -> ConcatResult:
        """
        Concatenate video clips with optimal strategy
        
        Args:
            clips: List of video file paths or ProcessedClip objects
            output_path: Output file path
            concat_mode: Concatenation mode ('auto', 'progressive', etc.)
            
        Returns:
            ConcatResult with operation outcome
        """
        if not clips:
            return ConcatResult(
                success=False,
                error="No clips provided for concatenation"
            )
        
        start_time = time.time()
        
        # Extract file paths if clips are objects
        clip_paths = self._extract_clip_paths(clips)
        
        # Analyze clips for optimal strategy
        analysis = self.format_analyzer.analyze_clips(clip_paths)
        
        if not analysis["compatible"]:
            return ConcatResult(
                success=False,
                error=f"Clips not compatible: {analysis['reason']}",
                processing_time=time.time() - start_time
            )
        
        # Select concatenation method
        if concat_mode == "auto":
            selected_mode = analysis["recommended_mode"]
        else:
            try:
                selected_mode = ConcatMode(concat_mode)
            except ValueError:
                selected_mode = ConcatMode.PROGRESSIVE
        
        logger.info(f"Concatenating {len(clip_paths)} clips using {selected_mode.value} mode")
        
        # Perform concatenation with retries
        for attempt in range(self.max_retries):
            try:
                result = await self._concatenate_with_method(
                    clip_paths, output_path, selected_mode, analysis
                )
                
                if result.success:
                    result.processing_time = time.time() - start_time
                    result.method_used = selected_mode.value
                    
                    # Get output file size
                    if os.path.exists(output_path):
                        result.file_size = os.path.getsize(output_path)
                    
                    logger.success(f"Concatenation completed in {result.processing_time:.2f}s "
                                 f"({result.file_size / 1024 / 1024:.1f}MB)")
                    return result
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"Concatenation attempt {attempt + 1} failed, retrying...")
                    # Try fallback method on retry
                    selected_mode = ConcatMode.PROGRESSIVE
                
            except Exception as e:
                logger.error(f"Concatenation attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    return ConcatResult(
                        success=False,
                        error=f"All concatenation attempts failed: {str(e)}",
                        processing_time=time.time() - start_time
                    )
        
        return ConcatResult(
            success=False,
            error="Maximum retry attempts exceeded",
            processing_time=time.time() - start_time
        )
    
    async def _concatenate_with_method(
        self,
        clip_paths: List[str],
        output_path: str,
        mode: ConcatMode,
        analysis: Dict[str, Any]
    ) -> ConcatResult:
        """Execute concatenation with specific method"""
        
        if mode == ConcatMode.DEMUXER:
            return await self._concat_demuxer(clip_paths, output_path)
        elif mode == ConcatMode.FILTER_COMPLEX:
            return await self._concat_filter_complex(clip_paths, output_path, analysis)
        elif mode == ConcatMode.COPY_CONCAT:
            return await self._concat_copy(clip_paths, output_path)
        else:  # PROGRESSIVE
            return await self._concat_progressive(clip_paths, output_path, analysis)
    
    async def _concat_demuxer(self, clip_paths: List[str], output_path: str) -> ConcatResult:
        """Fast concatenation using demuxer (no re-encoding)"""
        try:
            # Create file list for ffmpeg concat demuxer
            list_file = self.temp_dir / f"concat_list_{int(time.time())}.txt"
            
            with open(list_file, 'w') as f:
                for path in clip_paths:
                    f.write(f"file '{os.path.abspath(path)}'\n")
            
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(list_file),
                '-c', 'copy',
                '-y',
                output_path
            ]
            
            result = await self._execute_ffmpeg_command(cmd)
            
            # Cleanup
            list_file.unlink(missing_ok=True)
            
            return ConcatResult(success=result)
            
        except Exception as e:
            return ConcatResult(success=False, error=str(e))
    
    async def _concat_filter_complex(
        self, clip_paths: List[str], output_path: str, analysis: Dict[str, Any]
    ) -> ConcatResult:
        """Concatenation with scaling using filter_complex"""
        try:
            # Build filter_complex command for scaling and concatenation
            inputs = []
            filter_parts = []
            
            for i, path in enumerate(clip_paths):
                inputs.extend(['-i', path])
                filter_parts.append(f'[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,'
                                  f'pad=1920:1080:(ow-iw)/2:(oh-ih)/2[v{i}]')
            
            # Concatenate video and audio
            video_concat = ''.join(f'[v{i}]' for i in range(len(clip_paths)))
            audio_concat = ''.join(f'[{i}:a]' for i in range(len(clip_paths)))
            
            filter_complex = ';'.join(filter_parts) + f';{video_concat}concat=n={len(clip_paths)}:v=1:a=0[outv];{audio_concat}concat=n={len(clip_paths)}:v=0:a=1[outa]'
            
            cmd = [
                'ffmpeg',
                *inputs,
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-map', '[outa]',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-y',
                output_path
            ]
            
            result = await self._execute_ffmpeg_command(cmd)
            return ConcatResult(success=result)
            
        except Exception as e:
            return ConcatResult(success=False, error=str(e))
    
    async def _concat_copy(self, clip_paths: List[str], output_path: str) -> ConcatResult:
        """Simple copy concatenation for identical formats"""
        try:
            if len(clip_paths) == 1:
                # Single file, just copy
                import shutil
                shutil.copy2(clip_paths[0], output_path)
                return ConcatResult(success=True)
            
            # Use concat protocol for binary concatenation
            concat_input = 'concat:' + '|'.join(clip_paths)
            
            cmd = [
                'ffmpeg',
                '-i', concat_input,
                '-c', 'copy',
                '-y',
                output_path
            ]
            
            result = await self._execute_ffmpeg_command(cmd)
            return ConcatResult(success=result)
            
        except Exception as e:
            return ConcatResult(success=False, error=str(e))
    
    async def _concat_progressive(
        self, clip_paths: List[str], output_path: str, analysis: Dict[str, Any]
    ) -> ConcatResult:
        """Progressive concatenation with re-encoding"""
        try:
            # Build progressive concatenation command
            inputs = []
            for path in clip_paths:
                inputs.extend(['-i', path])
            
            # Create filter for concatenation
            filter_str = f'concat=n={len(clip_paths)}:v=1:a=1'
            
            cmd = [
                'ffmpeg',
                *inputs,
                '-filter_complex', filter_str,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
            
            result = await self._execute_ffmpeg_command(cmd)
            return ConcatResult(success=result)
            
        except Exception as e:
            return ConcatResult(success=False, error=str(e))
    
    async def _execute_ffmpeg_command(self, cmd: List[str]) -> bool:
        """Execute FFmpeg command with timeout"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Calculate timeout based on file sizes
            timeout = self._calculate_timeout(cmd)
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            
            if process.returncode != 0:
                logger.error(f"FFmpeg failed: {stderr.decode()}")
                return False
            
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"FFmpeg command timed out after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"FFmpeg execution failed: {str(e)}")
            return False
    
    def _calculate_timeout(self, cmd: List[str]) -> int:
        """Calculate timeout based on input files"""
        try:
            # Estimate total file size
            total_size = 0
            for arg in cmd:
                if os.path.isfile(arg):
                    total_size += os.path.getsize(arg)
            
            # Calculate timeout: base 60s + 5min per GB
            size_gb = total_size / (1024**3)
            timeout = 60 + int(size_gb * self.timeout_per_gb)
            
            return min(timeout, 3600)  # Max 1 hour
            
        except Exception:
            return 600  # Default 10 minutes
    
    def _extract_clip_paths(self, clips) -> List[str]:
        """Extract file paths from clips (strings or objects)"""
        paths = []
        for clip in clips:
            if isinstance(clip, str):
                paths.append(clip)
            elif hasattr(clip, 'processed_path'):
                paths.append(clip.processed_path)
            elif hasattr(clip, 'file_path'):
                paths.append(clip.file_path)
            elif hasattr(clip, 'path'):
                paths.append(clip.path)
        return paths
    
    def is_healthy(self) -> bool:
        """Check if concatenation service is healthy"""
        try:
            # Test ffmpeg availability
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False


# Import asyncio for async methods
import asyncio