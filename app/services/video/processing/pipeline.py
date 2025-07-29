"""
Processing Pipeline - Parallel video clip processing

This module provides high-performance parallel video clip processing
with resource management, fault tolerance, and adaptive optimization.

Author: MoneyPrinterTurbo Enhanced System
Version: 1.0.0
"""

import asyncio
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import psutil


@dataclass
class ProcessedClip:
    """Processed video clip information"""
    original_path: str
    processed_path: str
    dimensions: Dict[str, int]
    duration: float
    file_size: int = 0
    processing_time: float = 0.0


@dataclass
class ProcessingResult:
    """Result of clip processing operation"""
    success: bool
    clip: Optional[ProcessedClip] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class ResourcePool:
    """Thread-safe resource pool for managing processing resources"""
    
    def __init__(self, max_concurrent: int):
        """Initialize resource pool"""
        self.max_concurrent = max_concurrent
        self.active_resources = set()
        self.available_resources = set(range(max_concurrent))
        self._lock = asyncio.Lock()
    
    async def acquire_resource(self, timeout: float = 30.0) -> Optional[int]:
        """Acquire a processing resource"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with self._lock:
                if self.available_resources:
                    resource_id = self.available_resources.pop()
                    self.active_resources.add(resource_id)
                    return resource_id
            
            await asyncio.sleep(0.1)
        
        return None
    
    async def release_resource(self, resource_id: int):
        """Release a processing resource"""
        async with self._lock:
            if resource_id in self.active_resources:
                self.active_resources.remove(resource_id)
                self.available_resources.add(resource_id)
    
    def cleanup_batch(self):
        """Cleanup resources between batches"""
        # Force garbage collection and memory cleanup
        import gc
        gc.collect()


class ClipProcessor:
    """Individual clip processing with optimization"""
    
    def __init__(self):
        """Initialize clip processor"""
        self.temp_dir = Path("/tmp/video_processing")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Supported processing methods
        self.processing_methods = ['ffmpeg', 'moviepy']
    
    async def process_clip(
        self,
        clip_path: str,
        target_dimensions: Dict[str, int],
        quality_settings: Dict[str, Any],
        resource_id: int
    ) -> ProcessingResult:
        """
        Process a single video clip
        
        Args:
            clip_path: Path to input video clip
            target_dimensions: Target width and height
            quality_settings: Quality and encoding settings
            resource_id: Allocated resource ID
            
        Returns:
            ProcessingResult with processing outcome
        """
        start_time = time.time()
        
        try:
            # Generate output path
            output_path = self._generate_output_path(clip_path, resource_id)
            
            # Choose processing method based on clip characteristics
            method = self._select_processing_method(clip_path, quality_settings)
            
            # Process the clip
            if method == 'ffmpeg':
                success = await self._process_with_ffmpeg(
                    clip_path, output_path, target_dimensions, quality_settings
                )
            else:
                success = await self._process_with_moviepy(
                    clip_path, output_path, target_dimensions, quality_settings
                )
            
            if not success:
                return ProcessingResult(
                    success=False,
                    error=f"Processing failed with {method}",
                    processing_time=time.time() - start_time
                )
            
            # Validate output
            if not Path(output_path).exists():
                return ProcessingResult(
                    success=False,
                    error="Output file was not created",
                    processing_time=time.time() - start_time
                )
            
            # Create processed clip info
            processed_clip = ProcessedClip(
                original_path=clip_path,
                processed_path=output_path,
                dimensions=target_dimensions,
                duration=self._get_video_duration(output_path),
                file_size=Path(output_path).stat().st_size,
                processing_time=time.time() - start_time
            )
            
            return ProcessingResult(
                success=True,
                clip=processed_clip,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Clip processing failed: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _generate_output_path(self, input_path: str, resource_id: int) -> str:
        """Generate unique output path for processed clip"""
        input_file = Path(input_path)
        timestamp = int(time.time() * 1000)
        output_name = f"{input_file.stem}_processed_{resource_id}_{timestamp}.mp4"
        return str(self.temp_dir / output_name)
    
    def _select_processing_method(
        self, clip_path: str, quality_settings: Dict[str, Any]
    ) -> str:
        """Select optimal processing method based on clip characteristics"""
        # For now, prefer ffmpeg for better performance
        # Future: analyze clip properties to make intelligent choice
        return quality_settings.get('preferred_method', 'ffmpeg')
    
    async def _process_with_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        target_dimensions: Dict[str, int],
        quality_settings: Dict[str, Any]
    ) -> bool:
        """Process clip using FFmpeg"""
        try:
            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vf', f"scale={target_dimensions['width']}:{target_dimensions['height']}",
                '-c:v', quality_settings.get('video_codec', 'libx264'),
                '-preset', quality_settings.get('preset', 'medium'),
                '-crf', str(quality_settings.get('crf', 23)),
                '-y',  # Overwrite output file
                output_path
            ]
            
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=300
            )
            
            return process.returncode == 0
            
        except Exception as e:
            logger.error(f"FFmpeg processing failed: {str(e)}")
            return False
    
    async def _process_with_moviepy(
        self,
        input_path: str,
        output_path: str,
        target_dimensions: Dict[str, int],
        quality_settings: Dict[str, Any]
    ) -> bool:
        """Process clip using MoviePy"""
        try:
            # Import MoviePy in the processing function to avoid blocking
            from moviepy.editor import VideoFileClip
            
            # Process in thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            def process_sync():
                with VideoFileClip(input_path) as clip:
                    # Resize clip
                    resized = clip.resize(
                        (target_dimensions['width'], target_dimensions['height'])
                    )
                    
                    # Write output
                    resized.write_videofile(
                        output_path,
                        codec=quality_settings.get('video_codec', 'libx264'),
                        audio_codec=quality_settings.get('audio_codec', 'aac'),
                        verbose=False,
                        logger=None
                    )
                    
                    resized.close()
                
                return True
            
            return await loop.run_in_executor(None, process_sync)
            
        except Exception as e:
            logger.error(f"MoviePy processing failed: {str(e)}")
            return False
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe"""
        try:
            import subprocess
            import json
            
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data['format']['duration'])
            
            return 0.0
            
        except Exception:
            return 0.0


class ProcessingPipeline:
    """
    High-performance parallel video clip processing pipeline
    
    Provides parallel processing with resource management, fault tolerance,
    and adaptive performance optimization.
    """
    
    def __init__(self):
        """Initialize processing pipeline"""
        self.clip_processor = ClipProcessor()
        self.max_threads = min(multiprocessing.cpu_count() * 2, 16)
        self.resource_pool = ResourcePool(self.max_threads // 2)
        
        # Performance settings
        self.batch_size = 8
        self.min_success_rate = 0.8
        self.timeout_per_clip = 300  # 5 minutes
        
        logger.info(f"ProcessingPipeline initialized with {self.max_threads} max threads")
    
    async def process_clips(
        self,
        clips: List[str],
        target_dimensions: Dict[str, int],
        quality_settings: Dict[str, Any]
    ) -> List[ProcessedClip]:
        """
        Process multiple clips in parallel with fault tolerance
        
        Args:
            clips: List of video file paths
            target_dimensions: Target width and height
            quality_settings: Quality and encoding settings
            
        Returns:
            List of successfully processed clips
        """
        if not clips:
            return []
        
        logger.info(f"Starting parallel processing of {len(clips)} clips")
        start_time = time.time()
        
        # Calculate optimal batch size based on available memory
        optimal_batch_size = self._calculate_optimal_batch_size(len(clips))
        
        processed_clips = []
        failed_clips = []
        
        # Process clips in batches to manage memory
        for batch_start in range(0, len(clips), optimal_batch_size):
            batch_end = min(batch_start + optimal_batch_size, len(clips))
            batch_clips = clips[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//optimal_batch_size + 1} "
                       f"({len(batch_clips)} clips)")
            
            # Process current batch
            batch_results = await self._process_batch(
                batch_clips, target_dimensions, quality_settings
            )
            
            # Collect results
            for result in batch_results:
                if result.success and result.clip:
                    processed_clips.append(result.clip)
                else:
                    failed_clips.append(result.error)
            
            # Cleanup between batches
            self.resource_pool.cleanup_batch()
            await asyncio.sleep(0.1)  # Brief pause for system recovery
        
        # Check success rate
        total_clips = len(clips)
        success_rate = len(processed_clips) / total_clips if total_clips > 0 else 0
        
        if success_rate < self.min_success_rate:
            logger.warning(f"Low success rate: {success_rate:.2%} "
                          f"({len(failed_clips)} failures)")
        
        processing_time = time.time() - start_time
        logger.success(f"Processed {len(processed_clips)}/{total_clips} clips "
                      f"in {processing_time:.2f}s (success rate: {success_rate:.2%})")
        
        return processed_clips
    
    async def _process_batch(
        self,
        batch_clips: List[str],
        target_dimensions: Dict[str, int],
        quality_settings: Dict[str, Any]
    ) -> List[ProcessingResult]:
        """Process a batch of clips in parallel"""
        
        # Create processing tasks
        tasks = []
        for clip_path in batch_clips:
            task = self._process_single_clip_with_resource(
                clip_path, target_dimensions, quality_settings
            )
            tasks.append(task)
        
        # Execute tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_per_clip * len(batch_clips)
            )
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(ProcessingResult(
                        success=False,
                        error=f"Task exception: {str(result)}"
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except asyncio.TimeoutError:
            logger.error(f"Batch processing timed out after {self.timeout_per_clip * len(batch_clips)}s")
            return [ProcessingResult(success=False, error="Batch timeout") for _ in batch_clips]
    
    async def _process_single_clip_with_resource(
        self,
        clip_path: str,
        target_dimensions: Dict[str, int],
        quality_settings: Dict[str, Any]
    ) -> ProcessingResult:
        """Process single clip with resource management"""
        
        # Acquire resource
        resource_id = await self.resource_pool.acquire_resource(timeout=30)
        if resource_id is None:
            return ProcessingResult(
                success=False,
                error="Failed to acquire processing resource"
            )
        
        try:
            # Process the clip
            result = await self.clip_processor.process_clip(
                clip_path, target_dimensions, quality_settings, resource_id
            )
            
            return result
            
        finally:
            # Always release the resource
            await self.resource_pool.release_resource(resource_id)
    
    def _calculate_optimal_batch_size(self, total_clips: int) -> int:
        """Calculate optimal batch size based on system resources"""
        # Get available memory
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024**3)
        
        # Estimate memory per clip (conservative estimate)
        memory_per_clip_gb = 0.5  # 500MB per clip
        
        # Calculate memory-based batch size
        memory_based_batch = max(1, int(available_memory_gb / memory_per_clip_gb))
        
        # Consider thread limit
        thread_based_batch = self.max_threads // 2
        
        # Use the smaller of the two, but at least 2
        optimal_batch = max(2, min(memory_based_batch, thread_based_batch, self.batch_size))
        
        # Don't exceed total clips
        return min(optimal_batch, total_clips)
    
    def is_healthy(self) -> bool:
        """Check if processing pipeline is healthy"""
        try:
            # Basic health checks
            return (
                self.clip_processor is not None and
                self.resource_pool is not None and
                self.max_threads > 0
            )
        except Exception:
            return False