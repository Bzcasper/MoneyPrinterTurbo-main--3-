"""
Video Processor Agent (Worker-4)
Specialized agent for video processing, format conversion, and enhancement
"""

import asyncio
import os
import tempfile
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from app.services.hive_memory import get_hive_memory, log_swarm_event, store_swarm_memory, retrieve_swarm_memory

logger = logging.getLogger(__name__)


class VideoFormat(Enum):
    """Supported video formats"""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"
    MKV = "mkv"
    FLV = "flv"


class VideoCodec(Enum):
    """Supported video codecs"""
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    VP8 = "vp8"
    AV1 = "av1"


class ProcessingQuality(Enum):
    """Video processing quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class TaskStatus(Enum):
    """Video processing task status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VideoTask:
    """Video processing task definition"""
    task_id: str
    input_path: str
    output_path: str
    target_format: VideoFormat
    target_codec: VideoCodec
    quality: ProcessingQuality
    enhancement_options: Dict[str, Any]
    priority: int = 5  # 1-10, higher is more urgent
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.QUEUED
    progress: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['target_format'] = self.target_format.value
        data['target_codec'] = self.target_codec.value
        data['quality'] = self.quality.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


class VideoProcessorAgent:
    """Video processing agent for handling video conversion and enhancement"""
    
    def __init__(self, agent_id: str, session_id: str):
        self.agent_id = agent_id
        self.session_id = session_id
        self.hive_memory = get_hive_memory()
        
        # Processing configuration
        self.max_concurrent_tasks = 3
        self.processing_timeout = 1800  # 30 minutes
        self.temp_dir = Path(tempfile.gettempdir()) / "video_processor"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Task management
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, VideoTask] = {}
        self.completed_tasks: Dict[str, VideoTask] = {}
        self.processing_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Performance metrics
        self.metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "formats_processed": {},
            "errors": []
        }
        
        # Agent status
        self.is_running = False
        self.last_heartbeat = datetime.now()
        
        logger.info(f"Video Processor Agent {agent_id} initialized")
    
    async def start(self):
        """Start the video processor agent"""
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._process_queue())
        asyncio.create_task(self._cleanup_temp_files())
        asyncio.create_task(self._update_metrics())
        
        # Log startup event
        log_swarm_event(
            session_id=self.session_id,
            agent_id=self.agent_id,
            event_type="agent_started",
            event_data={"agent_type": "video_processor", "status": "active"}
        )
        
        logger.info(f"Video Processor Agent {self.agent_id} started")
    
    async def stop(self):
        """Stop the video processor agent"""
        self.is_running = False
        
        # Cancel active tasks
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)
        
        # Log shutdown event
        log_swarm_event(
            session_id=self.session_id,
            agent_id=self.agent_id,
            event_type="agent_stopped",
            event_data={"agent_type": "video_processor", "status": "stopped"}
        )
        
        logger.info(f"Video Processor Agent {self.agent_id} stopped")
    
    async def submit_task(self, task: VideoTask) -> bool:
        """Submit a video processing task"""
        try:
            # Validate task
            if not await self._validate_task(task):
                return False
            
            # Store task in hive memory
            task_data = task.to_dict()
            store_swarm_memory(
                key=f"video_task_{task.task_id}",
                value=task_data,
                session_id=self.session_id,
                agent_id=self.agent_id
            )
            
            # Add to queue (priority queue uses negative priority for max-heap)
            await self.task_queue.put((-task.priority, task.task_id, task))
            
            # Log task submission
            log_swarm_event(
                session_id=self.session_id,
                agent_id=self.agent_id,
                event_type="task_submitted",
                event_data={
                    "task_id": task.task_id,
                    "input_path": task.input_path,
                    "target_format": task.target_format.value,
                    "priority": task.priority
                }
            )
            
            logger.info(f"Video task {task.task_id} submitted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit video task {task.task_id}: {e}")
            return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a specific task"""
        try:
            # Check active tasks first
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].to_dict()
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].to_dict()
            
            # Check hive memory
            task_data = retrieve_swarm_memory(
                key=f"video_task_{task_id}",
                session_id=self.session_id,
                agent_id=self.agent_id
            )
            
            return task_data
            
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a video processing task"""
        try:
            # Check if task is active
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                
                # Move to completed tasks
                self.completed_tasks[task_id] = task
                del self.active_tasks[task_id]
                
                # Update in hive memory
                store_swarm_memory(
                    key=f"video_task_{task_id}",
                    value=task.to_dict(),
                    session_id=self.session_id,
                    agent_id=self.agent_id
                )
                
                logger.info(f"Video task {task_id} cancelled")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def _process_queue(self):
        """Main queue processing loop"""
        while self.is_running:
            try:
                # Wait for a task with timeout
                try:
                    _, task_id, task = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process task with semaphore to limit concurrency
                async with self.processing_semaphore:
                    await self._process_task(task)
                
            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
                await asyncio.sleep(1)
    
    async def _process_task(self, task: VideoTask):
        """Process a single video task"""
        try:
            # Update task status
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            self.active_tasks[task.task_id] = task
            
            # Update in hive memory
            store_swarm_memory(
                key=f"video_task_{task.task_id}",
                value=task.to_dict(),
                session_id=self.session_id,
                agent_id=self.agent_id
            )
            
            # Log processing start
            log_swarm_event(
                session_id=self.session_id,
                agent_id=self.agent_id,
                event_type="task_processing_started",
                event_data={"task_id": task.task_id}
            )
            
            # Simulate video processing (in real implementation, use FFmpeg or similar)
            success = await self._simulate_video_processing(task)
            
            # Update task completion
            task.completed_at = datetime.now()
            if success:
                task.status = TaskStatus.COMPLETED
                task.progress = 100.0
                self.metrics["tasks_processed"] += 1
            else:
                task.status = TaskStatus.FAILED
                self.metrics["tasks_failed"] += 1
            
            # Calculate processing time
            processing_time = (task.completed_at - task.started_at).total_seconds()
            self.metrics["total_processing_time"] += processing_time
            self.metrics["average_processing_time"] = (
                self.metrics["total_processing_time"] / 
                max(1, self.metrics["tasks_processed"] + self.metrics["tasks_failed"])
            )
            
            # Update format statistics
            format_key = task.target_format.value
            if format_key not in self.metrics["formats_processed"]:
                self.metrics["formats_processed"][format_key] = 0
            self.metrics["formats_processed"][format_key] += 1
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Update in hive memory
            store_swarm_memory(
                key=f"video_task_{task.task_id}",
                value=task.to_dict(),
                session_id=self.session_id,
                agent_id=self.agent_id
            )
            
            # Log completion
            log_swarm_event(
                session_id=self.session_id,
                agent_id=self.agent_id,
                event_type="task_completed",
                event_data={
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "processing_time": processing_time
                }
            )
            
            logger.info(f"Video task {task.task_id} completed with status: {task.status.value}")
            
        except Exception as e:
            # Handle processing error
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            self.metrics["tasks_failed"] += 1
            self.metrics["errors"].append({
                "task_id": task.task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 10 errors
            if len(self.metrics["errors"]) > 10:
                self.metrics["errors"] = self.metrics["errors"][-10:]
            
            logger.error(f"Video task {task.task_id} failed: {e}")
    
    async def _simulate_video_processing(self, task: VideoTask) -> bool:
        """Simulate video processing (replace with actual FFmpeg implementation)"""
        try:
            # Simulate processing time based on quality
            processing_times = {
                ProcessingQuality.LOW: 2,
                ProcessingQuality.MEDIUM: 5,
                ProcessingQuality.HIGH: 10,
                ProcessingQuality.ULTRA: 20
            }
            
            total_time = processing_times.get(task.quality, 5)
            steps = 10
            step_time = total_time / steps
            
            for i in range(steps):
                if not self.is_running or task.status == TaskStatus.CANCELLED:
                    return False
                
                # Update progress
                task.progress = (i + 1) * 10
                
                # Simulate processing step
                await asyncio.sleep(step_time)
                
                # Update progress in memory occasionally
                if i % 3 == 0:
                    store_swarm_memory(
                        key=f"video_task_{task.task_id}",
                        value=task.to_dict(),
                        session_id=self.session_id,
                        agent_id=self.agent_id
                    )
            
            # Simulate output file creation
            output_path = Path(task.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a dummy output file
            with open(output_path, 'w') as f:
                f.write(f"Processed video: {task.task_id}\n")
                f.write(f"Format: {task.target_format.value}\n")
                f.write(f"Codec: {task.target_codec.value}\n")
                f.write(f"Quality: {task.quality.value}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Simulation error for task {task.task_id}: {e}")
            return False
    
    async def _validate_task(self, task: VideoTask) -> bool:
        """Validate a video processing task"""
        try:
            # Check if input file exists
            if not Path(task.input_path).exists():
                task.error_message = f"Input file not found: {task.input_path}"
                return False
            
            # Check if output directory is writable
            output_dir = Path(task.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate enhancement options
            if task.enhancement_options:
                allowed_options = ['denoise', 'sharpen', 'upscale', 'color_correction']
                for option in task.enhancement_options:
                    if option not in allowed_options:
                        task.error_message = f"Invalid enhancement option: {option}"
                        return False
            
            return True
            
        except Exception as e:
            task.error_message = f"Validation error: {e}"
            return False
    
    async def _cleanup_temp_files(self):
        """Cleanup temporary files periodically"""
        while self.is_running:
            try:
                # Clean up files older than 1 hour
                cutoff_time = datetime.now() - timedelta(hours=1)
                
                for file_path in self.temp_dir.glob("*"):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_path.unlink(missing_ok=True)
                            logger.debug(f"Cleaned up temp file: {file_path}")
                
                # Sleep for 10 minutes before next cleanup
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"Error in temp file cleanup: {e}")
                await asyncio.sleep(600)
    
    async def _update_metrics(self):
        """Update performance metrics periodically"""
        while self.is_running:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Store metrics in hive memory
                store_swarm_memory(
                    key=f"video_processor_metrics_{self.agent_id}",
                    value=self.metrics,
                    session_id=self.session_id,
                    agent_id=self.agent_id
                )
                
                # Log metrics
                log_swarm_event(
                    session_id=self.session_id,
                    agent_id=self.agent_id,
                    event_type="metrics_update",
                    event_data={
                        "active_tasks": len(self.active_tasks),
                        "queue_size": self.task_queue.qsize(),
                        "completed_tasks": len(self.completed_tasks),
                        "metrics": self.metrics
                    }
                )
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(30)
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "status": "active" if self.is_running else "inactive",
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "completed_tasks": len(self.completed_tasks),
            "metrics": self.metrics,
            "configuration": {
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "processing_timeout": self.processing_timeout,
                "temp_dir": str(self.temp_dir)
            }
        }


# Utility functions for easy task creation
def create_video_task(
    task_id: str,
    input_path: str,
    output_path: str,
    target_format: VideoFormat = VideoFormat.MP4,
    target_codec: VideoCodec = VideoCodec.H264,
    quality: ProcessingQuality = ProcessingQuality.MEDIUM,
    enhancement_options: Dict[str, Any] = None,
    priority: int = 5
) -> VideoTask:
    """Create a video processing task with default values"""
    return VideoTask(
        task_id=task_id,
        input_path=input_path,
        output_path=output_path,
        target_format=target_format,
        target_codec=target_codec,
        quality=quality,
        enhancement_options=enhancement_options or {},
        priority=priority
    )


def create_conversion_task(
    input_path: str,
    output_path: str,
    target_format: VideoFormat,
    quality: ProcessingQuality = ProcessingQuality.MEDIUM
) -> VideoTask:
    """Create a simple format conversion task"""
    task_id = f"convert_{hashlib.md5(input_path.encode()).hexdigest()[:8]}"
    
    # Select appropriate codec for format
    codec_mapping = {
        VideoFormat.MP4: VideoCodec.H264,
        VideoFormat.WEBM: VideoCodec.VP9,
        VideoFormat.AVI: VideoCodec.H264,
        VideoFormat.MOV: VideoCodec.H264,
        VideoFormat.MKV: VideoCodec.H265
    }
    
    target_codec = codec_mapping.get(target_format, VideoCodec.H264)
    
    return create_video_task(
        task_id=task_id,
        input_path=input_path,
        output_path=output_path,
        target_format=target_format,
        target_codec=target_codec,
        quality=quality
    )