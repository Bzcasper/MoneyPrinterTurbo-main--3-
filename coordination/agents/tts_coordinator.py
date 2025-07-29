"""
TTS Coordinator Agent (Worker-5)
Specialized agent for text-to-speech coordination, voice management, and audio generation
"""

import asyncio
import os
import tempfile
import json
import hashlib
import base64
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from app.services.hive_memory import get_hive_memory, log_swarm_event, store_swarm_memory, retrieve_swarm_memory

logger = logging.getLogger(__name__)


class TTSEngine(Enum):
    """Supported TTS engines"""
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    GOOGLE = "google"
    AZURE = "azure"
    AWS = "aws"


class VoiceType(Enum):
    """Voice types and characteristics"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"
    CHILD = "child"
    ELDERLY = "elderly"
    ROBOTIC = "robotic"


class AudioFormat(Enum):
    """Supported audio output formats"""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    AAC = "aac"


class TaskPriority(Enum):
    """TTS task priority levels"""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    URGENT = 10


class TaskStatus(Enum):
    """TTS task processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VoiceProfile:
    """Voice profile configuration"""
    voice_id: str
    name: str
    engine: TTSEngine
    voice_type: VoiceType
    language: str
    accent: Optional[str] = None
    gender: Optional[str] = None
    age_range: Optional[str] = None
    description: Optional[str] = None
    sample_rate: int = 22050
    is_custom: bool = False
    clone_source: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TTSRequest:
    """Text-to-speech generation request"""
    request_id: str
    text: str
    voice_profile: VoiceProfile
    output_format: AudioFormat
    output_path: str
    priority: TaskPriority = TaskPriority.MEDIUM
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.QUEUED
    progress: float = 0.0
    error_message: Optional[str] = None
    audio_duration: Optional[float] = None
    file_size: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['voice_profile'] = asdict(self.voice_profile)
        data['voice_profile']['engine'] = self.voice_profile.engine.value
        data['voice_profile']['voice_type'] = self.voice_profile.voice_type.value
        data['output_format'] = self.output_format.value
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


class TTSCoordinatorAgent:
    """TTS coordination agent for managing text-to-speech generation"""
    
    def __init__(self, agent_id: str, session_id: str):
        self.agent_id = agent_id
        self.session_id = session_id
        self.hive_memory = get_hive_memory()
        
        # Processing configuration
        self.max_concurrent_requests = 5
        self.request_timeout = 300  # 5 minutes
        self.temp_dir = Path(tempfile.gettempdir()) / "tts_coordinator"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Queue management
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_requests: Dict[str, TTSRequest] = {}
        self.completed_requests: Dict[str, TTSRequest] = {}
        self.processing_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Voice management
        self.voice_profiles: Dict[str, VoiceProfile] = {}
        self.engine_clients: Dict[TTSEngine, Any] = {}
        
        # Performance metrics
        self.metrics = {
            "requests_processed": 0,
            "requests_failed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "total_audio_generated": 0.0,  # in seconds
            "engines_used": {},
            "voice_usage": {},
            "errors": []
        }
        
        # Agent status
        self.is_running = False
        self.last_heartbeat = datetime.now()
        
        # Initialize default voice profiles
        self._initialize_default_voices()
        
        logger.info(f"TTS Coordinator Agent {agent_id} initialized")
    
    def _initialize_default_voices(self):
        """Initialize default voice profiles for each engine"""
        default_voices = [
            VoiceProfile(
                voice_id="alloy",
                name="Alloy (OpenAI)",
                engine=TTSEngine.OPENAI,
                voice_type=VoiceType.NEUTRAL,
                language="en-US",
                description="Clear, versatile voice suitable for most content"
            ),
            VoiceProfile(
                voice_id="echo",
                name="Echo (OpenAI)",
                engine=TTSEngine.OPENAI,
                voice_type=VoiceType.MALE,
                language="en-US",
                description="Confident, articulate male voice"
            ),
            VoiceProfile(
                voice_id="nova",
                name="Nova (OpenAI)",
                engine=TTSEngine.OPENAI,
                voice_type=VoiceType.FEMALE,
                language="en-US",
                description="Warm, engaging female voice"
            )
        ]
        
        for voice in default_voices:
            self.voice_profiles[voice.voice_id] = voice
    
    async def start(self):
        """Start the TTS coordinator agent"""
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._process_queue())
        asyncio.create_task(self._cleanup_temp_files())
        asyncio.create_task(self._update_metrics())
        asyncio.create_task(self._voice_health_check())
        
        # Log startup event
        log_swarm_event(
            session_id=self.session_id,
            agent_id=self.agent_id,
            event_type="agent_started",
            event_data={"agent_type": "tts_coordinator", "status": "active"}
        )
        
        logger.info(f"TTS Coordinator Agent {self.agent_id} started")
    
    async def stop(self):
        """Stop the TTS coordinator agent"""
        self.is_running = False
        
        # Cancel active requests
        for request_id in list(self.active_requests.keys()):
            await self.cancel_request(request_id)
        
        # Log shutdown event
        log_swarm_event(
            session_id=self.session_id,
            agent_id=self.agent_id,
            event_type="agent_stopped",
            event_data={"agent_type": "tts_coordinator", "status": "stopped"}
        )
        
        logger.info(f"TTS Coordinator Agent {self.agent_id} stopped")
    
    async def submit_request(self, request: TTSRequest) -> bool:
        """Submit a TTS generation request"""
        try:
            # Validate request
            if not await self._validate_request(request):
                return False
            
            # Store request in hive memory
            request_data = request.to_dict()
            store_swarm_memory(
                key=f"tts_request_{request.request_id}",
                value=request_data,
                session_id=self.session_id,
                agent_id=self.agent_id
            )
            
            # Add to queue (priority queue uses negative priority for max-heap)
            await self.request_queue.put((-request.priority.value, request.request_id, request))
            
            # Log request submission
            log_swarm_event(
                session_id=self.session_id,
                agent_id=self.agent_id,
                event_type="request_submitted",
                event_data={
                    "request_id": request.request_id,
                    "text_length": len(request.text),
                    "voice_id": request.voice_profile.voice_id,
                    "engine": request.voice_profile.engine.value,
                    "priority": request.priority.value
                }
            )
            
            logger.info(f"TTS request {request.request_id} submitted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit TTS request {request.request_id}: {e}")
            return False
    
    async def get_request_status(self, request_id: str) -> Optional[Dict]:
        """Get status of a specific TTS request"""
        try:
            # Check active requests first
            if request_id in self.active_requests:
                return self.active_requests[request_id].to_dict()
            
            # Check completed requests
            if request_id in self.completed_requests:
                return self.completed_requests[request_id].to_dict()
            
            # Check hive memory
            request_data = retrieve_swarm_memory(
                key=f"tts_request_{request_id}",
                session_id=self.session_id,
                agent_id=self.agent_id
            )
            
            return request_data
            
        except Exception as e:
            logger.error(f"Failed to get request status for {request_id}: {e}")
            return None
    
    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a TTS generation request"""
        try:
            # Check if request is active
            if request_id in self.active_requests:
                request = self.active_requests[request_id]
                request.status = TaskStatus.CANCELLED
                request.completed_at = datetime.now()
                
                # Move to completed requests
                self.completed_requests[request_id] = request
                del self.active_requests[request_id]
                
                # Update in hive memory
                store_swarm_memory(
                    key=f"tts_request_{request_id}",
                    value=request.to_dict(),
                    session_id=self.session_id,
                    agent_id=self.agent_id
                )
                
                logger.info(f"TTS request {request_id} cancelled")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel request {request_id}: {e}")
            return False
    
    async def add_voice_profile(self, voice_profile: VoiceProfile) -> bool:
        """Add a new voice profile"""
        try:
            self.voice_profiles[voice_profile.voice_id] = voice_profile
            
            # Store in hive memory
            store_swarm_memory(
                key=f"voice_profile_{voice_profile.voice_id}",
                value=asdict(voice_profile),
                session_id=self.session_id,
                agent_id=self.agent_id
            )
            
            logger.info(f"Voice profile {voice_profile.voice_id} added")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add voice profile {voice_profile.voice_id}: {e}")
            return False
    
    async def clone_voice(self, voice_id: str, audio_sample_path: str, voice_name: str) -> Optional[VoiceProfile]:
        """Clone a voice from an audio sample"""
        try:
            # This is a placeholder for voice cloning functionality
            # In a real implementation, this would use AI voice cloning services
            
            cloned_profile = VoiceProfile(
                voice_id=voice_id,
                name=voice_name,
                engine=TTSEngine.ELEVENLABS,  # Assuming ElevenLabs for cloning
                voice_type=VoiceType.NEUTRAL,
                language="en-US",
                is_custom=True,
                clone_source=audio_sample_path,
                description=f"Cloned voice from {audio_sample_path}"
            )
            
            # Add to voice profiles
            await self.add_voice_profile(cloned_profile)
            
            logger.info(f"Voice cloned successfully: {voice_id}")
            return cloned_profile
            
        except Exception as e:
            logger.error(f"Failed to clone voice {voice_id}: {e}")
            return None
    
    async def _process_queue(self):
        """Main queue processing loop"""
        while self.is_running:
            try:
                # Wait for a request with timeout
                try:
                    _, request_id, request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process request with semaphore to limit concurrency
                async with self.processing_semaphore:
                    await self._process_request(request)
                
            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
                await asyncio.sleep(1)
    
    async def _process_request(self, request: TTSRequest):
        """Process a single TTS request"""
        try:
            # Update request status
            request.status = TaskStatus.PROCESSING
            request.started_at = datetime.now()
            self.active_requests[request.request_id] = request
            
            # Update in hive memory
            store_swarm_memory(
                key=f"tts_request_{request.request_id}",
                value=request.to_dict(),
                session_id=self.session_id,
                agent_id=self.agent_id
            )
            
            # Log processing start
            log_swarm_event(
                session_id=self.session_id,
                agent_id=self.agent_id,
                event_type="request_processing_started",
                event_data={"request_id": request.request_id}
            )
            
            # Generate audio (simulated)
            success, audio_info = await self._generate_audio(request)
            
            # Update request completion
            request.completed_at = datetime.now()
            if success:
                request.status = TaskStatus.COMPLETED
                request.progress = 100.0
                if audio_info:
                    request.audio_duration = audio_info.get("duration", 0.0)
                    request.file_size = audio_info.get("file_size", 0)
                
                self.metrics["requests_processed"] += 1
                self.metrics["total_audio_generated"] += request.audio_duration or 0.0
            else:
                request.status = TaskStatus.FAILED
                self.metrics["requests_failed"] += 1
            
            # Calculate processing time
            processing_time = (request.completed_at - request.started_at).total_seconds()
            self.metrics["total_processing_time"] += processing_time
            self.metrics["average_processing_time"] = (
                self.metrics["total_processing_time"] / 
                max(1, self.metrics["requests_processed"] + self.metrics["requests_failed"])
            )
            
            # Update engine and voice usage statistics
            engine_key = request.voice_profile.engine.value
            if engine_key not in self.metrics["engines_used"]:
                self.metrics["engines_used"][engine_key] = 0
            self.metrics["engines_used"][engine_key] += 1
            
            voice_key = request.voice_profile.voice_id
            if voice_key not in self.metrics["voice_usage"]:
                self.metrics["voice_usage"][voice_key] = 0
            self.metrics["voice_usage"][voice_key] += 1
            
            # Move to completed requests
            self.completed_requests[request.request_id] = request
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            
            # Update in hive memory
            store_swarm_memory(
                key=f"tts_request_{request.request_id}",
                value=request.to_dict(),
                session_id=self.session_id,
                agent_id=self.agent_id
            )
            
            # Log completion
            log_swarm_event(
                session_id=self.session_id,
                agent_id=self.agent_id,
                event_type="request_completed",
                event_data={
                    "request_id": request.request_id,
                    "status": request.status.value,
                    "processing_time": processing_time,
                    "audio_duration": request.audio_duration
                }
            )
            
            logger.info(f"TTS request {request.request_id} completed with status: {request.status.value}")
            
        except Exception as e:
            # Handle processing error
            request.status = TaskStatus.FAILED
            request.error_message = str(e)
            request.completed_at = datetime.now()
            
            self.metrics["requests_failed"] += 1
            self.metrics["errors"].append({
                "request_id": request.request_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 10 errors
            if len(self.metrics["errors"]) > 10:
                self.metrics["errors"] = self.metrics["errors"][-10:]
            
            logger.error(f"TTS request {request.request_id} failed: {e}")
    
    async def _generate_audio(self, request: TTSRequest) -> Tuple[bool, Optional[Dict]]:
        """Generate audio for TTS request (simulated)"""
        try:
            # Simulate processing time based on text length
            text_length = len(request.text)
            processing_time = min(max(text_length / 100, 1.0), 10.0)  # 1-10 seconds
            
            steps = 10
            step_time = processing_time / steps
            
            for i in range(steps):
                if not self.is_running or request.status == TaskStatus.CANCELLED:
                    return False, None
                
                # Update progress
                request.progress = (i + 1) * 10
                
                # Simulate processing step
                await asyncio.sleep(step_time)
                
                # Update progress in memory occasionally
                if i % 3 == 0:
                    store_swarm_memory(
                        key=f"tts_request_{request.request_id}",
                        value=request.to_dict(),
                        session_id=self.session_id,
                        agent_id=self.agent_id
                    )
            
            # Simulate audio generation
            output_path = Path(request.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a dummy audio file
            with open(output_path, 'w') as f:
                f.write(f"Generated audio for: {request.request_id}\n")
                f.write(f"Text: {request.text[:100]}...\n")
                f.write(f"Voice: {request.voice_profile.name}\n")
                f.write(f"Engine: {request.voice_profile.engine.value}\n")
                f.write(f"Format: {request.output_format.value}\n")
            
            # Calculate estimated audio duration (roughly 150 words per minute)
            word_count = len(request.text.split())
            estimated_duration = (word_count / 150) * 60  # in seconds
            estimated_duration *= request.speed  # Adjust for speed
            
            audio_info = {
                "duration": estimated_duration,
                "file_size": len(request.text) * 100,  # Rough estimate
                "sample_rate": request.voice_profile.sample_rate,
                "format": request.output_format.value
            }
            
            return True, audio_info
            
        except Exception as e:
            logger.error(f"Audio generation error for request {request.request_id}: {e}")
            return False, None
    
    async def _validate_request(self, request: TTSRequest) -> bool:
        """Validate a TTS request"""
        try:
            # Check text length
            if not request.text or len(request.text.strip()) == 0:
                request.error_message = "Text cannot be empty"
                return False
            
            if len(request.text) > 10000:  # Arbitrary limit
                request.error_message = "Text too long (max 10000 characters)"
                return False
            
            # Check voice profile
            if request.voice_profile.voice_id not in self.voice_profiles:
                request.error_message = f"Voice profile not found: {request.voice_profile.voice_id}"
                return False
            
            # Check output directory
            output_dir = Path(request.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate audio parameters
            if not (0.1 <= request.speed <= 3.0):
                request.error_message = "Speed must be between 0.1 and 3.0"
                return False
            
            if not (0.1 <= request.pitch <= 2.0):
                request.error_message = "Pitch must be between 0.1 and 2.0"
                return False
            
            if not (0.1 <= request.volume <= 2.0):
                request.error_message = "Volume must be between 0.1 and 2.0"
                return False
            
            return True
            
        except Exception as e:
            request.error_message = f"Validation error: {e}"
            return False
    
    async def _cleanup_temp_files(self):
        """Cleanup temporary files periodically"""
        while self.is_running:
            try:
                # Clean up files older than 2 hours
                cutoff_time = datetime.now() - timedelta(hours=2)
                
                for file_path in self.temp_dir.glob("*"):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_path.unlink(missing_ok=True)
                            logger.debug(f"Cleaned up temp file: {file_path}")
                
                # Sleep for 15 minutes before next cleanup
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error(f"Error in temp file cleanup: {e}")
                await asyncio.sleep(900)
    
    async def _voice_health_check(self):
        """Check voice engine health periodically"""
        while self.is_running:
            try:
                # Simulate health checks for different engines
                for engine in TTSEngine:
                    # In a real implementation, this would ping the actual services
                    health_status = "healthy"  # Simulated
                    
                    log_swarm_event(
                        session_id=self.session_id,
                        agent_id=self.agent_id,
                        event_type="engine_health_check",
                        event_data={
                            "engine": engine.value,
                            "status": health_status
                        }
                    )
                
                # Sleep for 5 minutes between health checks
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in voice health check: {e}")
                await asyncio.sleep(300)
    
    async def _update_metrics(self):
        """Update performance metrics periodically"""
        while self.is_running:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Store metrics in hive memory
                store_swarm_memory(
                    key=f"tts_coordinator_metrics_{self.agent_id}",
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
                        "active_requests": len(self.active_requests),
                        "queue_size": self.request_queue.qsize(),
                        "completed_requests": len(self.completed_requests),
                        "voice_profiles": len(self.voice_profiles),
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
            "active_requests": len(self.active_requests),
            "queue_size": self.request_queue.qsize(),
            "completed_requests": len(self.completed_requests),
            "voice_profiles": len(self.voice_profiles),
            "metrics": self.metrics,
            "configuration": {
                "max_concurrent_requests": self.max_concurrent_requests,
                "request_timeout": self.request_timeout,
                "temp_dir": str(self.temp_dir)
            }
        }


# Utility functions for easy request creation
def create_tts_request(
    request_id: str,
    text: str,
    voice_id: str,
    output_path: str,
    voice_profiles: Dict[str, VoiceProfile],
    output_format: AudioFormat = AudioFormat.MP3,
    priority: TaskPriority = TaskPriority.MEDIUM,
    speed: float = 1.0,
    pitch: float = 1.0,
    volume: float = 1.0
) -> Optional[TTSRequest]:
    """Create a TTS request with validation"""
    if voice_id not in voice_profiles:
        logger.error(f"Voice profile not found: {voice_id}")
        return None
    
    return TTSRequest(
        request_id=request_id,
        text=text,
        voice_profile=voice_profiles[voice_id],
        output_format=output_format,
        output_path=output_path,
        priority=priority,
        speed=speed,
        pitch=pitch,
        volume=volume
    )


def create_batch_requests(
    texts: List[str],
    voice_id: str,
    output_dir: str,
    voice_profiles: Dict[str, VoiceProfile],
    output_format: AudioFormat = AudioFormat.MP3,
    priority: TaskPriority = TaskPriority.MEDIUM
) -> List[TTSRequest]:
    """Create multiple TTS requests for batch processing"""
    requests = []
    
    for i, text in enumerate(texts):
        request_id = f"batch_{hashlib.md5(f'{voice_id}_{i}_{text[:50]}'.encode()).hexdigest()[:8]}"
        output_path = f"{output_dir}/audio_{i:03d}.{output_format.value}"
        
        request = create_tts_request(
            request_id=request_id,
            text=text,
            voice_id=voice_id,
            output_path=output_path,
            voice_profiles=voice_profiles,
            output_format=output_format,
            priority=priority
        )
        
        if request:
            requests.append(request)
    
    return requests