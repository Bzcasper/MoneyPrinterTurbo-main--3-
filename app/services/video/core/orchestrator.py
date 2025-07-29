"""
Video Processing Orchestrator - Main workflow coordinator

This module provides the main orchestration for video processing workflows,
coordinating between validation, processing, concatenation, and monitoring components.

Author: MoneyPrinterTurbo Enhanced System
Version: 1.0.0
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger
import psutil

from app.security.config_manager import ConfigManager
from app.security.input_validator import InputValidator
from app.security.audit_logger import AuditLogger
from app.services.video.validation.engine import ValidationEngine
from app.services.video.processing.pipeline import ProcessingPipeline
from app.services.video.concatenation.service import ConcatenationService
from app.services.video.memory.manager import MemoryManager
from app.services.video.monitoring.performance import PerformanceMonitor


class WorkflowStatus(Enum):
    """Video processing workflow status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VideoParams:
    """Video processing parameters"""
    clips: List[str]
    dimensions: Dict[str, int]
    quality: Dict[str, Any]
    output_path: str
    audio_path: Optional[str] = None
    subtitle_path: Optional[str] = None
    output_format: str = "mp4"
    concat_mode: str = "progressive"
    
    def __post_init__(self):
        """Validate parameters after initialization"""
        if not self.clips:
            raise ValueError("Clips list cannot be empty")
        if not self.output_path:
            raise ValueError("Output path is required")


@dataclass
class VideoResult:
    """Video processing result"""
    success: bool
    output_file: Optional[str] = None
    processing_time: float = 0.0
    metrics: Optional[Dict] = None
    error_message: Optional[str] = None
    workflow_id: Optional[str] = None


@dataclass
class HealthStatus:
    """System health status"""
    validation_engine: bool = False
    processing_pipeline: bool = False
    concatenation_service: bool = False
    memory_usage: float = 0.0
    performance_metrics: Optional[Dict] = None
    overall_healthy: bool = False


class VideoProcessingOrchestrator:
    """
    Main orchestrator for video processing workflows
    
    Coordinates validation, processing, concatenation, and monitoring
    with comprehensive error handling and resource management.
    """
    
    def __init__(
        self,
        validation_engine=None,
        processing_pipeline=None,
        concatenation_service=None,
        performance_monitor=None,
        memory_manager=None,
        config_manager: Optional[ConfigManager] = None
    ):
        """Initialize orchestrator with optional dependencies for testing compatibility"""
        # Initialize with real components or dependency injection for testing
        self.validation_engine = validation_engine or ValidationEngine()
        self.processing_pipeline = processing_pipeline or ProcessingPipeline()
        self.concatenation_service = concatenation_service or ConcatenationService()
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.memory_manager = memory_manager or MemoryManager()
        self.config_manager = config_manager or ConfigManager()
        
        # Security components
        self.input_validator = InputValidator()
        self.audit_logger = AuditLogger()
        
        # Active workflows tracking
        self.active_workflows: Dict[str, Dict] = {}
        
        logger.info("VideoProcessingOrchestrator initialized successfully")
    
    async def process_video_request(self, params: VideoParams) -> VideoResult:
        """
        Main entry point for video processing requests
        
        Args:
            params: Video processing parameters
            
        Returns:
            VideoResult with processing outcome
        """
        workflow_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Input validation
        if not self._validate_request(params):
            return VideoResult(
                success=False,
                error_message="Invalid request parameters",
                workflow_id=workflow_id
            )
        
        # Log workflow start
        self.audit_logger.log_event(
            "video_processing_started",
            {"workflow_id": workflow_id, "clips_count": len(params.clips)}
        )
        
        try:
            # Phase 1: Input validation and preparation
            validation_result = await self._validate_inputs(workflow_id, params)
            if not validation_result["is_valid"]:
                return self._create_error_result(
                    workflow_id, "Validation failed", validation_result["errors"]
                )
            
            # Phase 2: Initialize performance monitoring
            await self._initialize_monitoring(workflow_id, params)
            
            # Phase 3: Process individual clips
            processed_clips = await self._process_clips(workflow_id, params)
            if not processed_clips:
                return self._create_error_result(workflow_id, "Clip processing failed")
            
            # Phase 4: Concatenate processed clips
            concatenated_video = await self._concatenate_clips(
                workflow_id, processed_clips, params
            )
            if not concatenated_video:
                return self._create_error_result(workflow_id, "Concatenation failed")
            
            # Phase 5: Final validation and cleanup
            final_result = await self._finalize_processing(
                workflow_id, concatenated_video, params
            )
            
            processing_time = time.time() - start_time
            metrics = self.performance_monitor.get_workflow_summary(workflow_id)
            
            # Log successful completion
            self.audit_logger.log_event(
                "video_processing_completed",
                {
                    "workflow_id": workflow_id,
                    "processing_time": processing_time,
                    "output_file": final_result
                }
            )
            
            return VideoResult(
                success=True,
                output_file=final_result,
                processing_time=processing_time,
                metrics=metrics,
                workflow_id=workflow_id
            )
            
        except Exception as e:
            logger.error(f"Processing failed for workflow {workflow_id}: {str(e)}")
            self.audit_logger.log_event(
                "video_processing_failed",
                {"workflow_id": workflow_id, "error": str(e)}
            )
            return self._create_error_result(workflow_id, "Processing failed", str(e))
        
        finally:
            # Cleanup workflow resources
            await self._cleanup_workflow(workflow_id)
    
    async def process_batch_requests(
        self, batch: List[VideoParams]
    ) -> List[VideoResult]:
        """
        Process multiple video requests in batch
        
        Args:
            batch: List of video processing parameters
            
        Returns:
            List of processing results
        """
        if not batch:
            return []
        
        max_batch_size = self.config_manager.get("max_batch_size", 10)
        if len(batch) > max_batch_size:
            logger.warning(f"Batch size {len(batch)} exceeds limit {max_batch_size}")
            batch = batch[:max_batch_size]
        
        results = []
        for params in batch:
            # Check memory availability before processing each item
            if not self.memory_manager.is_memory_available():
                logger.warning("Memory pressure detected, forcing cleanup")
                self.memory_manager.force_cleanup()
                await asyncio.sleep(1)  # Brief pause for cleanup
            
            result = await self.process_video_request(params)
            results.append(result)
        
        return results
    
    def health_check(self) -> HealthStatus:
        """
        Perform comprehensive health check
        
        Returns:
            Current system health status
        """
        status = HealthStatus()
        
        try:
            # Check individual components
            status.validation_engine = self._check_component_health(
                self.validation_engine
            )
            status.processing_pipeline = self._check_component_health(
                self.processing_pipeline
            )
            status.concatenation_service = self._check_component_health(
                self.concatenation_service
            )
            
            # Memory usage check
            status.memory_usage = self.memory_manager.get_usage_percentage()
            
            # Performance metrics
            status.performance_metrics = self.performance_monitor.get_system_metrics()
            
            # Overall health assessment
            status.overall_healthy = (
                status.validation_engine and
                status.processing_pipeline and
                status.concatenation_service and
                status.memory_usage < 80.0
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            status.overall_healthy = False
        
        return status
    
    def _validate_request(self, params: VideoParams) -> bool:
        """Validate incoming request parameters"""
        try:
            # Security validation
            for clip_path in params.clips:
                if not self.input_validator.validate_file_path(clip_path):
                    logger.warning(f"Invalid file path: {clip_path}")
                    return False
            
            if not self.input_validator.validate_file_path(params.output_path):
                logger.warning(f"Invalid output path: {params.output_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Request validation failed: {str(e)}")
            return False
    
    async def _validate_inputs(self, workflow_id: str, params: VideoParams) -> Dict:
        """Phase 1: Input validation and preparation"""
        try:
            validation_result = self.validation_engine.validate_inputs(params)
            
            self.active_workflows[workflow_id] = {
                "status": WorkflowStatus.RUNNING,
                "start_time": time.time(),
                "params": params
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Input validation failed for {workflow_id}: {str(e)}")
            return {"is_valid": False, "errors": [str(e)]}
    
    async def _initialize_monitoring(self, workflow_id: str, params: VideoParams):
        """Phase 2: Initialize performance monitoring"""
        self.performance_monitor.start_workflow_monitoring(workflow_id)
        self.memory_manager.initialize_for_workflow(workflow_id)
    
    async def _process_clips(self, workflow_id: str, params: VideoParams) -> List:
        """Phase 3: Process individual clips"""
        try:
            processed_clips = await self.processing_pipeline.process_clips(
                clips=params.clips,
                target_dimensions=params.dimensions,
                quality_settings=params.quality
            )
            
            return processed_clips
            
        except Exception as e:
            logger.error(f"Clip processing failed for {workflow_id}: {str(e)}")
            return []
    
    async def _concatenate_clips(
        self, workflow_id: str, processed_clips: List, params: VideoParams
    ) -> Optional[str]:
        """Phase 4: Concatenate processed clips"""
        try:
            result = await self.concatenation_service.concatenate(
                clips=processed_clips,
                output_path=params.output_path,
                concat_mode=params.concat_mode
            )
            
            return result.output_path if result.success else None
            
        except Exception as e:
            logger.error(f"Concatenation failed for {workflow_id}: {str(e)}")
            return None
    
    async def _finalize_processing(
        self, workflow_id: str, video_path: str, params: VideoParams
    ) -> str:
        """Phase 5: Final validation and cleanup"""
        # Validate output file
        validation_result = self.validation_engine.validate_output(video_path)
        if not validation_result.is_valid:
            raise Exception(f"Output validation failed: {validation_result.errors}")
        
        return video_path
    
    async def _cleanup_workflow(self, workflow_id: str):
        """Clean up workflow resources"""
        try:
            self.memory_manager.cleanup_workflow(workflow_id)
            self.performance_monitor.finish_workflow_monitoring(workflow_id)
            
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
                
        except Exception as e:
            logger.error(f"Cleanup failed for workflow {workflow_id}: {str(e)}")
    
    def _create_error_result(
        self, workflow_id: str, message: str, details: Any = None
    ) -> VideoResult:
        """Create standardized error result"""
        return VideoResult(
            success=False,
            error_message=f"{message}: {details}" if details else message,
            workflow_id=workflow_id
        )
    
    def _check_component_health(self, component) -> bool:
        """Check if a component is healthy"""
        try:
            return hasattr(component, 'is_healthy') and component.is_healthy()
        except Exception:
            return False
    
    def process_video(self, input_path: str):
        """
        Simple process_video method for TDD testing
        
        Args:
            input_path: Path to input video file
            
        Raises:
            ValueError: If input_path is None or invalid
            NotImplementedError: If processing logic is not implemented
        """
        if input_path is None:
            raise ValueError("Input path cannot be None")
        
        # For now, raise NotImplementedError to satisfy TDD red phase
        raise NotImplementedError("Video processing not yet implemented")