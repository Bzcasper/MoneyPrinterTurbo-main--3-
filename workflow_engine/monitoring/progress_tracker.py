"""
Progress tracking system for workflow execution.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import threading


logger = logging.getLogger(__name__)


@dataclass
class ProgressInfo:
    """Progress information for a workflow."""
    workflow_id: str
    total_steps: int
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    running_steps: int = 0
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    current_phase: str = "initializing"
    estimated_completion: Optional[float] = None
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100.0
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed execution time."""
        return time.time() - self.start_time
    
    @property
    def steps_per_second(self) -> float:
        """Calculate processing rate."""
        elapsed = self.elapsed_time
        if elapsed <= 0:
            return 0.0
        return self.completed_steps / elapsed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "skipped_steps": self.skipped_steps,
            "running_steps": self.running_steps,
            "completion_percentage": self.completion_percentage,
            "elapsed_time": self.elapsed_time,
            "steps_per_second": self.steps_per_second,
            "current_phase": self.current_phase,
            "estimated_completion": self.estimated_completion,
            "last_update": datetime.fromtimestamp(self.last_update).isoformat()
        }


class ProgressTracker:
    """
    Tracks progress of workflow execution with real-time updates.
    
    Features:
    - Real-time progress tracking
    - Completion percentage calculation
    - Performance metrics (steps/second)
    - Estimated completion time
    - Thread-safe operations
    - Historical progress data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._progress_data: Dict[str, ProgressInfo] = {}
        self._lock = threading.RLock()
        self._observers: List[callable] = []
    
    def start_workflow(self, workflow_id: str, total_steps: int):
        """Start tracking a new workflow."""
        with self._lock:
            progress = ProgressInfo(
                workflow_id=workflow_id,
                total_steps=total_steps,
                current_phase="starting"
            )
            self._progress_data[workflow_id] = progress
            
            self.logger.debug(f"Started tracking workflow '{workflow_id}' with {total_steps} steps")
            self._notify_observers(workflow_id, progress)
    
    def update_step_status(self, 
                          workflow_id: str, 
                          completed: int = 0, 
                          failed: int = 0, 
                          skipped: int = 0, 
                          running: int = 0,
                          phase: Optional[str] = None):
        """Update step status counts."""
        with self._lock:
            progress = self._progress_data.get(workflow_id)
            if not progress:
                self.logger.warning(f"No progress tracking found for workflow '{workflow_id}'")
                return
            
            if completed > 0:
                progress.completed_steps += completed
            if failed > 0:
                progress.failed_steps += failed
            if skipped > 0:
                progress.skipped_steps += skipped
            
            progress.running_steps = running
            progress.last_update = time.time()
            
            if phase:
                progress.current_phase = phase
            
            # Update estimated completion
            progress.estimated_completion = self._calculate_estimated_completion(progress)
            
            self.logger.debug(f"Updated progress for workflow '{workflow_id}': {progress.completion_percentage:.1f}%")
            self._notify_observers(workflow_id, progress)
    
    def complete_workflow(self, workflow_id: str):
        """Mark workflow as completed."""
        with self._lock:
            progress = self._progress_data.get(workflow_id)
            if progress:
                progress.current_phase = "completed"
                progress.running_steps = 0
                progress.last_update = time.time()
                
                self.logger.info(f"Workflow '{workflow_id}' tracking completed: {progress.completion_percentage:.1f}%")
                self._notify_observers(workflow_id, progress)
    
    def fail_workflow(self, workflow_id: str, error: str = ""):
        """Mark workflow as failed."""
        with self._lock:
            progress = self._progress_data.get(workflow_id)
            if progress:
                progress.current_phase = f"failed: {error}" if error else "failed"
                progress.running_steps = 0
                progress.last_update = time.time()
                
                self.logger.info(f"Workflow '{workflow_id}' tracking failed at {progress.completion_percentage:.1f}%")
                self._notify_observers(workflow_id, progress)
    
    def get_progress(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for a workflow."""
        with self._lock:
            progress = self._progress_data.get(workflow_id)
            return progress.to_dict() if progress else None
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows being tracked."""
        with self._lock:
            return [progress.to_dict() for progress in self._progress_data.values()]
    
    def remove_workflow(self, workflow_id: str) -> bool:
        """Remove workflow from tracking."""
        with self._lock:
            if workflow_id in self._progress_data:
                del self._progress_data[workflow_id]
                self.logger.debug(f"Removed tracking for workflow '{workflow_id}'")
                return True
            return False
    
    def _calculate_estimated_completion(self, progress: ProgressInfo) -> Optional[float]:
        """Calculate estimated completion time."""
        if progress.completed_steps <= 0:
            return None
        
        steps_per_second = progress.steps_per_second
        if steps_per_second <= 0:
            return None
        
        remaining_steps = progress.total_steps - progress.completed_steps
        estimated_seconds = remaining_steps / steps_per_second
        
        return time.time() + estimated_seconds
    
    def add_observer(self, observer: callable):
        """Add a progress observer callback."""
        self._observers.append(observer)
        self.logger.debug(f"Added progress observer: {observer.__name__}")
    
    def remove_observer(self, observer: callable):
        """Remove a progress observer."""
        if observer in self._observers:
            self._observers.remove(observer)
            self.logger.debug(f"Removed progress observer: {observer.__name__}")
    
    def _notify_observers(self, workflow_id: str, progress: ProgressInfo):
        """Notify all observers of progress updates."""
        for observer in self._observers:
            try:
                observer(workflow_id, progress.to_dict())
            except Exception as e:
                self.logger.error(f"Error in progress observer {observer.__name__}: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all tracked workflows."""
        with self._lock:
            if not self._progress_data:
                return {
                    "total_workflows": 0,
                    "active_workflows": 0,
                    "completed_workflows": 0,
                    "failed_workflows": 0,
                    "total_steps": 0,
                    "completed_steps": 0,
                    "average_completion": 0.0
                }
            
            total_workflows = len(self._progress_data)
            completed_workflows = sum(1 for p in self._progress_data.values() if p.current_phase == "completed")
            failed_workflows = sum(1 for p in self._progress_data.values() if p.current_phase.startswith("failed"))
            active_workflows = total_workflows - completed_workflows - failed_workflows
            
            total_steps = sum(p.total_steps for p in self._progress_data.values())
            completed_steps = sum(p.completed_steps for p in self._progress_data.values())
            
            average_completion = (completed_steps / total_steps * 100) if total_steps > 0 else 0.0
            
            return {
                "total_workflows": total_workflows,
                "active_workflows": active_workflows,
                "completed_workflows": completed_workflows,
                "failed_workflows": failed_workflows,
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "average_completion": average_completion
            }
    
    def cleanup(self):
        """Cleanup tracker resources."""
        with self._lock:
            self._progress_data.clear()
            self._observers.clear()
            self.logger.info("Progress tracker cleanup completed")