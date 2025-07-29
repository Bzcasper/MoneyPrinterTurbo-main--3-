"""
Memory Manager - Smart memory allocation and cleanup

This module provides intelligent memory management for video processing
with automatic cleanup, memory pressure detection, and resource optimization.

Author: MoneyPrinterTurbo Enhanced System
Version: 1.0.0
"""

import os
import gc
import time
import threading
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import shutil

from loguru import logger
import psutil


@dataclass
class MemoryStats:
    """Memory statistics"""
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    swap_usage: float
    cache_size: int = 0


@dataclass
class WorkflowMemory:
    """Memory tracking for individual workflows"""
    workflow_id: str
    allocated_memory: int = 0
    temp_files: Set[str] = field(default_factory=set)
    start_time: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    priority: int = 1  # 1=high, 2=medium, 3=low


class MemoryCache:
    """LRU cache with memory-aware eviction"""
    
    def __init__(self, max_size_mb: int = 512):
        """Initialize memory cache"""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.item_sizes: Dict[str, int] = {}
        self.current_size = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any, size: Optional[int] = None) -> bool:
        """Put item in cache"""
        with self._lock:
            # Estimate size if not provided
            if size is None:
                size = self._estimate_size(value)
            
            # Check if item would fit
            if size > self.max_size_bytes:
                return False
            
            # Make space if needed
            while self.current_size + size > self.max_size_bytes:
                if not self._evict_lru():
                    return False
            
            # Store item
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.item_sizes[key] = size
            self.current_size += size
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove item from cache"""
        with self._lock:
            if key in self.cache:
                size = self.item_sizes.pop(key, 0)
                self.current_size -= size
                del self.cache[key]
                self.access_times.pop(key, None)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.item_sizes.clear()
            self.current_size = 0
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item"""
        if not self.access_times:
            return False
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        return self.remove(oldest_key)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            import sys
            return sys.getsizeof(obj)
        except Exception:
            return 1024  # Default 1KB


class TempFileManager:
    """Manage temporary files with automatic cleanup"""
    
    def __init__(self):
        """Initialize temp file manager"""
        self.temp_dirs: Set[Path] = set()
        self.temp_files: Set[Path] = set()
        self.workflow_files: Dict[str, Set[Path]] = {}
        self._lock = threading.RLock()
        
        # Create main temp directory
        self.base_temp_dir = Path(tempfile.gettempdir()) / "video_processing"
        self.base_temp_dir.mkdir(exist_ok=True)
        self.temp_dirs.add(self.base_temp_dir)
    
    def create_temp_file(
        self, workflow_id: str, suffix: str = ".tmp", prefix: str = "video_"
    ) -> Path:
        """Create a temporary file for a workflow"""
        with self._lock:
            temp_file = self.base_temp_dir / f"{prefix}{workflow_id}_{int(time.time())}_{suffix}"
            
            self.temp_files.add(temp_file)
            
            if workflow_id not in self.workflow_files:
                self.workflow_files[workflow_id] = set()
            self.workflow_files[workflow_id].add(temp_file)
            
            return temp_file
    
    def create_temp_dir(self, workflow_id: str) -> Path:
        """Create a temporary directory for a workflow"""
        with self._lock:
            temp_dir = self.base_temp_dir / f"workflow_{workflow_id}_{int(time.time())}"
            temp_dir.mkdir(exist_ok=True)
            
            self.temp_dirs.add(temp_dir)
            
            if workflow_id not in self.workflow_files:
                self.workflow_files[workflow_id] = set()
            self.workflow_files[workflow_id].add(temp_dir)
            
            return temp_dir
    
    def cleanup_workflow(self, workflow_id: str):
        """Clean up all temporary files for a workflow"""
        with self._lock:
            if workflow_id in self.workflow_files:
                for path in self.workflow_files[workflow_id].copy():
                    self._remove_path(path)
                del self.workflow_files[workflow_id]
    
    def cleanup_all(self):
        """Clean up all temporary files"""
        with self._lock:
            for path in list(self.temp_files):
                self._remove_path(path)
            
            for path in list(self.temp_dirs):
                if path != self.base_temp_dir:
                    self._remove_path(path)
            
            self.temp_files.clear()
            self.workflow_files.clear()
    
    def get_temp_usage(self) -> int:
        """Get total size of temporary files in bytes"""
        total_size = 0
        
        with self._lock:
            for path in self.temp_files.union(self.temp_dirs):
                try:
                    if path.is_file():
                        total_size += path.stat().st_size
                    elif path.is_dir():
                        total_size += sum(
                            f.stat().st_size for f in path.rglob('*') if f.is_file()
                        )
                except Exception:
                    continue
        
        return total_size
    
    def _remove_path(self, path: Path):
        """Remove file or directory"""
        try:
            if path.is_file():
                path.unlink(missing_ok=True)
                self.temp_files.discard(path)
            elif path.is_dir() and path != self.base_temp_dir:
                shutil.rmtree(path, ignore_errors=True)
                self.temp_dirs.discard(path)
        except Exception as e:
            logger.warning(f"Failed to remove {path}: {str(e)}")


class MemoryManager:
    """
    Intelligent memory management for video processing
    
    Provides automatic memory monitoring, cleanup, and resource optimization
    with workflow-specific tracking and pressure-based actions.
    """
    
    def __init__(self):
        """Initialize memory manager"""
        self.workflows: Dict[str, WorkflowMemory] = {}
        self.memory_cache = MemoryCache()
        self.temp_file_manager = TempFileManager()
        
        # Configuration
        self.memory_warning_threshold = 0.8  # 80%
        self.memory_critical_threshold = 0.9  # 90%
        self.cleanup_interval = 300  # 5 minutes
        self.max_workflow_age = 3600  # 1 hour
        
        # State tracking
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._monitoring_enabled = True
        
        logger.info("MemoryManager initialized successfully")
    
    def initialize_for_workflow(self, workflow_id: str, priority: int = 1):
        """Initialize memory tracking for a workflow"""
        with self._lock:
            self.workflows[workflow_id] = WorkflowMemory(
                workflow_id=workflow_id,
                priority=priority
            )
            
            logger.debug(f"Initialized memory tracking for workflow {workflow_id}")
    
    def allocate_memory(self, workflow_id: str, size_bytes: int) -> bool:
        """Allocate memory for a workflow"""
        with self._lock:
            # Check if allocation would exceed available memory
            stats = self.get_memory_stats()
            
            if stats.memory_percent > self.memory_critical_threshold * 100:
                logger.warning("Memory critical - forcing cleanup before allocation")
                self.force_cleanup()
                
                # Re-check after cleanup
                stats = self.get_memory_stats()
                if stats.memory_percent > self.memory_critical_threshold * 100:
                    logger.error("Insufficient memory for allocation")
                    return False
            
            # Track allocation
            if workflow_id in self.workflows:
                self.workflows[workflow_id].allocated_memory += size_bytes
                self.workflows[workflow_id].last_access = time.time()
            
            return True
    
    def deallocate_memory(self, workflow_id: str, size_bytes: int):
        """Deallocate memory for a workflow"""
        with self._lock:
            if workflow_id in self.workflows:
                self.workflows[workflow_id].allocated_memory = max(
                    0, self.workflows[workflow_id].allocated_memory - size_bytes
                )
                self.workflows[workflow_id].last_access = time.time()
    
    def register_temp_file(self, workflow_id: str, file_path: str):
        """Register a temporary file for cleanup"""
        with self._lock:
            if workflow_id in self.workflows:
                self.workflows[workflow_id].temp_files.add(file_path)
    
    def is_memory_available(self, required_mb: int = 100) -> bool:
        """Check if enough memory is available"""
        stats = self.get_memory_stats()
        available_mb = stats.available_memory / (1024 * 1024)
        
        return available_mb >= required_mb
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return MemoryStats(
                total_memory=memory.total,
                available_memory=memory.available,
                used_memory=memory.used,
                memory_percent=memory.percent,
                swap_usage=swap.percent,
                cache_size=self.memory_cache.current_size
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {str(e)}")
            return MemoryStats(0, 0, 0, 0.0, 0.0)
    
    def get_usage_percentage(self) -> float:
        """Get current memory usage percentage"""
        stats = self.get_memory_stats()
        return stats.memory_percent
    
    def cleanup_workflow(self, workflow_id: str):
        """Clean up all resources for a workflow"""
        with self._lock:
            if workflow_id not in self.workflows:
                return
            
            workflow = self.workflows[workflow_id]
            
            # Clean up temporary files
            for temp_file in workflow.temp_files.copy():
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        logger.debug(f"Removed temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_file}: {str(e)}")
            
            # Clean up workflow temp files via temp manager
            self.temp_file_manager.cleanup_workflow(workflow_id)
            
            # Remove from tracking
            del self.workflows[workflow_id]
            
            logger.debug(f"Cleaned up workflow {workflow_id}")
    
    def force_cleanup(self):
        """Force immediate cleanup of all resources"""
        logger.info("Forcing memory cleanup...")
        
        with self._lock:
            # Clean up old workflows
            current_time = time.time()
            old_workflows = [
                wid for wid, workflow in self.workflows.items()
                if current_time - workflow.start_time > self.max_workflow_age
            ]
            
            for workflow_id in old_workflows:
                logger.info(f"Cleaning up old workflow: {workflow_id}")
                self.cleanup_workflow(workflow_id)
            
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clean up temporary files
            self.temp_file_manager.cleanup_all()
            
            # Force garbage collection
            gc.collect()
            
            self._last_cleanup = current_time
            
            # Log results
            stats = self.get_memory_stats()
            logger.info(f"Cleanup completed - Memory usage: {stats.memory_percent:.1f}%")
    
    def auto_cleanup_if_needed(self):
        """Perform automatic cleanup if needed"""
        current_time = time.time()
        
        # Check if cleanup interval has passed
        if current_time - self._last_cleanup < self.cleanup_interval:
            return
        
        stats = self.get_memory_stats()
        
        # Check memory pressure
        if stats.memory_percent > self.memory_warning_threshold * 100:
            logger.warning(f"Memory warning threshold exceeded: {stats.memory_percent:.1f}%")
            self.force_cleanup()
        elif current_time - self._last_cleanup > self.cleanup_interval * 2:
            # Periodic cleanup even if memory is okay
            self._cleanup_old_workflows()
    
    def get_workflow_memory_usage(self, workflow_id: str) -> int:
        """Get memory usage for a specific workflow"""
        with self._lock:
            if workflow_id in self.workflows:
                return self.workflows[workflow_id].allocated_memory
            return 0
    
    def get_total_allocated_memory(self) -> int:
        """Get total allocated memory across all workflows"""
        with self._lock:
            return sum(w.allocated_memory for w in self.workflows.values())
    
    def is_healthy(self) -> bool:
        """Check if memory manager is healthy"""
        try:
            stats = self.get_memory_stats()
            return (
                stats.memory_percent < self.memory_critical_threshold * 100 and
                stats.swap_usage < 50.0 and
                len(self.workflows) < 100  # Reasonable workflow limit
            )
        except Exception:
            return False
    
    def _cleanup_old_workflows(self):
        """Clean up old workflows based on age and priority"""
        current_time = time.time()
        
        with self._lock:
            # Sort workflows by priority and age
            workflow_items = list(self.workflows.items())
            workflow_items.sort(
                key=lambda x: (x[1].priority, current_time - x[1].last_access),
                reverse=True
            )
            
            # Clean up oldest low-priority workflows
            cleaned_count = 0
            for workflow_id, workflow in workflow_items:
                age = current_time - workflow.start_time
                idle_time = current_time - workflow.last_access
                
                if (age > self.max_workflow_age or 
                    (workflow.priority >= 3 and idle_time > 600)):  # 10 min idle for low priority
                    
                    self.cleanup_workflow(workflow_id)
                    cleaned_count += 1
                    
                    if cleaned_count >= 5:  # Limit cleanup batch size
                        break
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old workflows")
            
            self._last_cleanup = current_time
    
    def create_temp_file(self, workflow_id: str, suffix: str = ".tmp") -> Path:
        """Create a temporary file for a workflow"""
        return self.temp_file_manager.create_temp_file(workflow_id, suffix)
    
    def create_temp_dir(self, workflow_id: str) -> Path:
        """Create a temporary directory for a workflow"""
        return self.temp_file_manager.create_temp_dir(workflow_id)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size_bytes": self.memory_cache.current_size,
            "size_mb": self.memory_cache.current_size / (1024 * 1024),
            "max_size_mb": self.memory_cache.max_size_bytes / (1024 * 1024),
            "item_count": len(self.memory_cache.cache),
            "utilization": (self.memory_cache.current_size / 
                          self.memory_cache.max_size_bytes * 100)
        }