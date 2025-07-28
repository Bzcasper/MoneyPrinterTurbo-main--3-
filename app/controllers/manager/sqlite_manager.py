"""
SQLite Task Manager - Replacement for Redis Task Manager
Provides comprehensive SQLite-based task management with connection pooling
"""

import json
import time
import threading
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from queue import Queue, Empty
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from loguru import logger

from app.controllers.manager.base_manager import TaskManager
from app.models.database import (
    Base, Task, TaskStatus, TaskType, create_database_engine,
    get_session_maker, PERFORMANCE_VIEWS
)
from app.models.schema import VideoParams
from app.repositories import TaskRepository
from app.services import task as tm

# Function mapping for task execution
FUNC_MAP = {
    "start": tm.start,
}


class SQLiteTaskManager(TaskManager):
    """SQLite-based task manager with connection pooling and advanced features"""
    
    def __init__(self, max_concurrent_tasks: int, database_path: str = None):
        self.database_path = database_path or "./storage/moneyprinterturbo.db"
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Initialize database engine and session maker
        self.engine = create_database_engine(self.database_path, echo=False)
        self.SessionMaker = get_session_maker(self.engine)
        
        # Connection pool for worker threads
        self._connection_pool = Queue()
        self._pool_lock = threading.Lock()
        self._initialize_connection_pool()
        
        # Task execution tracking
        self._active_tasks = {}
        self._task_lock = threading.Lock()
        
        # Initialize database tables and views
        self._initialize_database()
        
        super().__init__(max_concurrent_tasks)
        
        logger.info(f"SQLite Task Manager initialized with database: {self.database_path}")
    
    def _initialize_connection_pool(self):
        """Initialize connection pool for worker threads"""
        try:
            # Pre-create connections for the pool
            for _ in range(self.max_concurrent_tasks + 2):  # Extra connections for management
                session = self.SessionMaker()
                self._connection_pool.put(session)
            
            logger.debug(f"Initialized connection pool with {self._connection_pool.qsize()} connections")
        except Exception as e:
            logger.error(f"Error initializing connection pool: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize database tables and performance views"""
        try:
            # Create all tables
            Base.metadata.create_all(self.engine)
            
            # Create performance views
            with self.engine.connect() as conn:
                for view_sql in PERFORMANCE_VIEWS:
                    try:
                        conn.execute(text(view_sql))
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"Error creating performance view: {e}")
            
            logger.info("Database tables and views initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get database session from pool"""
        session = None
        try:
            # Get session from pool with timeout
            try:
                session = self._connection_pool.get(timeout=10)
            except Empty:
                # If pool is empty, create new session
                session = self.SessionMaker()
                logger.warning("Connection pool exhausted, created new session")
            
            yield session
            
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            if session:
                try:
                    # Return session to pool
                    self._connection_pool.put(session)
                except Exception as e:
                    logger.error(f"Error returning session to pool: {e}")
                    # Close the session if we can't return it
                    try:
                        session.close()
                    except:
                        pass
    
    def create_queue(self):
        """Create task queue (returns identifier for SQLite implementation)"""
        return "sqlite_task_queue"
    
    def enqueue(self, task: Dict):
        """Add task to SQLite-based queue"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                
                # Extract task parameters
                task_id = task.get('task_id')
                task_type = task.get('func').__name__ if callable(task.get('func')) else str(task.get('func'))
                kwargs = task.get('kwargs', {})
                
                # Convert VideoParams to dict if needed
                parameters = {}
                if 'params' in kwargs and isinstance(kwargs['params'], VideoParams):
                    parameters = kwargs['params'].dict()
                else:
                    parameters = kwargs.get('params', {})
                
                # Create task in database
                db_task = task_repo.create_task(
                    task_type=task_type,
                    parameters={
                        'task_function': task_type,
                        'function_kwargs': kwargs,
                        'video_params': parameters,
                        'enqueued_at': int(time.time())
                    },
                    user_id=kwargs.get('user_id'),
                    project_id=kwargs.get('project_id'),
                    priority=kwargs.get('priority', 5)
                )
                
                # If task_id was provided, update the database task
                if task_id and task_id != db_task.id:
                    # Store original task_id in parameters for reference
                    updated_params = db_task.parameters.copy()
                    updated_params['original_task_id'] = task_id
                    task_repo.update(db_task.id, parameters=updated_params)
                
                logger.debug(f"Enqueued task {db_task.id} of type {task_type}")
                
        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
            raise
    
    def dequeue(self) -> Optional[Dict]:
        """Get next task from SQLite-based queue"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                
                # Get highest priority pending task
                pending_tasks = task_repo.get_pending_tasks(limit=1)
                
                if not pending_tasks:
                    return None
                
                db_task = pending_tasks[0]
                
                # Update task status to processing
                task_repo.update_task_status(db_task.id, TaskStatus.PROCESSING)
                
                # Reconstruct task info for execution
                parameters = db_task.parameters or {}
                func_name = parameters.get('task_function', 'start')
                function_kwargs = parameters.get('function_kwargs', {})
                
                # Reconstruct VideoParams if available
                if 'video_params' in parameters and parameters['video_params']:
                    function_kwargs['params'] = VideoParams(**parameters['video_params'])
                
                task_info = {
                    'task_id': db_task.id,
                    'func': FUNC_MAP.get(func_name, tm.start),
                    'kwargs': function_kwargs
                }
                
                # Track active task
                with self._task_lock:
                    self._active_tasks[db_task.id] = {
                        'started_at': time.time(),
                        'task_type': db_task.task_type,
                        'thread_id': threading.current_thread().ident
                    }
                
                logger.debug(f"Dequeued task {db_task.id}")
                return task_info
                
        except Exception as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None
    
    def is_queue_empty(self) -> bool:
        """Check if task queue is empty"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                pending_count = task_repo.count(status=TaskStatus.PENDING.value)
                return pending_count == 0
        except Exception as e:
            logger.error(f"Error checking queue status: {e}")
            return True
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive queue status"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                return task_repo.get_queue_status()
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {}
    
    def update_task_progress(self, task_id: str, progress: int, 
                           stage: str = None, message: str = None):
        """Update task progress and add processing log"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                
                # Update progress
                task_repo.update_task_progress(
                    task_id=task_id,
                    progress=progress,
                    current_stage=stage,
                    stage_completed=stage if progress == 100 else None
                )
                
                # Add processing log if message provided
                if message:
                    task_repo.add_processing_log(
                        task_id=task_id,
                        stage=stage or "unknown",
                        message=message,
                        level="INFO"
                    )
                
                logger.debug(f"Updated task {task_id} progress to {progress}%")
                
        except Exception as e:
            logger.error(f"Error updating task progress: {e}")
    
    def mark_task_completed(self, task_id: str, result_data: Dict[str, Any] = None):
        """Mark task as completed with results"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                
                # Update task status and result data
                update_data = {'status': TaskStatus.COMPLETED.value}
                if result_data:
                    update_data['result_data'] = result_data
                
                task_repo.update(task_id, **update_data)
                
                # Remove from active tasks
                with self._task_lock:
                    if task_id in self._active_tasks:
                        del self._active_tasks[task_id]
                
                logger.info(f"Task {task_id} completed successfully")
                
        except Exception as e:
            logger.error(f"Error marking task completed: {e}")
    
    def mark_task_failed(self, task_id: str, error_message: str = None):
        """Mark task as failed with error details"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                
                # Update task status
                task_repo.update_task_status(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error_message=error_message
                )
                
                # Add error log
                if error_message:
                    task_repo.add_processing_log(
                        task_id=task_id,
                        stage="error",
                        message=error_message,
                        level="ERROR"
                    )
                
                # Remove from active tasks
                with self._task_lock:
                    if task_id in self._active_tasks:
                        del self._active_tasks[task_id]
                
                logger.error(f"Task {task_id} failed: {error_message}")
                
        except Exception as e:
            logger.error(f"Error marking task failed: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed task status"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                task = task_repo.get_by_id(task_id)
                
                if not task:
                    return None
                
                return {
                    'task_id': task.id,
                    'status': task.status,
                    'progress': task.progress,
                    'current_stage': task.current_stage,
                    'stages_completed': task.stages_completed or [],
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'error_message': task.error_message,
                    'result_data': task.result_data or {},
                    'duration_seconds': task.actual_duration
                }
                
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return None
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get currently active tasks"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                active_tasks = task_repo.get_active_tasks()
                
                result = []
                for task in active_tasks:
                    task_info = {
                        'task_id': task.id,
                        'task_type': task.task_type,
                        'progress': task.progress,
                        'current_stage': task.current_stage,
                        'started_at': task.started_at.isoformat() if task.started_at else None
                    }
                    
                    # Add runtime info if available
                    with self._task_lock:
                        if task.id in self._active_tasks:
                            runtime_info = self._active_tasks[task.id]
                            task_info['runtime_seconds'] = time.time() - runtime_info['started_at']
                            task_info['thread_id'] = runtime_info['thread_id']
                    
                    result.append(task_info)
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting active tasks: {e}")
            return []
    
    def cleanup_old_tasks(self, days_old: int = 30, keep_completed: bool = True) -> int:
        """Clean up old tasks"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                return task_repo.cleanup_old_tasks(days_old, keep_completed)
        except Exception as e:
            logger.error(f"Error cleaning up old tasks: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive task statistics"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                
                # Basic statistics
                stats = task_repo.get_task_statistics()
                
                # Add queue status
                queue_status = task_repo.get_queue_status()
                stats.update(queue_status)
                
                # Add active task details
                stats['active_task_details'] = self.get_active_tasks()
                
                # Add system health
                stats['connection_pool_size'] = self._connection_pool.qsize()
                stats['max_concurrent_tasks'] = self.max_concurrent_tasks
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def optimize_database(self):
        """Optimize database performance"""
        try:
            with self.engine.connect() as conn:
                # Run VACUUM to reclaim space and defragment
                conn.execute(text("VACUUM"))
                
                # Analyze tables for query optimization
                conn.execute(text("ANALYZE"))
                
                # Update table statistics
                conn.execute(text("PRAGMA optimize"))
                
                conn.commit()
                
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            import shutil
            shutil.copy2(self.database_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and health metrics"""
        try:
            with self.engine.connect() as conn:
                # Database size and page info
                page_count = conn.execute(text("PRAGMA page_count")).scalar()
                page_size = conn.execute(text("PRAGMA page_size")).scalar()
                database_size_mb = (page_count * page_size) / 1024 / 1024
                
                # Database integrity
                integrity_check = conn.execute(text("PRAGMA integrity_check")).scalar()
                
                # WAL mode status
                journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar()
                
                # Cache statistics
                cache_size = conn.execute(text("PRAGMA cache_size")).scalar()
                
                return {
                    'database_path': self.database_path,
                    'size_mb': round(database_size_mb, 2),
                    'page_count': page_count,
                    'page_size': page_size,
                    'journal_mode': journal_mode,
                    'cache_size': cache_size,
                    'integrity_ok': integrity_check == 'ok',
                    'connection_pool_size': self._connection_pool.qsize(),
                    'active_connections': self.max_concurrent_tasks + 2 - self._connection_pool.qsize()
                }
                
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {}
    
    def close(self):
        """Close all database connections and cleanup"""
        try:
            # Close all pooled connections
            while not self._connection_pool.empty():
                try:
                    session = self._connection_pool.get_nowait()
                    session.close()
                except Empty:
                    break
                except Exception as e:
                    logger.error(f"Error closing pooled session: {e}")
            
            # Dispose of the engine
            self.engine.dispose()
            
            logger.info("SQLite Task Manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing SQLite Task Manager: {e}")
    
    def __del__(self):
        """Cleanup on object destruction"""
        try:
            self.close()
        except:
            pass