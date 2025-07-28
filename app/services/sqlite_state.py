"""
SQLite-based state management for MoneyPrinterTurbo
Replaces Redis and Memory state with comprehensive SQLite storage
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
import json
import logging

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from app.models.database import (
    create_database_engine, get_session_maker, Base,
    Task, VideoMetadata, VideoProject, User, TaskStatus
)
from app.repositories import (
    TaskRepository, VideoRepository, ProjectRepository, 
    UserRepository, AnalyticsRepository
)
from app.services.state import BaseState

logger = logging.getLogger(__name__)


class SQLiteState(BaseState):
    """SQLite-based state management with comprehensive features"""
    
    def __init__(self, database_path: str = None):
        self.database_path = database_path or "./storage/moneyprinterturbo.db"
        
        # Initialize database connection
        self.engine = create_database_engine(self.database_path, echo=False)
        self.SessionMaker = get_session_maker(self.engine)
        
        # Initialize database tables
        self._initialize_database()
        
        logger.info(f"SQLite State initialized with database: {self.database_path}")
    
    def _initialize_database(self):
        """Initialize database tables"""
        try:
            Base.metadata.create_all(self.engine)
            logger.debug("Database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with automatic cleanup"""
        session = self.SessionMaker()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def update_task(self, task_id: str, state: int, progress: int = 0, **kwargs):
        """Update task state and progress"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                
                # Get or create task
                task = task_repo.get_by_id(task_id)
                if not task:
                    # Create new task if it doesn't exist
                    task = task_repo.create_task(
                        task_type=kwargs.get('task_type', 'video_generation'),
                        parameters=kwargs.get('parameters', {}),
                        user_id=kwargs.get('user_id'),
                        project_id=kwargs.get('project_id')
                    )
                
                # Map state constants to status enum
                status_mapping = {
                    1: TaskStatus.PENDING.value,      # TASK_STATE_PENDING
                    2: TaskStatus.PROCESSING.value,   # TASK_STATE_PROCESSING
                    3: TaskStatus.COMPLETED.value,    # TASK_STATE_COMPLETE
                    4: TaskStatus.FAILED.value,       # TASK_STATE_FAILED
                }
                
                status = status_mapping.get(state, TaskStatus.PENDING.value)
                
                # Update task
                update_data = {
                    'status': status,
                    'progress': progress
                }
                
                # Add additional data from kwargs
                if 'script' in kwargs:
                    result_data = task.result_data or {}
                    result_data['script'] = kwargs['script']
                    update_data['result_data'] = result_data
                
                if 'terms' in kwargs:
                    result_data = task.result_data or {}
                    result_data['terms'] = kwargs['terms']
                    update_data['result_data'] = result_data
                
                if 'videos' in kwargs:
                    result_data = task.result_data or {}
                    result_data['videos'] = kwargs['videos']
                    update_data['result_data'] = result_data
                
                if 'combined_videos' in kwargs:
                    result_data = task.result_data or {}
                    result_data['combined_videos'] = kwargs['combined_videos']
                    update_data['result_data'] = result_data
                
                if 'audio_file' in kwargs:
                    result_data = task.result_data or {}
                    result_data['audio_file'] = kwargs['audio_file']
                    update_data['result_data'] = result_data
                
                if 'subtitle_path' in kwargs:
                    result_data = task.result_data or {}
                    result_data['subtitle_path'] = kwargs['subtitle_path']
                    update_data['result_data'] = result_data
                
                if 'materials' in kwargs:
                    result_data = task.result_data or {}
                    result_data['materials'] = kwargs['materials']
                    update_data['result_data'] = result_data
                
                task_repo.update(task_id, **update_data)
                
                # Create video metadata if videos are provided
                if 'videos' in kwargs and kwargs['videos']:
                    self._create_video_metadata(session, task_id, kwargs)
                
                logger.debug(f"Updated task {task_id} to state {state} with progress {progress}%")
                
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {e}")
    
    def _create_video_metadata(self, session: Session, task_id: str, task_data: Dict[str, Any]):
        """Create video metadata entries for completed videos"""
        try:
            video_repo = VideoRepository(session)
            task_repo = TaskRepository(session)
            
            # Get task details
            task = task_repo.get_by_id(task_id)
            if not task:
                return
            
            videos = task_data.get('videos', [])
            if not videos:
                return
            
            # Extract video parameters from task
            video_params = task.parameters.get('video_params', {})
            
            for video_path in videos:
                try:
                    # Create video metadata
                    video_metadata = video_repo.create_video_metadata(
                        file_path=video_path,
                        task_id=task_id,
                        project_id=task.project_id,
                        duration=video_params.get('video_duration', 0),
                        width=1080,  # Default values, should be extracted from actual video
                        height=1920,
                        fps=30.0,
                        aspect_ratio=video_params.get('video_aspect', '9:16'),
                        subject=video_params.get('video_subject', ''),
                        script_text=task_data.get('script', ''),
                        search_terms=task_data.get('terms', []),
                        language=video_params.get('video_language', 'auto'),
                        voice_name=video_params.get('voice_name', ''),
                        bgm_type=video_params.get('bgm_type', 'none'),
                        subtitle_enabled=video_params.get('subtitle_enabled', True),
                        video_source=video_params.get('video_source', 'pexels')
                    )
                    
                    logger.debug(f"Created video metadata for {video_path}")
                    
                except Exception as e:
                    logger.error(f"Error creating video metadata for {video_path}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error creating video metadata for task {task_id}: {e}")
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                task = task_repo.get_by_id(task_id)
                
                if not task:
                    return None
                
                # Map status back to state constants
                state_mapping = {
                    TaskStatus.PENDING.value: 1,      # TASK_STATE_PENDING
                    TaskStatus.PROCESSING.value: 2,   # TASK_STATE_PROCESSING
                    TaskStatus.COMPLETED.value: 3,    # TASK_STATE_COMPLETE
                    TaskStatus.FAILED.value: 4,       # TASK_STATE_FAILED
                }
                
                state = state_mapping.get(task.status, 1)
                
                # Build task data
                task_data = {
                    'task_id': task.id,
                    'state': state,
                    'progress': task.progress,
                    'task_type': task.task_type,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'error_message': task.error_message
                }
                
                # Add result data
                if task.result_data:
                    task_data.update(task.result_data)
                
                return task_data
                
        except Exception as e:
            logger.error(f"Error getting task {task_id}: {e}")
            return None
    
    def get_all_tasks(self, page: int, page_size: int) -> Tuple[List[Dict[str, Any]], int]:
        """Get all tasks with pagination"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                
                # Get paginated results
                pagination_result = task_repo.paginate(page=page, per_page=page_size)
                
                tasks = []
                for task in pagination_result['items']:
                    # Map status to state
                    state_mapping = {
                        TaskStatus.PENDING.value: 1,
                        TaskStatus.PROCESSING.value: 2,
                        TaskStatus.COMPLETED.value: 3,
                        TaskStatus.FAILED.value: 4,
                    }
                    
                    state = state_mapping.get(task.status, 1)
                    
                    task_data = {
                        'task_id': task.id,
                        'state': state,
                        'progress': task.progress,
                        'task_type': task.task_type,
                        'created_at': task.created_at.isoformat(),
                        'started_at': task.started_at.isoformat() if task.started_at else None,
                        'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                        'error_message': task.error_message
                    }
                    
                    # Add result data
                    if task.result_data:
                        task_data.update(task.result_data)
                    
                    tasks.append(task_data)
                
                return tasks, pagination_result['total']
                
        except Exception as e:
            logger.error(f"Error getting all tasks: {e}")
            return [], 0
    
    def delete_task(self, task_id: str):
        """Delete task by ID"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                success = task_repo.delete(task_id)
                
                if success:
                    logger.debug(f"Deleted task {task_id}")
                else:
                    logger.warning(f"Task {task_id} not found for deletion")
                    
        except Exception as e:
            logger.error(f"Error deleting task {task_id}: {e}")
    
    # Enhanced methods for comprehensive state management
    
    def get_task_statistics(self, user_id: str = None, project_id: str = None) -> Dict[str, Any]:
        """Get comprehensive task statistics"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                return task_repo.get_task_statistics(user_id=user_id, project_id=project_id)
        except Exception as e:
            logger.error(f"Error getting task statistics: {e}")
            return {}
    
    def get_user_activity(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user activity summary"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                video_repo = VideoRepository(session)
                project_repo = ProjectRepository(session)
                
                # User tasks
                user_tasks = task_repo.get_tasks_by_user(user_id, limit=100)
                
                # User videos
                start_date = datetime.utcnow() - timedelta(days=days)
                user_videos = session.query(VideoMetadata).join(Task).filter(
                    and_(
                        Task.user_id == user_id,
                        VideoMetadata.created_at >= start_date
                    )
                ).all()
                
                # User projects
                user_projects = project_repo.get_projects_by_user(user_id)
                
                return {
                    'user_id': user_id,
                    'period_days': days,
                    'task_count': len(user_tasks),
                    'video_count': len(user_videos),
                    'project_count': len(user_projects),
                    'total_video_duration': sum(v.duration for v in user_videos),
                    'avg_video_quality': sum(v.quality_score or 0 for v in user_videos) / max(1, len(user_videos))
                }
                
        except Exception as e:
            logger.error(f"Error getting user activity: {e}")
            return {}
    
    def search_videos(self, search_term: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search videos with advanced filtering"""
        try:
            with self.get_session() as session:
                video_repo = VideoRepository(session)
                
                videos = video_repo.search_videos(
                    search_term=search_term,
                    language=filters.get('language') if filters else None,
                    aspect_ratio=filters.get('aspect_ratio') if filters else None,
                    duration_range=filters.get('duration_range') if filters else None,
                    limit=filters.get('limit', 50) if filters else 50
                )
                
                result = []
                for video in videos:
                    result.append({
                        'video_id': video.id,
                        'filename': video.filename,
                        'file_path': video.file_path,
                        'duration': video.duration,
                        'aspect_ratio': video.aspect_ratio,
                        'subject': video.subject,
                        'language': video.language,
                        'quality_score': video.quality_score,
                        'created_at': video.created_at.isoformat()
                    })
                
                return result
                
        except Exception as e:
            logger.error(f"Error searching videos: {e}")
            return []
    
    def get_system_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            with self.get_session() as session:
                analytics_repo = AnalyticsRepository(session)
                return analytics_repo.get_system_health_summary(hours=hours)
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def record_performance_metric(self, metric_type: str, metric_name: str, 
                                value: float, unit: str = None) -> bool:
        """Record a performance metric"""
        try:
            with self.get_session() as session:
                analytics_repo = AnalyticsRepository(session)
                analytics_repo.record_metric(
                    metric_type=metric_type,
                    metric_name=metric_name,
                    value=value,
                    unit=unit
                )
                return True
        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")
            return False
    
    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old data across all entities"""
        try:
            with self.get_session() as session:
                cleanup_stats = {}
                
                # Clean up old tasks
                task_repo = TaskRepository(session)
                cleanup_stats['tasks'] = task_repo.cleanup_old_tasks(days_old, keep_completed=True)
                
                # Clean up orphaned videos
                video_repo = VideoRepository(session)
                cleanup_stats['orphaned_videos'] = video_repo.cleanup_orphaned_videos()
                
                # Clean up old metrics
                analytics_repo = AnalyticsRepository(session)
                cleanup_stats['metrics'] = analytics_repo.cleanup_old_metrics(days_old)
                
                logger.info(f"Cleanup completed: {cleanup_stats}")
                return cleanup_stats
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {}
    
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
        """Get database information and health"""
        try:
            from sqlalchemy import text
            
            with self.engine.connect() as conn:
                # Database size
                try:
                    page_count = conn.execute(text("PRAGMA page_count")).scalar()
                    page_size = conn.execute(text("PRAGMA page_size")).scalar()
                    size_mb = (page_count * page_size) / 1024 / 1024
                except:
                    size_mb = 0
                
                # Basic counts
                with self.get_session() as session:
                    task_count = session.query(Task).count()
                    video_count = session.query(VideoMetadata).count()
                    project_count = session.query(VideoProject).count()
                    user_count = session.query(User).count()
                
                return {
                    'database_path': self.database_path,
                    'size_mb': round(size_mb, 2),
                    'table_counts': {
                        'tasks': task_count,
                        'videos': video_count,
                        'projects': project_count,
                        'users': user_count
                    },
                    'last_backup': None  # Could be enhanced to track backups
                }
                
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {}
    
    def close(self):
        """Close database connections"""
        try:
            self.engine.dispose()
            logger.info("SQLite State closed successfully")
        except Exception as e:
            logger.error(f"Error closing SQLite State: {e}")


# Factory function to create the appropriate state instance
def create_sqlite_state(database_path: str = None) -> SQLiteState:
    """Create SQLite state instance"""
    return SQLiteState(database_path=database_path)


# Enhanced state with additional utility methods
class EnhancedSQLiteState(SQLiteState):
    """Enhanced SQLite state with additional utility methods"""
    
    def __init__(self, database_path: str = None):
        super().__init__(database_path)
        self._cache = {}  # Simple in-memory cache for frequently accessed data
        self._cache_timeout = 300  # 5 minutes
    
    def get_cached_or_fetch(self, cache_key: str, fetch_func, timeout: int = None):
        """Get data from cache or fetch and cache it"""
        timeout = timeout or self._cache_timeout
        now = datetime.utcnow().timestamp()
        
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if now - cached_time < timeout:
                return cached_data
        
        # Fetch fresh data
        data = fetch_func()
        self._cache[cache_key] = (data, now)
        return data
    
    def clear_cache(self, pattern: str = None):
        """Clear cache entries"""
        if pattern:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()
        
        logger.debug(f"Cleared cache entries matching pattern: {pattern}")
    
    def get_popular_content(self, days: int = 30, limit: int = 10) -> Dict[str, Any]:
        """Get popular content analytics"""
        cache_key = f"popular_content_{days}_{limit}"
        
        def fetch_popular_content():
            try:
                with self.get_session() as session:
                    video_repo = VideoRepository(session)
                    return {
                        'popular_subjects': video_repo.get_popular_subjects(limit=limit, days=days),
                        'production_stats': video_repo.get_video_production_stats(days=days)
                    }
            except Exception as e:
                logger.error(f"Error fetching popular content: {e}")
                return {}
        
        return self.get_cached_or_fetch(cache_key, fetch_popular_content)
    
    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        try:
            with self.get_session() as session:
                task_repo = TaskRepository(session)
                
                # Current queue status
                queue_status = task_repo.get_queue_status()
                
                # Recent activity (last hour)
                one_hour_ago = datetime.utcnow() - timedelta(hours=1)
                recent_tasks = session.query(Task).filter(
                    Task.created_at >= one_hour_ago
                ).count()
                
                recent_completions = session.query(Task).filter(
                    and_(
                        Task.status == TaskStatus.COMPLETED.value,
                        Task.completed_at >= one_hour_ago
                    )
                ).count()
                
                # System health
                system_health = self.get_system_metrics(hours=1)
                
                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'queue_status': queue_status,
                    'recent_activity': {
                        'tasks_created_last_hour': recent_tasks,
                        'tasks_completed_last_hour': recent_completions
                    },
                    'system_health': system_health
                }
                
        except Exception as e:
            logger.error(f"Error getting real-time dashboard data: {e}")
            return {}