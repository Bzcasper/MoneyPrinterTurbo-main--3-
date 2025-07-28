"""
Task repository for video generation task management
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc
from datetime import datetime, timedelta
import logging

from app.models.database import Task, ProcessingLog, TaskStatus, TaskType
from app.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class TaskRepository(BaseRepository[Task]):
    """Repository for task operations"""
    
    def __init__(self, session: Session):
        super().__init__(session, Task)
    
    def create_task(self, task_type: str, parameters: Dict[str, Any], 
                   user_id: str = None, project_id: str = None, 
                   priority: int = 5) -> Task:
        """Create a new task with specific parameters"""
        try:
            task = Task(
                task_type=task_type,
                status=TaskStatus.PENDING.value,
                parameters=parameters,
                user_id=user_id,
                project_id=project_id,
                priority=priority
            )
            
            self.session.add(task)
            self.session.commit()
            self.session.refresh(task)
            
            logger.info(f"Created task {task.id} of type {task_type}")
            return task
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating task: {e}")
            raise
    
    def update_task_progress(self, task_id: str, progress: int, 
                           current_stage: str = None, 
                           stage_completed: str = None) -> Optional[Task]:
        """Update task progress and stage information"""
        try:
            task = self.get_by_id(task_id)
            if not task:
                return None
            
            task.progress = min(100, max(0, progress))
            
            if current_stage:
                task.current_stage = current_stage
            
            if stage_completed and stage_completed not in task.stages_completed:
                stages = task.stages_completed or []
                stages.append(stage_completed)
                task.stages_completed = stages
            
            # Update status based on progress
            if progress >= 100:
                task.status = TaskStatus.COMPLETED.value
                task.completed_at = datetime.utcnow()
            elif progress > 0 and task.status == TaskStatus.PENDING.value:
                task.status = TaskStatus.PROCESSING.value
                if not task.started_at:
                    task.started_at = datetime.utcnow()
            
            # Calculate actual duration if completed
            if task.completed_at and task.started_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                task.actual_duration = int(duration)
            
            self.session.commit()
            self.session.refresh(task)
            
            logger.debug(f"Updated task {task_id} progress to {progress}%")
            return task
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating task progress: {e}")
            raise
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          error_message: str = None) -> Optional[Task]:
        """Update task status"""
        try:
            task = self.get_by_id(task_id)
            if not task:
                return None
            
            old_status = task.status
            task.status = status.value
            
            if error_message:
                task.error_message = error_message
            
            # Update timestamps based on status
            if status == TaskStatus.PROCESSING and not task.started_at:
                task.started_at = datetime.utcnow()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if not task.completed_at:
                    task.completed_at = datetime.utcnow()
                
                # Calculate actual duration
                if task.started_at:
                    duration = (task.completed_at - task.started_at).total_seconds()
                    task.actual_duration = int(duration)
                
                # Set progress to 100 if completed
                if status == TaskStatus.COMPLETED:
                    task.progress = 100
            
            self.session.commit()
            self.session.refresh(task)
            
            logger.info(f"Updated task {task_id} status from {old_status} to {status.value}")
            return task
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating task status: {e}")
            raise
    
    def get_tasks_by_status(self, status: TaskStatus, limit: int = None) -> List[Task]:
        """Get tasks by status"""
        try:
            query = self.session.query(Task).filter(Task.status == status.value)
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error getting tasks by status {status}: {e}")
            return []
    
    def get_pending_tasks(self, limit: int = None) -> List[Task]:
        """Get pending tasks ordered by priority"""
        try:
            query = self.session.query(Task).filter(
                Task.status == TaskStatus.PENDING.value
            ).order_by(desc(Task.priority), Task.created_at)
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error getting pending tasks: {e}")
            return []
    
    def get_active_tasks(self, user_id: str = None) -> List[Task]:
        """Get currently active (processing) tasks"""
        try:
            query = self.session.query(Task).filter(
                Task.status == TaskStatus.PROCESSING.value
            )
            
            if user_id:
                query = query.filter(Task.user_id == user_id)
            
            return query.order_by(Task.started_at).all()
        except Exception as e:
            logger.error(f"Error getting active tasks: {e}")
            return []
    
    def get_tasks_by_project(self, project_id: str, 
                           status: TaskStatus = None) -> List[Task]:
        """Get tasks for a specific project"""
        try:
            query = self.session.query(Task).filter(Task.project_id == project_id)
            
            if status:
                query = query.filter(Task.status == status.value)
            
            return query.order_by(desc(Task.created_at)).all()
        except Exception as e:
            logger.error(f"Error getting project tasks: {e}")
            return []
    
    def get_tasks_by_user(self, user_id: str, status: TaskStatus = None, 
                         limit: int = None) -> List[Task]:
        """Get tasks for a specific user"""
        try:
            query = self.session.query(Task).filter(Task.user_id == user_id)
            
            if status:
                query = query.filter(Task.status == status.value)
            
            query = query.order_by(desc(Task.created_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error getting user tasks: {e}")
            return []
    
    def get_task_with_logs(self, task_id: str) -> Optional[Task]:
        """Get task with processing logs"""
        try:
            return self.session.query(Task).options(
                joinedload(Task.processing_logs)
            ).filter(Task.id == task_id).first()
        except Exception as e:
            logger.error(f"Error getting task with logs: {e}")
            return None
    
    def add_processing_log(self, task_id: str, stage: str, message: str, 
                          level: str = "INFO", details: Dict[str, Any] = None,
                          stage_duration: float = None) -> bool:
        """Add processing log entry for task"""
        try:
            log_entry = ProcessingLog(
                task_id=task_id,
                stage=stage,
                level=level.upper(),
                message=message,
                details=details or {},
                stage_duration=stage_duration
            )
            
            self.session.add(log_entry)
            self.session.commit()
            
            logger.debug(f"Added {level} log for task {task_id}: {message}")
            return True
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error adding processing log: {e}")
            return False
    
    def get_task_statistics(self, user_id: str = None, 
                          project_id: str = None) -> Dict[str, Any]:
        """Get task statistics"""
        try:
            base_query = self.session.query(Task)
            
            if user_id:
                base_query = base_query.filter(Task.user_id == user_id)
            if project_id:
                base_query = base_query.filter(Task.project_id == project_id)
            
            # Get status counts
            status_counts = {}
            for status in TaskStatus:
                count = base_query.filter(Task.status == status.value).count()
                status_counts[status.value] = count
            
            # Get type counts
            type_counts = {}
            for task_type in TaskType:
                count = base_query.filter(Task.task_type == task_type.value).count()
                type_counts[task_type.value] = count
            
            # Get performance metrics
            completed_tasks = base_query.filter(
                Task.status == TaskStatus.COMPLETED.value,
                Task.actual_duration.isnot(None)
            )
            
            avg_duration = completed_tasks.with_entities(
                func.avg(Task.actual_duration)
            ).scalar() or 0
            
            avg_memory = completed_tasks.with_entities(
                func.avg(Task.memory_peak)
            ).filter(Task.memory_peak.isnot(None)).scalar() or 0
            
            # Get recent activity (last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_tasks = base_query.filter(Task.created_at >= yesterday).count()
            
            return {
                'total_tasks': sum(status_counts.values()),
                'status_counts': status_counts,
                'type_counts': type_counts,
                'avg_duration_seconds': round(avg_duration, 2),
                'avg_memory_mb': round(avg_memory, 2),
                'recent_tasks_24h': recent_tasks,
                'success_rate': round(
                    (status_counts.get('completed', 0) / max(1, sum(status_counts.values()))) * 100, 2
                )
            }
        except Exception as e:
            logger.error(f"Error getting task statistics: {e}")
            return {}
    
    def cleanup_old_tasks(self, days_old: int = 30, 
                         keep_completed: bool = True) -> int:
        """Clean up old tasks"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            query = self.session.query(Task).filter(Task.created_at < cutoff_date)
            
            # Keep completed tasks if requested
            if keep_completed:
                query = query.filter(Task.status != TaskStatus.COMPLETED.value)
            
            # Only delete failed, cancelled, or old pending tasks
            query = query.filter(Task.status.in_([
                TaskStatus.FAILED.value,
                TaskStatus.CANCELLED.value,
                TaskStatus.PENDING.value if not keep_completed else None
            ]))
            
            deleted_count = query.delete(synchronize_session=False)
            self.session.commit()
            
            logger.info(f"Cleaned up {deleted_count} old tasks older than {days_old} days")
            return deleted_count
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error cleaning up old tasks: {e}")
            return 0
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current task queue status"""
        try:
            pending_count = self.session.query(Task).filter(
                Task.status == TaskStatus.PENDING.value
            ).count()
            
            processing_count = self.session.query(Task).filter(
                Task.status == TaskStatus.PROCESSING.value
            ).count()
            
            # Get highest priority pending task
            highest_priority = self.session.query(func.max(Task.priority)).filter(
                Task.status == TaskStatus.PENDING.value
            ).scalar() or 0
            
            # Get estimated queue time (rough calculation)
            avg_duration = self.session.query(func.avg(Task.actual_duration)).filter(
                Task.status == TaskStatus.COMPLETED.value,
                Task.actual_duration.isnot(None)
            ).scalar() or 300  # Default 5 minutes
            
            estimated_queue_time = pending_count * avg_duration
            
            return {
                'pending_tasks': pending_count,
                'processing_tasks': processing_count,
                'highest_priority': highest_priority,
                'estimated_queue_time_seconds': int(estimated_queue_time),
                'avg_task_duration_seconds': int(avg_duration)
            }
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {}
    
    def get_failed_tasks_summary(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get summary of recent failed tasks for debugging"""
        try:
            failed_tasks = self.session.query(Task).filter(
                Task.status == TaskStatus.FAILED.value
            ).order_by(desc(Task.completed_at)).limit(limit).all()
            
            summary = []
            for task in failed_tasks:
                summary.append({
                    'task_id': task.id,
                    'task_type': task.task_type,
                    'failed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'error_message': task.error_message,
                    'duration_seconds': task.actual_duration,
                    'current_stage': task.current_stage,
                    'stages_completed': task.stages_completed or []
                })
            
            return summary
        except Exception as e:
            logger.error(f"Error getting failed tasks summary: {e}")
            return []