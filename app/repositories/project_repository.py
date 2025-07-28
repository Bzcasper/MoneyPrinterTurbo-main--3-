"""
Project repository for video project management
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc
from datetime import datetime, timedelta
import logging

from app.models.database import VideoProject, Task, VideoMetadata, TaskStatus
from app.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ProjectRepository(BaseRepository[VideoProject]):
    """Repository for video project operations"""
    
    def __init__(self, session: Session):
        super().__init__(session, VideoProject)
    
    def create_project(self, name: str, subject: str, user_id: str,
                      description: str = None, default_params: Dict[str, Any] = None,
                      tags: List[str] = None, language: str = None) -> VideoProject:
        """Create a new video project"""
        try:
            project = VideoProject(
                name=name,
                subject=subject,
                user_id=user_id,
                description=description,
                default_params=default_params or {},
                tags=tags or [],
                language=language
            )
            
            self.session.add(project)
            self.session.commit()
            self.session.refresh(project)
            
            logger.info(f"Created project '{name}' for user {user_id}")
            return project
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating project: {e}")
            raise
    
    def get_projects_by_user(self, user_id: str, status: str = None,
                           limit: int = None) -> List[VideoProject]:
        """Get projects for a specific user"""
        try:
            query = self.session.query(VideoProject).filter(
                VideoProject.user_id == user_id
            )
            
            if status:
                query = query.filter(VideoProject.status == status)
            
            query = query.order_by(desc(VideoProject.updated_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error getting user projects: {e}")
            return []
    
    def get_project_with_tasks(self, project_id: str) -> Optional[VideoProject]:
        """Get project with its tasks"""
        try:
            return self.session.query(VideoProject).options(
                joinedload(VideoProject.tasks),
                joinedload(VideoProject.videos)
            ).filter(VideoProject.id == project_id).first()
        except Exception as e:
            logger.error(f"Error getting project with tasks: {e}")
            return None
    
    def update_project_metrics(self, project_id: str) -> bool:
        """Update project metrics (video counts, duration, etc.)"""
        try:
            project = self.get_by_id(project_id)
            if not project:
                return False
            
            # Get video statistics
            video_stats = self.session.query(
                func.count(VideoMetadata.id).label('total_videos'),
                func.sum(VideoMetadata.duration).label('total_duration')
            ).filter(VideoMetadata.project_id == project_id).first()
            
            # Get completed task count
            completed_tasks = self.session.query(func.count(Task.id)).filter(
                and_(
                    Task.project_id == project_id,
                    Task.status == TaskStatus.COMPLETED.value
                )
            ).scalar()
            
            # Update project metrics
            project.total_videos = video_stats.total_videos or 0
            project.total_duration = video_stats.total_duration or 0.0
            project.completed_videos = completed_tasks or 0
            project.updated_at = datetime.utcnow()
            
            self.session.commit()
            logger.debug(f"Updated metrics for project {project_id}")
            return True
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating project metrics: {e}")
            return False
    
    def add_project_tag(self, project_id: str, tag: str) -> bool:
        """Add a tag to project"""
        try:
            project = self.get_by_id(project_id)
            if not project:
                return False
            
            tags = project.tags or []
            if tag not in tags:
                tags.append(tag)
                project.tags = tags
                project.updated_at = datetime.utcnow()
                self.session.commit()
                logger.debug(f"Added tag '{tag}' to project {project_id}")
            
            return True
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error adding project tag: {e}")
            return False
    
    def remove_project_tag(self, project_id: str, tag: str) -> bool:
        """Remove a tag from project"""
        try:
            project = self.get_by_id(project_id)
            if not project:
                return False
            
            tags = project.tags or []
            if tag in tags:
                tags.remove(tag)
                project.tags = tags
                project.updated_at = datetime.utcnow()
                self.session.commit()
                logger.debug(f"Removed tag '{tag}' from project {project_id}")
            
            return True
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error removing project tag: {e}")
            return False
    
    def search_projects(self, search_term: str, user_id: str = None,
                       tags: List[str] = None, status: str = None,
                       limit: int = 50) -> List[VideoProject]:
        """Search projects by various criteria"""
        try:
            query = self.session.query(VideoProject)
            
            # Text search in name, subject, and description
            if search_term:
                search_filter = or_(
                    VideoProject.name.like(f"%{search_term}%"),
                    VideoProject.subject.like(f"%{search_term}%"),
                    VideoProject.description.like(f"%{search_term}%")
                )
                query = query.filter(search_filter)
            
            # User filter
            if user_id:
                query = query.filter(VideoProject.user_id == user_id)
            
            # Status filter
            if status:
                query = query.filter(VideoProject.status == status)
            
            # Tag filters (projects that contain any of the specified tags)
            if tags:
                tag_filters = []
                for tag in tags:
                    tag_filters.append(
                        func.json_extract(VideoProject.tags, '$').like(f"%{tag}%")
                    )
                if tag_filters:
                    query = query.filter(or_(*tag_filters))
            
            query = query.order_by(desc(VideoProject.updated_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error searching projects: {e}")
            return []
    
    def get_projects_by_tags(self, tags: List[str], user_id: str = None) -> List[VideoProject]:
        """Get projects that contain any of the specified tags"""
        try:
            query = self.session.query(VideoProject)
            
            if user_id:
                query = query.filter(VideoProject.user_id == user_id)
            
            # Build tag filters
            tag_filters = []
            for tag in tags:
                tag_filters.append(
                    func.json_extract(VideoProject.tags, '$').like(f"%{tag}%")
                )
            
            if tag_filters:
                query = query.filter(or_(*tag_filters))
            
            return query.order_by(desc(VideoProject.updated_at)).all()
        except Exception as e:
            logger.error(f"Error getting projects by tags: {e}")
            return []
    
    def get_project_activity_summary(self, project_id: str, days: int = 30) -> Dict[str, Any]:
        """Get project activity summary"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Task activity
            task_stats = self.session.query(
                Task.status,
                func.count(Task.id).label('count')
            ).filter(
                and_(
                    Task.project_id == project_id,
                    Task.created_at >= start_date
                )
            ).group_by(Task.status).all()
            
            task_counts = {status: count for status, count in task_stats}
            
            # Video production
            video_stats = self.session.query(
                func.count(VideoMetadata.id).label('video_count'),
                func.sum(VideoMetadata.duration).label('total_duration'),
                func.avg(VideoMetadata.quality_score).label('avg_quality')
            ).filter(
                and_(
                    VideoMetadata.project_id == project_id,
                    VideoMetadata.created_at >= start_date
                )
            ).first()
            
            # Recent activity (tasks and videos by day)
            daily_activity = self.session.query(
                func.date(Task.created_at).label('date'),
                func.count(Task.id).label('task_count')
            ).filter(
                and_(
                    Task.project_id == project_id,
                    Task.created_at >= start_date
                )
            ).group_by(func.date(Task.created_at)).all()
            
            return {
                'period_days': days,
                'task_summary': task_counts,
                'video_production': {
                    'count': video_stats.video_count or 0,
                    'total_duration': video_stats.total_duration or 0.0,
                    'avg_quality': round(video_stats.avg_quality or 0, 2)
                },
                'daily_activity': [
                    {
                        'date': str(activity.date),
                        'task_count': activity.task_count
                    } for activity in daily_activity
                ]
            }
        except Exception as e:
            logger.error(f"Error getting project activity summary: {e}")
            return {}
    
    def get_project_performance_metrics(self, project_id: str) -> Dict[str, Any]:
        """Get detailed project performance metrics"""
        try:
            # Task performance
            task_metrics = self.session.query(
                func.count(Task.id).label('total_tasks'),
                func.avg(Task.actual_duration).label('avg_duration'),
                func.avg(Task.memory_peak).label('avg_memory'),
                func.sum(func.case([(Task.status == TaskStatus.COMPLETED.value, 1)], else_=0)).label('completed_tasks'),
                func.sum(func.case([(Task.status == TaskStatus.FAILED.value, 1)], else_=0)).label('failed_tasks')
            ).filter(Task.project_id == project_id).first()
            
            # Video metrics
            video_metrics = self.session.query(
                func.count(VideoMetadata.id).label('total_videos'),
                func.avg(VideoMetadata.duration).label('avg_video_duration'),
                func.sum(VideoMetadata.duration).label('total_video_duration'),
                func.avg(VideoMetadata.quality_score).label('avg_quality'),
                func.avg(VideoMetadata.processing_time).label('avg_processing_time')
            ).filter(VideoMetadata.project_id == project_id).first()
            
            # Success rate calculation
            total_tasks = task_metrics.total_tasks or 0
            completed_tasks = task_metrics.completed_tasks or 0
            success_rate = (completed_tasks / max(1, total_tasks)) * 100
            
            return {
                'task_metrics': {
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'failed_tasks': task_metrics.failed_tasks or 0,
                    'success_rate': round(success_rate, 2),
                    'avg_duration_seconds': round(task_metrics.avg_duration or 0, 2),
                    'avg_memory_mb': round(task_metrics.avg_memory or 0, 2)
                },
                'video_metrics': {
                    'total_videos': video_metrics.total_videos or 0,
                    'total_duration_seconds': round(video_metrics.total_video_duration or 0, 2),
                    'avg_video_duration_seconds': round(video_metrics.avg_video_duration or 0, 2),
                    'avg_quality_score': round(video_metrics.avg_quality or 0, 2),
                    'avg_processing_time_seconds': round(video_metrics.avg_processing_time or 0, 2)
                }
            }
        except Exception as e:
            logger.error(f"Error getting project performance metrics: {e}")
            return {}
    
    def archive_project(self, project_id: str) -> bool:
        """Archive a project (set status to archived)"""
        try:
            project = self.update(project_id, status='archived')
            if project:
                logger.info(f"Archived project {project_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error archiving project: {e}")
            return False
    
    def restore_project(self, project_id: str) -> bool:
        """Restore an archived project"""
        try:
            project = self.update(project_id, status='active')
            if project:
                logger.info(f"Restored project {project_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error restoring project: {e}")
            return False
    
    def get_popular_tags(self, user_id: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most popular project tags"""
        try:
            query = self.session.query(VideoProject)
            
            if user_id:
                query = query.filter(VideoProject.user_id == user_id)
            
            projects = query.all()
            
            # Count tag usage
            tag_counts = {}
            for project in projects:
                if project.tags:
                    for tag in project.tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Sort by usage count
            popular_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
            
            return [
                {'tag': tag, 'usage_count': count}
                for tag, count in popular_tags
            ]
        except Exception as e:
            logger.error(f"Error getting popular tags: {e}")
            return []
    
    def get_user_project_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive project statistics for a user"""
        try:
            # Basic project counts
            project_counts = self.session.query(
                VideoProject.status,
                func.count(VideoProject.id).label('count')
            ).filter(VideoProject.user_id == user_id).group_by(VideoProject.status).all()
            
            status_counts = {status: count for status, count in project_counts}
            
            # Total metrics across all projects
            total_metrics = self.session.query(
                func.sum(VideoProject.total_videos).label('total_videos'),
                func.sum(VideoProject.total_duration).label('total_duration'),
                func.count(VideoProject.id).label('total_projects')
            ).filter(VideoProject.user_id == user_id).first()
            
            # Recent activity (projects created in last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_projects = self.session.query(func.count(VideoProject.id)).filter(
                and_(
                    VideoProject.user_id == user_id,
                    VideoProject.created_at >= thirty_days_ago
                )
            ).scalar()
            
            return {
                'total_projects': total_metrics.total_projects or 0,
                'status_distribution': status_counts,
                'total_videos_produced': total_metrics.total_videos or 0,
                'total_video_duration_seconds': total_metrics.total_duration or 0.0,
                'projects_created_last_30_days': recent_projects or 0,
                'avg_videos_per_project': round(
                    (total_metrics.total_videos or 0) / max(1, total_metrics.total_projects or 1), 2
                )
            }
        except Exception as e:
            logger.error(f"Error getting user project stats: {e}")
            return {}