"""
Analytics repository for system metrics and performance tracking
"""

from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, text
from datetime import datetime, timedelta
import logging
import json

from app.models.database import (
    SystemMetrics, CacheEntry, Task, VideoMetadata, 
    User, VideoProject, TaskStatus
)
from app.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class AnalyticsRepository(BaseRepository[SystemMetrics]):
    """Repository for analytics and system metrics"""
    
    def __init__(self, session: Session):
        super().__init__(session, SystemMetrics)
    
    def record_metric(self, metric_type: str, metric_name: str, value: float,
                     unit: str = None, hostname: str = None, 
                     process_id: int = None, context_data: Dict[str, Any] = None) -> SystemMetrics:
        """Record a system metric"""
        try:
            metric = SystemMetrics(
                metric_type=metric_type,
                metric_name=metric_name,
                value=value,
                unit=unit,
                hostname=hostname,
                process_id=process_id,
                context_data=context_data or {}
            )
            
            self.session.add(metric)
            self.session.commit()
            self.session.refresh(metric)
            
            return metric
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error recording metric: {e}")
            raise
    
    def get_metrics_by_type(self, metric_type: str, hours: int = 24,
                           limit: int = None) -> List[SystemMetrics]:
        """Get metrics by type within time period"""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            query = self.session.query(SystemMetrics).filter(
                and_(
                    SystemMetrics.metric_type == metric_type,
                    SystemMetrics.timestamp >= start_time
                )
            ).order_by(desc(SystemMetrics.timestamp))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error getting metrics by type: {e}")
            return []
    
    def get_system_health_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get system health summary"""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Get latest metrics for each type
            health_metrics = {}
            
            # CPU metrics
            cpu_metrics = self.session.query(SystemMetrics).filter(
                and_(
                    SystemMetrics.metric_type == 'cpu',
                    SystemMetrics.timestamp >= start_time
                )
            ).order_by(desc(SystemMetrics.timestamp)).limit(10).all()
            
            if cpu_metrics:
                cpu_values = [m.value for m in cpu_metrics]
                health_metrics['cpu'] = {
                    'current': cpu_values[0],
                    'average': sum(cpu_values) / len(cpu_values),
                    'max': max(cpu_values),
                    'unit': cpu_metrics[0].unit
                }
            
            # Memory metrics
            memory_metrics = self.session.query(SystemMetrics).filter(
                and_(
                    SystemMetrics.metric_type == 'memory',
                    SystemMetrics.timestamp >= start_time
                )
            ).order_by(desc(SystemMetrics.timestamp)).limit(10).all()
            
            if memory_metrics:
                memory_values = [m.value for m in memory_metrics]
                health_metrics['memory'] = {
                    'current': memory_values[0],
                    'average': sum(memory_values) / len(memory_values),
                    'max': max(memory_values),
                    'unit': memory_metrics[0].unit
                }
            
            # Task throughput
            task_throughput = self.session.query(
                func.count(Task.id).label('completed_tasks')
            ).filter(
                and_(
                    Task.status == TaskStatus.COMPLETED.value,
                    Task.completed_at >= start_time
                )
            ).scalar()
            
            health_metrics['task_throughput'] = {
                'completed_last_hour': task_throughput or 0,
                'rate_per_minute': round((task_throughput or 0) / (hours * 60), 2)
            }
            
            return health_metrics
        except Exception as e:
            logger.error(f"Error getting system health summary: {e}")
            return {}
    
    def get_performance_trends(self, metric_type: str, days: int = 7) -> Dict[str, Any]:
        """Get performance trends over time"""
        try:
            start_time = datetime.utcnow() - timedelta(days=days)
            
            # Daily aggregates
            daily_metrics = self.session.query(
                func.date(SystemMetrics.timestamp).label('date'),
                func.avg(SystemMetrics.value).label('avg_value'),
                func.min(SystemMetrics.value).label('min_value'),
                func.max(SystemMetrics.value).label('max_value'),
                func.count(SystemMetrics.id).label('sample_count')
            ).filter(
                and_(
                    SystemMetrics.metric_type == metric_type,
                    SystemMetrics.timestamp >= start_time
                )
            ).group_by(func.date(SystemMetrics.timestamp)).all()
            
            # Calculate trend (simple linear regression slope)
            if len(daily_metrics) >= 2:
                x_values = list(range(len(daily_metrics)))
                y_values = [float(m.avg_value) for m in daily_metrics]
                
                n = len(x_values)
                sum_x = sum(x_values)
                sum_y = sum(y_values)
                sum_xy = sum(x * y for x, y in zip(x_values, y_values))
                sum_x2 = sum(x * x for x in x_values)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if n * sum_x2 - sum_x * sum_x != 0 else 0
                trend = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
            else:
                slope = 0
                trend = "insufficient_data"
            
            return {
                'metric_type': metric_type,
                'period_days': days,
                'trend': trend,
                'slope': round(slope, 4),
                'daily_data': [
                    {
                        'date': str(m.date),
                        'avg_value': round(float(m.avg_value), 2),
                        'min_value': round(float(m.min_value), 2),
                        'max_value': round(float(m.max_value), 2),
                        'sample_count': m.sample_count
                    } for m in daily_metrics
                ]
            }
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return {}
    
    def get_usage_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive usage analytics"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # User activity
            active_users = self.session.query(
                func.count(func.distinct(Task.user_id))
            ).filter(
                and_(
                    Task.created_at >= start_date,
                    Task.user_id.isnot(None)
                )
            ).scalar()
            
            # Task statistics
            task_stats = self.session.query(
                Task.status,
                func.count(Task.id).label('count')
            ).filter(Task.created_at >= start_date).group_by(Task.status).all()
            
            task_counts = {status: count for status, count in task_stats}
            
            # Video production
            video_production = self.session.query(
                func.count(VideoMetadata.id).label('total_videos'),
                func.sum(VideoMetadata.duration).label('total_duration'),
                func.avg(VideoMetadata.quality_score).label('avg_quality')
            ).filter(VideoMetadata.created_at >= start_date).first()
            
            # Daily activity
            daily_activity = self.session.query(
                func.date(Task.created_at).label('date'),
                func.count(Task.id).label('task_count'),
                func.count(func.distinct(Task.user_id)).label('active_users')
            ).filter(Task.created_at >= start_date).group_by(
                func.date(Task.created_at)
            ).all()
            
            # Popular video aspects and languages
            aspect_stats = dict(
                self.session.query(
                    VideoMetadata.aspect_ratio,
                    func.count(VideoMetadata.id)
                ).filter(
                    VideoMetadata.created_at >= start_date
                ).group_by(VideoMetadata.aspect_ratio).all()
            )
            
            language_stats = dict(
                self.session.query(
                    VideoMetadata.language,
                    func.count(VideoMetadata.id)
                ).filter(
                    and_(
                        VideoMetadata.created_at >= start_date,
                        VideoMetadata.language.isnot(None)
                    )
                ).group_by(VideoMetadata.language).all()
            )
            
            return {
                'period_days': days,
                'user_activity': {
                    'active_users': active_users or 0,
                    'total_registered': self.session.query(User).filter(User.is_active == True).count()
                },
                'task_statistics': task_counts,
                'video_production': {
                    'total_videos': video_production.total_videos or 0,
                    'total_duration_hours': round((video_production.total_duration or 0) / 3600, 2),
                    'avg_quality_score': round(video_production.avg_quality or 0, 2)
                },
                'daily_activity': [
                    {
                        'date': str(day.date),
                        'task_count': day.task_count,
                        'active_users': day.active_users
                    } for day in daily_activity
                ],
                'content_preferences': {
                    'aspect_ratios': aspect_stats,
                    'languages': language_stats
                }
            }
        except Exception as e:
            logger.error(f"Error getting usage analytics: {e}")
            return {}
    
    def get_error_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get error and failure analytics"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Failed tasks
            failed_tasks = self.session.query(Task).filter(
                and_(
                    Task.status == TaskStatus.FAILED.value,
                    Task.completed_at >= start_date
                )
            ).all()
            
            # Error patterns
            error_patterns = {}
            stage_failures = {}
            
            for task in failed_tasks:
                # Count error types
                if task.error_message:
                    # Simple error categorization
                    error_key = task.error_message[:50] + "..." if len(task.error_message) > 50 else task.error_message
                    error_patterns[error_key] = error_patterns.get(error_key, 0) + 1
                
                # Count stage failures
                if task.current_stage:
                    stage_failures[task.current_stage] = stage_failures.get(task.current_stage, 0) + 1
            
            # Failure rate by task type
            task_type_failures = self.session.query(
                Task.task_type,
                func.count(func.case([(Task.status == TaskStatus.FAILED.value, 1)])).label('failed'),
                func.count(Task.id).label('total')
            ).filter(Task.created_at >= start_date).group_by(Task.task_type).all()
            
            failure_rates = []
            for task_type, failed, total in task_type_failures:
                if total > 0:
                    failure_rates.append({
                        'task_type': task_type,
                        'failure_rate': round((failed / total) * 100, 2),
                        'failed_count': failed,
                        'total_count': total
                    })
            
            return {
                'period_days': days,
                'total_failures': len(failed_tasks),
                'common_errors': sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10],
                'stage_failures': sorted(stage_failures.items(), key=lambda x: x[1], reverse=True),
                'failure_rates_by_type': sorted(failure_rates, key=lambda x: x['failure_rate'], reverse=True)
            }
        except Exception as e:
            logger.error(f"Error getting error analytics: {e}")
            return {}
    
    def cleanup_old_metrics(self, days_old: int = 30) -> int:
        """Clean up old metrics data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            deleted_count = self.session.query(SystemMetrics).filter(
                SystemMetrics.timestamp < cutoff_date
            ).delete(synchronize_session=False)
            
            self.session.commit()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old metrics older than {days_old} days")
            
            return deleted_count
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error cleaning up old metrics: {e}")
            return 0


class CacheRepository(BaseRepository[CacheEntry]):
    """Repository for cache management"""
    
    def __init__(self, session: Session):
        super().__init__(session, CacheEntry)
    
    def set_cache(self, cache_key: str, data: Any, cache_type: str = "general",
                 expires_in_seconds: int = 3600) -> CacheEntry:
        """Set cache entry"""
        try:
            # Serialize data to JSON
            serialized_data = json.dumps(data) if not isinstance(data, str) else data
            size_bytes = len(serialized_data.encode('utf-8'))
            
            # Check if cache entry exists
            existing_entry = self.session.query(CacheEntry).filter(
                CacheEntry.cache_key == cache_key
            ).first()
            
            if existing_entry:
                # Update existing entry
                existing_entry.cache_data = serialized_data
                existing_entry.cache_type = cache_type
                existing_entry.size_bytes = size_bytes
                existing_entry.expires_at = datetime.utcnow() + timedelta(seconds=expires_in_seconds)
                existing_entry.last_accessed = datetime.utcnow()
                cache_entry = existing_entry
            else:
                # Create new entry
                expires_at = datetime.utcnow() + timedelta(seconds=expires_in_seconds)
                cache_entry = CacheEntry(
                    cache_key=cache_key,
                    cache_data=serialized_data,
                    cache_type=cache_type,
                    size_bytes=size_bytes,
                    expires_at=expires_at
                )
                self.session.add(cache_entry)
            
            self.session.commit()
            self.session.refresh(cache_entry)
            
            return cache_entry
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error setting cache: {e}")
            raise
    
    def get_cache(self, cache_key: str) -> Optional[Any]:
        """Get cache entry"""
        try:
            cache_entry = self.session.query(CacheEntry).filter(
                and_(
                    CacheEntry.cache_key == cache_key,
                    or_(
                        CacheEntry.expires_at.is_(None),
                        CacheEntry.expires_at > datetime.utcnow()
                    )
                )
            ).first()
            
            if cache_entry:
                # Update access statistics
                cache_entry.hit_count += 1
                cache_entry.last_accessed = datetime.utcnow()
                self.session.commit()
                
                # Deserialize data
                try:
                    return json.loads(cache_entry.cache_data)
                except json.JSONDecodeError:
                    return cache_entry.cache_data
                
            return None
        except Exception as e:
            logger.error(f"Error getting cache: {e}")
            return None
    
    def delete_cache(self, cache_key: str) -> bool:
        """Delete cache entry"""
        try:
            deleted_count = self.session.query(CacheEntry).filter(
                CacheEntry.cache_key == cache_key
            ).delete()
            
            self.session.commit()
            return deleted_count > 0
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting cache: {e}")
            return False
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries"""
        try:
            deleted_count = self.session.query(CacheEntry).filter(
                and_(
                    CacheEntry.expires_at.isnot(None),
                    CacheEntry.expires_at <= datetime.utcnow()
                )
            ).delete(synchronize_session=False)
            
            self.session.commit()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired cache entries")
            
            return deleted_count
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error cleaning up expired cache: {e}")
            return 0
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        try:
            # Total cache entries and size
            total_stats = self.session.query(
                func.count(CacheEntry.id).label('total_entries'),
                func.sum(CacheEntry.size_bytes).label('total_size'),
                func.sum(CacheEntry.hit_count).label('total_hits')
            ).first()
            
            # Cache by type
            type_stats = dict(
                self.session.query(
                    CacheEntry.cache_type,
                    func.count(CacheEntry.id)
                ).group_by(CacheEntry.cache_type).all()
            )
            
            # Most accessed entries
            popular_entries = self.session.query(
                CacheEntry.cache_key,
                CacheEntry.cache_type,
                CacheEntry.hit_count
            ).order_by(desc(CacheEntry.hit_count)).limit(10).all()
            
            return {
                'total_entries': total_stats.total_entries or 0,
                'total_size_mb': round((total_stats.total_size or 0) / 1024 / 1024, 2),
                'total_hits': total_stats.total_hits or 0,
                'entries_by_type': type_stats,
                'most_accessed': [
                    {
                        'key': entry.cache_key,
                        'type': entry.cache_type,
                        'hit_count': entry.hit_count
                    } for entry in popular_entries
                ]
            }
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {}