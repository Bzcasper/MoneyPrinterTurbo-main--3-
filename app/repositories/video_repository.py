"""
Video repository for video metadata and file management
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc, asc
from datetime import datetime, timedelta
import logging
import os

from app.models.database import VideoMetadata, MaterialMetadata
from app.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class VideoRepository(BaseRepository[VideoMetadata]):
    """Repository for video metadata operations"""
    
    def __init__(self, session: Session):
        super().__init__(session, VideoMetadata)
    
    def create_video_metadata(self, file_path: str, task_id: str = None, 
                            project_id: str = None, **metadata) -> VideoMetadata:
        """Create video metadata entry"""
        try:
            # Get file stats
            file_stats = os.stat(file_path) if os.path.exists(file_path) else None
            file_size = file_stats.st_size if file_stats else 0
            
            # Extract filename
            filename = os.path.basename(file_path)
            
            video = VideoMetadata(
                filename=filename,
                file_path=file_path,
                file_size=file_size,
                task_id=task_id,
                project_id=project_id,
                **metadata
            )
            
            self.session.add(video)
            self.session.commit()
            self.session.refresh(video)
            
            logger.info(f"Created video metadata for {filename}")
            return video
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating video metadata: {e}")
            raise
    
    def get_videos_by_project(self, project_id: str, 
                            limit: int = None) -> List[VideoMetadata]:
        """Get videos for a specific project"""
        try:
            query = self.session.query(VideoMetadata).filter(
                VideoMetadata.project_id == project_id
            ).order_by(desc(VideoMetadata.created_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error getting project videos: {e}")
            return []
    
    def get_videos_by_task(self, task_id: str) -> List[VideoMetadata]:
        """Get videos for a specific task"""
        try:
            return self.session.query(VideoMetadata).filter(
                VideoMetadata.task_id == task_id
            ).order_by(VideoMetadata.created_at).all()
        except Exception as e:
            logger.error(f"Error getting task videos: {e}")
            return []
    
    def get_video_with_materials(self, video_id: str) -> Optional[VideoMetadata]:
        """Get video with its material metadata"""
        try:
            return self.session.query(VideoMetadata).options(
                joinedload(VideoMetadata.materials)
            ).filter(VideoMetadata.id == video_id).first()
        except Exception as e:
            logger.error(f"Error getting video with materials: {e}")
            return None
    
    def search_videos(self, search_term: str, language: str = None,
                     aspect_ratio: str = None, duration_range: Dict[str, float] = None,
                     limit: int = 50) -> List[VideoMetadata]:
        """Search videos by various criteria"""
        try:
            query = self.session.query(VideoMetadata)
            
            # Text search in subject and script
            if search_term:
                search_filter = or_(
                    VideoMetadata.subject.like(f"%{search_term}%"),
                    VideoMetadata.script_text.like(f"%{search_term}%"),
                    VideoMetadata.filename.like(f"%{search_term}%")
                )
                query = query.filter(search_filter)
            
            # Language filter
            if language:
                query = query.filter(VideoMetadata.language == language)
            
            # Aspect ratio filter
            if aspect_ratio:
                query = query.filter(VideoMetadata.aspect_ratio == aspect_ratio)
            
            # Duration range filter
            if duration_range:
                if 'min' in duration_range:
                    query = query.filter(VideoMetadata.duration >= duration_range['min'])
                if 'max' in duration_range:
                    query = query.filter(VideoMetadata.duration <= duration_range['max'])
            
            # Order by relevance (most recent first)
            query = query.order_by(desc(VideoMetadata.created_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error searching videos: {e}")
            return []
    
    def get_videos_by_tags(self, search_terms: List[str], 
                          limit: int = 50) -> List[VideoMetadata]:
        """Get videos that match search terms"""
        try:
            query = self.session.query(VideoMetadata)
            
            # Build OR conditions for each search term
            term_filters = []
            for term in search_terms:
                # Search in subject, script, and search_terms JSON
                term_filter = or_(
                    VideoMetadata.subject.like(f"%{term}%"),
                    VideoMetadata.script_text.like(f"%{term}%"),
                    func.json_extract(VideoMetadata.search_terms, '$').like(f"%{term}%")
                )
                term_filters.append(term_filter)
            
            if term_filters:
                query = query.filter(or_(*term_filters))
            
            query = query.order_by(desc(VideoMetadata.created_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except Exception as e:
            logger.error(f"Error getting videos by tags: {e}")
            return []
    
    def get_videos_by_duration_range(self, min_duration: float, 
                                   max_duration: float = None) -> List[VideoMetadata]:
        """Get videos within duration range"""
        try:
            query = self.session.query(VideoMetadata).filter(
                VideoMetadata.duration >= min_duration
            )
            
            if max_duration:
                query = query.filter(VideoMetadata.duration <= max_duration)
            
            return query.order_by(VideoMetadata.duration).all()
        except Exception as e:
            logger.error(f"Error getting videos by duration: {e}")
            return []
    
    def get_videos_by_quality_range(self, min_quality: float, 
                                   max_quality: float = 100.0) -> List[VideoMetadata]:
        """Get videos within quality score range"""
        try:
            return self.session.query(VideoMetadata).filter(
                and_(
                    VideoMetadata.quality_score >= min_quality,
                    VideoMetadata.quality_score <= max_quality,
                    VideoMetadata.quality_score.isnot(None)
                )
            ).order_by(desc(VideoMetadata.quality_score)).all()
        except Exception as e:
            logger.error(f"Error getting videos by quality: {e}")
            return []
    
    def add_material_metadata(self, video_id: str, materials: List[Dict[str, Any]]) -> bool:
        """Add material metadata for a video"""
        try:
            for material_data in materials:
                material = MaterialMetadata(
                    video_id=video_id,
                    **material_data
                )
                self.session.add(material)
            
            self.session.commit()
            logger.debug(f"Added {len(materials)} materials for video {video_id}")
            return True
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error adding material metadata: {e}")
            return False
    
    def get_material_usage_stats(self, source: str = None) -> Dict[str, Any]:
        """Get material usage statistics"""
        try:
            query = self.session.query(MaterialMetadata)
            
            if source:
                query = query.filter(MaterialMetadata.source == source)
            
            total_materials = query.count()
            
            # Get source distribution
            source_counts = dict(
                self.session.query(
                    MaterialMetadata.source,
                    func.count(MaterialMetadata.id)
                ).group_by(MaterialMetadata.source).all()
            )
            
            # Get average file sizes by source
            avg_sizes = dict(
                self.session.query(
                    MaterialMetadata.source,
                    func.avg(MaterialMetadata.file_size)
                ).group_by(MaterialMetadata.source).all()
            )
            
            # Get most used search terms
            popular_terms = self.session.query(
                MaterialMetadata.search_term,
                func.count(MaterialMetadata.id).label('usage_count')
            ).filter(
                MaterialMetadata.search_term.isnot(None)
            ).group_by(
                MaterialMetadata.search_term
            ).order_by(desc('usage_count')).limit(10).all()
            
            return {
                'total_materials': total_materials,
                'source_distribution': source_counts,
                'avg_file_sizes_mb': {k: round(v / 1024 / 1024, 2) for k, v in avg_sizes.items()},
                'popular_search_terms': [
                    {'term': term, 'usage_count': count} for term, count in popular_terms
                ]
            }
        except Exception as e:
            logger.error(f"Error getting material usage stats: {e}")
            return {}
    
    def get_video_production_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get video production statistics"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Total videos in period
            total_videos = self.session.query(VideoMetadata).filter(
                VideoMetadata.created_at >= start_date
            ).count()
            
            # Videos by day
            daily_stats = self.session.query(
                func.date(VideoMetadata.created_at).label('date'),
                func.count(VideoMetadata.id).label('count'),
                func.sum(VideoMetadata.duration).label('total_duration')
            ).filter(
                VideoMetadata.created_at >= start_date
            ).group_by(func.date(VideoMetadata.created_at)).all()
            
            # Average duration and quality
            avg_stats = self.session.query(
                func.avg(VideoMetadata.duration).label('avg_duration'),
                func.avg(VideoMetadata.quality_score).label('avg_quality'),
                func.avg(VideoMetadata.processing_time).label('avg_processing_time')
            ).filter(VideoMetadata.created_at >= start_date).first()
            
            # Aspect ratio distribution
            aspect_ratios = dict(
                self.session.query(
                    VideoMetadata.aspect_ratio,
                    func.count(VideoMetadata.id)
                ).filter(
                    VideoMetadata.created_at >= start_date
                ).group_by(VideoMetadata.aspect_ratio).all()
            )
            
            # Language distribution
            languages = dict(
                self.session.query(
                    VideoMetadata.language,
                    func.count(VideoMetadata.id)
                ).filter(
                    VideoMetadata.created_at >= start_date,
                    VideoMetadata.language.isnot(None)
                ).group_by(VideoMetadata.language).all()
            )
            
            return {
                'period_days': days,
                'total_videos': total_videos,
                'daily_production': [
                    {
                        'date': str(stat.date),
                        'video_count': stat.count,
                        'total_duration': round(stat.total_duration or 0, 2)
                    } for stat in daily_stats
                ],
                'averages': {
                    'duration_seconds': round(avg_stats.avg_duration or 0, 2),
                    'quality_score': round(avg_stats.avg_quality or 0, 2),
                    'processing_time_seconds': round(avg_stats.avg_processing_time or 0, 2)
                },
                'aspect_ratio_distribution': aspect_ratios,
                'language_distribution': languages
            }
        except Exception as e:
            logger.error(f"Error getting video production stats: {e}")
            return {}
    
    def cleanup_orphaned_videos(self) -> int:
        """Clean up video metadata for files that no longer exist"""
        try:
            orphaned_count = 0
            videos = self.session.query(VideoMetadata).all()
            
            for video in videos:
                if not os.path.exists(video.file_path):
                    logger.info(f"Removing metadata for missing file: {video.file_path}")
                    self.session.delete(video)
                    orphaned_count += 1
            
            if orphaned_count > 0:
                self.session.commit()
                logger.info(f"Cleaned up {orphaned_count} orphaned video metadata entries")
            
            return orphaned_count
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error cleaning up orphaned videos: {e}")
            return 0
    
    def update_video_quality_score(self, video_id: str, quality_score: float) -> bool:
        """Update video quality score"""
        try:
            video = self.get_by_id(video_id)
            if video:
                video.quality_score = max(0, min(100, quality_score))
                self.session.commit()
                logger.debug(f"Updated quality score for video {video_id}: {quality_score}")
                return True
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating video quality score: {e}")
            return False
    
    def get_popular_subjects(self, limit: int = 10, days: int = 30) -> List[Dict[str, Any]]:
        """Get most popular video subjects"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            popular_subjects = self.session.query(
                VideoMetadata.subject,
                func.count(VideoMetadata.id).label('video_count'),
                func.avg(VideoMetadata.quality_score).label('avg_quality')
            ).filter(
                VideoMetadata.created_at >= start_date,
                VideoMetadata.subject.isnot(None)
            ).group_by(
                VideoMetadata.subject
            ).order_by(desc('video_count')).limit(limit).all()
            
            return [
                {
                    'subject': subject,
                    'video_count': count,
                    'avg_quality': round(avg_quality or 0, 2)
                } for subject, count, avg_quality in popular_subjects
            ]
        except Exception as e:
            logger.error(f"Error getting popular subjects: {e}")
            return []