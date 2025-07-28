"""
Database migrations for MoneyPrinterTurbo
Contains all database schema migration definitions
"""

from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

from .migration_manager import Migration

logger = logging.getLogger(__name__)


class Migration001_InitialSchema(Migration):
    """Initial database schema creation"""
    
    def __init__(self):
        super().__init__(
            version="001_initial_schema",
            description="Create initial database schema with all core tables"
        )
    
    def up(self, session: Session) -> bool:
        """Create initial schema"""
        try:
            # This migration is handled by SQLAlchemy's create_all()
            # since we're starting with a complete schema
            logger.info("Initial schema created by SQLAlchemy create_all()")
            return True
        except Exception as e:
            logger.error(f"Error in initial schema migration: {e}")
            return False
    
    def down(self, session: Session) -> bool:
        """Drop all tables"""
        try:
            # Drop all tables in reverse dependency order
            tables_to_drop = [
                'cache_entries',
                'system_metrics', 
                'material_metadata',
                'video_metadata',
                'processing_logs',
                'tasks',
                'video_projects',
                'user_sessions',
                'users',
                'database_migrations'
            ]
            
            for table in tables_to_drop:
                try:
                    session.execute(text(f"DROP TABLE IF EXISTS {table}"))
                except Exception as e:
                    logger.warning(f"Could not drop table {table}: {e}")
            
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Error dropping tables: {e}")
            return False


class Migration002_AddIndexes(Migration):
    """Add performance indexes"""
    
    def __init__(self):
        super().__init__(
            version="002_add_indexes",
            description="Add performance indexes for better query optimization"
        )
    
    def up(self, session: Session) -> bool:
        """Add indexes"""
        try:
            # Additional indexes for better performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_tasks_user_status ON tasks (user_id, status)",
                "CREATE INDEX IF NOT EXISTS idx_tasks_project_status ON tasks (project_id, status)",
                "CREATE INDEX IF NOT EXISTS idx_videos_project_created ON video_metadata (project_id, created_at)",
                "CREATE INDEX IF NOT EXISTS idx_videos_subject_language ON video_metadata (subject, language)",
                "CREATE INDEX IF NOT EXISTS idx_logs_task_timestamp ON processing_logs (task_id, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_sessions_user_active ON user_sessions (user_id, is_active)",
                "CREATE INDEX IF NOT EXISTS idx_materials_video_source ON material_metadata (video_id, source)",
                "CREATE INDEX IF NOT EXISTS idx_metrics_type_timestamp ON system_metrics (metric_type, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_cache_type_accessed ON cache_entries (cache_type, last_accessed)"
            ]
            
            for index_sql in indexes:
                session.execute(text(index_sql))
                logger.debug(f"Created index: {index_sql}")
            
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding indexes: {e}")
            return False
    
    def down(self, session: Session) -> bool:
        """Remove indexes"""
        try:
            indexes_to_drop = [
                "idx_tasks_user_status",
                "idx_tasks_project_status", 
                "idx_videos_project_created",
                "idx_videos_subject_language",
                "idx_logs_task_timestamp",
                "idx_sessions_user_active",
                "idx_materials_video_source",
                "idx_metrics_type_timestamp",
                "idx_cache_type_accessed"
            ]
            
            for index_name in indexes_to_drop:
                try:
                    session.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
                except Exception as e:
                    logger.warning(f"Could not drop index {index_name}: {e}")
            
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Error removing indexes: {e}")
            return False


class Migration003_AddViews(Migration):
    """Add performance views"""
    
    def __init__(self):
        super().__init__(
            version="003_add_views",
            description="Add database views for analytics and reporting"
        )
    
    def up(self, session: Session) -> bool:
        """Create views"""
        try:
            views = [
                """
                CREATE VIEW IF NOT EXISTS task_performance_summary AS
                SELECT 
                    t.task_type,
                    COUNT(*) as total_tasks,
                    AVG(t.actual_duration) as avg_duration,
                    AVG(t.memory_peak) as avg_memory,
                    SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
                    SUM(CASE WHEN t.status = 'failed' THEN 1 ELSE 0 END) as failed_tasks,
                    (SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as success_rate
                FROM tasks t
                WHERE t.created_at >= datetime('now', '-30 days')
                GROUP BY t.task_type
                """,
                
                """
                CREATE VIEW IF NOT EXISTS video_production_stats AS
                SELECT 
                    DATE(v.created_at) as production_date,
                    COUNT(*) as videos_created,
                    SUM(v.duration) as total_duration,
                    AVG(v.duration) as avg_duration,
                    AVG(v.quality_score) as avg_quality,
                    COUNT(DISTINCT v.project_id) as projects_active
                FROM video_metadata v
                WHERE v.created_at >= datetime('now', '-90 days')
                GROUP BY DATE(v.created_at)
                ORDER BY production_date DESC
                """,
                
                """
                CREATE VIEW IF NOT EXISTS user_activity_summary AS
                SELECT 
                    u.id as user_id,
                    u.username,
                    COUNT(DISTINCT p.id) as project_count,
                    COUNT(DISTINCT t.id) as task_count,
                    COUNT(DISTINCT v.id) as video_count,
                    SUM(v.duration) as total_video_duration,
                    MAX(t.created_at) as last_activity
                FROM users u
                LEFT JOIN video_projects p ON u.id = p.user_id
                LEFT JOIN tasks t ON u.id = t.user_id
                LEFT JOIN video_metadata v ON t.id = v.task_id
                WHERE u.is_active = 1
                GROUP BY u.id, u.username
                """,
                
                """
                CREATE VIEW IF NOT EXISTS system_health_overview AS
                SELECT 
                    sm.metric_type,
                    AVG(sm.value) as avg_value,
                    MIN(sm.value) as min_value,
                    MAX(sm.value) as max_value,
                    COUNT(*) as sample_count,
                    sm.unit
                FROM system_metrics sm
                WHERE sm.timestamp >= datetime('now', '-1 hour')
                GROUP BY sm.metric_type, sm.unit
                """
            ]
            
            for view_sql in views:
                session.execute(text(view_sql))
                logger.debug("Created database view")
            
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating views: {e}")
            return False
    
    def down(self, session: Session) -> bool:
        """Drop views"""
        try:
            views_to_drop = [
                "task_performance_summary",
                "video_production_stats",
                "user_activity_summary", 
                "system_health_overview"
            ]
            
            for view_name in views_to_drop:
                try:
                    session.execute(text(f"DROP VIEW IF EXISTS {view_name}"))
                except Exception as e:
                    logger.warning(f"Could not drop view {view_name}: {e}")
            
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Error dropping views: {e}")
            return False


class Migration004_AddTriggers(Migration):
    """Add database triggers for data consistency"""
    
    def __init__(self):
        super().__init__(
            version="004_add_triggers", 
            description="Add database triggers for automatic data management"
        )
    
    def up(self, session: Session) -> bool:
        """Create triggers"""
        try:
            triggers = [
                # Update project metrics when tasks are completed
                """
                CREATE TRIGGER IF NOT EXISTS update_project_metrics_on_task_complete
                AFTER UPDATE OF status ON tasks
                WHEN NEW.status = 'completed' AND OLD.status != 'completed'
                BEGIN
                    UPDATE video_projects 
                    SET completed_videos = completed_videos + 1,
                        updated_at = datetime('now')
                    WHERE id = NEW.project_id;
                END
                """,
                
                # Update project duration when videos are added
                """
                CREATE TRIGGER IF NOT EXISTS update_project_duration_on_video_add
                AFTER INSERT ON video_metadata
                BEGIN
                    UPDATE video_projects 
                    SET total_videos = total_videos + 1,
                        total_duration = total_duration + NEW.duration,
                        updated_at = datetime('now')
                    WHERE id = NEW.project_id;
                END
                """,
                
                # Update user updated_at timestamp
                """
                CREATE TRIGGER IF NOT EXISTS update_user_timestamp
                AFTER UPDATE ON users
                BEGIN
                    UPDATE users SET updated_at = datetime('now') WHERE id = NEW.id;
                END
                """,
                
                # Clean up orphaned materials when video is deleted
                """
                CREATE TRIGGER IF NOT EXISTS cleanup_materials_on_video_delete
                AFTER DELETE ON video_metadata
                BEGIN
                    DELETE FROM material_metadata WHERE video_id = OLD.id;
                END
                """,
                
                # Clean up processing logs when task is deleted
                """
                CREATE TRIGGER IF NOT EXISTS cleanup_logs_on_task_delete
                AFTER DELETE ON tasks
                BEGIN
                    DELETE FROM processing_logs WHERE task_id = OLD.id;
                END
                """
            ]
            
            for trigger_sql in triggers:
                session.execute(text(trigger_sql))
                logger.debug("Created database trigger")
            
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating triggers: {e}")
            return False
    
    def down(self, session: Session) -> bool:
        """Drop triggers"""
        try:
            triggers_to_drop = [
                "update_project_metrics_on_task_complete",
                "update_project_duration_on_video_add",
                "update_user_timestamp",
                "cleanup_materials_on_video_delete", 
                "cleanup_logs_on_task_delete"
            ]
            
            for trigger_name in triggers_to_drop:
                try:
                    session.execute(text(f"DROP TRIGGER IF EXISTS {trigger_name}"))
                except Exception as e:
                    logger.warning(f"Could not drop trigger {trigger_name}: {e}")
            
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Error dropping triggers: {e}")
            return False


class Migration005_OptimizeSettings(Migration):
    """Optimize SQLite settings for performance"""
    
    def __init__(self):
        super().__init__(
            version="005_optimize_settings",
            description="Apply SQLite optimization settings"
        )
    
    def up(self, session: Session) -> bool:
        """Apply optimization settings"""
        try:
            optimizations = [
                "PRAGMA journal_mode=WAL",
                "PRAGMA synchronous=NORMAL", 
                "PRAGMA cache_size=10000",
                "PRAGMA temp_store=MEMORY",
                "PRAGMA mmap_size=268435456",  # 256MB
                "PRAGMA page_size=4096"
            ]
            
            for pragma_sql in optimizations:
                try:
                    session.execute(text(pragma_sql))
                    logger.debug(f"Applied optimization: {pragma_sql}")
                except Exception as e:
                    logger.warning(f"Could not apply optimization {pragma_sql}: {e}")
            
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Error applying optimizations: {e}")
            return False
    
    def down(self, session: Session) -> bool:
        """Revert to default settings"""
        try:
            # Reset to default values
            defaults = [
                "PRAGMA journal_mode=DELETE",
                "PRAGMA synchronous=FULL",
                "PRAGMA cache_size=2000", 
                "PRAGMA temp_store=DEFAULT",
                "PRAGMA mmap_size=0",
                "PRAGMA page_size=1024"
            ]
            
            for pragma_sql in defaults:
                try:
                    session.execute(text(pragma_sql))
                    logger.debug(f"Reverted setting: {pragma_sql}")
                except Exception as e:
                    logger.warning(f"Could not revert setting {pragma_sql}: {e}")
            
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Error reverting optimizations: {e}")
            return False


def get_all_migrations() -> List[Migration]:
    """Get all available migrations in order"""
    return [
        Migration001_InitialSchema(),
        Migration002_AddIndexes(),
        Migration003_AddViews(),
        Migration004_AddTriggers(),
        Migration005_OptimizeSettings()
    ]


def get_migration_by_version(version: str) -> Migration:
    """Get specific migration by version"""
    all_migrations = get_all_migrations()
    for migration in all_migrations:
        if migration.version == version:
            return migration
    raise ValueError(f"Migration {version} not found")


def get_latest_migration_version() -> str:
    """Get the latest migration version"""
    migrations = get_all_migrations()
    if migrations:
        return sorted(migrations, key=lambda m: m.version)[-1].version
    return None