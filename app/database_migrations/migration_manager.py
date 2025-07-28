"""
Database migration manager for MoneyPrinterTurbo
Handles database schema versioning and updates
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
import logging

from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from app.models.database import DatabaseMigration, Base, create_database_engine, get_session_maker

logger = logging.getLogger(__name__)


class Migration(ABC):
    """Base class for database migrations"""
    
    def __init__(self, version: str, description: str):
        self.version = version
        self.description = description
    
    @abstractmethod
    def up(self, session: Session) -> bool:
        """Apply the migration"""
        pass
    
    @abstractmethod  
    def down(self, session: Session) -> bool:
        """Rollback the migration"""
        pass
    
    def validate(self, session: Session) -> bool:
        """Validate that the migration was applied correctly"""
        return True


class MigrationManager:
    """Manages database migrations"""
    
    def __init__(self, database_path: str = None):
        self.database_path = database_path or "./storage/moneyprinterturbo.db"
        self.engine = create_database_engine(self.database_path, echo=False)
        self.SessionMaker = get_session_maker(self.engine)
        
        # Ensure migration table exists
        self._ensure_migration_table()
        
        logger.info(f"Migration manager initialized for database: {self.database_path}")
    
    def _ensure_migration_table(self):
        """Ensure the database_migrations table exists"""
        try:
            Base.metadata.create_all(self.engine)
            logger.debug("Migration table ensured")
        except Exception as e:
            logger.error(f"Error ensuring migration table: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session"""
        session = self.SessionMaker()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Migration session error: {e}")
            raise
        finally:
            session.close()
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions"""
        try:
            with self.get_session() as session:
                migrations = session.query(DatabaseMigration).filter(
                    DatabaseMigration.success == True
                ).order_by(DatabaseMigration.applied_at).all()
                
                return [m.version for m in migrations]
        except Exception as e:
            logger.error(f"Error getting applied migrations: {e}")
            return []
    
    def is_migration_applied(self, version: str) -> bool:
        """Check if a migration version is already applied"""
        return version in self.get_applied_migrations()
    
    def apply_migration(self, migration: Migration, dry_run: bool = False) -> bool:
        """Apply a single migration"""
        if self.is_migration_applied(migration.version):
            logger.info(f"Migration {migration.version} already applied, skipping")
            return True
        
        try:
            with self.get_session() as session:
                logger.info(f"Applying migration {migration.version}: {migration.description}")
                
                if dry_run:
                    logger.info("DRY RUN: Migration would be applied but no changes will be made")
                    return True
                
                # Record migration start
                migration_record = DatabaseMigration(
                    version=migration.version,
                    description=migration.description,
                    success=False
                )
                session.add(migration_record)
                session.commit()
                
                try:
                    # Apply the migration
                    success = migration.up(session)
                    
                    if success:
                        # Validate the migration
                        if migration.validate(session):
                            # Mark as successful
                            migration_record.success = True
                            session.commit()
                            logger.info(f"Migration {migration.version} applied successfully")
                            return True
                        else:
                            raise Exception("Migration validation failed")
                    else:
                        raise Exception("Migration up() method returned False")
                        
                except Exception as e:
                    # Mark as failed
                    migration_record.success = False
                    migration_record.error_message = str(e)
                    session.commit()
                    logger.error(f"Migration {migration.version} failed: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error applying migration {migration.version}: {e}")
            return False
    
    def rollback_migration(self, migration: Migration, dry_run: bool = False) -> bool:
        """Rollback a single migration"""
        if not self.is_migration_applied(migration.version):
            logger.info(f"Migration {migration.version} not applied, cannot rollback")
            return True
        
        try:
            with self.get_session() as session:
                logger.info(f"Rolling back migration {migration.version}: {migration.description}")
                
                if dry_run:
                    logger.info("DRY RUN: Migration would be rolled back but no changes will be made")
                    return True
                
                try:
                    # Rollback the migration
                    success = migration.down(session)
                    
                    if success:
                        # Remove migration record
                        session.query(DatabaseMigration).filter(
                            DatabaseMigration.version == migration.version
                        ).delete()
                        session.commit()
                        logger.info(f"Migration {migration.version} rolled back successfully")
                        return True
                    else:
                        raise Exception("Migration down() method returned False")
                        
                except Exception as e:
                    logger.error(f"Migration {migration.version} rollback failed: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error rolling back migration {migration.version}: {e}")
            return False
    
    def apply_migrations(self, migrations: List[Migration], dry_run: bool = False) -> Dict[str, Any]:
        """Apply multiple migrations in order"""
        results = {
            'total': len(migrations),
            'applied': 0,
            'skipped': 0,
            'failed': 0,
            'details': []
        }
        
        # Sort migrations by version
        sorted_migrations = sorted(migrations, key=lambda m: m.version)
        
        for migration in sorted_migrations:
            if self.is_migration_applied(migration.version):
                results['skipped'] += 1
                results['details'].append({
                    'version': migration.version,
                    'status': 'skipped',
                    'description': migration.description
                })
                continue
            
            success = self.apply_migration(migration, dry_run=dry_run)
            
            if success:
                results['applied'] += 1
                results['details'].append({
                    'version': migration.version,
                    'status': 'applied',
                    'description': migration.description
                })
            else:
                results['failed'] += 1
                results['details'].append({
                    'version': migration.version,
                    'status': 'failed',
                    'description': migration.description
                })
                
                # Stop on first failure
                logger.error(f"Migration {migration.version} failed, stopping migration process")
                break
        
        logger.info(f"Migration results: {results['applied']} applied, "
                   f"{results['skipped']} skipped, {results['failed']} failed")
        
        return results
    
    def rollback_migrations(self, migrations: List[Migration], dry_run: bool = False) -> Dict[str, Any]:
        """Rollback multiple migrations in reverse order"""
        results = {
            'total': len(migrations),
            'rolled_back': 0,
            'skipped': 0,
            'failed': 0,
            'details': []
        }
        
        # Sort migrations by version in reverse order
        sorted_migrations = sorted(migrations, key=lambda m: m.version, reverse=True)
        
        for migration in sorted_migrations:
            if not self.is_migration_applied(migration.version):
                results['skipped'] += 1
                results['details'].append({
                    'version': migration.version,
                    'status': 'skipped',
                    'description': migration.description
                })
                continue
            
            success = self.rollback_migration(migration, dry_run=dry_run)
            
            if success:
                results['rolled_back'] += 1
                results['details'].append({
                    'version': migration.version,
                    'status': 'rolled_back',
                    'description': migration.description
                })
            else:
                results['failed'] += 1
                results['details'].append({
                    'version': migration.version,
                    'status': 'failed',
                    'description': migration.description
                })
                
                # Stop on first failure
                logger.error(f"Migration {migration.version} rollback failed, stopping rollback process")
                break
        
        logger.info(f"Rollback results: {results['rolled_back']} rolled back, "
                   f"{results['skipped']} skipped, {results['failed']} failed")
        
        return results
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        try:
            with self.get_session() as session:
                # Get all migration records
                migrations = session.query(DatabaseMigration).order_by(
                    DatabaseMigration.applied_at
                ).all()
                
                applied_migrations = []
                failed_migrations = []
                
                for migration in migrations:
                    migration_info = {
                        'version': migration.version,
                        'description': migration.description,
                        'applied_at': migration.applied_at.isoformat() if migration.applied_at else None,
                        'success': migration.success,
                        'error_message': migration.error_message
                    }
                    
                    if migration.success:
                        applied_migrations.append(migration_info)
                    else:
                        failed_migrations.append(migration_info)
                
                return {
                    'database_path': self.database_path,
                    'total_migrations': len(migrations),
                    'applied_count': len(applied_migrations),
                    'failed_count': len(failed_migrations),
                    'applied_migrations': applied_migrations,
                    'failed_migrations': failed_migrations,
                    'latest_migration': applied_migrations[-1]['version'] if applied_migrations else None
                }
                
        except Exception as e:
            logger.error(f"Error getting migration status: {e}")
            return {
                'error': str(e)
            }
    
    def create_backup(self, backup_path: str = None) -> str:
        """Create database backup before migrations"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.database_path}.backup_{timestamp}"
        
        try:
            import shutil
            shutil.copy2(self.database_path, backup_path)
            logger.info(f"Database backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    def validate_database_integrity(self) -> bool:
        """Validate database integrity"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("PRAGMA integrity_check")).scalar()
                
                if result == 'ok':
                    logger.info("Database integrity check passed")
                    return True
                else:
                    logger.error(f"Database integrity check failed: {result}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error checking database integrity: {e}")
            return False
    
    def optimize_database(self):
        """Optimize database after migrations"""
        try:
            with self.engine.connect() as conn:
                # Vacuum to reclaim space
                conn.execute(text("VACUUM"))
                
                # Analyze tables for query optimization
                conn.execute(text("ANALYZE"))
                
                # Update SQLite optimizer statistics
                conn.execute(text("PRAGMA optimize"))
                
                conn.commit()
                
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        try:
            with self.engine.connect() as conn:
                # Get database size and info
                page_count = conn.execute(text("PRAGMA page_count")).scalar()
                page_size = conn.execute(text("PRAGMA page_size")).scalar()
                size_mb = (page_count * page_size) / 1024 / 1024
                
                # Get schema version
                schema_version = conn.execute(text("PRAGMA schema_version")).scalar()
                
                # Get journal mode
                journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar()
                
                return {
                    'database_path': self.database_path,
                    'size_mb': round(size_mb, 2),
                    'page_count': page_count,
                    'page_size': page_size,
                    'schema_version': schema_version,
                    'journal_mode': journal_mode
                }
                
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {'error': str(e)}
    
    def close(self):
        """Close migration manager"""
        try:
            self.engine.dispose()
            logger.info("Migration manager closed")
        except Exception as e:
            logger.error(f"Error closing migration manager: {e}")


def create_migration_manager(database_path: str = None) -> MigrationManager:
    """Factory function to create migration manager"""
    return MigrationManager(database_path)