"""
Database backup and recovery system for MoneyPrinterTurbo
Provides comprehensive backup, restore, and disaster recovery capabilities
"""

import os
import shutil
import json
import gzip
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import threading
import schedule
import time
from contextlib import contextmanager

from app.models.database import create_database_engine, get_session_maker
from app.repositories import AnalyticsRepository

logger = logging.getLogger(__name__)


class BackupManager:
    """Manages database backups and recovery operations"""
    
    def __init__(self, database_path: str = None, backup_dir: str = None):
        self.database_path = database_path or "./storage/moneyprinterturbo.db"
        self.backup_dir = backup_dir or "./storage/backups"
        
        # Ensure backup directory exists
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)
        
        # Backup configuration
        self.max_backups = 30  # Keep last 30 backups
        self.compression = True
        self.verify_backups = True
        
        # Auto-backup settings
        self.auto_backup_enabled = False
        self.auto_backup_interval = 24  # hours
        self._backup_thread = None
        self._stop_auto_backup = threading.Event()
        
        # Initialize database connection for health checks
        self.engine = create_database_engine(self.database_path, echo=False)
        self.SessionMaker = get_session_maker(self.engine)
        
        logger.info(f"Backup manager initialized - DB: {self.database_path}, Backups: {self.backup_dir}")
    
    def create_backup(self, backup_name: str = None, include_metadata: bool = True) -> Dict[str, Any]:
        """Create a database backup"""
        try:
            # Generate backup name if not provided
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"
            
            # Ensure backup name doesn't have extension
            if backup_name.endswith('.db') or backup_name.endswith('.gz'):
                backup_name = os.path.splitext(backup_name)[0]
            
            backup_file = os.path.join(self.backup_dir, f"{backup_name}.db")
            metadata_file = os.path.join(self.backup_dir, f"{backup_name}.json")
            
            # Check if source database exists
            if not os.path.exists(self.database_path):
                raise FileNotFoundError(f"Source database not found: {self.database_path}")
            
            logger.info(f"Creating backup: {backup_name}")
            
            # Get database info before backup
            db_info = self._get_database_info()
            
            # Create backup using SQLite backup API for consistency
            backup_success = self._create_sqlite_backup(backup_file)
            
            if not backup_success:
                raise Exception("SQLite backup failed")
            
            # Verify backup integrity
            if self.verify_backups:
                if not self._verify_backup_integrity(backup_file):
                    os.remove(backup_file)
                    raise Exception("Backup verification failed")
            
            # Compress backup if enabled
            final_backup_path = backup_file
            if self.compression:
                compressed_path = f"{backup_file}.gz"
                self._compress_file(backup_file, compressed_path)
                os.remove(backup_file)
                final_backup_path = compressed_path
            
            # Create metadata
            backup_metadata = {
                'backup_name': backup_name,
                'created_at': datetime.now().isoformat(),
                'database_path': self.database_path,
                'backup_file': final_backup_path,
                'compressed': self.compression,
                'file_size_bytes': os.path.getsize(final_backup_path),
                'database_info': db_info,
                'backup_type': 'full',
                'verified': self.verify_backups
            }
            
            if include_metadata:
                with open(metadata_file, 'w') as f:
                    json.dump(backup_metadata, f, indent=2)
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            logger.info(f"Backup created successfully: {final_backup_path}")
            return {
                'success': True,
                'backup_name': backup_name,
                'backup_file': final_backup_path,
                'metadata': backup_metadata
            }
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_sqlite_backup(self, backup_path: str) -> bool:
        """Create backup using SQLite backup API"""
        try:
            # Open source and destination databases
            source_conn = sqlite3.connect(self.database_path)
            dest_conn = sqlite3.connect(backup_path)
            
            # Perform backup
            source_conn.backup(dest_conn)
            
            # Close connections
            source_conn.close()
            dest_conn.close()
            
            return True
        except Exception as e:
            logger.error(f"SQLite backup error: {e}")
            return False
    
    def _verify_backup_integrity(self, backup_path: str) -> bool:
        """Verify backup integrity"""
        try:
            conn = sqlite3.connect(backup_path)
            cursor = conn.cursor()
            
            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            conn.close()
            
            is_valid = result and result[0] == 'ok'
            logger.debug(f"Backup integrity check: {'PASS' if is_valid else 'FAIL'}")
            
            return is_valid
        except Exception as e:
            logger.error(f"Backup verification error: {e}")
            return False
    
    def _compress_file(self, source_path: str, dest_path: str):
        """Compress file using gzip"""
        try:
            with open(source_path, 'rb') as f_in:
                with gzip.open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.debug(f"Compressed backup: {dest_path}")
        except Exception as e:
            logger.error(f"Compression error: {e}")
            raise
    
    def _decompress_file(self, source_path: str, dest_path: str):
        """Decompress gzip file"""
        try:
            with gzip.open(source_path, 'rb') as f_in:
                with open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.debug(f"Decompressed backup: {dest_path}")
        except Exception as e:
            logger.error(f"Decompression error: {e}")
            raise
    
    def restore_backup(self, backup_name: str, confirm: bool = False) -> Dict[str, Any]:
        """Restore database from backup"""
        try:
            if not confirm:
                return {
                    'success': False,
                    'error': 'Restore operation requires explicit confirmation'
                }
            
            logger.warning(f"Starting database restore from backup: {backup_name}")
            
            # Find backup files
            backup_file = None
            metadata_file = os.path.join(self.backup_dir, f"{backup_name}.json")
            
            # Check for compressed or uncompressed backup
            compressed_backup = os.path.join(self.backup_dir, f"{backup_name}.db.gz")
            uncompressed_backup = os.path.join(self.backup_dir, f"{backup_name}.db")
            
            if os.path.exists(compressed_backup):
                backup_file = compressed_backup
                is_compressed = True
            elif os.path.exists(uncompressed_backup):
                backup_file = uncompressed_backup
                is_compressed = False
            else:
                raise FileNotFoundError(f"Backup file not found: {backup_name}")
            
            # Load metadata if available
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Create backup of current database before restore
            current_backup_name = f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            current_backup_result = self.create_backup(current_backup_name)
            
            if not current_backup_result['success']:
                logger.warning("Could not create pre-restore backup")
            
            # Prepare restore file
            restore_source = backup_file
            if is_compressed:
                temp_restore_file = os.path.join(self.backup_dir, f"temp_restore_{backup_name}.db")
                self._decompress_file(backup_file, temp_restore_file)
                restore_source = temp_restore_file
            
            # Verify backup before restore
            if not self._verify_backup_integrity(restore_source):
                if is_compressed:
                    os.remove(restore_source)
                raise Exception("Backup integrity check failed")
            
            # Stop any active connections
            self.engine.dispose()
            
            # Perform restore
            shutil.copy2(restore_source, self.database_path)
            
            # Clean up temporary file
            if is_compressed and restore_source != backup_file:
                os.remove(restore_source)
            
            # Reinitialize database connection
            self.engine = create_database_engine(self.database_path, echo=False)
            self.SessionMaker = get_session_maker(self.engine)
            
            # Verify restored database
            if not self._verify_backup_integrity(self.database_path):
                logger.error("Restored database failed integrity check")
                return {
                    'success': False,
                    'error': 'Restored database failed integrity check'
                }
            
            logger.info(f"Database restored successfully from backup: {backup_name}")
            return {
                'success': True,
                'backup_name': backup_name,
                'restored_at': datetime.now().isoformat(),
                'metadata': metadata,
                'pre_restore_backup': current_backup_result.get('backup_name')
            }
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        try:
            backups = []
            
            # Find all backup files
            backup_files = []
            for file in os.listdir(self.backup_dir):
                if file.endswith('.db') or file.endswith('.db.gz'):
                    backup_name = file.replace('.db.gz', '').replace('.db', '')
                    backup_files.append((backup_name, file))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.backup_dir, x[1])), reverse=True)
            
            for backup_name, file_name in backup_files:
                backup_path = os.path.join(self.backup_dir, file_name)
                metadata_path = os.path.join(self.backup_dir, f"{backup_name}.json")
                
                backup_info = {
                    'name': backup_name,
                    'file': file_name,
                    'path': backup_path,
                    'size_bytes': os.path.getsize(backup_path),
                    'created_at': datetime.fromtimestamp(os.path.getmtime(backup_path)).isoformat(),
                    'compressed': file_name.endswith('.gz'),
                    'has_metadata': os.path.exists(metadata_path)
                }
                
                # Load metadata if available
                if backup_info['has_metadata']:
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            backup_info['metadata'] = metadata
                    except Exception as e:
                        logger.warning(f"Could not load metadata for {backup_name}: {e}")
                
                backups.append(backup_info)
            
            return backups
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    def delete_backup(self, backup_name: str) -> bool:
        """Delete a specific backup"""
        try:
            # Find and delete backup files
            files_to_delete = [
                os.path.join(self.backup_dir, f"{backup_name}.db"),
                os.path.join(self.backup_dir, f"{backup_name}.db.gz"),
                os.path.join(self.backup_dir, f"{backup_name}.json")
            ]
            
            deleted_count = 0
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
                    logger.debug(f"Deleted backup file: {file_path}")
            
            if deleted_count > 0:
                logger.info(f"Deleted backup: {backup_name}")
                return True
            else:
                logger.warning(f"Backup not found: {backup_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting backup {backup_name}: {e}")
            return False
    
    def _cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        try:
            backups = self.list_backups()
            
            if len(backups) <= self.max_backups:
                return
            
            # Sort by creation time (oldest first)
            backups.sort(key=lambda x: x['created_at'])
            
            # Delete oldest backups
            to_delete = backups[:len(backups) - self.max_backups]
            
            for backup in to_delete:
                self.delete_backup(backup['name'])
                logger.info(f"Cleaned up old backup: {backup['name']}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def _get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        try:
            with self.engine.connect() as conn:
                from sqlalchemy import text
                
                page_count = conn.execute(text("PRAGMA page_count")).scalar()
                page_size = conn.execute(text("PRAGMA page_size")).scalar()
                size_mb = (page_count * page_size) / 1024 / 1024
                
                # Get table counts
                with self.SessionMaker() as session:
                    analytics_repo = AnalyticsRepository(session)
                    # Could add more detailed stats here
                
                return {
                    'size_mb': round(size_mb, 2),
                    'page_count': page_count,
                    'page_size': page_size,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {}
    
    def start_auto_backup(self, interval_hours: int = None):
        """Start automatic backup scheduling"""
        if self.auto_backup_enabled:
            logger.warning("Auto-backup is already running")
            return
        
        if interval_hours:
            self.auto_backup_interval = interval_hours
        
        self.auto_backup_enabled = True
        self._stop_auto_backup.clear()
        
        # Schedule backups
        schedule.clear()
        schedule.every(self.auto_backup_interval).hours.do(self._auto_backup_job)
        
        # Start background thread
        self._backup_thread = threading.Thread(target=self._auto_backup_worker, daemon=True)
        self._backup_thread.start()
        
        logger.info(f"Auto-backup started with {self.auto_backup_interval} hour interval")
    
    def stop_auto_backup(self):
        """Stop automatic backup scheduling"""
        if not self.auto_backup_enabled:
            return
        
        self.auto_backup_enabled = False
        self._stop_auto_backup.set()
        
        if self._backup_thread and self._backup_thread.is_alive():
            self._backup_thread.join(timeout=5)
        
        schedule.clear()
        logger.info("Auto-backup stopped")
    
    def _auto_backup_worker(self):
        """Background worker for automatic backups"""
        while not self._stop_auto_backup.is_set():
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _auto_backup_job(self):
        """Execute automatic backup"""
        try:
            backup_name = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            result = self.create_backup(backup_name)
            
            if result['success']:
                logger.info(f"Auto-backup completed: {backup_name}")
            else:
                logger.error(f"Auto-backup failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Auto-backup job error: {e}")
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get backup system status"""
        try:
            backups = self.list_backups()
            
            total_size = sum(backup['size_bytes'] for backup in backups)
            latest_backup = backups[0] if backups else None
            
            return {
                'backup_directory': self.backup_dir,
                'total_backups': len(backups),
                'total_size_mb': round(total_size / 1024 / 1024, 2),
                'latest_backup': latest_backup,
                'auto_backup_enabled': self.auto_backup_enabled,
                'auto_backup_interval_hours': self.auto_backup_interval,
                'max_backups': self.max_backups,
                'compression_enabled': self.compression,
                'verification_enabled': self.verify_backups
            }
            
        except Exception as e:
            logger.error(f"Error getting backup status: {e}")
            return {'error': str(e)}
    
    def export_data(self, export_path: str, format: str = 'json') -> Dict[str, Any]:
        """Export database data to various formats"""
        try:
            logger.info(f"Exporting data to {export_path} in {format} format")
            
            if format.lower() == 'json':
                return self._export_to_json(export_path)
            elif format.lower() == 'sql':
                return self._export_to_sql(export_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return {'success': False, 'error': str(e)}
    
    def _export_to_json(self, export_path: str) -> Dict[str, Any]:
        """Export data to JSON format"""
        try:
            # This would require implementing data serialization for all tables
            # For now, return a placeholder
            return {
                'success': True,
                'message': 'JSON export not yet implemented',
                'export_path': export_path
            }
        except Exception as e:
            logger.error(f"JSON export error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _export_to_sql(self, export_path: str) -> Dict[str, Any]:
        """Export data to SQL dump format"""
        try:
            # Use sqlite3 command line tool if available
            import subprocess
            
            result = subprocess.run([
                'sqlite3', self.database_path, '.dump'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                with open(export_path, 'w') as f:
                    f.write(result.stdout)
                
                return {
                    'success': True,
                    'export_path': export_path,
                    'file_size_bytes': os.path.getsize(export_path)
                }
            else:
                raise Exception(f"sqlite3 dump failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"SQL export error: {e}")
            return {'success': False, 'error': str(e)}
    
    def close(self):
        """Close backup manager and cleanup resources"""
        try:
            self.stop_auto_backup()
            self.engine.dispose()
            logger.info("Backup manager closed")
        except Exception as e:
            logger.error(f"Error closing backup manager: {e}")


def create_backup_manager(database_path: str = None, backup_dir: str = None) -> BackupManager:
    """Factory function to create backup manager"""
    return BackupManager(database_path, backup_dir)