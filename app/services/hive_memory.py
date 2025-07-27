"""
Hive Memory SQL Backend for Swarm Coordination
Advanced memory management system for Claude Flow hive mind
"""

import sqlite3
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from contextlib import contextmanager

from app.config import config

logger = logging.getLogger(__name__)


class HiveMemoryManager:
    """SQL-based memory management for swarm coordination"""
    
    def __init__(self):
        self.db_type = config.hive_memory.get("db_type", "sqlite")
        self.db_path = config.hive_memory.get("db_path", "./memory/hive_memory.db")
        self.max_connections = config.hive_memory.get("max_connections", 10)
        self.auto_create_tables = config.hive_memory.get("auto_create_tables", True)
        self.retention_days = config.hive_memory.get("retention_days", 30)
        
        # Thread-local storage for database connections
        self._local = threading.local()
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database and create tables if needed"""
        if self.auto_create_tables:
            self._create_tables()
    
    def _get_connection(self):
        """Get a database connection (thread-safe)"""
        if not hasattr(self._local, 'connection'):
            if self.db_type == "sqlite":
                # Ensure directory exists
                db_path = Path(self.db_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                
                self._local.connection = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=30.0
                )
                self._local.connection.row_factory = sqlite3.Row
            else:
                raise NotImplementedError(f"Database type {self.db_type} not yet supported")
        
        return self._local.connection
    
    @contextmanager
    def _get_cursor(self):
        """Get a database cursor with transaction management"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            cursor.close()
    
    def _create_tables(self):
        """Create necessary tables for hive memory"""
        with self._get_cursor() as cursor:
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hive_sessions (
                    session_id TEXT PRIMARY KEY,
                    topology TEXT NOT NULL,
                    max_agents INTEGER,
                    strategy TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    metadata TEXT
                )
            """)
            
            # Agents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hive_agents (
                    agent_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    agent_type TEXT NOT NULL,
                    agent_name TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES hive_sessions (session_id)
                )
            """)
            
            # Memory entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hive_memory (
                    memory_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    agent_id TEXT,
                    memory_key TEXT NOT NULL,
                    memory_type TEXT DEFAULT 'general',
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES hive_sessions (session_id),
                    FOREIGN KEY (agent_id) REFERENCES hive_agents (agent_id)
                )
            """)
            
            # Tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hive_tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    agent_id TEXT,
                    task_type TEXT NOT NULL,
                    task_status TEXT DEFAULT 'pending',
                    task_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES hive_sessions (session_id),
                    FOREIGN KEY (agent_id) REFERENCES hive_agents (agent_id)
                )
            """)
            
            # Coordination events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hive_coordination (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    agent_id TEXT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES hive_sessions (session_id),
                    FOREIGN KEY (agent_id) REFERENCES hive_agents (agent_id)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_session ON hive_memory (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_key ON hive_memory (memory_key)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON hive_memory (memory_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session ON hive_tasks (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON hive_tasks (task_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_coordination_session ON hive_coordination (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_coordination_type ON hive_coordination (event_type)")
    
    def create_session(self, session_id: str, topology: str, max_agents: int = 8, 
                      strategy: str = "adaptive", metadata: Dict = None) -> bool:
        """Create a new hive session"""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO hive_sessions 
                    (session_id, topology, max_agents, strategy, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id, 
                    topology, 
                    max_agents, 
                    strategy, 
                    json.dumps(metadata or {})
                ))
            logger.info(f"Created hive session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            return False
    
    def register_agent(self, agent_id: str, session_id: str, agent_type: str, 
                      agent_name: str = None, metadata: Dict = None) -> bool:
        """Register an agent in the hive"""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO hive_agents 
                    (agent_id, session_id, agent_type, agent_name, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    agent_id, 
                    session_id, 
                    agent_type, 
                    agent_name or agent_type, 
                    json.dumps(metadata or {})
                ))
            logger.debug(f"Registered agent {agent_id} in session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False
    
    def store_memory(self, memory_key: str, content: Any, session_id: str = None, 
                    agent_id: str = None, memory_type: str = "general", 
                    expires_in: int = None, metadata: Dict = None) -> bool:
        """Store memory entry"""
        try:
            memory_id = f"{memory_key}_{int(time.time())}"
            expires_at = None
            if expires_in:
                expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            with self._get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO hive_memory 
                    (memory_id, session_id, agent_id, memory_key, memory_type, 
                     content, expires_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_id,
                    session_id,
                    agent_id,
                    memory_key,
                    memory_type,
                    json.dumps(content) if not isinstance(content, str) else content,
                    expires_at.isoformat() if expires_at else None,
                    json.dumps(metadata or {})
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to store memory {memory_key}: {e}")
            return False
    
    def retrieve_memory(self, memory_key: str, session_id: str = None, 
                       agent_id: str = None, memory_type: str = None) -> Optional[Any]:
        """Retrieve memory entry"""
        try:
            with self._get_cursor() as cursor:
                query = """
                    SELECT content, memory_type, created_at, expires_at 
                    FROM hive_memory 
                    WHERE memory_key = ?
                """
                params = [memory_key]
                
                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)
                
                if agent_id:
                    query += " AND agent_id = ?"
                    params.append(agent_id)
                
                if memory_type:
                    query += " AND memory_type = ?"
                    params.append(memory_type)
                
                query += " AND (expires_at IS NULL OR expires_at > datetime('now'))"
                query += " ORDER BY created_at DESC LIMIT 1"
                
                cursor.execute(query, params)
                row = cursor.fetchone()
                
                if row:
                    content = row['content']
                    try:
                        # Try to parse as JSON
                        return json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        # Return as string if not valid JSON
                        return content
                
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_key}: {e}")
            return None
    
    def list_memories(self, session_id: str = None, agent_id: str = None, 
                     memory_type: str = None, pattern: str = None) -> List[Dict]:
        """List memory entries with optional filtering"""
        try:
            with self._get_cursor() as cursor:
                query = """
                    SELECT memory_key, memory_type, content, created_at, expires_at, metadata
                    FROM hive_memory 
                    WHERE (expires_at IS NULL OR expires_at > datetime('now'))
                """
                params = []
                
                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)
                
                if agent_id:
                    query += " AND agent_id = ?"
                    params.append(agent_id)
                
                if memory_type:
                    query += " AND memory_type = ?"
                    params.append(memory_type)
                
                if pattern:
                    query += " AND memory_key LIKE ?"
                    params.append(f"%{pattern}%")
                
                query += " ORDER BY created_at DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                memories = []
                for row in rows:
                    content = row['content']
                    try:
                        content = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    
                    memories.append({
                        'key': row['memory_key'],
                        'type': row['memory_type'],
                        'content': content,
                        'created_at': row['created_at'],
                        'expires_at': row['expires_at'],
                        'metadata': json.loads(row['metadata'] or '{}')
                    })
                
                return memories
        except Exception as e:
            logger.error(f"Failed to list memories: {e}")
            return []
    
    def create_task(self, task_id: str, session_id: str, agent_id: str, 
                   task_type: str, task_data: Dict = None, metadata: Dict = None) -> bool:
        """Create a new task"""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO hive_tasks 
                    (task_id, session_id, agent_id, task_type, task_data, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    task_id,
                    session_id,
                    agent_id,
                    task_type,
                    json.dumps(task_data or {}),
                    json.dumps(metadata or {})
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to create task {task_id}: {e}")
            return False
    
    def update_task_status(self, task_id: str, status: str, 
                          completed_at: datetime = None) -> bool:
        """Update task status"""
        try:
            with self._get_cursor() as cursor:
                if status == 'completed' and not completed_at:
                    completed_at = datetime.now()
                
                cursor.execute("""
                    UPDATE hive_tasks 
                    SET task_status = ?, updated_at = datetime('now'), completed_at = ?
                    WHERE task_id = ?
                """, (status, completed_at.isoformat() if completed_at else None, task_id))
            return True
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {e}")
            return False
    
    def log_coordination_event(self, session_id: str, agent_id: str, event_type: str, 
                              event_data: Dict = None, metadata: Dict = None) -> bool:
        """Log a coordination event"""
        try:
            event_id = f"{event_type}_{agent_id}_{int(time.time())}"
            with self._get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO hive_coordination 
                    (event_id, session_id, agent_id, event_type, event_data, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    session_id,
                    agent_id,
                    event_type,
                    json.dumps(event_data or {}),
                    json.dumps(metadata or {})
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to log coordination event: {e}")
            return False
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get session status and statistics"""
        try:
            with self._get_cursor() as cursor:
                # Get session info
                cursor.execute("""
                    SELECT * FROM hive_sessions WHERE session_id = ?
                """, (session_id,))
                session = cursor.fetchone()
                
                if not session:
                    return None
                
                # Get agent count
                cursor.execute("""
                    SELECT COUNT(*) as agent_count FROM hive_agents 
                    WHERE session_id = ? AND status = 'active'
                """, (session_id,))
                agent_count = cursor.fetchone()['agent_count']
                
                # Get task statistics
                cursor.execute("""
                    SELECT task_status, COUNT(*) as count 
                    FROM hive_tasks WHERE session_id = ?
                    GROUP BY task_status
                """, (session_id,))
                task_stats = {row['task_status']: row['count'] for row in cursor.fetchall()}
                
                # Get memory count
                cursor.execute("""
                    SELECT COUNT(*) as memory_count FROM hive_memory 
                    WHERE session_id = ? AND (expires_at IS NULL OR expires_at > datetime('now'))
                """, (session_id,))
                memory_count = cursor.fetchone()['memory_count']
                
                return {
                    'session_id': session['session_id'],
                    'topology': session['topology'],
                    'max_agents': session['max_agents'],
                    'strategy': session['strategy'],
                    'status': session['status'],
                    'created_at': session['created_at'],
                    'updated_at': session['updated_at'],
                    'agent_count': agent_count,
                    'task_stats': task_stats,
                    'memory_count': memory_count,
                    'metadata': json.loads(session['metadata'] or '{}')
                }
        except Exception as e:
            logger.error(f"Failed to get session status: {e}")
            return None
    
    def cleanup_expired(self) -> int:
        """Clean up expired memory entries and old data"""
        try:
            cleaned_count = 0
            with self._get_cursor() as cursor:
                # Clean expired memory entries
                cursor.execute("""
                    DELETE FROM hive_memory 
                    WHERE expires_at IS NOT NULL AND expires_at <= datetime('now')
                """)
                cleaned_count += cursor.rowcount
                
                # Clean old coordination events (beyond retention period)
                retention_date = datetime.now() - timedelta(days=self.retention_days)
                cursor.execute("""
                    DELETE FROM hive_coordination 
                    WHERE timestamp < ?
                """, (retention_date.isoformat(),))
                cleaned_count += cursor.rowcount
                
                logger.info(f"Cleaned up {cleaned_count} expired entries")
                return cleaned_count
        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {e}")
            return 0
    
    def close(self):
        """Close database connections"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')


# Global instance
_hive_memory = None
_memory_lock = threading.Lock()


def get_hive_memory() -> HiveMemoryManager:
    """Get the global hive memory manager instance"""
    global _hive_memory
    if _hive_memory is None:
        with _memory_lock:
            if _hive_memory is None:
                _hive_memory = HiveMemoryManager()
    return _hive_memory


# Convenience functions for easy access
def store_swarm_memory(key: str, value: Any, session_id: str = None, 
                      agent_id: str = None, expires_in: int = None) -> bool:
    """Store memory in the hive"""
    return get_hive_memory().store_memory(
        key, value, session_id, agent_id, expires_in=expires_in
    )


def retrieve_swarm_memory(key: str, session_id: str = None, 
                         agent_id: str = None) -> Optional[Any]:
    """Retrieve memory from the hive"""
    return get_hive_memory().retrieve_memory(key, session_id, agent_id)


def log_swarm_event(session_id: str, agent_id: str, event_type: str, 
                   event_data: Dict = None) -> bool:
    """Log a coordination event"""
    return get_hive_memory().log_coordination_event(
        session_id, agent_id, event_type, event_data
    )