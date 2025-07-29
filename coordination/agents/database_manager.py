"""
Database Manager Agent (Worker-6)
Specialized agent for database operations, connection management, and data coordination
"""

import asyncio
import os
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from app.services.hive_memory import get_hive_memory, log_swarm_event, store_swarm_memory, retrieve_swarm_memory

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    SQLITE = "sqlite"
    ELASTICSEARCH = "elasticsearch"


class QueryType(Enum):
    """Database query types"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    AGGREGATE = "aggregate"
    INDEX = "index"


class TransactionState(Enum):
    """Transaction states"""
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class ConnectionState(Enum):
    """Database connection states"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class BackupStatus(Enum):
    """Backup operation status"""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    database_id: str
    database_type: DatabaseType
    host: str
    port: int
    database_name: str
    username: str
    password: str  # Should be encrypted/hashed in production
    max_connections: int = 10
    connection_timeout: int = 30
    query_timeout: int = 300
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    replica_hosts: List[str] = None
    read_preference: str = "primary"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.replica_hosts is None:
            self.replica_hosts = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryRequest:
    """Database query request"""
    request_id: str
    database_id: str
    query_type: QueryType
    query: str
    parameters: Dict[str, Any] = None
    options: Dict[str, Any] = None
    timeout: int = 300
    priority: int = 5
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    rows_affected: Optional[int] = None
    result_size: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.parameters is None:
            self.parameters = {}
        if self.options is None:
            self.options = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Transaction:
    """Database transaction"""
    transaction_id: str
    database_id: str
    queries: List[QueryRequest]
    state: TransactionState = TransactionState.ACTIVE
    isolation_level: str = "READ_COMMITTED"
    timeout: int = 300
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    rollback_reason: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class BackupOperation:
    """Database backup operation"""
    backup_id: str
    database_id: str
    backup_type: str  # full, incremental, differential
    output_path: str
    compression: bool = True
    encryption: bool = False
    status: BackupStatus = BackupStatus.SCHEDULED
    scheduled_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    file_size: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.scheduled_at is None:
            self.scheduled_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class DatabaseConnection:
    """Database connection wrapper"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        self.state = ConnectionState.DISCONNECTED
        self.last_activity = datetime.now()
        self.connection_attempts = 0
        self.max_retries = 3
        self.query_count = 0
        self.error_count = 0
    
    async def connect(self) -> bool:
        """Establish database connection"""
        try:
            # Simulated connection logic
            # In real implementation, use appropriate database drivers
            await asyncio.sleep(0.1)  # Simulate connection time
            
            self.connection = f"mock_connection_{self.config.database_id}"
            self.state = ConnectionState.CONNECTED
            self.last_activity = datetime.now()
            self.connection_attempts += 1
            
            logger.info(f"Connected to database {self.config.database_id}")
            return True
            
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.error_count += 1
            logger.error(f"Failed to connect to database {self.config.database_id}: {e}")
            return False
    
    async def disconnect(self):
        """Close database connection"""
        try:
            if self.connection:
                # Simulated disconnection
                self.connection = None
                self.state = ConnectionState.DISCONNECTED
                logger.info(f"Disconnected from database {self.config.database_id}")
        except Exception as e:
            logger.error(f"Error disconnecting from database {self.config.database_id}: {e}")
    
    async def execute_query(self, query_request: QueryRequest) -> Tuple[bool, Any]:
        """Execute database query"""
        try:
            if self.state != ConnectionState.CONNECTED:
                await self.connect()
            
            # Simulate query execution
            start_time = time.time()
            await asyncio.sleep(0.05)  # Simulate query time
            
            # Mock result based on query type
            if query_request.query_type == QueryType.SELECT:
                result = [{"id": i, "data": f"row_{i}"} for i in range(10)]
                query_request.rows_affected = len(result)
            elif query_request.query_type in [QueryType.INSERT, QueryType.UPDATE, QueryType.DELETE]:
                result = {"affected_rows": 1, "last_insert_id": 123}
                query_request.rows_affected = 1
            else:
                result = {"status": "success"}
                query_request.rows_affected = 0
            
            query_request.execution_time = time.time() - start_time
            query_request.result_size = len(str(result))
            
            self.query_count += 1
            self.last_activity = datetime.now()
            
            return True, result
            
        except Exception as e:
            self.error_count += 1
            query_request.error_message = str(e)
            logger.error(f"Query execution failed for {self.config.database_id}: {e}")
            return False, None
    
    def is_healthy(self) -> bool:
        """Check connection health"""
        if self.state != ConnectionState.CONNECTED:
            return False
        
        # Check if connection is too old (5 minutes)
        if (datetime.now() - self.last_activity).total_seconds() > 300:
            return False
        
        # Check error rate
        if self.query_count > 0 and (self.error_count / self.query_count) > 0.1:
            return False
        
        return True


class DatabaseManagerAgent:
    """Database management agent for coordinating database operations"""
    
    def __init__(self, agent_id: str, session_id: str):
        self.agent_id = agent_id
        self.session_id = session_id
        self.hive_memory = get_hive_memory()
        
        # Database management
        self.databases: Dict[str, DatabaseConfig] = {}
        self.connections: Dict[str, DatabaseConnection] = {}
        self.active_transactions: Dict[str, Transaction] = {}
        self.query_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.backup_queue: asyncio.Queue = asyncio.Queue()
        
        # Processing configuration
        self.max_concurrent_queries = 10
        self.query_processing_semaphore = asyncio.Semaphore(self.max_concurrent_queries)
        self.connection_pool_size = 20
        
        # Performance monitoring
        self.metrics = {
            "queries_executed": 0,
            "queries_failed": 0,
            "transactions_committed": 0,
            "transactions_rolled_back": 0,
            "connections_active": 0,
            "connections_failed": 0,
            "average_query_time": 0.0,
            "slowest_queries": [],
            "database_sizes": {},
            "backup_count": 0,
            "errors": []
        }
        
        # Agent status
        self.is_running = False
        self.last_heartbeat = datetime.now()
        
        logger.info(f"Database Manager Agent {agent_id} initialized")
    
    async def start(self):
        """Start the database manager agent"""
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._process_query_queue())
        asyncio.create_task(self._process_backup_queue())
        asyncio.create_task(self._monitor_connections())
        asyncio.create_task(self._cleanup_old_data())
        asyncio.create_task(self._update_metrics())
        
        # Log startup event
        log_swarm_event(
            session_id=self.session_id,
            agent_id=self.agent_id,
            event_type="agent_started",
            event_data={"agent_type": "database_manager", "status": "active"}
        )
        
        logger.info(f"Database Manager Agent {self.agent_id} started")
    
    async def stop(self):
        """Stop the database manager agent"""
        self.is_running = False
        
        # Close all connections
        for connection in self.connections.values():
            await connection.disconnect()
        
        # Rollback active transactions
        for transaction in self.active_transactions.values():
            if transaction.state == TransactionState.ACTIVE:
                await self.rollback_transaction(transaction.transaction_id)
        
        # Log shutdown event
        log_swarm_event(
            session_id=self.session_id,
            agent_id=self.agent_id,
            event_type="agent_stopped",
            event_data={"agent_type": "database_manager", "status": "stopped"}
        )
        
        logger.info(f"Database Manager Agent {self.agent_id} stopped")
    
    async def add_database(self, config: DatabaseConfig) -> bool:
        """Add a new database configuration"""
        try:
            # Store configuration
            self.databases[config.database_id] = config
            
            # Create connection
            connection = DatabaseConnection(config)
            self.connections[config.database_id] = connection
            
            # Test connection
            if await connection.connect():
                self.metrics["connections_active"] += 1
                
                # Store in hive memory
                store_swarm_memory(
                    key=f"database_config_{config.database_id}",
                    value=asdict(config),
                    session_id=self.session_id,
                    agent_id=self.agent_id
                )
                
                logger.info(f"Database {config.database_id} added successfully")
                return True
            else:
                self.metrics["connections_failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Failed to add database {config.database_id}: {e}")
            return False
    
    async def remove_database(self, database_id: str) -> bool:
        """Remove a database configuration"""
        try:
            if database_id in self.connections:
                await self.connections[database_id].disconnect()
                del self.connections[database_id]
            
            if database_id in self.databases:
                del self.databases[database_id]
            
            logger.info(f"Database {database_id} removed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove database {database_id}: {e}")
            return False
    
    async def execute_query(self, query_request: QueryRequest) -> Tuple[bool, Any]:
        """Execute a database query"""
        try:
            # Validate database exists
            if query_request.database_id not in self.connections:
                query_request.error_message = f"Database {query_request.database_id} not found"
                return False, None
            
            # Add to queue for processing
            priority = -query_request.priority  # Negative for max-heap behavior
            await self.query_queue.put((priority, time.time(), query_request))
            
            # Log query submission
            log_swarm_event(
                session_id=self.session_id,
                agent_id=self.agent_id,
                event_type="query_submitted",
                event_data={
                    "request_id": query_request.request_id,
                    "database_id": query_request.database_id,
                    "query_type": query_request.query_type.value
                }
            )
            
            return True, {"status": "queued", "request_id": query_request.request_id}
            
        except Exception as e:
            logger.error(f"Failed to execute query {query_request.request_id}: {e}")
            return False, None
    
    async def get_query_result(self, request_id: str) -> Optional[Dict]:
        """Get result of a previously executed query"""
        try:
            result_data = retrieve_swarm_memory(
                key=f"query_result_{request_id}",
                session_id=self.session_id,
                agent_id=self.agent_id
            )
            return result_data
            
        except Exception as e:
            logger.error(f"Failed to get query result for {request_id}: {e}")
            return None
    
    async def begin_transaction(self, database_id: str, isolation_level: str = "READ_COMMITTED") -> Optional[str]:
        """Begin a new database transaction"""
        try:
            if database_id not in self.connections:
                return None
            
            transaction_id = f"txn_{hashlib.md5(f'{database_id}_{time.time()}'.encode()).hexdigest()[:8]}"
            
            transaction = Transaction(
                transaction_id=transaction_id,
                database_id=database_id,
                queries=[],
                isolation_level=isolation_level,
                started_at=datetime.now()
            )
            
            self.active_transactions[transaction_id] = transaction
            
            logger.info(f"Transaction {transaction_id} started for database {database_id}")
            return transaction_id
            
        except Exception as e:
            logger.error(f"Failed to begin transaction for database {database_id}: {e}")
            return None
    
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a database transaction"""
        try:
            if transaction_id not in self.active_transactions:
                return False
            
            transaction = self.active_transactions[transaction_id]
            
            # Simulate transaction commit
            await asyncio.sleep(0.01)
            
            transaction.state = TransactionState.COMMITTED
            transaction.completed_at = datetime.now()
            
            # Remove from active transactions
            del self.active_transactions[transaction_id]
            
            self.metrics["transactions_committed"] += 1
            
            logger.info(f"Transaction {transaction_id} committed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to commit transaction {transaction_id}: {e}")
            return False
    
    async def rollback_transaction(self, transaction_id: str, reason: str = None) -> bool:
        """Rollback a database transaction"""
        try:
            if transaction_id not in self.active_transactions:
                return False
            
            transaction = self.active_transactions[transaction_id]
            
            # Simulate transaction rollback
            await asyncio.sleep(0.01)
            
            transaction.state = TransactionState.ROLLED_BACK
            transaction.completed_at = datetime.now()
            transaction.rollback_reason = reason
            
            # Remove from active transactions
            del self.active_transactions[transaction_id]
            
            self.metrics["transactions_rolled_back"] += 1
            
            logger.info(f"Transaction {transaction_id} rolled back: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback transaction {transaction_id}: {e}")
            return False
    
    async def schedule_backup(self, backup_operation: BackupOperation) -> bool:
        """Schedule a database backup operation"""
        try:
            # Add to backup queue
            await self.backup_queue.put(backup_operation)
            
            # Store in hive memory
            store_swarm_memory(
                key=f"backup_operation_{backup_operation.backup_id}",
                value=asdict(backup_operation),
                session_id=self.session_id,
                agent_id=self.agent_id
            )
            
            logger.info(f"Backup {backup_operation.backup_id} scheduled for database {backup_operation.database_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule backup {backup_operation.backup_id}: {e}")
            return False
    
    async def get_database_stats(self, database_id: str) -> Optional[Dict]:
        """Get database statistics"""
        try:
            if database_id not in self.connections:
                return None
            
            connection = self.connections[database_id]
            
            stats = {
                "database_id": database_id,
                "connection_state": connection.state.value,
                "query_count": connection.query_count,
                "error_count": connection.error_count,
                "last_activity": connection.last_activity.isoformat(),
                "is_healthy": connection.is_healthy(),
                "configuration": asdict(connection.config)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for database {database_id}: {e}")
            return None
    
    async def optimize_database(self, database_id: str) -> bool:
        """Optimize database performance"""
        try:
            if database_id not in self.connections:
                return False
            
            connection = self.connections[database_id]
            
            # Simulate optimization operations
            optimization_queries = [
                "ANALYZE TABLE",
                "OPTIMIZE TABLE", 
                "REINDEX",
                "VACUUM ANALYZE"
            ]
            
            for query in optimization_queries:
                await asyncio.sleep(0.1)  # Simulate optimization time
            
            logger.info(f"Database {database_id} optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize database {database_id}: {e}")
            return False
    
    async def _process_query_queue(self):
        """Process queued database queries"""
        while self.is_running:
            try:
                # Wait for query with timeout
                try:
                    _, timestamp, query_request = await asyncio.wait_for(
                        self.query_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process query with semaphore to limit concurrency
                async with self.query_processing_semaphore:
                    await self._execute_single_query(query_request)
                
            except Exception as e:
                logger.error(f"Error in query queue processing: {e}")
                await asyncio.sleep(1)
    
    async def _execute_single_query(self, query_request: QueryRequest):
        """Execute a single database query"""
        try:
            query_request.started_at = datetime.now()
            
            # Get database connection
            connection = self.connections.get(query_request.database_id)
            if not connection:
                query_request.error_message = f"Database {query_request.database_id} not found"
                return
            
            # Execute query
            success, result = await connection.execute_query(query_request)
            
            query_request.completed_at = datetime.now()
            
            if success:
                self.metrics["queries_executed"] += 1
                
                # Update average query time
                if query_request.execution_time:
                    current_avg = self.metrics["average_query_time"]
                    total_queries = self.metrics["queries_executed"]
                    self.metrics["average_query_time"] = (
                        (current_avg * (total_queries - 1) + query_request.execution_time) / total_queries
                    )
                
                # Track slow queries
                if query_request.execution_time and query_request.execution_time > 1.0:
                    slow_query = {
                        "request_id": query_request.request_id,
                        "database_id": query_request.database_id,
                        "execution_time": query_request.execution_time,
                        "query": query_request.query[:100]  # First 100 chars
                    }
                    self.metrics["slowest_queries"].append(slow_query)
                    # Keep only last 10 slow queries
                    if len(self.metrics["slowest_queries"]) > 10:
                        self.metrics["slowest_queries"] = self.metrics["slowest_queries"][-10:]
                
                # Store result in hive memory
                store_swarm_memory(
                    key=f"query_result_{query_request.request_id}",
                    value={
                        "success": True,
                        "result": result,
                        "query_request": asdict(query_request),
                        "timestamp": datetime.now().isoformat()
                    },
                    session_id=self.session_id,
                    agent_id=self.agent_id
                )
            else:
                self.metrics["queries_failed"] += 1
                
                # Store error result
                store_swarm_memory(
                    key=f"query_result_{query_request.request_id}",
                    value={
                        "success": False,
                        "error": query_request.error_message,
                        "query_request": asdict(query_request),
                        "timestamp": datetime.now().isoformat()
                    },
                    session_id=self.session_id,
                    agent_id=self.agent_id
                )
            
            # Log query completion
            log_swarm_event(
                session_id=self.session_id,
                agent_id=self.agent_id,
                event_type="query_completed",
                event_data={
                    "request_id": query_request.request_id,
                    "database_id": query_request.database_id,
                    "success": success,
                    "execution_time": query_request.execution_time
                }
            )
            
        except Exception as e:
            self.metrics["queries_failed"] += 1
            query_request.error_message = str(e)
            logger.error(f"Query execution error for {query_request.request_id}: {e}")
    
    async def _process_backup_queue(self):
        """Process scheduled database backups"""
        while self.is_running:
            try:
                # Wait for backup operation
                backup_operation = await asyncio.wait_for(
                    self.backup_queue.get(), timeout=5.0
                )
                
                await self._execute_backup(backup_operation)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in backup queue processing: {e}")
                await asyncio.sleep(5)
    
    async def _execute_backup(self, backup_operation: BackupOperation):
        """Execute a database backup operation"""
        try:
            backup_operation.status = BackupStatus.IN_PROGRESS
            backup_operation.started_at = datetime.now()
            
            # Simulate backup process
            backup_time = 5.0 if backup_operation.backup_type == "full" else 2.0
            await asyncio.sleep(backup_time)
            
            # Create backup file (simulated)
            output_path = Path(backup_operation.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            backup_content = f"Database backup for {backup_operation.database_id}\n"
            backup_content += f"Backup ID: {backup_operation.backup_id}\n"
            backup_content += f"Type: {backup_operation.backup_type}\n"
            backup_content += f"Created: {backup_operation.started_at}\n"
            
            with open(output_path, 'w') as f:
                f.write(backup_content)
            
            backup_operation.status = BackupStatus.COMPLETED
            backup_operation.completed_at = datetime.now()
            backup_operation.file_size = len(backup_content)
            
            self.metrics["backup_count"] += 1
            
            # Update in hive memory
            store_swarm_memory(
                key=f"backup_operation_{backup_operation.backup_id}",
                value=asdict(backup_operation),
                session_id=self.session_id,
                agent_id=self.agent_id
            )
            
            logger.info(f"Backup {backup_operation.backup_id} completed successfully")
            
        except Exception as e:
            backup_operation.status = BackupStatus.FAILED
            backup_operation.error_message = str(e)
            backup_operation.completed_at = datetime.now()
            
            logger.error(f"Backup {backup_operation.backup_id} failed: {e}")
    
    async def _monitor_connections(self):
        """Monitor database connection health"""
        while self.is_running:
            try:
                for database_id, connection in self.connections.items():
                    if not connection.is_healthy():
                        logger.warning(f"Connection to {database_id} is unhealthy, attempting reconnection")
                        
                        connection.state = ConnectionState.RECONNECTING
                        if await connection.connect():
                            logger.info(f"Successfully reconnected to {database_id}")
                        else:
                            logger.error(f"Failed to reconnect to {database_id}")
                
                # Sleep for 30 seconds before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_old_data(self):
        """Cleanup old query results and logs"""
        while self.is_running:
            try:
                # This would clean up old query results from memory
                # and perform database maintenance tasks
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in data cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def _update_metrics(self):
        """Update performance metrics periodically"""
        while self.is_running:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Update connection metrics
                self.metrics["connections_active"] = sum(
                    1 for conn in self.connections.values() 
                    if conn.state == ConnectionState.CONNECTED
                )
                
                # Store metrics in hive memory
                store_swarm_memory(
                    key=f"database_manager_metrics_{self.agent_id}",
                    value=self.metrics,
                    session_id=self.session_id,
                    agent_id=self.agent_id
                )
                
                # Log metrics
                log_swarm_event(
                    session_id=self.session_id,
                    agent_id=self.agent_id,
                    event_type="metrics_update",
                    event_data={
                        "active_connections": self.metrics["connections_active"],
                        "query_queue_size": self.query_queue.qsize(),
                        "active_transactions": len(self.active_transactions),
                        "queries_executed": self.metrics["queries_executed"],
                        "average_query_time": self.metrics["average_query_time"]
                    }
                )
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(30)
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "status": "active" if self.is_running else "inactive",
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "databases": list(self.databases.keys()),
            "active_connections": self.metrics["connections_active"],
            "query_queue_size": self.query_queue.qsize(),
            "active_transactions": len(self.active_transactions),
            "metrics": self.metrics,
            "configuration": {
                "max_concurrent_queries": self.max_concurrent_queries,
                "connection_pool_size": self.connection_pool_size
            }
        }


# Utility functions
def create_database_config(
    database_id: str,
    database_type: DatabaseType,
    host: str,
    port: int,
    database_name: str,
    username: str,
    password: str,
    **kwargs
) -> DatabaseConfig:
    """Create a database configuration"""
    return DatabaseConfig(
        database_id=database_id,
        database_type=database_type,
        host=host,
        port=port,
        database_name=database_name,
        username=username,
        password=password,
        **kwargs
    )


def create_query_request(
    request_id: str,
    database_id: str,
    query_type: QueryType,
    query: str,
    parameters: Dict[str, Any] = None,
    **kwargs
) -> QueryRequest:
    """Create a query request"""
    return QueryRequest(
        request_id=request_id,
        database_id=database_id,
        query_type=query_type,
        query=query,
        parameters=parameters or {},
        **kwargs
    )


def create_backup_operation(
    backup_id: str,
    database_id: str,
    backup_type: str,
    output_path: str,
    **kwargs
) -> BackupOperation:
    """Create a backup operation"""
    return BackupOperation(
        backup_id=backup_id,
        database_id=database_id,
        backup_type=backup_type,
        output_path=output_path,
        **kwargs
    )