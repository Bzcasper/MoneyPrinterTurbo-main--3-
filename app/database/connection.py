"""
Supabase Database Connection Manager

This module handles connections to Supabase PostgreSQL database and provides
a unified interface for database operations.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

import asyncpg
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

from ..models.exception import DatabaseConnectionError


@dataclass
class ConnectionConfig:
    """Database connection configuration."""
    supabase_url: str
    supabase_key: str
    supabase_service_key: Optional[str] = None
    database_url: Optional[str] = None
    pool_size: int = 10
    timeout: int = 30


class SupabaseConnectionError(Exception):
    """Custom exception for Supabase connection errors."""
    pass


class SupabaseConnection:
    """
    Manages Supabase database connections and operations.
    
    Provides both REST API client and direct PostgreSQL connections.
    """
    
    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._load_config()
        self.client: Optional[Client] = None
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.is_connected = False
        
    def _load_config(self) -> ConnectionConfig:
        """Load configuration from environment variables."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        database_url = os.getenv("SUPABASE_DATABASE_URL")
        
        if not supabase_url or not supabase_key:
            raise SupabaseConnectionError(
                "SUPABASE_URL and SUPABASE_ANON_KEY environment variables are required"
            )
            
        return ConnectionConfig(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            supabase_service_key=supabase_service_key,
            database_url=database_url
        )
    
    async def connect(self, use_service_key: bool = False) -> bool:
        """
        Establish connection to Supabase.
        
        Args:
            use_service_key: Whether to use service role key for admin operations
            
        Returns:
            True if connection successful
        """
        try:
            # Create Supabase client
            key = (self.config.supabase_service_key if use_service_key and self.config.supabase_service_key 
                   else self.config.supabase_key)
            
            self.client = create_client(
                self.config.supabase_url,
                key,
                options=ClientOptions(
                    auto_refresh_token=True,
                    persist_session=True
                )
            )
            
            # Test connection with a simple query
            result = self.client.table("_supabase_migrations").select("*").limit(1).execute()
            
            # Create PostgreSQL pool if database URL is available
            if self.config.database_url:
                try:
                    self.pg_pool = await asyncpg.create_pool(
                        self.config.database_url,
                        min_size=1,
                        max_size=self.config.pool_size,
                        command_timeout=self.config.timeout
                    )
                    self.logger.info("PostgreSQL connection pool created")
                except Exception as e:
                    self.logger.warning(f"Failed to create PostgreSQL pool: {str(e)}")
            
            self.is_connected = True
            self.logger.info("Successfully connected to Supabase")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Supabase: {str(e)}")
            raise SupabaseConnectionError(f"Connection failed: {str(e)}")
    
    async def disconnect(self):
        """Close all connections."""
        try:
            if self.pg_pool:
                await self.pg_pool.close()
                self.pg_pool = None
            
            self.client = None
            self.is_connected = False
            self.logger.info("Disconnected from Supabase")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {str(e)}")
    
    async def execute_sql(self, sql: str, params: Optional[List] = None) -> Dict[str, Any]:
        """
        Execute SQL command directly using PostgreSQL connection.
        
        Args:
            sql: SQL command to execute
            params: Optional parameters for the query
            
        Returns:
            Query result
        """
        if not self.pg_pool:
            raise SupabaseConnectionError("PostgreSQL connection not available")
        
        try:
            async with self.pg_pool.acquire() as conn:
                if params:
                    result = await conn.fetch(sql, *params)
                else:
                    result = await conn.fetch(sql)
                
                return {
                    "data": [dict(row) for row in result],
                    "count": len(result),
                    "status": "success"
                }
                
        except Exception as e:
            self.logger.error(f"SQL execution failed: {str(e)}")
            raise SupabaseConnectionError(f"SQL execution failed: {str(e)}")
    
    async def execute_rpc(self, function_name: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute Supabase RPC function.
        
        Args:
            function_name: Name of the RPC function
            params: Function parameters
            
        Returns:
            Function result
        """
        if not self.client:
            raise SupabaseConnectionError("Supabase client not connected")
        
        try:
            result = self.client.rpc(function_name, params or {}).execute()
            return {
                "data": result.data,
                "count": result.count,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"RPC execution failed: {str(e)}")
            raise SupabaseConnectionError(f"RPC execution failed: {str(e)}")
    
    def get_table(self, table_name: str):
        """
        Get a table reference for operations.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table reference
        """
        if not self.client:
            raise SupabaseConnectionError("Supabase client not connected")
        
        return self.client.table(table_name)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the connection.
        
        Returns:
            Health status information
        """
        status = {
            "connected": self.is_connected,
            "client_available": self.client is not None,
            "pg_pool_available": self.pg_pool is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.is_connected and self.client:
            try:
                # Test with a simple query
                result = self.client.from_("information_schema.tables").select("table_name").limit(1).execute()
                status["api_responsive"] = True
                status["test_query_success"] = True
            except Exception as e:
                status["api_responsive"] = False
                status["test_query_success"] = False
                status["error"] = str(e)
        
        if self.pg_pool:
            try:
                async with self.pg_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                status["pg_connection_healthy"] = True
            except Exception as e:
                status["pg_connection_healthy"] = False
                status["pg_error"] = str(e)
        
        return status


# Global connection instance
_connection_instance: Optional[SupabaseConnection] = None


def get_supabase_connection() -> SupabaseConnection:
    """
    Get the global Supabase connection instance.
    
    Returns:
        SupabaseConnection instance
    """
    global _connection_instance
    
    if _connection_instance is None:
        _connection_instance = SupabaseConnection()
    
    return _connection_instance


async def ensure_connection(use_service_key: bool = False) -> SupabaseConnection:
    """
    Ensure Supabase connection is established.
    
    Args:
        use_service_key: Whether to use service role key
        
    Returns:
        Connected SupabaseConnection instance
    """
    conn = get_supabase_connection()
    
    if not conn.is_connected:
        await conn.connect(use_service_key=use_service_key)
    
    return conn


async def close_connection():
    """Close the global connection."""
    global _connection_instance
    
    if _connection_instance:
        await _connection_instance.disconnect()
        _connection_instance = None