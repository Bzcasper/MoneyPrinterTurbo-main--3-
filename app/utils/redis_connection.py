"""
Redis Connection Manager with Health Checks and Retry Logic

Provides robust Redis connectivity for MoneyPrinterTurbo MCP service with:
- Automatic connection retry with exponential backoff
- Health monitoring and reconnection
- Environment-based configuration
- Graceful error handling
- Connection pooling support
"""

import os
import asyncio
import time
import signal
from typing import Optional, Dict, Any
import redis
import redis.asyncio as redis_async
from loguru import logger


class RedisConnectionError(Exception):
    """Custom exception for Redis connection issues"""
    pass


class RedisConnectionManager:
    """
    Robust Redis connection manager with health checks and auto-retry.
    
    Features:
    - Automatic retry with exponential backoff
    - Health monitoring
    - Connection pooling
    - Graceful shutdown handling
    - Environment configuration
    """
    
    def __init__(self, 
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 password: Optional[str] = None,
                 db: int = 0,
                 max_retries: int = 10,
                 retry_delay: float = 2.0,
                 max_retry_delay: float = 60.0,
                 health_check_interval: float = 30.0):
        
        # Environment-based configuration with fallbacks
        self.host = host or os.getenv("REDIS_HOST", "redis")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self.password = password or os.getenv("REDIS_PASSWORD", "")
        self.db = db or int(os.getenv("REDIS_DB", 0))
        
        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        self.health_check_interval = health_check_interval
        
        # Connection state
        self.client: Optional[redis.Redis] = None
        self.async_client: Optional[redis_async.Redis] = None
        self.is_connected = False
        self.last_health_check = 0
        self.connection_attempts = 0
        self.shutdown_event = asyncio.Event()
        
        logger.info(f"Redis manager initialized for {self.host}:{self.port}")
    
    def _build_redis_url(self) -> str:
        """Build Redis connection URL from configuration"""
        auth_part = f":{self.password}@" if self.password else ""
        return f"redis://{auth_part}{self.host}:{self.port}/{self.db}"
    
    async def connect(self, force_reconnect: bool = False) -> bool:
        """
        Establish connection to Redis with retry logic.
        
        Args:
            force_reconnect: Force reconnection even if already connected
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.is_connected and not force_reconnect:
            return True
        
        redis_url = self._build_redis_url()
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Attempting Redis connection (attempt {attempt}/{self.max_retries})")
                
                # Create synchronous client
                self.client = redis.Redis.from_url(redis_url, decode_responses=True)
                
                # Create asynchronous client
                self.async_client = redis_async.Redis.from_url(redis_url, decode_responses=True)
                
                # Test connection
                await self._test_connection()
                
                self.is_connected = True
                self.connection_attempts = attempt
                self.last_health_check = time.time()
                
                logger.success(f"✅ Connected to Redis on attempt {attempt}")
                return True
                
            except Exception as e:
                logger.warning(f"❌ Redis connection failed (attempt {attempt}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries:
                    # Calculate exponential backoff delay
                    delay = min(self.retry_delay * (2 ** (attempt - 1)), self.max_retry_delay)
                    logger.info(f"🔁 Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"🛑 Failed to connect to Redis after {self.max_retries} attempts")
                    break
        
        self.is_connected = False
        return False
    
    async def _test_connection(self):
        """Test Redis connection with both sync and async clients"""
        # Test sync client
        if self.client:
            self.client.ping()
        
        # Test async client
        if self.async_client:
            await self.async_client.ping()
    
    async def health_check(self) -> bool:
        """
        Perform health check and reconnect if needed.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        current_time = time.time()
        
        # Skip if recently checked
        if current_time - self.last_health_check < self.health_check_interval:
            return self.is_connected
        
        try:
            if not self.is_connected:
                logger.info("🔍 Redis not connected, attempting reconnection...")
                return await self.connect()
            
            # Test existing connection
            await self._test_connection()
            self.last_health_check = current_time
            logger.debug("✅ Redis health check passed")
            return True
            
        except Exception as e:
            logger.warning(f"❌ Redis health check failed: {e}")
            self.is_connected = False
            
            # Attempt reconnection
            logger.info("🔄 Attempting to reconnect to Redis...")
            return await self.connect(force_reconnect=True)
    
    async def get_client(self) -> redis.Redis:
        """
        Get synchronous Redis client, ensuring connection is healthy.
        
        Returns:
            redis.Redis: Connected Redis client
            
        Raises:
            RedisConnectionError: If connection cannot be established
        """
        if not await self.health_check():
            raise RedisConnectionError("Redis connection is not available")
        
        return self.client
    
    async def get_async_client(self) -> redis_async.Redis:
        """
        Get asynchronous Redis client, ensuring connection is healthy.
        
        Returns:
            redis_async.Redis: Connected async Redis client
            
        Raises:
            RedisConnectionError: If connection cannot be established
        """
        if not await self.health_check():
            raise RedisConnectionError("Redis connection is not available")
        
        return self.async_client
    
    async def execute_command(self, command: str, *args, **kwargs) -> Any:
        """
        Execute Redis command with automatic retry on connection failure.
        
        Args:
            command: Redis command name
            *args: Command arguments
            **kwargs: Command keyword arguments
            
        Returns:
            Any: Command result
        """
        client = await self.get_async_client()
        
        try:
            method = getattr(client, command.lower())
            return await method(*args, **kwargs)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis command failed, attempting reconnection: {e}")
            
            # Try to reconnect and retry once
            if await self.connect(force_reconnect=True):
                client = await self.get_async_client()
                method = getattr(client, command.lower())
                return await method(*args, **kwargs)
            else:
                raise RedisConnectionError(f"Failed to execute Redis command: {command}")
    
    async def set(self, key: str, value: Any, **kwargs) -> bool:
        """Set key-value pair in Redis"""
        return await self.execute_command("set", key, value, **kwargs)
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        return await self.execute_command("get", key)
    
    async def delete(self, *keys) -> int:
        """Delete keys from Redis"""
        return await self.execute_command("delete", *keys)
    
    async def exists(self, *keys) -> int:
        """Check if keys exist in Redis"""
        return await self.execute_command("exists", *keys)
    
    async def expire(self, key: str, time: int) -> bool:
        """Set expiration time for key"""
        return await self.execute_command("expire", key, time)
    
    async def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """Set hash fields"""
        return await self.execute_command("hset", name, mapping=mapping)
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value"""
        return await self.execute_command("hget", name, key)
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields and values"""
        return await self.execute_command("hgetall", name)
    
    async def lpush(self, name: str, *values) -> int:
        """Push values to list"""
        return await self.execute_command("lpush", name, *values)
    
    async def rpop(self, name: str) -> Optional[str]:
        """Pop value from list"""
        return await self.execute_command("rpop", name)
    
    async def close(self):
        """Close Redis connections gracefully"""
        logger.info("🧹 Closing Redis connections...")
        
        try:
            if self.async_client:
                await self.async_client.close()
            
            if self.client:
                self.client.close()
            
            self.is_connected = False
            logger.info("✅ Redis connections closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Redis connections: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            logger.info(f"🛑 Received signal {sig}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)


# Global Redis connection manager instance
_redis_manager: Optional[RedisConnectionManager] = None


async def get_redis_connection() -> RedisConnectionManager:
    """
    Get global Redis connection manager instance.
    
    Returns:
        RedisConnectionManager: Configured and connected Redis manager
    """
    global _redis_manager
    
    if _redis_manager is None:
        _redis_manager = RedisConnectionManager()
        await _redis_manager.connect()
    
    return _redis_manager


async def wait_for_redis(max_wait_time: float = 120.0) -> bool:
    """
    Wait for Redis to become available with timeout.
    
    Args:
        max_wait_time: Maximum time to wait in seconds
        
    Returns:
        bool: True if Redis becomes available, False if timeout
    """
    logger.info(f"🔍 Waiting for Redis to become available (max {max_wait_time}s)...")
    
    start_time = time.time()
    manager = RedisConnectionManager()
    
    while time.time() - start_time < max_wait_time:
        try:
            if await manager.connect():
                logger.success("✅ Redis is now available!")
                await manager.close()
                return True
        except Exception:
            pass
        
        await asyncio.sleep(2)
    
    logger.error(f"❌ Redis did not become available within {max_wait_time} seconds")
    await manager.close()
    return False


# Utility function for MCP server integration
async def ensure_redis_ready():
    """Ensure Redis is ready before starting MCP server"""
    logger.info("🚀 Ensuring Redis is ready for MCP server...")
    
    if not await wait_for_redis():
        raise RedisConnectionError("Redis is not available - cannot start MCP server")
    
    logger.success("✅ Redis is ready for MCP server")


# Export main components
__all__ = [
    "RedisConnectionManager",
    "RedisConnectionError", 
    "get_redis_connection",
    "wait_for_redis",
    "ensure_redis_ready"
]
