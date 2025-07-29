# Redis Integration for MoneyPrinterTurbo MCP Service

This document describes the robust Redis integration implemented for the MoneyPrinterTurbo MCP (Model Context Protocol) service.

## 🎯 Overview

The Redis integration provides:
- **Robust Connection Management**: Automatic retry with exponential backoff
- **Health Monitoring**: Continuous health checks and auto-reconnection
- **Graceful Error Handling**: Proper error handling and logging
- **Production Ready**: Docker-optimized with wait-for-redis logic
- **Connection Pooling**: Efficient async/sync Redis client management

## 🏗️ Architecture

### Components

1. **RedisConnectionManager** (`app/utils/redis_connection.py`)
   - Main Redis connection management class
   - Handles retry logic, health checks, and connection pooling
   - Supports both sync and async Redis operations

2. **MCP Server Integration** (`app/mcp/server.py`)
   - Integrated Redis health checks at startup
   - Graceful shutdown handling
   - Background health monitoring

3. **Middleware Integration** (`app/middleware/supabase_middleware.py`)
   - Request-scoped Redis access
   - Error handling for Redis failures
   - Utility functions for route handlers

4. **Docker Configuration** (`app/docker-compose.yml`)
   - Redis service with health checks
   - Dependency management
   - Environment variable configuration

5. **Startup Scripts** (`scripts/`)
   - `wait-for-redis.sh`: Wait for Redis availability
   - `start_mcp_server.sh`: Production startup script
   - `test_redis_connection.py`: Comprehensive connection tests

## 🚀 Quick Start

### 1. Start with Docker Compose

```bash
cd app/
docker-compose up --build
```

The system will:
1. Start Redis with health checks
2. Wait for Redis to be ready
3. Run connection tests
4. Start the MCP server

### 2. Manual Testing

```bash
# Test Redis connection
python3 scripts/test_redis_connection.py

# Check Redis directly
redis-cli -h redis -p 6379 ping
```

## 📋 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `redis` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_PASSWORD` | `""` | Redis password (if required) |
| `REDIS_DB` | `0` | Redis database number |
| `ENABLE_REDIS` | `true` | Enable/disable Redis |
| `REDIS_TIMEOUT` | `60` | Connection timeout in seconds |

### Docker Compose Configuration

```yaml
services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: [
      "redis-server",
      "--maxmemory", "512mb",
      "--maxmemory-policy", "allkeys-lru",
      "--appendonly", "yes"
    ]
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  mcp-server:
    depends_on:
      redis:
        condition: service_healthy
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - ENABLE_REDIS=true
```

## 🔧 Usage Examples

### Basic Redis Operations

```python
from app.utils.redis_connection import get_redis_connection

# Get Redis manager
redis_manager = await get_redis_connection()

# Basic operations
await redis_manager.set("key", "value", ex=3600)
value = await redis_manager.get("key")

# Hash operations
await redis_manager.hset("user:123", {"name": "John", "email": "john@example.com"})
user_data = await redis_manager.hgetall("user:123")

# List operations
await redis_manager.lpush("queue", "task1", "task2")
task = await redis_manager.rpop("queue")
```

### In FastAPI Routes

```python
from fastapi import Request
from app.middleware.supabase_middleware import get_redis_client_from_request

async def my_route(request: Request):
    try:
        redis_client = await get_redis_client_from_request(request)
        
        # Use Redis client
        await redis_client.set("session:user123", "data")
        
    except HTTPException as e:
        # Handle Redis unavailable
        logger.warning("Redis not available, using fallback")
```

### Custom Redis Commands

```python
# Execute any Redis command
result = await redis_manager.execute_command("PING")
result = await redis_manager.execute_command("INFO", "memory")
```

## 🏥 Health Monitoring

### Automatic Health Checks

The system performs automatic health checks:
- **Interval**: Every 30 seconds (configurable)
- **Timeout**: 5 seconds per check
- **Auto-reconnect**: On health check failure
- **Exponential backoff**: For failed reconnection attempts

### Manual Health Check

```python
# Check Redis health
healthy = await redis_manager.health_check()
if not healthy:
    logger.warning("Redis is unhealthy")
```

### Monitoring Endpoints

The MCP server exposes health information:
- `ws://localhost:8081/health` - WebSocket health check
- Service discovery includes Redis status

## 🔄 Error Handling

### Connection Failures

The system handles various failure scenarios:

1. **Redis Unavailable at Startup**
   - Waits up to 60 seconds for Redis
   - Logs detailed error messages
   - Graceful service degradation

2. **Connection Lost During Operation**
   - Automatic reconnection attempts
   - Exponential backoff (2s, 4s, 8s, ...)
   - Maximum retry limit (10 attempts)

3. **Command Execution Errors**
   - Automatic retry for connection errors
   - Detailed error logging
   - Fallback to service degradation

### Error Types

```python
from app.utils.redis_connection import RedisConnectionError

try:
    await redis_manager.set("key", "value")
except RedisConnectionError as e:
    logger.error(f"Redis operation failed: {e}")
    # Handle gracefully
```

## 🧪 Testing

### Comprehensive Test Suite

```bash
# Run all Redis tests
python3 scripts/test_redis_connection.py
```

Tests include:
- Basic connection establishment
- Health check functionality
- Set/Get operations
- Hash operations
- List operations
- Connection recovery
- Global connection manager

### Manual Testing Commands

```bash
# Test basic connectivity
redis-cli -h redis -p 6379 ping

# Test from within container
docker exec moneyprinterturbo-mcp-new redis-cli -h redis ping

# Check Redis info
redis-cli -h redis -p 6379 INFO

# Monitor Redis operations
redis-cli -h redis -p 6379 MONITOR
```

## 🔧 Troubleshooting

### Common Issues

1. **"Connection refused" Error**
   ```bash
   # Check if Redis container is running
   docker ps | grep redis
   
   # Check Redis logs
   docker logs moneyprinterturbo-redis-new
   
   # Test connectivity
   docker exec moneyprinterturbo-mcp-new redis-cli -h redis ping
   ```

2. **"Redis not ready" Message**
   ```bash
   # Increase timeout
   export REDIS_TIMEOUT=120
   
   # Check Redis health
   docker exec moneyprinterturbo-redis-new redis-cli ping
   ```

3. **Performance Issues**
   ```bash
   # Check Redis memory usage
   redis-cli -h redis INFO memory
   
   # Check slow queries
   redis-cli -h redis SLOWLOG get 10
   ```

### Debug Mode

Enable detailed logging:

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG

# Enable Redis command logging
export REDIS_LOG_COMMANDS=true
```

## 📈 Performance Tuning

### Redis Configuration

Optimized Redis settings in docker-compose.yml:
- `maxmemory`: 512MB limit
- `maxmemory-policy`: LRU eviction
- `appendonly`: Persistence enabled
- `tcp-keepalive`: Connection monitoring

### Connection Pooling

The system uses connection pooling:
- Async connections for high-throughput operations
- Sync connections for simple operations
- Automatic connection reuse

### Monitoring Metrics

Key metrics to monitor:
- Connection count
- Memory usage
- Command latency
- Error rates
- Health check status

## 🛠️ Advanced Configuration

### Custom Connection Parameters

```python
# Custom Redis manager
manager = RedisConnectionManager(
    host="custom-redis",
    port=6380,
    password="secret",
    max_retries=15,
    retry_delay=1.0,
    max_retry_delay=30.0,
    health_check_interval=60.0
)
```

### Production Deployment

For production environments:

1. **Use Redis Cluster** for high availability
2. **Enable authentication** with strong passwords
3. **Configure SSL/TLS** for encrypted connections
4. **Set up monitoring** with Redis monitoring tools
5. **Use persistent storage** for data durability

## 📚 API Reference

### RedisConnectionManager

Main class for Redis connection management.

#### Methods

- `connect(force_reconnect=False)` - Establish connection
- `health_check()` - Check connection health
- `get_client()` - Get sync Redis client
- `get_async_client()` - Get async Redis client
- `execute_command(command, *args, **kwargs)` - Execute Redis command
- `set(key, value, **kwargs)` - Set key-value
- `get(key)` - Get value
- `delete(*keys)` - Delete keys
- `hset(name, mapping)` - Set hash
- `hget(name, key)` - Get hash field
- `lpush(name, *values)` - Push to list
- `rpop(name)` - Pop from list
- `close()` - Close connections

### Utility Functions

- `get_redis_connection()` - Get global Redis manager
- `wait_for_redis(max_wait_time)` - Wait for Redis availability
- `ensure_redis_ready()` - Ensure Redis is ready

## 🤝 Contributing

When contributing to Redis integration:

1. **Test thoroughly** with the test suite
2. **Handle errors gracefully** with proper fallbacks
3. **Add logging** for debugging and monitoring
4. **Update documentation** for new features
5. **Follow async/await patterns** for consistency

## 📄 License

This Redis integration is part of the MoneyPrinterTurbo project and follows the same license terms.
