#!/bin/bash
set -e

# Production-ready MCP server startup script with Redis health checks
# This script ensures Redis is available before starting the MCP server

echo "🚀 MoneyPrinterTurbo MCP Server Startup"
echo "========================================"

# Environment variables with defaults
REDIS_HOST=${REDIS_HOST:-redis}
REDIS_PORT=${REDIS_PORT:-6379}
REDIS_TIMEOUT=${REDIS_TIMEOUT:-60}
MCP_HOST=${MCP_HOST:-0.0.0.0}
MCP_PORT=${MCP_PORT:-8081}
PYTHON_PATH=${PYTHONPATH:-/MoneyPrinterTurbo}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if Redis is ready
check_redis() {
    log_info "Checking Redis connection at $REDIS_HOST:$REDIS_PORT..."
    
    # Use the wait-for-redis script if available
    if [ -f "/MoneyPrinterTurbo/scripts/wait-for-redis.sh" ]; then
        log_info "Using wait-for-redis script..."
        if /MoneyPrinterTurbo/scripts/wait-for-redis.sh "$REDIS_HOST" "$REDIS_PORT" "$REDIS_TIMEOUT"; then
            log_success "Redis is ready via wait script"
            return 0
        else
            log_error "Redis not ready after $REDIS_TIMEOUT seconds"
            return 1
        fi
    fi
    
    # Fallback: manual check
    log_info "Manual Redis connectivity check..."
    local attempts=0
    local max_attempts=$((REDIS_TIMEOUT / 2))
    
    while [ $attempts -lt $max_attempts ]; do
        if timeout 2 bash -c "echo > /dev/tcp/$REDIS_HOST/$REDIS_PORT" 2>/dev/null; then
            log_success "Redis TCP connection successful"
            
            # Additional ping test if redis-cli is available
            if command -v redis-cli >/dev/null 2>&1; then
                if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping | grep -q PONG; then
                    log_success "Redis PING successful"
                    return 0
                fi
            else
                log_success "Redis TCP check passed (redis-cli not available)"
                return 0
            fi
        fi
        
        attempts=$((attempts + 1))
        log_info "Redis not ready, attempt $attempts/$max_attempts, sleeping 2s..."
        sleep 2
    done
    
    log_error "Redis connection failed after $max_attempts attempts"
    return 1
}

# Function to run Redis connection tests
test_redis_connection() {
    log_info "Running Redis connection tests..."
    
    if [ -f "/MoneyPrinterTurbo/scripts/test_redis_connection.py" ]; then
        cd /MoneyPrinterTurbo
        if python3 scripts/test_redis_connection.py; then
            log_success "Redis connection tests passed"
            return 0
        else
            log_warning "Redis connection tests failed"
            return 1
        fi
    else
        log_warning "Redis test script not found, skipping tests"
        return 0
    fi
}

# Function to start MCP server
start_mcp_server() {
    log_info "Starting MCP server on $MCP_HOST:$MCP_PORT..."
    
    cd /MoneyPrinterTurbo
    export PYTHONPATH="$PYTHON_PATH"
    export PYTHONUNBUFFERED=1
    
    # Start the MCP server with proper error handling
    python3 -c "
import asyncio
import sys
import signal
from loguru import logger
from app.mcp.server import MCPServer

async def main():
    try:
        server = MCPServer('$MCP_HOST', $MCP_PORT)
        await server.start_server()
    except KeyboardInterrupt:
        logger.info('🛑 MCP server interrupted')
        sys.exit(0)
    except Exception as e:
        logger.error(f'❌ MCP server failed: {e}')
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
"
}

# Function for graceful shutdown
cleanup() {
    log_info "🧹 Shutting down MCP server gracefully..."
    # Send SIGTERM to child processes
    jobs -p | xargs -r kill
    exit 0
}

# Setup signal handlers
trap cleanup SIGTERM SIGINT

# Main startup sequence
main() {
    log_info "Environment:"
    log_info "  REDIS_HOST: $REDIS_HOST"
    log_info "  REDIS_PORT: $REDIS_PORT"
    log_info "  REDIS_TIMEOUT: $REDIS_TIMEOUT"
    log_info "  MCP_HOST: $MCP_HOST"
    log_info "  MCP_PORT: $MCP_PORT"
    log_info "  PYTHONPATH: $PYTHON_PATH"
    
    # Step 1: Check Redis connectivity
    if ! check_redis; then
        log_error "Redis is not available. Cannot start MCP server."
        exit 1
    fi
    
    # Step 2: Run connection tests (optional)
    if [ "${RUN_REDIS_TESTS:-true}" = "true" ]; then
        test_redis_connection || log_warning "Redis tests failed but continuing..."
    fi
    
    # Step 3: Start MCP server
    log_success "Redis is ready. Starting MCP server..."
    start_mcp_server
}

# Run main function
main "$@"
