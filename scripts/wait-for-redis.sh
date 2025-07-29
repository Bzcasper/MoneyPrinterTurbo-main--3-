#!/bin/bash
set -e

host="$1"
port="${2:-6379}"
timeout="${3:-30}"

echo "🔍 Waiting for Redis at $host:$port (timeout: ${timeout}s)..."

start_time=$(date +%s)

while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -ge $timeout ]; then
        echo "❌ Timeout waiting for Redis after ${timeout} seconds"
        exit 1
    fi
    
    if timeout 1 bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null; then
        echo "✅ Redis is accepting connections at $host:$port"
        
        # Additional Redis ping test if redis-cli is available
        if command -v redis-cli >/dev/null 2>&1; then
            if redis-cli -h "$host" -p "$port" ping | grep -q PONG; then
                echo "✅ Redis PING successful"
                break
            else
                echo "⚠️ TCP connection OK but Redis PING failed, retrying..."
            fi
        else
            echo "✅ TCP connection successful (redis-cli not available for PING test)"
            break
        fi
    fi
    
    echo "🔁 Redis not ready, sleeping 2 seconds... (${elapsed}/${timeout}s)"
    sleep 2
done

echo "✅ Redis is ready!"
