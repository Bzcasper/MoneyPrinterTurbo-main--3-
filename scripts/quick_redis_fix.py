#!/usr/bin/env python3
"""
Quick Redis Fix for MoneyPrinterTurbo

This script addresses the Redis connection issues by:
1. Testing different Redis host configurations
2. Providing environment-specific recommendations
3. Validating the setup for both local and Docker environments
"""

import os
import sys
import socket
import subprocess
from pathlib import Path

def log_info(msg):
    print(f"ℹ️  {msg}")

def log_success(msg):
    print(f"✅ {msg}")

def log_warning(msg):
    print(f"⚠️  {msg}")

def log_error(msg):
    print(f"❌ {msg}")

def check_redis_port(host="localhost", port=6379):
    """Check if Redis port is accessible"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def check_redis_ping(host="localhost", port=6379):
    """Check Redis with PING command"""
    try:
        result = subprocess.run(
            ["redis-cli", "-h", host, "-p", str(port), "ping"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and "PONG" in result.stdout
    except Exception:
        return False

def check_docker_redis():
    """Check if Redis is running in Docker"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=redis"],
            capture_output=True, text=True, timeout=10
        )
        return "redis" in result.stdout
    except Exception:
        return False

def fix_redis_connection():
    """Main fix function"""
    log_info("🔧 MoneyPrinterTurbo Redis Quick Fix")
    print("=" * 40)
    
    # Check current directory
    if not Path("app").exists():
        log_error("Run this script from the MoneyPrinterTurbo root directory")
        return False
    
    # Test different Redis configurations
    redis_configs = [
        ("localhost", 6379),
        ("127.0.0.1", 6379),
        ("redis", 6379)  # Docker service name
    ]
    
    working_configs = []
    
    for host, port in redis_configs:
        log_info(f"Testing Redis at {host}:{port}...")
        
        # Check port accessibility
        if check_redis_port(host, port):
            log_success(f"Port {port} is accessible on {host}")
            
            # Check Redis PING
            if check_redis_ping(host, port):
                log_success(f"Redis PING successful on {host}:{port}")
                working_configs.append((host, port))
            else:
                log_warning(f"Port accessible but Redis PING failed on {host}:{port}")
        else:
            log_warning(f"Port {port} not accessible on {host}")
    
    # Check Docker Redis
    if check_docker_redis():
        log_success("Redis Docker container is running")
    else:
        log_info("No Redis Docker container found")
    
    # Provide recommendations
    print("\n🎯 Recommendations:")
    print("=" * 20)
    
    if working_configs:
        log_success("Redis is accessible! Working configurations:")
        for host, port in working_configs:
            print(f"   - {host}:{port}")
        
        # Set environment for tests
        best_host = working_configs[0][0]
        os.environ["REDIS_HOST"] = best_host
        os.environ["REDIS_PORT"] = str(working_configs[0][1])
        
        log_success(f"Environment set to use {best_host}:{working_configs[0][1]}")
        
        # Test Python connection
        try:
            log_info("Testing Python Redis connection...")
            sys.path.insert(0, str(Path("app").resolve()))
            
            from app.utils.redis_connection import RedisConnectionManager
            
            import asyncio
            async def test_connection():
                manager = RedisConnectionManager(host=best_host, port=working_configs[0][1])
                connected = await manager.connect()
                if connected:
                    await manager.set("test_key", "test_value", ex=10)
                    value = await manager.get("test_key")
                    await manager.delete("test_key")
                    await manager.close()
                    return value == "test_value"
                return False
            
            if asyncio.run(test_connection()):
                log_success("Python Redis connection test passed!")
            else:
                log_warning("Python Redis connection test failed")
                
        except Exception as e:
            log_warning(f"Python Redis test error: {e}")
    
    else:
        log_warning("No working Redis configurations found")
        print("\n💡 To fix this:")
        print("1. Start Redis locally:")
        print("   docker run -d --name redis -p 6379:6379 redis:7-alpine")
        print("2. Or use the full Docker Compose setup:")
        print("   cd app/ && docker-compose up --build")
        print("3. Or install Redis locally:")
        print("   Ubuntu: sudo apt install redis-server")
        print("   macOS: brew install redis")
        
    # Update test script for local environment
    if working_configs:
        log_info("Updating test script configuration...")
        
        # Create a local config file
        config_content = f"""# Redis configuration for local testing
export REDIS_HOST={working_configs[0][0]}
export REDIS_PORT={working_configs[0][1]}
export REDIS_DB=0
"""
        
        with open("scripts/redis_local_config.sh", "w") as f:
            f.write(config_content)
        
        log_success("Created scripts/redis_local_config.sh")
        print(f"   Run: source scripts/redis_local_config.sh")
    
    print("\n🚀 Next Steps:")
    print("=" * 15)
    if working_configs:
        print("✅ Redis is working! You can now:")
        print("   1. Run tests: python3 scripts/test_redis_connection.py")
        print("   2. Start Docker: cd app/ && docker-compose up --build")
        print("   3. The MCP service should connect successfully")
    else:
        print("🔧 Start Redis first, then run this script again")
    
    return len(working_configs) > 0

if __name__ == "__main__":
    try:
        success = fix_redis_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        log_info("Fix interrupted")
        sys.exit(1)
    except Exception as e:
        log_error(f"Fix failed: {e}")
        sys.exit(1)
