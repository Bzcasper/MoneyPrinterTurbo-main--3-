"""
Rate limiting middleware for MoneyPrinterTurbo API
"""

import time
from typing import Dict, Optional
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
import hashlib


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with different limits for different endpoint types
    """
    
    def __init__(self, app, redis_client=None):
        super().__init__(app)
        self.redis_client = redis_client
        self.memory_store: Dict[str, Dict] = {}
        
        # Rate limits (requests per minute)
        self.rate_limits = {
            "/videos": 10,      # Video generation - most resource intensive
            "/audios": 20,      # Audio generation
            "/subtitles": 30,   # Subtitle generation
            "/scripts": 50,     # Script generation
            "/terms": 50,       # Terms generation
            "default": 100      # All other endpoints
        }
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and static files
        if request.url.path in ["/health", "/ping", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Get rate limit for this endpoint
        rate_limit = self._get_rate_limit(request.url.path)
        
        # Check rate limit
        if not await self._check_rate_limit(client_id, request.url.path, rate_limit):
            logger.warning(f"Rate limit exceeded for {client_id} on {request.url.path}")
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "Rate limit exceeded",
                    "retry_after": 60,
                    "limit": rate_limit
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(
            await self._get_remaining_requests(client_id, request.url.path, rate_limit)
        )
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier (IP + User-Agent hash for privacy)"""
        ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        # Create a hash for privacy
        client_string = f"{ip}:{user_agent}"
        return hashlib.sha256(client_string.encode()).hexdigest()[:16]
    
    def _get_rate_limit(self, path: str) -> int:
        """Get rate limit for specific endpoint"""
        for endpoint, limit in self.rate_limits.items():
            if endpoint in path:
                return limit
        return self.rate_limits["default"]
    
    async def _check_rate_limit(self, client_id: str, path: str, limit: int) -> bool:
        """Check if client has exceeded rate limit"""
        current_time = int(time.time())
        window_start = current_time - 60  # 1-minute window
        
        if self.redis_client:
            return await self._check_redis_rate_limit(client_id, path, limit, current_time, window_start)
        else:
            return self._check_memory_rate_limit(client_id, path, limit, current_time, window_start)
    
    async def _check_redis_rate_limit(self, client_id: str, path: str, limit: int, current_time: int, window_start: int) -> bool:
        """Redis-based rate limiting"""
        try:
            key = f"rate_limit:{client_id}:{path}"
            
            # Remove old entries
            await self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            current_count = await self.redis_client.zcard(key)
            
            if current_count >= limit:
                return False
            
            # Add current request
            await self.redis_client.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            await self.redis_client.expire(key, 120)  # 2 minutes
            
            return True
            
        except Exception as e:
            logger.warning(f"Redis rate limiting error: {e}, falling back to memory")
            return self._check_memory_rate_limit(client_id, path, limit, current_time, window_start)
    
    def _check_memory_rate_limit(self, client_id: str, path: str, limit: int, current_time: int, window_start: int) -> bool:
        """Memory-based rate limiting"""
        key = f"{client_id}:{path}"
        
        if key not in self.memory_store:
            self.memory_store[key] = {"requests": [], "last_cleanup": current_time}
        
        client_data = self.memory_store[key]
        
        # Cleanup old entries
        if current_time - client_data["last_cleanup"] > 60:
            client_data["requests"] = [req_time for req_time in client_data["requests"] if req_time > window_start]
            client_data["last_cleanup"] = current_time
        
        # Check limit
        if len(client_data["requests"]) >= limit:
            return False
        
        # Add current request
        client_data["requests"].append(current_time)
        
        return True
    
    async def _get_remaining_requests(self, client_id: str, path: str, limit: int) -> int:
        """Get remaining requests for client"""
        current_time = int(time.time())
        window_start = current_time - 60
        
        if self.redis_client:
            try:
                key = f"rate_limit:{client_id}:{path}"
                current_count = await self.redis_client.zcard(key)
                return max(0, limit - current_count)
            except Exception:
                pass
        
        # Fallback to memory
        key = f"{client_id}:{path}"
        if key in self.memory_store:
            current_requests = [req for req in self.memory_store[key]["requests"] if req > window_start]
            return max(0, limit - len(current_requests))
        
        return limit