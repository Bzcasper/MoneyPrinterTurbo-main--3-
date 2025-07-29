"""
Supabase Middleware for MoneyPrinterTurbo

Provides comprehensive Supabase integration including:
- Database connection management
- Authentication handling
- Request/response logging
- Error handling with Supabase context
- Connection health monitoring
"""

import asyncio
import time
from typing import Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from app.database.connection import get_supabase_connection
from app.models.exception import DatabaseConnectionError
from app.mcp.supabase_tools import supabase_mcp_tools
from app.utils.redis_connection import get_redis_connection, RedisConnectionError


class SupabaseMiddleware(BaseHTTPMiddleware):
    """
    Middleware to ensure all routes have access to Supabase functionality.
    
    Features:
    - Automatic connection management
    - Request-scoped database access
    - Authentication validation
    - Error handling with proper logging
    - Health monitoring
    """
    
    def __init__(self, app, enable_auth: bool = True, enable_logging: bool = True, enable_redis: bool = True):
        super().__init__(app)
        self.enable_auth = enable_auth
        self.enable_logging = enable_logging
        self.enable_redis = enable_redis
        self.connection_pool = {}
        self.health_check_interval = 60  # seconds
        self.last_health_check = 0
        
    async def dispatch(self, request: Request, call_next):
        """Process each request with Supabase context."""
        request_id = self._generate_request_id()
        start_time = time.time()
        
        # Add request ID for tracking
        request.state.request_id = request_id
        
        try:
            # Skip Supabase for health checks and static routes
            if self._should_skip_supabase(request):
                return await call_next(request)
            
            # Ensure Supabase connection is available
            await self._ensure_supabase_connection(request)
            
            # Ensure Redis connection is available (if enabled)
            if self.enable_redis:
                await self._ensure_redis_connection(request)
            
            # Validate authentication if enabled
            if self.enable_auth and self._requires_auth(request):
                auth_result = await self._validate_authentication(request)
                if not auth_result["valid"]:
                    return JSONResponse(
                        status_code=401,
                        content={
                            "status": 401,
                            "message": auth_result["error"],
                            "request_id": request_id,
                            "path": str(request.url)
                        }
                    )
                request.state.user = auth_result.get("user")
            
            # Log request if enabled
            if self.enable_logging:
                await self._log_request(request, request_id)
            
            # Process the request
            response = await call_next(request)
            
            # Log response if enabled
            if self.enable_logging:
                await self._log_response(request, response, request_id, start_time)
            
            # Add Supabase context headers
            self._add_context_headers(response, request_id)
            
            return response
            
        except SupabaseConnectionError as e:
            logger.error(f"Supabase connection error for request {request_id}: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": 503,
                    "message": "Database service unavailable",
                    "request_id": request_id,
                    "path": str(request.url)
                }
            )
        except RedisConnectionError as e:
            logger.error(f"Redis connection error for request {request_id}: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": 503,
                    "message": "Cache service unavailable",
                    "request_id": request_id,
                    "path": str(request.url)
                }
            )
        except Exception as e:
            logger.error(f"Middleware error for request {request_id}: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": 500,
                    "message": "Internal server error",
                    "request_id": request_id,
                    "path": str(request.url)
                }
            )
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return f"req_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
    def _should_skip_supabase(self, request: Request) -> bool:
        """Determine if request should skip Supabase processing."""
        skip_paths = ["/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"]
        path = str(request.url.path)
        
        # Skip health checks and docs
        if any(path.startswith(skip_path) for skip_path in skip_paths):
            return True
        
        # Skip static files
        if path.startswith("/static/") or path.endswith((".css", ".js", ".png", ".jpg", ".ico")):
            return True
            
        return False
    
    async def _ensure_supabase_connection(self, request: Request):
        """Ensure Supabase connection is available for the request."""
        try:
            connection = get_supabase_connection()
            
            # Perform health check if needed
            current_time = time.time()
            if current_time - self.last_health_check > self.health_check_interval:
                if not connection.is_connected:
                    await connection.connect()
                await connection._health_check()
                self.last_health_check = current_time
            
            # Make connection available in request state
            request.state.supabase = connection
            request.state.supabase_client = connection.client if connection.is_connected else None
            
            # Make MCP tools available
            request.state.mcp_tools = supabase_mcp_tools
            
        except Exception as e:
            logger.error(f"Failed to ensure Supabase connection: {str(e)}")
            raise SupabaseConnectionError(f"Connection setup failed: {str(e)}")
    
    async def _ensure_redis_connection(self, request: Request):
        """Ensure Redis connection is available for the request."""
        try:
            redis_manager = await get_redis_connection()
            
            # Perform health check
            if not await redis_manager.health_check():
                raise RedisConnectionError("Redis health check failed")
            
            # Make Redis available in request state
            request.state.redis = redis_manager
            request.state.redis_client = await redis_manager.get_async_client()
            
        except Exception as e:
            logger.error(f"Failed to ensure Redis connection: {str(e)}")
            if self.enable_redis:
                raise RedisConnectionError(f"Redis connection setup failed: {str(e)}")
            else:
                logger.warning("Redis connection failed but not required")
    
    def _requires_auth(self, request: Request) -> bool:
        """Determine if route requires authentication."""
        # Define routes that don't require authentication
        public_routes = [
            "/",
            "/health",
            "/v1/mcp/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
        
        path = str(request.url.path)
        
        # Skip auth for public routes
        if path in public_routes:
            return False
        
        # Skip auth for health endpoints
        if "health" in path.lower():
            return False
            
        # All other routes require auth by default
        return True
    
    async def _validate_authentication(self, request: Request) -> Dict[str, Any]:
        """Validate request authentication."""
        try:
            # Check for Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                return {"valid": False, "error": "Missing Authorization header"}
            
            # Extract token
            if not auth_header.startswith("Bearer "):
                return {"valid": False, "error": "Invalid Authorization header format"}
            
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            # Validate token with Supabase
            if hasattr(request.state, 'supabase_client') and request.state.supabase_client:
                try:
                    # Verify JWT token
                    user_response = request.state.supabase_client.auth.get_user(token)
                    if user_response.user:
                        return {
                            "valid": True,
                            "user": {
                                "id": user_response.user.id,
                                "email": user_response.user.email,
                                "metadata": user_response.user.user_metadata
                            }
                        }
                except Exception as e:
                    logger.warning(f"Token validation failed: {str(e)}")
                    return {"valid": False, "error": "Invalid or expired token"}
            
            # Fallback: Accept any properly formatted token for development
            if len(token) > 20:  # Basic token length check
                return {
                    "valid": True,
                    "user": {"id": "dev_user", "email": "dev@example.com"}
                }
            
            return {"valid": False, "error": "Invalid token"}
            
        except Exception as e:
            logger.error(f"Authentication validation error: {str(e)}")
            return {"valid": False, "error": "Authentication service error"}
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request details."""
        try:
            log_data = {
                "request_id": request_id,
                "method": request.method,
                "path": str(request.url.path),
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
                "timestamp": time.time()
            }
            
            # Log to Supabase if connection is available
            if hasattr(request.state, 'supabase') and request.state.supabase.is_connected:
                try:
                    await request.state.supabase.execute_query(
                        table="request_logs",
                        operation="insert",
                        data=log_data
                    )
                except Exception as e:
                    logger.warning(f"Failed to log request to Supabase: {str(e)}")
            
            # Always log to file/console
            logger.info(f"Request {request_id}: {request.method} {request.url.path}")
            
        except Exception as e:
            logger.error(f"Request logging failed: {str(e)}")
    
    async def _log_response(self, request: Request, response: Response, 
                          request_id: str, start_time: float):
        """Log response details."""
        try:
            duration = time.time() - start_time
            
            log_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
                "response_size": len(response.body) if hasattr(response, 'body') else 0,
                "timestamp": time.time()
            }
            
            # Log to Supabase if connection is available
            if hasattr(request.state, 'supabase') and request.state.supabase.is_connected:
                try:
                    await request.state.supabase.execute_query(
                        table="response_logs",
                        operation="insert",
                        data=log_data
                    )
                except Exception as e:
                    logger.warning(f"Failed to log response to Supabase: {str(e)}")
            
            # Always log to file/console
            logger.info(f"Response {request_id}: {response.status_code} ({duration:.3f}s)")
            
        except Exception as e:
            logger.error(f"Response logging failed: {str(e)}")
    
    def _add_context_headers(self, response: Response, request_id: str):
        """Add context headers to response."""
        try:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Supabase-Enabled"] = "true"
            response.headers["X-MoneyPrinter-Version"] = "2.0.0"
        except Exception as e:
            logger.warning(f"Failed to add context headers: {str(e)}")


# Utility functions for route handlers
async def get_supabase_from_request(request: Request):
    """Get Supabase connection from request state."""
    if hasattr(request.state, 'supabase'):
        return request.state.supabase
    raise HTTPException(status_code=503, detail="Supabase connection not available")


async def get_supabase_client_from_request(request: Request):
    """Get Supabase client from request state."""
    if hasattr(request.state, 'supabase_client') and request.state.supabase_client:
        return request.state.supabase_client
    raise HTTPException(status_code=503, detail="Supabase client not available")


async def get_mcp_tools_from_request(request: Request):
    """Get MCP tools from request state."""
    if hasattr(request.state, 'mcp_tools'):
        return request.state.mcp_tools
    raise HTTPException(status_code=503, detail="MCP tools not available")


async def get_redis_from_request(request: Request):
    """Get Redis manager from request state."""
    if hasattr(request.state, 'redis'):
        return request.state.redis
    raise HTTPException(status_code=503, detail="Redis connection not available")


async def get_redis_client_from_request(request: Request):
    """Get Redis client from request state."""
    if hasattr(request.state, 'redis_client') and request.state.redis_client:
        return request.state.redis_client
    raise HTTPException(status_code=503, detail="Redis client not available")


def get_user_from_request(request: Request) -> Optional[Dict[str, Any]]:
    """Get authenticated user from request state."""
    return getattr(request.state, 'user', None)


# Export middleware and utilities
__all__ = [
    "SupabaseMiddleware",
    "get_supabase_from_request",
    "get_supabase_client_from_request", 
    "get_mcp_tools_from_request",
    "get_redis_from_request",
    "get_redis_client_from_request",
    "get_user_from_request"
]