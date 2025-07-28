"""
MCP Server Implementation for MoneyPrinterTurbo

Provides a complete MCP server that exposes video generation capabilities,
handles authentication, manages resources, and coordinates with AI models.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
import websockets
from websockets.server import WebSocketServerProtocol
from loguru import logger
import redis
from fastapi import HTTPException

from .protocol import (
    MCPRequest, MCPResponse, MCPError, MCPMethodType, MCPCapability,
    MCPBatchRequest, MCPBatchResponse, create_success_response, create_error_response,
    validate_mcp_message, MCPProtocolHandler
)
from .tools import MoneyPrinterMCPTools
from .auth import MCPAuthenticator
from .discovery import MCPServiceRegistry
from app.config import config


class MCPConnectionManager:
    """Manages MCP client connections and their state"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.connection_metadata: Dict[str, Dict] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        
    async def add_connection(self, connection_id: str, websocket: WebSocketServerProtocol, 
                           metadata: Optional[Dict] = None):
        """Add a new connection"""
        self.connections[connection_id] = websocket
        self.connection_metadata[connection_id] = metadata or {}
        self.rate_limits[connection_id] = []
        logger.info(f"Added MCP connection: {connection_id}")
        
    async def remove_connection(self, connection_id: str):
        """Remove a connection"""
        if connection_id in self.connections:
            del self.connections[connection_id]
            del self.connection_metadata[connection_id]
            del self.rate_limits[connection_id]
            logger.info(f"Removed MCP connection: {connection_id}")
            
    async def broadcast_message(self, message: MCPResponse, exclude: Optional[List[str]] = None):
        """Broadcast a message to all connected clients"""
        exclude = exclude or []
        message_data = message.model_dump_json()
        
        disconnected = []
        for connection_id, websocket in self.connections.items():
            if connection_id in exclude:
                continue
                
            try:
                await websocket.send(message_data)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(connection_id)
                
        # Clean up disconnected clients
        for connection_id in disconnected:
            await self.remove_connection(connection_id)
            
    def check_rate_limit(self, connection_id: str, max_requests: int = 100, 
                        window_seconds: int = 60) -> bool:
        """Check if connection is within rate limits"""
        now = time.time()
        if connection_id not in self.rate_limits:
            self.rate_limits[connection_id] = []
            
        # Remove old requests outside the window
        self.rate_limits[connection_id] = [
            req_time for req_time in self.rate_limits[connection_id]
            if now - req_time < window_seconds
        ]
        
        # Check if under limit
        if len(self.rate_limits[connection_id]) >= max_requests:
            return False
            
        # Add current request
        self.rate_limits[connection_id].append(now)
        return True


class MCPServer:
    """Main MCP server for MoneyPrinterTurbo"""
    
    def __init__(self, host: str = "localhost", port: int = 8081):
        self.host = host
        self.port = port
        self.connection_manager = MCPConnectionManager()
        self.protocol_handler = MCPProtocolHandler()
        self.tools = MoneyPrinterMCPTools()
        self.authenticator = MCPAuthenticator()
        self.service_registry = MCPServiceRegistry()
        
        # Configuration
        self.max_connections = config.app.get("mcp_max_connections", 100)
        self.rate_limit_requests = config.app.get("mcp_rate_limit_requests", 100)
        self.rate_limit_window = config.app.get("mcp_rate_limit_window", 60)
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.app.get("mcp_circuit_breaker_threshold", 5),
            recovery_timeout=config.app.get("mcp_circuit_breaker_timeout", 60)
        )
        
        # Redis for distributed state (optional)
        self.redis_client = None
        if config.app.get("enable_redis", False):
            try:
                redis_url = f"redis://:{config.app.get('redis_password', '')}@{config.app.get('redis_host', 'redis')}:{config.app.get('redis_port', 6379)}/{config.app.get('redis_db', 0)}"
                self.redis_client = redis.Redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for MCP: {e}")
        
        self._setup_capabilities()
        self._setup_middleware()
        
    def _setup_capabilities(self):
        """Setup server capabilities"""
        video_capability = MCPCapability(
            name="video_generation",
            version="1.0.0",
            description="Video generation and processing capabilities",
            methods=[
                MCPMethodType.VIDEO_GENERATE,
                MCPMethodType.VIDEO_STATUS,
                MCPMethodType.SCRIPT_GENERATE,
                MCPMethodType.VOICE_SYNTHESIZE
            ]
        )
        
        batch_capability = MCPCapability(
            name="batch_processing",
            version="1.0.0",
            description="Batch processing capabilities",
            methods=[
                MCPMethodType.BATCH_SUBMIT,
                MCPMethodType.BATCH_STATUS
            ]
        )
        
        self.protocol_handler.add_capability(video_capability)
        self.protocol_handler.add_capability(batch_capability)
        
        # Add tools to protocol handler
        for tool in self.tools.registry.get_tools():
            self.protocol_handler.add_tool(tool)
            
    def _setup_middleware(self):
        """Setup middleware for request processing"""
        self.tools.registry.add_middleware(self._authentication_middleware)
        self.tools.registry.add_middleware(self._logging_middleware)
        self.tools.registry.add_middleware(self._caching_middleware)
        
    async def _authentication_middleware(self, tool_name: str, params: Dict[str, Any], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Authentication middleware"""
        connection_id = context.get("connection_id")
        if connection_id:
            metadata = self.connection_manager.connection_metadata.get(connection_id, {})
            if not metadata.get("authenticated", False):
                raise HTTPException(status_code=401, detail="Authentication required")
        return params
        
    async def _logging_middleware(self, tool_name: str, params: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Logging middleware"""
        connection_id = context.get("connection_id", "unknown")
        logger.info(f"MCP tool call: {tool_name} from {connection_id}")
        return params
        
    async def _caching_middleware(self, tool_name: str, params: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Caching middleware"""
        if self.redis_client and tool_name in ["generate_video_script", "generate_video_terms"]:
            cache_key = f"mcp_cache:{tool_name}:{hash(json.dumps(params, sort_keys=True))}"
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                context["cached_result"] = json.loads(cached_result)
        return params
        
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        connection_id = f"conn_{int(time.time() * 1000)}"
        
        try:
            # Check connection limits
            if len(self.connection_manager.connections) >= self.max_connections:
                await websocket.close(code=1008, reason="Connection limit exceeded")
                return
                
            await self.connection_manager.add_connection(connection_id, websocket)
            
            # Send welcome message
            welcome_response = create_success_response(
                "welcome",
                {
                    "server": "MoneyPrinterTurbo MCP Server",
                    "version": "1.0.0",
                    "capabilities": [cap.model_dump() for cap in self.protocol_handler.get_capabilities()],
                    "connection_id": connection_id
                }
            )
            await websocket.send(welcome_response.model_dump_json())
            
            # Handle messages
            async for message in websocket:
                await self.handle_message(connection_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"MCP connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"Error handling MCP connection {connection_id}: {str(e)}")
        finally:
            await self.connection_manager.remove_connection(connection_id)
            
    async def handle_message(self, connection_id: str, message: str):
        """Handle incoming MCP message"""
        try:
            # Check rate limits
            if not self.connection_manager.check_rate_limit(
                connection_id, self.rate_limit_requests, self.rate_limit_window
            ):
                error_response = create_error_response(
                    "rate_limit",
                    MCPError.RATE_LIMIT_EXCEEDED,
                    "Rate limit exceeded"
                )
                await self._send_response(connection_id, error_response)
                return
                
            # Parse message
            message_data = json.loads(message)
            parsed_message = validate_mcp_message(message_data)
            
            if isinstance(parsed_message, MCPError):
                error_response = create_error_response(
                    message_data.get("id", "unknown"),
                    parsed_message.code,
                    parsed_message.message
                )
                await self._send_response(connection_id, error_response)
                return
                
            if isinstance(parsed_message, MCPRequest):
                response = await self.handle_request(connection_id, parsed_message)
                await self._send_response(connection_id, response)
                
        except json.JSONDecodeError:
            error_response = create_error_response(
                "parse_error",
                MCPError.PARSE_ERROR,
                "Invalid JSON"
            )
            await self._send_response(connection_id, error_response)
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {str(e)}")
            error_response = create_error_response(
                "internal_error",
                MCPError.INTERNAL_ERROR,
                "Internal server error"
            )
            await self._send_response(connection_id, error_response)
            
    async def handle_request(self, connection_id: str, request: MCPRequest) -> MCPResponse:
        """Handle MCP request"""
        try:
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                return create_error_response(
                    request.id,
                    MCPError.SERVICE_UNAVAILABLE,
                    "Service temporarily unavailable"
                )
                
            context = {
                "connection_id": connection_id,
                "request_id": request.id,
                "timestamp": datetime.utcnow()
            }
            
            # Handle different method types
            if request.method == MCPMethodType.TOOLS_LIST:
                result = [tool.model_dump() for tool in self.tools.registry.get_tools()]
                
            elif request.method == MCPMethodType.TOOLS_CALL:
                tool_name = request.params.get("name")
                tool_params = request.params.get("arguments", {})
                
                if not tool_name:
                    return create_error_response(
                        request.id,
                        MCPError.INVALID_PARAMS,
                        "Tool name is required"
                    )
                    
                # Check for cached result
                if context.get("cached_result"):
                    result = context["cached_result"]
                else:
                    result = await self.tools.registry.call_tool(tool_name, tool_params, context)
                    
                    # Cache result if Redis is available
                    if self.redis_client and tool_name in ["generate_video_script", "generate_video_terms"]:
                        cache_key = f"mcp_cache:{tool_name}:{hash(json.dumps(tool_params, sort_keys=True))}"
                        self.redis_client.setex(cache_key, 3600, json.dumps(result))  # Cache for 1 hour
                        
            elif request.method == MCPMethodType.SERVICE_CAPABILITIES:
                result = {
                    "capabilities": [cap.model_dump() for cap in self.protocol_handler.get_capabilities()],
                    "tools": [tool.model_dump() for tool in self.protocol_handler.get_tools()],
                    "resources": [resource.model_dump() for resource in self.protocol_handler.get_resources()],
                    "prompts": [prompt.model_dump() for prompt in self.protocol_handler.get_prompts()]
                }
                
            elif request.method == MCPMethodType.SERVICE_STATUS:
                result = {
                    "status": "healthy",
                    "uptime": time.time() - self.start_time,
                    "connections": len(self.connection_manager.connections),
                    "circuit_breaker": {
                        "state": self.circuit_breaker.state,
                        "failure_count": self.circuit_breaker.failure_count
                    }
                }
                
            elif request.method == MCPMethodType.BATCH_SUBMIT:
                batch_request = MCPBatchRequest(**request.params)
                result = await self.handle_batch_request(connection_id, batch_request)
                
            else:
                return create_error_response(
                    request.id,
                    MCPError.METHOD_NOT_FOUND,
                    f"Method '{request.method}' not found"
                )
                
            self.circuit_breaker.record_success()
            return create_success_response(request.id, result)
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Error handling request {request.id}: {str(e)}")
            return create_error_response(
                request.id,
                MCPError.INTERNAL_ERROR,
                f"Internal error: {str(e)}"
            )
            
    async def handle_batch_request(self, connection_id: str, batch_request: MCPBatchRequest) -> Dict[str, Any]:
        """Handle batch processing request"""
        start_time = time.time()
        responses = []
        
        if batch_request.parallel:
            # Process requests in parallel
            semaphore = asyncio.Semaphore(batch_request.max_concurrent)
            
            async def process_request(req):
                async with semaphore:
                    return await self.handle_request(connection_id, req)
                    
            tasks = [process_request(req) for req in batch_request.requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process requests sequentially
            for req in batch_request.requests:
                response = await self.handle_request(connection_id, req)
                responses.append(response)
                
        completed = sum(1 for r in responses if not isinstance(r, Exception) and not r.error)
        failed = len(responses) - completed
        duration = time.time() - start_time
        
        return {
            "batch_id": f"batch_{int(time.time() * 1000)}",
            "responses": [r.model_dump() if hasattr(r, 'model_dump') else str(r) for r in responses],
            "completed": completed,
            "failed": failed,
            "duration": duration
        }
        
    async def _send_response(self, connection_id: str, response: MCPResponse):
        """Send response to client"""
        if connection_id in self.connection_manager.connections:
            websocket = self.connection_manager.connections[connection_id]
            try:
                await websocket.send(response.model_dump_json())
            except websockets.exceptions.ConnectionClosed:
                await self.connection_manager.remove_connection(connection_id)
                
    async def start_server(self):
        """Start the MCP server"""
        self.start_time = time.time()
        logger.info(f"Starting MCP server on {self.host}:{self.port}")
        
        # Register with service discovery
        await self.service_registry.register_service(
            "moneyprinter_mcp",
            f"ws://{self.host}:{self.port}",
            {
                "capabilities": [cap.model_dump() for cap in self.protocol_handler.get_capabilities()],
                "tools": len(self.tools.registry.get_tools()),
                "version": "1.0.0"
            }
        )
        
        async with websockets.serve(self.handle_connection, self.host, self.port):
            logger.info(f"MCP server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever
            
    async def stop_server(self):
        """Stop the MCP server"""
        logger.info("Stopping MCP server")
        await self.service_registry.unregister_service("moneyprinter_mcp")


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True
            
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"
        
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"