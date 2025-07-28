"""
MCP Client Implementation

Provides Python client SDK for connecting to MCP servers, with connection pooling,
retry logic, caching, and comprehensive error handling.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
import websockets
from websockets.client import WebSocketClientProtocol
from loguru import logger
import aiohttp
from dataclasses import dataclass
from enum import Enum

from .protocol import (
    MCPRequest, MCPResponse, MCPError, MCPMethodType, MCPTool,
    create_success_response, create_error_response, validate_mcp_message
)
from .auth import MCPAuthenticator


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class MCPClientConfig:
    """Configuration for MCP client"""
    server_url: str
    auth_type: str = "api_key"  # api_key, jwt, hmac
    api_key: Optional[str] = None
    jwt_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    
    # Connection settings
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_timeout: float = 10.0
    heartbeat_interval: float = 30.0
    
    # Pool settings
    max_connections: int = 5
    idle_timeout: float = 300.0
    
    # Cache settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour


class MCPConnection:
    """Represents a single MCP connection"""
    
    def __init__(self, config: MCPClientConfig):
        self.config = config
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.connection_id: Optional[str] = None
        self.state = ConnectionState.DISCONNECTED
        self.last_activity = time.time()
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Heartbeat task
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_handler_task: Optional[asyncio.Task] = None
        
    async def connect(self) -> bool:
        """Establish connection to MCP server"""
        try:
            self.state = ConnectionState.CONNECTING
            logger.info(f"Connecting to MCP server: {self.config.server_url}")
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.config.server_url,
                timeout=self.config.connection_timeout
            )
            
            # Start message handler
            self._message_handler_task = asyncio.create_task(self._handle_messages())
            
            # Wait for welcome message
            welcome_response = await asyncio.wait_for(
                self._wait_for_welcome(),
                timeout=self.config.connection_timeout
            )
            
            if welcome_response:
                self.connection_id = welcome_response.get("result", {}).get("connection_id")
                self.state = ConnectionState.CONNECTED
                logger.info(f"Connected to MCP server with ID: {self.connection_id}")
                
                # Authenticate if required
                if await self._authenticate():
                    # Start heartbeat
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    return True
                else:
                    await self.disconnect()
                    return False
            else:
                await self.disconnect()
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            self.state = ConnectionState.FAILED
            await self.disconnect()
            return False
            
    async def disconnect(self):
        """Disconnect from MCP server"""
        self.state = ConnectionState.DISCONNECTED
        
        # Cancel tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._message_handler_task:
            self._message_handler_task.cancel()
            
        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            
        # Reject pending requests
        for future in self.pending_requests.values():
            if not future.done():
                future.set_exception(ConnectionError("Connection closed"))
        self.pending_requests.clear()
        
        logger.info("Disconnected from MCP server")
        
    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """Send request and wait for response"""
        if self.state != ConnectionState.CONNECTED:
            raise ConnectionError("Not connected to MCP server")
            
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request.id] = future
        
        try:
            # Send request
            message = request.model_dump_json()
            await self.websocket.send(message)
            self.last_activity = time.time()
            
            # Wait for response
            response = await asyncio.wait_for(
                future,
                timeout=self.config.connection_timeout
            )
            
            return response
            
        except asyncio.TimeoutError:
            # Remove from pending
            if request.id in self.pending_requests:
                del self.pending_requests[request.id]
            raise TimeoutError(f"Request {request.id} timed out")
            
        except Exception as e:
            # Remove from pending
            if request.id in self.pending_requests:
                del self.pending_requests[request.id]
            raise
            
    async def _wait_for_welcome(self) -> Optional[MCPResponse]:
        """Wait for welcome message from server"""
        try:
            message = await self.websocket.recv()
            data = json.loads(message)
            response = validate_mcp_message(data)
            
            if isinstance(response, MCPResponse) and response.id == "welcome":
                return response
            return None
            
        except Exception as e:
            logger.error(f"Error waiting for welcome: {str(e)}")
            return None
            
    async def _authenticate(self) -> bool:
        """Authenticate with the server"""
        try:
            auth_data = {"type": self.config.auth_type}
            
            if self.config.auth_type == "api_key":
                auth_data["api_key"] = self.config.api_key
            elif self.config.auth_type == "jwt":
                auth_data["token"] = self.config.jwt_token
            elif self.config.auth_type == "hmac":
                timestamp = str(int(time.time()))
                auth_data.update({
                    "client_id": self.config.client_id,
                    "timestamp": timestamp,
                    "signature": self._generate_hmac_signature(timestamp)
                })
                
            # Send authentication request
            auth_request = MCPRequest(
                method="auth/authenticate",
                params=auth_data
            )
            
            response = await self.send_request(auth_request)
            
            if response.error:
                logger.error(f"Authentication failed: {response.error.message}")
                return False
                
            logger.info("Successfully authenticated with MCP server")
            return True
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
            
    def _generate_hmac_signature(self, timestamp: str) -> str:
        """Generate HMAC signature for authentication"""
        import hmac
        import hashlib
        
        message = f"{self.config.client_id}{timestamp}"
        signature = hmac.new(
            self.config.client_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
        
    async def _handle_messages(self):
        """Handle incoming messages from server"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    parsed_message = validate_mcp_message(data)
                    
                    if isinstance(parsed_message, MCPResponse):
                        # Handle response
                        request_id = parsed_message.id
                        if request_id in self.pending_requests:
                            future = self.pending_requests.pop(request_id)
                            if not future.done():
                                future.set_result(parsed_message)
                        else:
                            # Unsolicited response or notification
                            await self._handle_notification(parsed_message)
                            
                    elif isinstance(parsed_message, MCPRequest):
                        # Handle server-initiated request (rare)
                        await self._handle_server_request(parsed_message)
                        
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON message")
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in message handler: {str(e)}")
            
    async def _handle_notification(self, response: MCPResponse):
        """Handle notifications from server"""
        event_type = "notification"
        handlers = self.event_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                await handler(response)
            except Exception as e:
                logger.error(f"Error in notification handler: {str(e)}")
                
    async def _handle_server_request(self, request: MCPRequest):
        """Handle requests from server"""
        # For now, just log
        logger.info(f"Received server request: {request.method}")
        
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                # Send ping
                ping_request = MCPRequest(method="ping")
                await self.send_request(ping_request)
                
            except Exception as e:
                logger.warning(f"Heartbeat failed: {str(e)}")
                break
                
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        
    @property
    def is_connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED
        
    @property
    def is_idle(self) -> bool:
        return time.time() - self.last_activity > self.config.idle_timeout


class MCPConnectionPool:
    """Connection pool for MCP connections"""
    
    def __init__(self, config: MCPClientConfig):
        self.config = config
        self.connections: List[MCPConnection] = []
        self.available_connections: asyncio.Queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        
    async def get_connection(self) -> MCPConnection:
        """Get an available connection from the pool"""
        # Try to get from available connections
        try:
            connection = self.available_connections.get_nowait()
            if connection.is_connected:
                return connection
        except asyncio.QueueEmpty:
            pass
            
        # Create new connection if under limit
        async with self.lock:
            if len(self.connections) < self.config.max_connections:
                connection = MCPConnection(self.config)
                if await connection.connect():
                    self.connections.append(connection)
                    return connection
                    
        # Wait for available connection
        connection = await self.available_connections.get()
        return connection
        
    async def return_connection(self, connection: MCPConnection):
        """Return connection to the pool"""
        if connection.is_connected and not connection.is_idle:
            await self.available_connections.put(connection)
        else:
            # Remove from pool
            async with self.lock:
                if connection in self.connections:
                    self.connections.remove(connection)
            await connection.disconnect()
            
    async def close_all(self):
        """Close all connections in the pool"""
        async with self.lock:
            for connection in self.connections:
                await connection.disconnect()
            self.connections.clear()
            
        # Clear queue
        while not self.available_connections.empty():
            try:
                self.available_connections.get_nowait()
            except asyncio.QueueEmpty:
                break


class MCPClient:
    """High-level MCP client with caching and retry logic"""
    
    def __init__(self, config: MCPClientConfig):
        self.config = config
        self.connection_pool = MCPConnectionPool(config)
        self.cache: Dict[str, Dict] = {}
        self.available_tools: List[MCPTool] = []
        
    async def connect(self):
        """Initialize client"""
        # Get initial connection to load capabilities
        connection = await self.connection_pool.get_connection()
        try:
            # Load available tools
            await self._load_tools(connection)
        finally:
            await self.connection_pool.return_connection(connection)
            
    async def disconnect(self):
        """Disconnect client"""
        await self.connection_pool.close_all()
        
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any],
                       use_cache: bool = True) -> Dict[str, Any]:
        """Call an MCP tool"""
        # Check cache first
        if use_cache and self.config.enable_caching:
            cache_key = self._get_cache_key(tool_name, parameters)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
                
        # Get connection and make request
        connection = await self.connection_pool.get_connection()
        
        try:
            request = MCPRequest(
                method=MCPMethodType.TOOLS_CALL,
                params={
                    "name": tool_name,
                    "arguments": parameters
                }
            )
            
            response = await self._send_with_retry(connection, request)
            
            if response.error:
                raise Exception(f"Tool call failed: {response.error.message}")
                
            result = response.result
            
            # Cache result
            if use_cache and self.config.enable_caching:
                cache_key = self._get_cache_key(tool_name, parameters)
                self._cache_result(cache_key, result)
                
            return result
            
        finally:
            await self.connection_pool.return_connection(connection)
            
    async def list_tools(self) -> List[MCPTool]:
        """List available tools"""
        if self.available_tools:
            return self.available_tools
            
        connection = await self.connection_pool.get_connection()
        try:
            await self._load_tools(connection)
            return self.available_tools
        finally:
            await self.connection_pool.return_connection(connection)
            
    async def get_service_status(self) -> Dict[str, Any]:
        """Get server status"""
        connection = await self.connection_pool.get_connection()
        try:
            request = MCPRequest(method=MCPMethodType.SERVICE_STATUS)
            response = await self._send_with_retry(connection, request)
            
            if response.error:
                raise Exception(f"Status check failed: {response.error.message}")
                
            return response.result
            
        finally:
            await self.connection_pool.return_connection(connection)
            
    async def _load_tools(self, connection: MCPConnection):
        """Load available tools from server"""
        request = MCPRequest(method=MCPMethodType.TOOLS_LIST)
        response = await self._send_with_retry(connection, request)
        
        if response.error:
            logger.error(f"Failed to load tools: {response.error.message}")
            return
            
        self.available_tools = [
            MCPTool(**tool_data) for tool_data in response.result
        ]
        
        logger.info(f"Loaded {len(self.available_tools)} MCP tools")
        
    async def _send_with_retry(self, connection: MCPConnection, 
                              request: MCPRequest) -> MCPResponse:
        """Send request with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    await asyncio.sleep(self.config.retry_delay * attempt)
                    
                return await connection.send_request(request)
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                
                # Try to reconnect if connection failed
                if not connection.is_connected:
                    await connection.connect()
                    
        raise last_exception
        
    def _get_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        import hashlib
        key_data = f"{tool_name}:{json.dumps(parameters, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if still valid"""
        if cache_key not in self.cache:
            return None
            
        cached_data = self.cache[cache_key]
        if time.time() - cached_data["timestamp"] > self.config.cache_ttl:
            del self.cache[cache_key]
            return None
            
        return cached_data["result"]
        
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result"""
        self.cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        # Simple cache cleanup - remove oldest entries if cache gets too large
        if len(self.cache) > 1000:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]


# Convenience functions

async def create_mcp_client(server_url: str, api_key: str, **kwargs) -> MCPClient:
    """Create and connect MCP client with API key authentication"""
    config = MCPClientConfig(
        server_url=server_url,
        auth_type="api_key",
        api_key=api_key,
        **kwargs
    )
    
    client = MCPClient(config)
    await client.connect()
    return client


async def create_mcp_client_jwt(server_url: str, jwt_token: str, **kwargs) -> MCPClient:
    """Create and connect MCP client with JWT authentication"""
    config = MCPClientConfig(
        server_url=server_url,
        auth_type="jwt",
        jwt_token=jwt_token,
        **kwargs
    )
    
    client = MCPClient(config)
    await client.connect()
    return client