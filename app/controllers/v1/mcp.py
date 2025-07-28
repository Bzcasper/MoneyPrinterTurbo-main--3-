"""
MCP Controller for FastAPI Integration

Provides REST API endpoints that bridge to MCP protocol, allowing HTTP clients
to interact with MCP services through familiar REST endpoints.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from fastapi import Request, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from app.controllers.v1.base import new_router
from app.models.schema import BaseResponse
from app.utils import utils
from app.mcp.server import MCPServer
from app.mcp.client import MCPClient, MCPClientConfig
from app.mcp.tools import mcp_tools
from app.mcp.protocol import MCPRequest, MCPMethodType, MCPBatchRequest
from app.config import config


# Request/Response models for REST API
class MCPToolCallRequest(BaseModel):
    """Request model for calling MCP tools via REST"""
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    use_cache: bool = Field(default=True, description="Whether to use caching")
    timeout: Optional[int] = Field(default=30, description="Request timeout in seconds")


class MCPToolCallResponse(BaseResponse):
    """Response model for MCP tool calls"""
    class MCPToolCallData(BaseModel):
        tool_name: str
        result: Any
        execution_time: Optional[float] = None
        cached: bool = False
        
    data: MCPToolCallData


class MCPBatchRequest(BaseModel):
    """Request model for batch MCP operations"""
    requests: List[MCPToolCallRequest] = Field(..., description="List of tool calls")
    max_concurrent: int = Field(default=5, description="Maximum concurrent executions")
    fail_fast: bool = Field(default=False, description="Stop on first failure")


class MCPBatchResponse(BaseResponse):
    """Response model for batch MCP operations"""
    class MCPBatchData(BaseModel):
        results: List[Dict[str, Any]]
        completed: int
        failed: int
        execution_time: float
        
    data: MCPBatchData


class MCPServiceStatusResponse(BaseResponse):
    """Response model for MCP service status"""
    class MCPServiceStatusData(BaseModel):
        server_status: str
        tools_available: int
        active_connections: int
        uptime: float
        circuit_breaker_state: str
        
    data: MCPServiceStatusData


class MCPToolListResponse(BaseResponse):
    """Response model for MCP tool listing"""
    class MCPToolListData(BaseModel):
        tools: List[Dict[str, Any]]
        categories: List[str]
        total_count: int
        
    data: MCPToolListData


# Router setup
router = new_router()

# Global MCP server instance (will be initialized on startup)
mcp_server: Optional[MCPServer] = None


async def get_mcp_server() -> MCPServer:
    """Dependency to get the MCP server instance"""
    global mcp_server
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP server not available")
    return mcp_server


@router.post("/tools/call", response_model=MCPToolCallResponse, summary="Call an MCP tool")
async def call_mcp_tool(
    request: Request,
    body: MCPToolCallRequest,
    background_tasks: BackgroundTasks,
    server: MCPServer = Depends(get_mcp_server)
):
    """Call an MCP tool via REST API"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Create MCP request
        mcp_request = MCPRequest(
            method=MCPMethodType.TOOLS_CALL,
            params={
                "name": body.tool_name,
                "arguments": body.parameters
            }
        )
        
        # Execute tool call
        context = {
            "connection_id": f"rest_{utils.get_uuid()}",
            "request_id": mcp_request.id,
            "use_cache": body.use_cache
        }
        
        result = await server.tools.registry.call_tool(
            body.tool_name,
            body.parameters,
            context
        )
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        response_data = MCPToolCallResponse.MCPToolCallData(
            tool_name=body.tool_name,
            result=result,
            execution_time=execution_time,
            cached=context.get("cached_result") is not None
        )
        
        return utils.get_response(200, response_data.model_dump())
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except TimeoutError as e:
        raise HTTPException(status_code=408, detail=f"Request timeout: {str(e)}")
    except Exception as e:
        logger.error(f"Error calling MCP tool {body.tool_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


@router.post("/tools/batch", response_model=MCPBatchResponse, summary="Execute multiple MCP tools")
async def batch_call_mcp_tools(
    request: Request,
    body: MCPBatchRequest,
    background_tasks: BackgroundTasks,
    server: MCPServer = Depends(get_mcp_server)
):
    """Execute multiple MCP tools in batch"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        results = []
        completed = 0
        failed = 0
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(body.max_concurrent)
        
        async def execute_single_tool(tool_request: MCPToolCallRequest) -> Dict[str, Any]:
            async with semaphore:
                try:
                    context = {
                        "connection_id": f"batch_{utils.get_uuid()}",
                        "request_id": utils.get_uuid(),
                        "use_cache": tool_request.use_cache
                    }
                    
                    result = await server.tools.registry.call_tool(
                        tool_request.tool_name,
                        tool_request.parameters,
                        context
                    )
                    
                    return {
                        "tool_name": tool_request.tool_name,
                        "status": "success",
                        "result": result,
                        "cached": context.get("cached_result") is not None
                    }
                    
                except Exception as e:
                    return {
                        "tool_name": tool_request.tool_name,
                        "status": "error",
                        "error": str(e)
                    }
        
        # Execute all tools
        if body.fail_fast:
            # Execute sequentially and stop on first failure
            for tool_request in body.requests:
                result = await execute_single_tool(tool_request)
                results.append(result)
                
                if result["status"] == "success":
                    completed += 1
                else:
                    failed += 1
                    break
        else:
            # Execute in parallel
            tasks = [execute_single_tool(tool_request) for tool_request in body.requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({
                        "status": "error",
                        "error": str(result)
                    })
                    failed += 1
                else:
                    processed_results.append(result)
                    if result["status"] == "success":
                        completed += 1
                    else:
                        failed += 1
            
            results = processed_results
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        response_data = MCPBatchResponse.MCPBatchData(
            results=results,
            completed=completed,
            failed=failed,
            execution_time=execution_time
        )
        
        return utils.get_response(200, response_data.model_dump())
        
    except Exception as e:
        logger.error(f"Error in batch MCP tool execution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch execution failed: {str(e)}")


@router.get("/tools", response_model=MCPToolListResponse, summary="List available MCP tools")
async def list_mcp_tools(
    request: Request,
    category: Optional[str] = None,
    server: MCPServer = Depends(get_mcp_server)
):
    """List all available MCP tools"""
    try:
        tools = server.tools.registry.get_tools()
        
        # Filter by category if specified
        if category:
            tools = [tool for tool in tools if tool.category == category]
        
        # Get unique categories
        categories = list(set(tool.category or "general" for tool in server.tools.registry.get_tools()))
        categories.sort()
        
        response_data = MCPToolListResponse.MCPToolListData(
            tools=[tool.model_dump() for tool in tools],
            categories=categories,
            total_count=len(tools)
        )
        
        return utils.get_response(200, response_data.model_dump())
        
    except Exception as e:
        logger.error(f"Error listing MCP tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@router.get("/tools/{tool_name}", summary="Get MCP tool details")
async def get_mcp_tool(
    request: Request,
    tool_name: str,
    server: MCPServer = Depends(get_mcp_server)
):
    """Get details for a specific MCP tool"""
    try:
        tools = server.tools.registry.get_tools()
        tool = next((t for t in tools if t.name == tool_name), None)
        
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        
        return utils.get_response(200, tool.model_dump())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting MCP tool {tool_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tool details: {str(e)}")


@router.get("/status", response_model=MCPServiceStatusResponse, summary="Get MCP service status")
async def get_mcp_status(
    request: Request,
    server: MCPServer = Depends(get_mcp_server)
):
    """Get MCP service status and health information"""
    try:
        # Get server status
        uptime = asyncio.get_event_loop().time() - server.start_time if hasattr(server, 'start_time') else 0
        
        response_data = MCPServiceStatusResponse.MCPServiceStatusData(
            server_status="healthy",
            tools_available=len(server.tools.registry.get_tools()),
            active_connections=len(server.connection_manager.connections),
            uptime=uptime,
            circuit_breaker_state=server.circuit_breaker.state
        )
        
        return utils.get_response(200, response_data.model_dump())
        
    except Exception as e:
        logger.error(f"Error getting MCP status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/health", summary="MCP health check endpoint")
async def mcp_health_check(request: Request):
    """Simple health check endpoint for MCP service"""
    try:
        global mcp_server
        if not mcp_server:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "MCP server not initialized"}
            )
        
        return JSONResponse(
            status_code=200,
            content={"status": "healthy", "timestamp": utils.get_timestamp()}
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@router.websocket("/ws")
async def mcp_websocket_endpoint(websocket):
    """WebSocket endpoint for direct MCP protocol communication"""
    global mcp_server
    if not mcp_server:
        await websocket.close(code=1011, reason="MCP server not available")
        return
    
    # Delegate to MCP server's WebSocket handler
    await mcp_server.handle_connection(websocket, "/ws")


# Utility functions for integration

async def initialize_mcp_server():
    """Initialize the global MCP server instance"""
    global mcp_server
    
    try:
        import os
        
        # Check if we're running in a container environment where MCP server is already running
        if os.getenv("ENVIRONMENT") == "production" or os.path.exists("/.dockerenv"):
            logger.info("Detected container environment - MCP server already running, creating client-only instance")
            
            # Create MCP server instance without starting the WebSocket server
            # This allows the REST API endpoints to work while the actual server runs in a separate container
            host = config.app.get("mcp_server_host", "localhost")
            port = config.app.get("mcp_server_port", 8081)
            
            mcp_server = MCPServer(host=host, port=port)
            
            # Initialize components without starting the WebSocket server
            mcp_server.start_time = time.time()
            logger.info(f"MCP server client initialized (server running separately on {host}:{port})")
            
        else:
            # Development environment - start embedded MCP server
            host = config.app.get("mcp_server_host", "localhost")
            port = config.app.get("mcp_server_port", 8081)
            
            mcp_server = MCPServer(host=host, port=port)
            
            # Start server in background
            asyncio.create_task(mcp_server.start_server())
            
            logger.info(f"MCP server initialized and started on {host}:{port}")
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP server: {str(e)}")
        raise


async def shutdown_mcp_server():
    """Shutdown the global MCP server instance"""
    global mcp_server
    
    if mcp_server:
        try:
            await mcp_server.stop_server()
            logger.info("MCP server shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down MCP server: {str(e)}")


# Convenience functions for internal use

async def call_mcp_tool_internal(tool_name: str, parameters: Dict[str, Any]) -> Any:
    """Internal function to call MCP tools from other parts of the application"""
    global mcp_server
    
    if not mcp_server:
        raise RuntimeError("MCP server not available")
    
    try:
        context = {
            "connection_id": f"internal_{utils.get_uuid()}",
            "request_id": utils.get_uuid()
        }
        
        result = await mcp_server.tools.registry.call_tool(
            tool_name,
            parameters,
            context
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error calling MCP tool {tool_name} internally: {str(e)}")
        raise


def create_mcp_client_for_service(service_url: str) -> MCPClient:
    """Create MCP client for connecting to other MCP services"""
    config_dict = MCPClientConfig(
        server_url=service_url,
        auth_type="api_key",
        api_key=config.app.get("mcp_api_key", "default_admin"),
        max_retries=3,
        connection_timeout=10.0
    )
    
    return MCPClient(config_dict)