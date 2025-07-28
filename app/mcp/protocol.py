"""
Core MCP Protocol Implementation

Defines the base message types, request/response handling, and protocol validation
for Model Context Protocol communication.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class MCPMethodType(str, Enum):
    """Supported MCP method types"""
    # Tool methods
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    
    # Resource methods
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    
    # Prompt methods
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    
    # Service methods
    SERVICE_STATUS = "service/status"
    SERVICE_CAPABILITIES = "service/capabilities"
    
    # Video generation methods
    VIDEO_GENERATE = "video/generate"
    VIDEO_STATUS = "video/status"
    SCRIPT_GENERATE = "script/generate"
    VOICE_SYNTHESIZE = "voice/synthesize"
    
    # Batch processing
    BATCH_SUBMIT = "batch/submit"
    BATCH_STATUS = "batch/status"


class MCPMessage(BaseModel):
    """Base MCP message structure"""
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: Optional[Union[str, int]] = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MCPRequest(MCPMessage):
    """MCP request message"""
    method: MCPMethodType = Field(..., description="Method to call")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    
class MCPResponse(MCPMessage):
    """MCP response message"""
    result: Optional[Any] = None
    error: Optional['MCPError'] = None
    
    
class MCPError(BaseModel):
    """MCP error information"""
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    data: Optional[Dict[str, Any]] = None
    
    # Standard MCP error codes
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Custom error codes
    AUTHENTICATION_FAILED = -32001
    AUTHORIZATION_FAILED = -32002
    RESOURCE_NOT_FOUND = -32003
    RATE_LIMIT_EXCEEDED = -32004
    SERVICE_UNAVAILABLE = -32005


class MCPCapability(BaseModel):
    """MCP service capability definition"""
    name: str = Field(..., description="Capability name")
    version: str = Field(..., description="Capability version")
    description: Optional[str] = None
    methods: List[MCPMethodType] = Field(default_factory=list)
    

class MCPTool(BaseModel):
    """MCP tool definition"""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    input_schema: Dict[str, Any] = Field(..., description="JSON schema for input parameters")
    output_schema: Optional[Dict[str, Any]] = None
    category: Optional[str] = None
    version: str = "1.0.0"
    

class MCPResource(BaseModel):
    """MCP resource definition"""
    uri: str = Field(..., description="Resource URI")
    name: str = Field(..., description="Resource name")
    description: Optional[str] = None
    mime_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    

class MCPPrompt(BaseModel):
    """MCP prompt template definition"""
    name: str = Field(..., description="Prompt name")
    description: str = Field(..., description="Prompt description")
    template: str = Field(..., description="Prompt template")
    variables: List[str] = Field(default_factory=list)
    category: Optional[str] = None


class MCPBatchRequest(BaseModel):
    """MCP batch processing request"""
    requests: List[MCPRequest] = Field(..., description="List of requests to process")
    parallel: bool = Field(default=True, description="Process requests in parallel")
    max_concurrent: int = Field(default=5, description="Maximum concurrent requests")
    timeout: int = Field(default=300, description="Timeout in seconds")
    

class MCPBatchResponse(BaseModel):
    """MCP batch processing response"""
    responses: List[MCPResponse] = Field(..., description="List of responses")
    completed: int = Field(..., description="Number of completed requests")
    failed: int = Field(..., description="Number of failed requests")
    duration: float = Field(..., description="Processing duration in seconds")


def create_error_response(request_id: Union[str, int], error_code: int, 
                         error_message: str, error_data: Optional[Dict] = None) -> MCPResponse:
    """Create a standardized error response"""
    return MCPResponse(
        id=request_id,
        error=MCPError(
            code=error_code,
            message=error_message,
            data=error_data
        )
    )


def create_success_response(request_id: Union[str, int], result: Any) -> MCPResponse:
    """Create a standardized success response"""
    return MCPResponse(
        id=request_id,
        result=result
    )


def validate_mcp_message(data: Dict[str, Any]) -> Union[MCPRequest, MCPResponse, MCPError]:
    """Validate and parse MCP message from raw data"""
    try:
        # Check for required fields
        if "jsonrpc" not in data or data["jsonrpc"] != "2.0":
            raise ValueError("Invalid JSON-RPC version")
            
        if "id" not in data:
            raise ValueError("Missing message ID")
            
        # Determine message type
        if "method" in data:
            return MCPRequest(**data)
        elif "result" in data or "error" in data:
            return MCPResponse(**data)
        else:
            raise ValueError("Invalid message structure")
            
    except Exception as e:
        return MCPError(
            code=MCPError.PARSE_ERROR,
            message=f"Failed to parse MCP message: {str(e)}"
        )


class MCPProtocolHandler:
    """Protocol handler for MCP message processing"""
    
    def __init__(self):
        self.capabilities: List[MCPCapability] = []
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        
    def add_capability(self, capability: MCPCapability):
        """Add a capability to the handler"""
        self.capabilities.append(capability)
        
    def add_tool(self, tool: MCPTool):
        """Add a tool to the handler"""
        self.tools[tool.name] = tool
        
    def add_resource(self, resource: MCPResource):
        """Add a resource to the handler"""
        self.resources[resource.uri] = resource
        
    def add_prompt(self, prompt: MCPPrompt):
        """Add a prompt template to the handler"""
        self.prompts[prompt.name] = prompt
        
    def get_capabilities(self) -> List[MCPCapability]:
        """Get all registered capabilities"""
        return self.capabilities
        
    def get_tools(self) -> List[MCPTool]:
        """Get all registered tools"""
        return list(self.tools.values())
        
    def get_resources(self) -> List[MCPResource]:
        """Get all registered resources"""
        return list(self.resources.values())
        
    def get_prompts(self) -> List[MCPPrompt]:
        """Get all registered prompts"""
        return list(self.prompts.values())