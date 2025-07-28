"""
Core MCP Protocol Implementation

Defines the base message types, request/response handling, and protocol validation
for Model Context Protocol communication.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MCPMethodType(str, Enum):
    """Supported MCP method types."""
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    SERVICE_STATUS = "service/status"
    SERVICE_CAPABILITIES = "service/capabilities"
    VIDEO_GENERATE = "video/generate"
    VIDEO_STATUS = "video/status"
    SCRIPT_GENERATE = "script/generate"
    VOICE_SYNTHESIZE = "voice/synthesize"
    BATCH_SUBMIT = "batch/submit"
    BATCH_STATUS = "batch/status"


class MCPMessage(BaseModel):
    """Base MCP message structure."""
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: Optional[Union[str, int]] = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class MCPError(BaseModel):
    """MCP error information."""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None

    # Standard MCP error codes
    PARSE_ERROR: ClassVar[int] = -32700
    INVALID_REQUEST: ClassVar[int] = -32600
    METHOD_NOT_FOUND: ClassVar[int] = -32601
    INVALID_PARAMS: ClassVar[int] = -32602
    INTERNAL_ERROR: ClassVar[int] = -32603

    # Custom error codes
    AUTHENTICATION_FAILED: ClassVar[int] = -32001
    AUTHORIZATION_FAILED: ClassVar[int] = -32002
    RESOURCE_NOT_FOUND: ClassVar[int] = -32003
    RATE_LIMIT_EXCEEDED: ClassVar[int] = -32004
    SERVICE_UNAVAILABLE: ClassVar[int] = -32005


class MCPRequest(MCPMessage):
    """MCP request message."""
    method: MCPMethodType
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MCPResponse(MCPMessage):
    """MCP response message."""
    result: Optional[Any] = None
    error: Optional[MCPError] = None


class MCPCapability(BaseModel):
    """MCP service capability definition."""
    name: str
    version: str
    description: Optional[str] = None
    methods: List[MCPMethodType] = Field(default_factory=list)


class MCPTool(BaseModel):
    """MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    category: Optional[str] = None
    version: str = "1.0.0"


class MCPResource(BaseModel):
    """MCP resource definition."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MCPPrompt(BaseModel):
    """MCP prompt template definition."""
    name: str
    description: str
    template: str
    variables: List[str] = Field(default_factory=list)
    category: Optional[str] = None


class MCPBatchRequest(BaseModel):
    """MCP batch processing request."""
    requests: List[MCPRequest]
    parallel: bool = True
    max_concurrent: int = 5
    timeout: int = 300


class MCPBatchResponse(BaseModel):
    """MCP batch processing response."""
    responses: List[MCPResponse]
    completed: int
    failed: int
    duration: float


def create_error_response(
    request_id: Union[str, int],
    error_code: int,
    error_message: str,
    error_data: Optional[Dict[str, Any]] = None,
) -> MCPResponse:
    """Create a standardized error response."""
    return MCPResponse(
        id=request_id,
        error=MCPError(code=error_code, message=error_message, data=error_data),
    )


def create_success_response(
    request_id: Union[str, int],
    result: Any,
) -> MCPResponse:
    """Create a standardized success response."""
    return MCPResponse(id=request_id, result=result)


def validate_mcp_message(
    data: Dict[str, Any]
) -> Union[MCPRequest, MCPResponse, MCPError]:
    """Validate and parse MCP message from raw data."""
    try:
        if data.get("jsonrpc") != "2.0":
            raise ValueError("Invalid JSON-RPC version")
        if "id" not in data:
            raise ValueError("Missing message ID")

        if "method" in data:
            return MCPRequest(**data)
        if "result" in data or "error" in data:
            return MCPResponse(**data)

        raise ValueError("Invalid message structure")
    except Exception as e:
        return MCPError(
            code=MCPError.PARSE_ERROR,
            message=f"Failed to parse MCP message: {e}",
        )


class MCPProtocolHandler:
    """Protocol handler for MCP message processing."""

    def __init__(self):
        self.capabilities: List[MCPCapability] = []
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}

    def add_capability(self, capability: MCPCapability):
        self.capabilities.append(capability)

    def add_tool(self, tool: MCPTool):
        self.tools[tool.name] = tool

    def add_resource(self, resource: MCPResource):
        self.resources[resource.uri] = resource

    def add_prompt(self, prompt: MCPPrompt):
        self.prompts[prompt.name] = prompt

    def get_capabilities(self) -> List[MCPCapability]:
        return self.capabilities

    def get_tools(self) -> List[MCPTool]:
        return list(self.tools.values())

    def get_resources(self) -> List[MCPResource]:
        return list(self.resources.values())

    def get_prompts(self) -> List[MCPPrompt]:
        return list(self.prompts.values())
