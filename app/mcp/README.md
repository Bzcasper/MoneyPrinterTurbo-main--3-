# MoneyPrinterTurbo MCP Integration

This module provides a complete Model Context Protocol (MCP) implementation for MoneyPrinterTurbo, enabling advanced AI model communication and coordination for video generation workflows.

## Features

### 🚀 Core MCP Implementation
- **Full MCP Protocol Support**: Complete implementation of the Model Context Protocol specification
- **WebSocket Communication**: Real-time bidirectional communication between clients and servers
- **JSON-RPC 2.0**: Standards-compliant message format with proper error handling
- **Request/Response Patterns**: Synchronous and asynchronous operation support

### 🔧 Video Generation Tools
- **Script Generation**: AI-powered video script creation with customizable parameters
- **Voice Synthesis**: Text-to-speech conversion with multiple voice options
- **Video Creation**: Complete video generation pipeline integration
- **Search Term Generation**: Intelligent keyword extraction for video materials
- **Batch Processing**: Efficient handling of multiple concurrent operations

### 🔐 Security & Authentication
- **Multiple Auth Methods**: API keys, JWT tokens, and HMAC signatures
- **Role-Based Access Control**: Granular permissions for different user types
- **Rate Limiting**: Per-user and per-connection request throttling
- **Connection Security**: Secure WebSocket connections with authentication

### 🌐 Service Discovery & Load Balancing
- **Service Registry**: Automatic service registration and discovery
- **Load Balancing**: Multiple strategies (round-robin, least connections, weighted)
- **Health Checking**: Automatic service health monitoring
- **Circuit Breaker**: Fault tolerance with automatic recovery

### 📊 Monitoring & Analytics
- **Real-time Metrics**: Performance, usage, and health metrics collection
- **Alert System**: Configurable thresholds with multi-channel notifications
- **Dashboard Integration**: Ready-to-use monitoring dashboard data
- **Distributed Tracing**: Request tracking across service boundaries

### 💻 Client SDKs
- **Python Client**: Full-featured async client with connection pooling
- **TypeScript Client**: Browser and Node.js compatible client library
- **Connection Management**: Automatic reconnection and retry logic
- **Caching**: Intelligent response caching for improved performance

## Quick Start

### 1. Server Setup

```python
from app.mcp.server import MCPServer

# Initialize MCP server
server = MCPServer(host="localhost", port=8081)

# Start server
await server.start_server()
```

### 2. Client Connection

```python
from app.mcp.client import create_mcp_client

# Connect with API key
client = await create_mcp_client(
    server_url="ws://localhost:8081",
    api_key="your_api_key"
)

# List available tools
tools = await client.list_tools()
print(f"Available tools: {[tool.name for tool in tools]}")
```

### 3. Tool Usage

```python
# Generate video script
result = await client.call_tool(
    "generate_video_script",
    {
        "video_subject": "AI in Healthcare",
        "language": "en",
        "paragraph_number": 2
    }
)

print(f"Generated script: {result['script']}")
```

## Available MCP Tools

### Content Generation Tools

#### `generate_video_script`
Generates video scripts based on subject and parameters.

**Parameters:**
- `video_subject` (string, required): The subject/topic for the video
- `language` (string, optional): Language for the script
- `paragraph_number` (integer, optional): Number of paragraphs (1-10)

**Returns:**
- `script` (string): The generated script content
- `word_count` (integer): Number of words in the script
- `estimated_duration` (number): Estimated reading time in seconds

#### `generate_video_terms`
Generates search terms for finding relevant video materials.

**Parameters:**
- `video_subject` (string, required): The video subject
- `video_script` (string, required): The video script content
- `amount` (integer, optional): Number of terms to generate (1-20)

**Returns:**
- `terms` (array): Array of search term strings

### Video Generation Tools

#### `create_video`
Creates a complete video with script, voice, and visual elements.

**Parameters:**
- `video_subject` (string, required): The video subject/topic
- `video_script` (string, optional): Pre-written script
- `video_aspect` (string, optional): Video aspect ratio ("16:9", "9:16", "1:1")
- `voice_name` (string, optional): Voice to use for narration
- `bgm_type` (string, optional): Background music type
- `subtitle_enabled` (boolean, optional): Enable subtitles

**Returns:**
- `task_id` (string): Unique task identifier
- `status` (string): Current task status
- `estimated_completion` (string): Estimated completion time

#### `synthesize_voice`
Converts text to speech using various voice options.

**Parameters:**
- `text` (string, required): Text to convert to speech
- `voice_name` (string, required): Voice to use for synthesis
- `voice_rate` (number, optional): Speech rate (0.5-2.0)
- `voice_volume` (number, optional): Voice volume (0.0-1.0)

**Returns:**
- `audio_file` (string): Path to generated audio file
- `duration` (number): Audio duration in seconds
- `format` (string): Audio format

### Batch Processing Tools

#### `batch_video_generation`
Generates multiple videos from a list of subjects.

**Parameters:**
- `subjects` (array, required): List of video subjects
- `template_params` (object, optional): Common parameters for all videos
- `max_concurrent` (integer, optional): Maximum concurrent generations (1-10)

**Returns:**
- `batch_id` (string): Unique batch identifier
- `total_videos` (integer): Number of videos to generate
- `estimated_completion` (string): Estimated completion time

### Analysis Tools

#### `analyze_video_content`
Analyzes video content and provides insights.

**Parameters:**
- `video_uri` (string, required): URI of the video to analyze
- `analysis_type` (string, optional): Type of analysis ("quality", "content", "performance", "full")

**Returns:**
- `analysis_results` (object): Detailed analysis results
- `recommendations` (array): List of improvement recommendations
- `quality_score` (number): Overall quality score (0-10)

### Monitoring Tools

#### `get_generation_status`
Gets the status of video generation tasks.

**Parameters:**
- `task_id` (string, required): Task ID to check status for

**Returns:**
- `status` (string): Current task status
- `progress` (number): Completion progress (0.0-1.0)
- `estimated_remaining` (string): Estimated remaining time
- `result_urls` (array): URLs to completed results

## REST API Integration

The MCP server also provides REST API endpoints for HTTP-based access:

### Tool Execution
```http
POST /v1/mcp/tools/call
Content-Type: application/json

{
  "tool_name": "generate_video_script",
  "parameters": {
    "video_subject": "Space exploration",
    "paragraph_number": 2
  },
  "use_cache": true
}
```

### Batch Processing
```http
POST /v1/mcp/tools/batch
Content-Type: application/json

{
  "requests": [
    {
      "tool_name": "generate_video_script",
      "parameters": {"video_subject": "AI in Healthcare"}
    },
    {
      "tool_name": "generate_video_script", 
      "parameters": {"video_subject": "Climate Change"}
    }
  ],
  "max_concurrent": 5
}
```

### Service Status
```http
GET /v1/mcp/status
```

### Available Tools
```http
GET /v1/mcp/tools
```

## Configuration

### Server Configuration

Add these settings to your `config.toml`:

```toml
[app]
# MCP Server Settings
mcp_server_host = "localhost"
mcp_server_port = 8081
mcp_max_connections = 100
mcp_rate_limit_requests = 100
mcp_rate_limit_window = 60

# Authentication
mcp_jwt_secret = "your-secret-key"
mcp_jwt_expiration = 3600

# API Keys
[app.mcp_api_keys]
admin_key = { name = "Admin User", role = "admin", active = true }
user_key = { name = "Regular User", role = "user", active = true }

# Circuit Breaker
mcp_circuit_breaker_threshold = 5
mcp_circuit_breaker_timeout = 60

# Service Discovery
mcp_heartbeat_interval = 30
mcp_health_check_interval = 10
mcp_service_timeout = 90
```

### Client Configuration

```python
from app.mcp.client import MCPClientConfig, MCPClient

config = MCPClientConfig(
    server_url="ws://localhost:8081",
    auth_type="api_key",
    api_key="your_api_key",
    max_retries=3,
    retry_delay=1.0,
    connection_timeout=10.0,
    max_connections=5,
    enable_caching=True,
    cache_ttl=3600
)

client = MCPClient(config)
```

## Authentication Methods

### API Key Authentication
```python
client = await create_mcp_client(
    server_url="ws://localhost:8081",
    api_key="your_api_key"
)
```

### JWT Token Authentication
```python
client = await create_mcp_client_jwt(
    server_url="ws://localhost:8081",
    jwt_token="your_jwt_token"
)
```

### HMAC Authentication
```python
config = MCPClientConfig(
    server_url="ws://localhost:8081",
    auth_type="hmac",
    client_id="your_client_id",
    client_secret="your_client_secret"
)
client = MCPClient(config)
```

## Monitoring and Alerting

### Built-in Metrics
- `mcp.operations.started` - Operations initiated
- `mcp.operations.duration` - Operation execution time
- `mcp.operations.success` - Successful operations
- `mcp.operations.error` - Failed operations
- `mcp.connections.active` - Active connections
- `mcp.tools.calls` - Tool invocations

### Custom Alerts
```python
from app.mcp.monitoring import AlertRule, mcp_monitor

# Add custom alert rule
rule = AlertRule(
    name="high_error_rate",
    metric="mcp.operations.error",
    condition="> 10",
    duration=300,
    cooldown=900,
    severity="warning"
)

mcp_monitor.alert_manager.add_alert_rule(rule)
```

### Alert Handlers
```python
async def email_alert_handler(alert_data):
    # Send email notification
    print(f"Alert: {alert_data['rule_name']}")

mcp_monitor.alert_manager.add_alert_handler(email_alert_handler)
```

## Error Handling

### Client-Side Error Handling
```python
from app.mcp.protocol import MCPError

try:
    result = await client.call_tool("generate_video_script", params)
except ConnectionError as e:
    print(f"Connection failed: {e}")
except TimeoutError as e:
    print(f"Request timed out: {e}")
except Exception as e:
    print(f"Tool execution failed: {e}")
```

### Server-Side Error Responses
The server returns structured error responses:
```json
{
  "jsonrpc": "2.0",
  "id": "request_id",
  "error": {
    "code": -32603,
    "message": "Internal server error",
    "data": {
      "details": "Specific error information"
    }
  }
}
```

## Performance Optimization

### Connection Pooling
- Automatic connection reuse and pooling
- Configurable pool size and idle timeout
- Load balancing across multiple connections

### Caching
- Intelligent response caching for repeated requests
- Configurable TTL and cache size limits
- Cache invalidation strategies

### Batch Processing
- Concurrent execution of multiple operations
- Configurable concurrency limits
- Fail-fast or continue-on-error modes

## Development and Testing

### Running Examples
```bash
cd app/mcp/examples
python example_usage.py
```

### TypeScript Client Development
```bash
cd app/mcp/clients/typescript
npm install
npm run build
npm test
```

### Testing Tools
The implementation includes comprehensive test utilities and mock servers for development and testing.

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility when possible

## License

This MCP integration is part of the MoneyPrinterTurbo project and follows the same licensing terms.

## Support

For issues and questions:
1. Check the examples and documentation
2. Review the test files for usage patterns
3. Create an issue in the main repository
4. Refer to the MCP specification for protocol details