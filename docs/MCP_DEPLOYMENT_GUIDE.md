# MCP Server Deployment Guide

## Overview

This guide covers the deployment and verification of the MoneyPrinterTurbo MCP (Model Context Protocol) server integration. The MCP server provides external API access to video generation capabilities through WebSocket connections.

## Architecture

The MCP integration consists of:

- **MCP Server** (`app/mcp/server.py`) - WebSocket server on port 8081
- **MCP Tools** (`app/mcp/tools.py`) - Tool registry and implementations
- **MCP Protocol** (`app/mcp/protocol.py`) - Protocol handlers and validation
- **MCP Authentication** (`app/mcp/auth.py`) - Authentication and authorization
- **MCP Monitoring** (`app/mcp/monitoring.py`) - Health checks and metrics

## Services Configuration

The Docker Compose setup includes:

1. **API Service** - FastAPI backend on port 8080
2. **WebUI Service** - Streamlit frontend on port 8501
3. **Redis Service** - Cache and session storage on port 6379 (internal)
4. **MCP Server Service** - WebSocket server on port 8081

## Deployment Steps

### 1. Prerequisites

- Docker and Docker Compose installed
- Required environment variables configured
- Storage directory permissions set

### 2. Deploy Services

Run the deployment script:

```bash
chmod +x deploy_and_check.sh
./deploy_and_check.sh
```

This script will:
- Stop existing containers
- Build and start all services
- Check service health
- Display logs for debugging
- Test connectivity

### 3. Manual Deployment

Alternatively, deploy manually:

```bash
cd app
docker compose down
docker compose up -d --build
```

## Service Verification

### Health Checks

1. **API Health**: `curl http://localhost:8080/health`
2. **WebUI**: Open `http://localhost:8501` in browser
3. **MCP Server**: Test WebSocket connection on `ws://localhost:8081`

### Log Monitoring

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f mcp-server
docker compose logs -f api
docker compose logs -f redis
```

### Service Status

```bash
# Check running services
docker compose ps

# Check resource usage
docker stats
```

## MCP Server Features

### Capabilities

- **Video Generation**: Generate videos from scripts and parameters
- **Script Generation**: Create video scripts from topics
- **Voice Synthesis**: Convert text to speech
- **Batch Processing**: Handle multiple requests in parallel
- **Rate Limiting**: Prevent API abuse
- **Circuit Breaker**: Handle service failures gracefully
- **Authentication**: Secure API access
- **Caching**: Redis-based response caching

### Available Tools

- `generate_video_script` - Generate video scripts
- `generate_video_terms` - Generate video search terms
- `synthesize_voice` - Text-to-speech conversion
- `generate_video` - Full video generation
- `get_video_status` - Check video generation status

### WebSocket Protocol

The MCP server uses WebSocket for real-time communication:

```javascript
// Connect to MCP server
const ws = new WebSocket('ws://localhost:8081');

// Send tool call request
ws.send(JSON.stringify({
  id: 'request-1',
  method: 'tools/call',
  params: {
    name: 'generate_video_script',
    arguments: {
      topic: 'AI Technology',
      duration: 60
    }
  }
}));
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**: Ensure ports 8080, 8081, 8501 are available
2. **Permission Errors**: Check storage directory permissions
3. **Memory Issues**: Monitor container memory usage
4. **Network Issues**: Verify Docker network configuration

### Debug Commands

```bash
# Check container logs
docker compose logs mcp-server --tail=50

# Access container shell
docker compose exec mcp-server bash

# Check network connectivity
docker compose exec api ping mcp-server

# Monitor Redis
docker compose exec redis redis-cli monitor
```

### Performance Monitoring

```bash
# Resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Container health
docker compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}"
```

## Security Considerations

1. **Authentication**: MCP server requires proper authentication
2. **Rate Limiting**: Prevents API abuse and DoS attacks
3. **Input Validation**: All inputs are validated and sanitized
4. **Environment Variables**: Sensitive data stored in environment
5. **Network Security**: Services communicate through Docker network

## Scaling

### Horizontal Scaling

```yaml
# Scale MCP server instances
docker compose up -d --scale mcp-server=3
```

### Load Balancing

For production deployment, consider:
- Nginx reverse proxy for load balancing
- Redis Cluster for distributed caching
- Database connection pooling
- Container orchestration (Kubernetes)

## Maintenance

### Updates

```bash
# Pull latest images
docker compose pull

# Rebuild and restart
docker compose up -d --build
```

### Backup

```bash
# Backup volumes
docker compose down
docker run --rm -v moneyprinterturbo_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis-backup.tar.gz /data
```

### Cleanup

```bash
# Remove unused containers and images
docker system prune -a

# Remove specific compose setup
docker compose down -v --rmi all
```

## API Integration Examples

### Python Client

```python
import asyncio
import websockets
import json

async def mcp_client():
    uri = "ws://localhost:8081"
    async with websockets.connect(uri) as websocket:
        # Send request
        request = {
            "id": "test-1",
            "method": "tools/call",
            "params": {
                "name": "generate_video_script",
                "arguments": {"topic": "Machine Learning"}
            }
        }
        await websocket.send(json.dumps(request))
        
        # Receive response
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(mcp_client())
```

### cURL Testing

```bash
# Test API health
curl -X GET http://localhost:8080/health

# Test video generation endpoint
curl -X POST http://localhost:8080/api/v1/video/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI Innovation", "duration": 30}'
```

## Support

For issues and support:
1. Check service logs first
2. Verify network connectivity
3. Review configuration files
4. Monitor resource usage
5. Consult documentation

---

**Last Updated**: 2025-01-28
**Version**: 1.0.0