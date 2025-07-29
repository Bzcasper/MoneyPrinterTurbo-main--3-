#!/bin/bash
echo "Starting MCP Server..."
cd /MoneyPrinterTurbo
python3 -c "
import asyncio
import os
from app.mcp.server import MCPServer

async def main():
    host = os.getenv('MCP_HOST', '0.0.0.0')
    port = int(os.getenv('MCP_PORT', 8081))
    server = MCPServer(host=host, port=port)
    await server.start_server()

if __name__ == '__main__':
    asyncio.run(main())
"
exec "$@"
