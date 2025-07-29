#!/bin/bash
echo "Starting MCP Server..."
cd /MoneyPrinterTurbo
python3 -m uvicorn app.mcp.server:app --host 0.0.0.0 --port 8081 --workers 2 --log-level info
exec "$@"
