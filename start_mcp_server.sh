#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "🚀 Starting MoneyPrinterTurbo MCP Server..."
echo "MCP server will listen on: 0.0.0.0:8081"
python3 run_mcp_server.py
