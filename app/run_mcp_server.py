"""
MCP Server standalone runner script
"""

import asyncio
import sys
import os

# Add the parent directory to sys.path to allow imports from the app package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Setup environment variables
os.environ["PYTHONPATH"] = parent_dir

async def main():
    # Import the MCP server implementation
    from app.mcp.server import MCPServer
    
    # Get configuration from command line or use defaults
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8081"))
    
    print(f"Starting MCP server on {host}:{port}...")
    
    # Initialize the MCP server
    server = MCPServer(host=host, port=port)
    
    # Start the server
    await server.start_server()
    
    # Keep the server running
    try:
        print("MCP server running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(3600)  # Sleep for a long time
    except KeyboardInterrupt:
        print("Shutting down MCP server...")
        await server.stop_server()
        print("MCP server stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        sys.exit(1)
