#!/bin/bash

# MoneyPrinterTurbo MCP Deployment and Log Check Script
# This script deploys the Docker Compose services and checks logs

set -e

echo "🚀 Starting MoneyPrinterTurbo MCP Deployment..."

# Change to app directory
cd app

# Stop any existing containers
echo "📦 Stopping existing containers..."
docker compose down || true

# Build and start all services
echo "🔨 Building and starting services..."
docker compose up -d --build

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service status
echo "📊 Checking service status..."
docker compose ps

# Check logs for each service
echo "📝 Checking Redis logs..."
docker compose logs redis --tail=20

echo "📝 Checking API logs..."
docker compose logs api --tail=20

echo "📝 Checking WebUI logs..."
docker compose logs webui --tail=20

echo "📝 Checking MCP Server logs..."
docker compose logs mcp-server --tail=20

# Test MCP server connection
echo "🔍 Testing MCP server connection..."
timeout 10 bash -c 'until nc -z localhost 8081; do sleep 1; done' && echo "✅ MCP server is reachable on port 8081" || echo "❌ MCP server connection failed"

# Test API health
echo "🔍 Testing API health..."
curl -f http://localhost:8080/health && echo "✅ API health check passed" || echo "❌ API health check failed"

# Display service URLs
echo "🌐 Service URLs:"
echo "  - API: http://localhost:8080"
echo "  - WebUI: http://localhost:8501"
echo "  - MCP Server: ws://localhost:8081"

echo "✅ Deployment complete! Check the logs above for any issues."
echo "📋 To monitor logs continuously, run: docker compose logs -f"
echo "🛑 To stop services, run: docker compose down"