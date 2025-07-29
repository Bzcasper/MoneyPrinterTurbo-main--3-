#!/bin/bash
# Docker deployment and verification script

echo "=== MoneyPrinterTurbo Docker Deployment & Verification ==="

# Step 1: Stop existing containers
echo "1. Stopping existing containers..."
cd /home/bobby/Downloads/moneyp/app
docker-compose down

# Step 2: Rebuild images
echo "2. Rebuilding Docker images..."
docker-compose build --no-cache

# Step 3: Start services
echo "3. Starting services..."
docker-compose up -d

# Step 4: Wait for services to be ready
echo "4. Waiting for services to initialize..."
sleep 30

# Step 5: Check container status
echo "5. Checking container status..."
docker-compose ps

# Step 6: Check logs for errors
echo "6. Checking recent logs..."
echo "=== API Logs ==="
docker-compose logs --tail=20 api

echo "=== WebUI Logs ==="
docker-compose logs --tail=20 webui

# Step 7: Test API endpoints
echo "7. Testing API endpoints..."

echo "Testing health endpoint..."
curl -s http://localhost:8080/health | jq '.' || echo "Health check failed"

echo "Testing root endpoint..."
curl -s http://localhost:8080/ | head -10 || echo "Root endpoint failed"

echo "Testing docs endpoint..."
curl -s -I http://localhost:8080/docs | head -5 || echo "Docs endpoint failed"

echo "Testing favicon..."
curl -s -I http://localhost:8080/favicon.ico | head -5 || echo "Favicon failed"

# Step 8: Check environment variables
echo "8. Checking environment variables in API container..."
docker exec moneyprinterturbo-api-new printenv | grep SUPABASE

echo "=== Verification Complete ==="
echo "Check the output above for any errors."
echo "If all tests pass, the application should be working correctly."
