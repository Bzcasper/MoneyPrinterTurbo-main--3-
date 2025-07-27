#!/bin/bash

# Docker Compose v1.29.2 ContainerConfig fix script
# This script handles the migration from docker-compose v1 to v2 or provides workarounds

echo "=== MoneyPrinterTurbo Docker Setup ==="

# Check docker-compose version
COMPOSE_VERSION=$(docker-compose --version 2>/dev/null || echo "not found")
echo "Current docker-compose: $COMPOSE_VERSION"

# Clean up existing containers and images
echo "Cleaning up existing containers..."
docker-compose down -v --remove-orphans 2>/dev/null || true
docker stop moneyprinterturbo-webui moneyprinterturbo-api 2>/dev/null || true
docker rm moneyprinterturbo-webui moneyprinterturbo-api 2>/dev/null || true
docker rm moneyprinterturbo-webui-new moneyprinterturbo-api-new 2>/dev/null || true

# Remove old images
echo "Removing old images..."
docker rmi moneyprinterturbo_webui:latest moneyprinterturbo_api:latest 2>/dev/null || true
docker rmi moneyprinterturbo-webui:latest moneyprinterturbo-api:latest 2>/dev/null || true

# Clean up volumes and networks
echo "Cleaning up volumes and networks..."
docker volume prune -f 2>/dev/null || true
docker network prune -f 2>/dev/null || true

# Check if Docker Compose v2 is available
if command -v docker compose &> /dev/null; then
    echo "Using Docker Compose v2..."
    docker compose -f docker-compose.yml up -d --build
elif command -v docker-compose &> /dev/null; then
    echo "Using Docker Compose v1 (legacy)..."
    echo "Attempting to use legacy mode..."
    
    # Try with legacy mode
    docker-compose -f docker-compose.yml up -d --build --force-recreate
    
    # If that fails, provide manual instructions
    if [ $? -ne 0 ]; then
        echo "Legacy docker-compose failed. Trying manual approach..."
        
        # Build images manually
        docker build -t moneyprinterturbo-webui:latest .
        docker build -t moneyprinterturbo-api:latest .
        
        # Run containers directly
        docker run -d --name moneyprinterturbo-webui-new \
            -p 8501:8501 \
            -v $(pwd):/MoneyPrinterTurbo \
            -v $(pwd)/storage:/MoneyPrinterTurbo/storage \
            -e PYTHONPATH=/MoneyPrinterTurbo \
            -e HOST=0.0.0.0 \
            -e PORT=8501 \
            moneyprinterturbo-webui:latest \
            streamlit run ./webui/Main.py --browser.serverAddress=0.0.0.0 --server.enableCORS=True --browser.gatherUsageStats=False
            
        docker run -d --name moneyprinterturbo-api-new \
            -p 8080:8080 \
            -v $(pwd):/MoneyPrinterTurbo \
            -v $(pwd)/storage:/MoneyPrinterTurbo/storage \
            -e PYTHONPATH=/MoneyPrinterTurbo \
            -e HOST=0.0.0.0 \
            -e PORT=8080 \
            moneyprinterturbo-api:latest \
            python3 main.py
    fi
else
    echo "Neither docker-compose nor docker compose found!"
    echo "Please install Docker Compose v2:"
    echo "  sudo apt-get update && sudo apt-get install docker-compose-plugin"
fi

echo "=== Setup Complete ==="
echo "WebUI should be available at: http://localhost:8501"
echo "API should be available at: http://localhost:8080"
