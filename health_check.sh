#!/bin/bash
echo "🔍 MoneyPrinterTurbo Health Check"
echo "================================"

# Check if API is running
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ API Service - Running"
else
    echo "❌ API Service - Not responding"
fi

# Check if WebUI is accessible
if curl -s http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ WebUI Service - Running"
else
    echo "❌ WebUI Service - Not responding"
fi

# System resources
echo "📊 System Resources:"
echo "  CPU: $(nproc) cores"
echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}') total"
echo "  Disk: $(df -h . | tail -1 | awk '{print $4}') available"

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "  GPU: Not detected (nvidia-smi not available)"
fi
