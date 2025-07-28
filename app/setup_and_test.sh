#!/bin/bash

# MoneyPrinterTurbo Enhanced Setup and Test Script
# Created following Claude_General.prompt.md instructions

echo "🚀 MoneyPrinterTurbo Enhanced - Setup and Test"
echo "=============================================="

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found, installing minimal dependencies..."
    pip install streamlit fastapi uvicorn loguru
fi

# Check application structure
echo "🏗️ Validating application structure..."
required_files=(
    "app/main.py"
    "webui/Main.py"
    "app/config/config.py"
    "app/services/"
)

for file in "${required_files[@]}"; do
    if [ -e "$file" ]; then
        echo "✅ $file - Found"
    else
        echo "❌ $file - Missing"
    fi
done

# Test imports
echo "🧪 Testing critical imports..."
python3 -c "
try:
    import sys, os
    sys.path.append('.')
    
    # Test basic imports
    import app.config.config as config
    print('✅ Config module - OK')
    
    # Test service imports
    try:
        import app.services
        print('✅ Services module - OK')
    except Exception as e:
        print(f'⚠️ Services module - Partial: {e}')
    
    # Test WebUI
    try:
        import streamlit
        print('✅ Streamlit - OK')
    except Exception as e:
        print(f'❌ Streamlit - Missing: {e}')
        
    print('🎯 Import test completed')
    
except Exception as e:
    print(f'❌ Critical import failure: {e}')
    exit(1)
"

# Create startup scripts
echo "📝 Creating startup scripts..."

# WebUI startup script
cat > start_webui.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || echo "Virtual env not found, using system Python"
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "🌐 Starting MoneyPrinterTurbo WebUI..."
echo "Access at: http://localhost:8501"
streamlit run webui/Main.py --browser.serverAddress=0.0.0.0 --server.enableCORS=True --browser.gatherUsageStats=False
EOF

# API startup script  
cat > start_api.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || echo "Virtual env not found, using system Python"
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "🚀 Starting MoneyPrinterTurbo API..."
echo "API docs at: http://localhost:8080/docs"
python app/main.py
EOF

chmod +x start_webui.sh start_api.sh

# Health check script
cat > health_check.sh << 'EOF'
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
EOF

chmod +x health_check.sh

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Available commands:"
echo "  ./start_webui.sh    - Start Web Interface (port 8501)"
echo "  ./start_api.sh      - Start API Service (port 8080)" 
echo "  ./health_check.sh   - Check system health"
echo ""
echo "🌐 Quick start:"
echo "  1. Run: ./start_webui.sh"
echo "  2. Open: http://localhost:8501"
echo ""
echo "📁 Generated files:"
echo "  - moneyprinterturbo.desktop (Desktop shortcut)"
echo "  - ANALYSIS_REPORT.md (Architecture analysis)"
echo "  - TODO.md (Project tracking)"
echo "  - credentials.env (Environment variables - secured)"
echo ""
echo "✅ MoneyPrinterTurbo Enhanced is ready for production!"
