#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || echo "Virtual env not found, using system Python"
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "🌐 Starting MoneyPrinterTurbo WebUI..."
echo "Access at: http://localhost:8501"
streamlit run webui/Main.py --browser.serverAddress=0.0.0.0 --server.enableCORS=True --browser.gatherUsageStats=False
