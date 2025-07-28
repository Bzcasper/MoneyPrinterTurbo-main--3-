#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || echo "Virtual env not found, using system Python"
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "🚀 Starting MoneyPrinterTurbo API..."
echo "API docs at: http://localhost:8080/docs"
python3 app/main.py
