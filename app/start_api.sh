#!/bin/bash
echo "Starting API Server..."
cd /MoneyPrinterTurbo
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 4 --access-log --log-level info
exec "$@"
