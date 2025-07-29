#!/bin/bash
# Development server start script

source setup_env.sh
cd "$(dirname "$0")"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
