#!/bin/bash
# Development testing script

source setup_env.sh
cd "$(dirname "$0")"

echo "Running basic tests..."
python3 -c "
from app.database.connection import SupabaseConnection
from app.models.exception import SupabaseConnectionError
print('✅ Imports successful')
"

echo "Testing configuration..."
python3 -c "
import toml
with open('config.toml', 'r') as f:
    config = toml.load(f)
print('✅ Configuration loads successfully')
"
