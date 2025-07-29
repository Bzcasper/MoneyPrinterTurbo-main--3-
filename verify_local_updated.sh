#!/bin/bash
# Updated verification script for virtual environment

echo "=== MoneyPrinterTurbo Local Development Verification ==="

# Check if in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "❌ Not in virtual environment! Please run: source venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment active: $VIRTUAL_ENV"

# Step 1: Set environment variables
echo "1. Setting up environment variables..."
source /workspaces/bobby/Downloads/moneyp/setup_env.sh

# Step 2: Check Python environment
echo "2. Checking Python environment..."
python --version
echo "Python path: $(which python)"

echo "3. Checking package versions..."
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import fastapi
    print(f'FastAPI: {fastapi.__version__}')
except ImportError as e:
    print(f'❌ FastAPI not available: {e}')

try:
    import supabase
    print(f'Supabase: {supabase.__version__}')
except ImportError as e:
    print(f'❌ Supabase not available: {e}')

try:
    import uvicorn
    print(f'Uvicorn: {uvicorn.__version__}')
except ImportError as e:
    print(f'❌ Uvicorn not available: {e}')

try:
    import typing_extensions
    print(f'typing_extensions: {typing_extensions.__version__}')
except ImportError as e:
    print(f'❌ typing_extensions not available: {e}')
"

# Step 3: Test configuration loading
echo "4. Testing configuration loading..."
cd /workspaces/bobby/Downloads/moneyp
python -c "
try:
    from app.database.connection import SupabaseConnection
    from app.config import config
    print('✅ Configuration and imports successful')
    
    # Test connection config
    conn = SupabaseConnection()
    print(f'✅ Supabase URL: {conn.config.supabase_url}')
    print('✅ Connection configuration loaded successfully')
except Exception as e:
    print(f'❌ Configuration test failed: {e}')
    import traceback
    traceback.print_exc()
"

# Step 4: Test exception imports
echo "5. Testing exception imports..."
python -c "
try:
    from app.models.exception import SupabaseConnectionError, DatabaseConnectionError
    print('✅ Exception classes imported successfully')
except Exception as e:
    print(f'❌ Exception import failed: {e}')
"

# Step 5: Test FastAPI startup (basic import test)
echo "6. Testing FastAPI application startup..."
python -c "
try:
    from app.main import app
    print('✅ FastAPI application imported successfully')
    print('✅ No typing_extensions compatibility issues detected')
except Exception as e:
    print(f'❌ FastAPI application import failed: {e}')
    import traceback
    traceback.print_exc()
"

echo "7. Verification complete!"
