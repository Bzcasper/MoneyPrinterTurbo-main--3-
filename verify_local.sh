#!/bin/bash
# Local development verification script

echo "=== MoneyPrinterTurbo Local Development Verification ==="

# Step 1: Set environment variables
echo "1. Setting up environment variables..."
source /home/bobby/Downloads/moneyp/setup_env.sh

# Step 2: Check Python environment
echo "2. Checking Python environment..."
python3 --version
pip3 list | grep -E "(fastapi|supabase|uvicorn)" || echo "Some packages may be missing"

# Step 3: Test configuration loading
echo "3. Testing configuration loading..."
cd /home/bobby/Downloads/moneyp
python3 -c "
try:
    from app.database.connection import SupabaseConnection
    from app.config import config
    print('✓ Configuration and imports successful')
    
    # Test connection config
    conn = SupabaseConnection()
    print(f'✓ Supabase URL: {conn.config.supabase_url}')
    print('✓ Connection configuration loaded successfully')
except Exception as e:
    print(f'✗ Configuration test failed: {e}')
"

# Step 4: Test exception imports
echo "4. Testing exception imports..."
python3 -c "
try:
    from app.models.exception import SupabaseConnectionError, DatabaseConnectionError
    print('✓ Exception classes imported successfully')
except Exception as e:
    print(f'✗ Exception import failed: {e}')
"

# Step 5: Run the application (background)
echo "5. Starting the application..."
cd /home/bobby/Downloads/moneyp
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 &
APP_PID=$!

# Wait for startup
echo "Waiting for application to start..."
sleep 10

# Step 6: Test endpoints
echo "6. Testing endpoints..."
curl -s http://localhost:8080/health | jq '.' || echo "Health check failed"
curl -s http://localhost:8080/ | head -10 || echo "Root endpoint failed"

# Step 7: Cleanup
echo "7. Stopping application..."
kill $APP_PID 2>/dev/null || echo "Process already stopped"

echo "=== Local Verification Complete ==="
