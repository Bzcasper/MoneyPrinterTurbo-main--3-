#!/bin/bash
# Development environment setup script - Updated for virtual environments

echo "=== MoneyPrinterTurbo Development Environment Setup ==="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if we're in a virtual environment
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo "❌ Not in a virtual environment!"
        echo "Please run: python3 -m venv venv && source venv/bin/activate"
        exit 1
    else
        echo "✅ Virtual environment active: $VIRTUAL_ENV"
    fi
}

# Check prerequisites
echo "1. Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check virtual environment
check_venv

if ! command_exists docker; then
    echo "⚠️  Docker not found - Docker deployment will not be available"
fi

echo "✅ Prerequisites check complete"

# Install Python dependencies
echo "2. Installing Python dependencies..."
cd /workspaces/bobby/Downloads/moneyp

# Use updated requirements
if [ -f requirements-updated.txt ]; then
    echo "Installing from updated requirements (Python 3.13 compatible)..."
    pip install -r requirements-updated.txt
else
    echo "Using original requirements..."
    pip install -r requirements.txt
fi

# Install app-specific requirements
if [ -f app/requirements-updated.txt ]; then
    echo "Installing app requirements (Python 3.13 compatible)..."
    pip install -r app/requirements-updated.txt
elif [ -f app/requirements.txt ]; then
    echo "Installing app requirements..."
    pip install -r app/requirements.txt
fi

# Setup environment variables
echo "3. Setting up environment configuration..."
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your actual Supabase credentials"
else
    echo "✅ .env file already exists"
fi

# Install development dependencies
echo "4. Installing development dependencies..."
pip install black flake8 mypy pytest

# Verify installation
echo "5. Verifying installation..."
python -c "
try:
    import fastapi
    import supabase
    import uvicorn
    print(f'✅ FastAPI: {fastapi.__version__}')
    print(f'✅ Supabase: {supabase.__version__}')
    print(f'✅ Uvicorn: {uvicorn.__version__}')
    
    # Test typing_extensions
    import typing_extensions
    print(f'✅ typing_extensions: {typing_extensions.__version__}')
    
except Exception as e:
    print(f'❌ Package verification failed: {e}')
"

echo "✅ Development environment setup complete!"
echo "💡 To activate this environment in future sessions:"
echo "   cd /workspaces/bobby/Downloads/moneyp && source venv/bin/activate"
