#!/bin/bash
# Development environment setup script

echo "=== MoneyPrinterTurbo Development Environment Setup ==="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "1. Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

if ! command_exists pip3; then
    echo "❌ pip3 is required but not installed"
    exit 1
fi

if ! command_exists docker; then
    echo "⚠️  Docker not found - Docker deployment will not be available"
fi

echo "✅ Prerequisites check complete"

# Install Python dependencies
echo "2. Installing Python dependencies..."
cd /home/bobby/Downloads/moneyp
pip3 install -r requirements.txt

# Setup environment variables
echo "3. Setting up environment configuration..."
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your actual Supabase credentials"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "4. Creating necessary directories..."
mkdir -p app/static
mkdir -p storage
mkdir -p logs

# Setup git hooks (if git repository)
if [ -d .git ]; then
    echo "5. Setting up git hooks..."
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook to check for common issues

echo "Running pre-commit checks..."

# Check for environment variable placeholders in committed files
if git diff --cached --name-only | xargs grep -l "\${.*}" 2>/dev/null; then
    echo "❌ Found environment variable placeholders in staged files"
    echo "Make sure to use .env files instead of committing secrets"
    exit 1
fi

# Check Python syntax
python3 -m py_compile app/main.py
if [ $? -ne 0 ]; then
    echo "❌ Python syntax errors found"
    exit 1
fi

echo "✅ Pre-commit checks passed"
EOF
    chmod +x .git/hooks/pre-commit
fi

# Create development scripts
echo "6. Creating development helper scripts..."

cat > dev_start.sh << 'EOF'
#!/bin/bash
# Development server start script

source setup_env.sh
cd "$(dirname "$0")"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
EOF

cat > dev_test.sh << 'EOF'
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
EOF

chmod +x dev_start.sh dev_test.sh

echo "7. Setting up linting and formatting tools..."
pip3 install black flake8 isort pylint || echo "⚠️  Linting tools installation failed"

echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit .env with your Supabase credentials"
echo "2. Run ./dev_test.sh to verify setup"
echo "3. Run ./dev_start.sh to start development server"
echo "4. For Docker: Run ./deploy_and_verify.sh"
echo ""
echo "Documentation: See CUSTOM_CHANGES.md for fork-specific information"
