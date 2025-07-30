#!/bin/bash
"""
Video Generation Test Runner

This script runs comprehensive tests for the video generation workflow
with CPU-optimized processing and proper environment setup.

Usage:
    ./run_video_tests.sh [quick|full|api-only]

Arguments:
    quick     - Run quick validation tests only (5-10 minutes)
    full      - Run comprehensive test suite (20-30 minutes)  
    api-only  - Test only API connectivity and script generation (2-3 minutes)
    (default) - Run quick tests by default
"""

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} ✅ $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')]${NC} ⚠️  $1"
}

print_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')]${NC} ❌ $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments
TEST_MODE="${1:-quick}"

print_status "🚀 Video Generation Test Runner"
print_status "📁 Working directory: $SCRIPT_DIR"
print_status "🎯 Test mode: $TEST_MODE"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found, checking system Python..."
else
    print_status "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
print_status "🐍 Python version: $PYTHON_VERSION"

# Verify Python version is compatible (3.8+)
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" || {
    print_error "Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
}

# Check required files
print_status "📋 Checking required files..."

required_files=(
    "config.toml"
    "app/config/config.py"
    "app/services/llm.py"
    "app/services/task.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "Required file missing: $file"
        exit 1
    fi
done

print_success "All required files found"

# Check Python dependencies
print_status "📦 Checking Python dependencies..."

python3 -c "
import sys
try:
    import google.generativeai
    import loguru
    import PIL
    print('✅ Core dependencies available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)
" || {
    print_error "Missing required Python dependencies"
    print_status "💡 Install dependencies with: pip install -r requirements.txt"
    exit 1
}

# Check configuration
print_status "🔧 Validating configuration..."

if [ ! -f "config.toml" ]; then
    print_error "config.toml not found"
    exit 1
fi

# Check if Gemini API key is configured
if ! grep -q "gemini_api_key.*=" config.toml; then
    print_error "Gemini API key not found in config.toml"
    exit 1
fi

# Check if API key looks valid (not placeholder)
if grep -q "your_.*_key_here\|YOUR_.*_KEY\|placeholder" config.toml; then
    print_error "Gemini API key appears to be placeholder text"
    print_status "💡 Update config.toml with your actual Gemini API key"
    exit 1
fi

print_success "Configuration validation passed"

# Check system requirements
print_status "💻 Checking system requirements..."

# Check available memory
AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
if [ "$AVAILABLE_MEM" -lt 1000 ]; then
    print_warning "Low memory available: ${AVAILABLE_MEM}MB (recommended: 1GB+)"
else
    print_status "📊 Available memory: ${AVAILABLE_MEM}MB"
fi

# Check CPU cores
CPU_CORES=$(nproc)
print_status "🖥️  CPU cores: $CPU_CORES"

# Check disk space
DISK_SPACE=$(df -BM . | awk 'NR==2 {print $4}' | sed 's/M//')
if [ "$DISK_SPACE" -lt 1000 ]; then
    print_warning "Low disk space: ${DISK_SPACE}MB (recommended: 1GB+)"
else
    print_status "💾 Available disk space: ${DISK_SPACE}MB"
fi

# Check FFmpeg availability
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | cut -d' ' -f3)
    print_status "🎬 FFmpeg version: $FFMPEG_VERSION"
else
    print_warning "FFmpeg not found - some video processing may fail"
    print_status "💡 Install FFmpeg: sudo apt-get install ffmpeg"
fi

# Check network connectivity
print_status "🌐 Testing network connectivity..."
if curl -s --max-time 10 https://generativelanguage.googleapis.com > /dev/null; then
    print_success "Network connectivity to Gemini API: OK"
else
    print_warning "Cannot reach Gemini API - check network connection"
fi

print_success "System requirements check completed"

# Create test output directory
TEST_OUTPUT_DIR="$SCRIPT_DIR/test_outputs"
mkdir -p "$TEST_OUTPUT_DIR"
print_status "📁 Test output directory: $TEST_OUTPUT_DIR"

# Run tests based on mode
case "$TEST_MODE" in
    "quick")
        print_status "🏃 Running quick validation tests..."
        python3 test_video_quick.py
        ;;
    
    "full")
        print_status "🧪 Running comprehensive test suite..."
        python3 test_video_generation_complete.py
        ;;
    
    "api-only")
        print_status "🔌 Running API-only tests..."
        python3 -c "
import sys
sys.path.insert(0, 'app')
from app.config import config
from app.services import llm
from loguru import logger

logger.remove()
logger.add(sys.stdout, level='INFO')

try:
    logger.info('🔧 Testing configuration...')
    assert config.app.get('llm_provider') == 'gemini'
    assert config.app.get('gemini_api_key')
    logger.success('✅ Configuration OK')
    
    logger.info('🔌 Testing API connectivity...')
    response = llm._generate_response('Test response')
    assert response and not response.startswith('Error:')
    logger.success('✅ API connectivity OK')
    
    logger.info('📝 Testing script generation...')
    script = llm.generate_script('Test subject', 'en', 1)
    assert script and len(script) > 50
    logger.success('✅ Script generation OK')
    
    logger.success('🎉 All API tests passed!')
    
except Exception as e:
    logger.error(f'❌ API test failed: {e}')
    sys.exit(1)
"
        ;;
    
    *)
        print_error "Invalid test mode: $TEST_MODE"
        print_status "Valid modes: quick, full, api-only"
        exit 1
        ;;
esac

TEST_EXIT_CODE=$?

# Print final results
print_status ""
print_status "🏁 Test execution completed"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "🎉 All tests passed successfully!"
    print_status "💡 Video generation system is ready for use"
    
    case "$TEST_MODE" in
        "quick"|"api-only")
            print_status "💭 To run full video generation tests: ./run_video_tests.sh full"
            ;;
    esac
else
    print_error "💥 Some tests failed"
    print_status "🔧 Review error messages above and fix issues before proceeding"
    print_status "💡 For help, check the configuration and ensure all dependencies are installed"
fi

# Clean up test outputs if successful and not full test
if [ $TEST_EXIT_CODE -eq 0 ] && [ "$TEST_MODE" != "full" ]; then
    if [ -d "$TEST_OUTPUT_DIR" ] && [ "$(ls -A $TEST_OUTPUT_DIR)" ]; then
        print_status "🧹 Cleaning up test outputs..."
        rm -rf "$TEST_OUTPUT_DIR"/*
    fi
fi

print_status "📊 Test summary saved to test_outputs/ (if any)"
print_status "🔚 Test runner finished"

exit $TEST_EXIT_CODE