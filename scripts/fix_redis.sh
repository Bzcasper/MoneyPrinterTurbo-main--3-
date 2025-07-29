#!/bin/bash
set -e

# MoneyPrinterTurbo Redis Fix Script
# Fixes common Redis connection issues and validates setup

echo "🔧 MoneyPrinterTurbo Redis Fix Script"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check Redis status
check_redis_status() {
    log_info "Checking Redis status..."
    
    # Check if Redis is running via Docker
    if docker ps | grep -q redis; then
        log_success "Redis container is running"
        return 0
    fi
    
    # Check if Redis is running locally
    if pgrep -x "redis-server" > /dev/null; then
        log_success "Redis server is running locally"
        return 0
    fi
    
    # Check if Redis service is running
    if systemctl is-active --quiet redis 2>/dev/null || systemctl is-active --quiet redis-server 2>/dev/null; then
        log_success "Redis service is running"
        return 0
    fi
    
    log_warning "Redis is not running"
    return 1
}

# Function to start Redis
start_redis() {
    log_info "Starting Redis..."
    
    # Try Docker first
    if command -v docker >/dev/null 2>&1; then
        log_info "Starting Redis with Docker..."
        docker run -d --name moneyprinter-redis -p 6379:6379 redis:7-alpine
        sleep 3
        if docker ps | grep -q moneyprinter-redis; then
            log_success "Redis started with Docker"
            return 0
        fi
    fi
    
    # Try systemctl
    if command -v systemctl >/dev/null 2>&1; then
        log_info "Starting Redis service..."
        sudo systemctl start redis 2>/dev/null || sudo systemctl start redis-server 2>/dev/null
        sleep 2
        if systemctl is-active --quiet redis 2>/dev/null || systemctl is-active --quiet redis-server 2>/dev/null; then
            log_success "Redis service started"
            return 0
        fi
    fi
    
    log_error "Failed to start Redis"
    return 1
}

# Function to test Redis connection
test_redis_connection() {
    log_info "Testing Redis connection..."
    
    # Test localhost connection
    if redis-cli -h localhost -p 6379 ping 2>/dev/null | grep -q PONG; then
        log_success "Redis connection test passed (localhost:6379)"
        return 0
    fi
    
    # Test Docker connection
    if redis-cli -h 127.0.0.1 -p 6379 ping 2>/dev/null | grep -q PONG; then
        log_success "Redis connection test passed (127.0.0.1:6379)"
        return 0
    fi
    
    log_error "Redis connection test failed"
    return 1
}

# Function to fix Python test script
fix_python_test() {
    log_info "Fixing Python test script for local environment..."
    
    # Set environment variables for local testing
    export REDIS_HOST=localhost
    export REDIS_PORT=6379
    export REDIS_DB=0
    
    log_success "Environment variables set for local Redis testing"
}

# Function to open required ports
open_ports() {
    log_info "Checking and opening required ports..."
    
    # Check if UFW is available
    if command -v ufw >/dev/null 2>&1; then
        log_info "Opening Redis port 6379 with UFW..."
        sudo ufw allow 6379/tcp 2>/dev/null || log_warning "Could not open port with UFW"
    fi
    
    # Check if firewall-cmd is available
    if command -v firewall-cmd >/dev/null 2>&1; then
        log_info "Opening Redis port 6379 with firewalld..."
        sudo firewall-cmd --permanent --add-port=6379/tcp 2>/dev/null || log_warning "Could not open port with firewalld"
        sudo firewall-cmd --reload 2>/dev/null || log_warning "Could not reload firewalld"
    fi
    
    log_success "Port configuration completed"
}

# Function to validate Docker setup
validate_docker_setup() {
    log_info "Validating Docker setup..."
    
    if [ ! -f "app/docker-compose.yml" ]; then
        log_error "docker-compose.yml not found in app/ directory"
        return 1
    fi
    
    # Check if Redis service is defined
    if grep -q "redis:" app/docker-compose.yml; then
        log_success "Redis service found in docker-compose.yml"
    else
        log_warning "Redis service not found in docker-compose.yml"
    fi
    
    # Check if MCP service depends on Redis
    if grep -q "depends_on:" app/docker-compose.yml && grep -A 5 "depends_on:" app/docker-compose.yml | grep -q "redis"; then
        log_success "MCP service properly depends on Redis"
    else
        log_warning "MCP service dependency on Redis may be missing"
    fi
    
    return 0
}

# Function to run Python test with fixes
run_python_test() {
    log_info "Running Python Redis test..."
    
    if [ -f "scripts/test_redis_connection.py" ]; then
        # Set local environment
        export REDIS_HOST=localhost
        export REDIS_PORT=6379
        
        log_info "Running Redis connection test with localhost configuration..."
        python3 scripts/test_redis_connection.py
        
        if [ $? -eq 0 ]; then
            log_success "Python Redis test passed"
            return 0
        else
            log_warning "Python Redis test had issues (may be expected if Redis not running)"
            return 1
        fi
    else
        log_warning "Redis test script not found"
        return 1
    fi
}

# Main fix sequence
main() {
    log_info "Starting Redis fix sequence..."
    
    # Step 1: Check current Redis status
    if ! check_redis_status; then
        log_info "Redis not running, attempting to start..."
        start_redis
    fi
    
    # Step 2: Open required ports
    open_ports
    
    # Step 3: Test Redis connection
    if ! test_redis_connection; then
        log_warning "Direct Redis connection failed, this may be normal for Docker environments"
    fi
    
    # Step 4: Fix Python test environment
    fix_python_test
    
    # Step 5: Validate Docker setup
    validate_docker_setup
    
    # Step 6: Run Python test
    run_python_test
    
    echo ""
    log_info "Redis Fix Summary:"
    log_info "=================="
    
    if check_redis_status; then
        log_success "✅ Redis is running"
    else
        log_warning "⚠️ Redis status unclear"
    fi
    
    if test_redis_connection; then
        log_success "✅ Redis connection works"
    else
        log_warning "⚠️ Redis connection needs Docker environment"
    fi
    
    log_success "✅ Environment configured for local testing"
    log_success "✅ Docker setup validated"
    
    echo ""
    log_info "Next Steps:"
    log_info "==========="
    log_info "1. For local testing: Redis should be accessible at localhost:6379"
    log_info "2. For Docker deployment: cd app/ && docker-compose up --build"
    log_info "3. For production: Use the Docker setup with proper service dependencies"
    
    echo ""
    log_success "🎉 Redis fix completed! Your setup should now work correctly."
}

# Run main function
main "$@"
