#!/bin/bash

# Northflank Deployment Script for MoneyPrinterTurbo
# This script helps automate the deployment process to Northflank

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="moneyprinterturbo"
REGION="europe-west-1"  # Change as needed
GITHUB_USERNAME=""  # Set your GitHub username
GITHUB_REPO="MoneyPrinterTurbo"

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

# Function to check if Northflank CLI is installed
check_cli() {
    if ! command -v northflank &> /dev/null; then
        log_warning "Northflank CLI not found. Installing..."
        npm install -g @northflank/cli
        log_success "Northflank CLI installed"
    else
        log_success "Northflank CLI found"
    fi
}

# Function to check authentication
check_auth() {
    log_info "Checking Northflank authentication..."
    if ! northflank auth whoami &> /dev/null; then
        log_warning "Not authenticated. Please login to Northflank"
        northflank auth login
    fi
    log_success "Authenticated with Northflank"
}

# Function to create project
create_project() {
    log_info "Creating project: $PROJECT_NAME"
    
    # Check if project exists
    if northflank project describe --project $PROJECT_NAME &> /dev/null; then
        log_warning "Project $PROJECT_NAME already exists"
        return
    fi
    
    northflank project create \
        --name $PROJECT_NAME \
        --description "AI-powered video generation platform with swarm intelligence" \
        --region $REGION
        
    log_success "Project created: $PROJECT_NAME"
}

# Function to create secrets
create_secrets() {
    log_info "Creating secrets..."
    
    # Create secret group
    northflank secret-group create \
        --project $PROJECT_NAME \
        --name mpt-secrets \
        --description "MoneyPrinterTurbo application secrets"
    
    log_warning "Please manually set the following secrets in the Northflank dashboard:"
    echo "Project: $PROJECT_NAME > Settings > Secrets > mpt-secrets"
    echo ""
    echo "Required secrets:"
    echo "- POSTGRES_USER"
    echo "- POSTGRES_PASSWORD" 
    echo "- POSTGRES_DB"
    echo "- DATABASE_URL"
    echo "- REDIS_HOST"
    echo "- REDIS_PORT"
    echo "- REDIS_PASSWORD"
    echo "- JWT_SECRET"
    echo "- MCP_JWT_SECRET"
    echo "- PEXELS_API_KEY"
    echo "- PIXABAY_API_KEY"
    echo "- OPENAI_API_KEY"
    echo "- SUPABASE_URL (optional)"
    echo "- SUPABASE_ANON_KEY (optional)"
    echo "- SUPABASE_SERVICE_ROLE_KEY (optional)"
    echo ""
    read -p "Press Enter after configuring secrets..."
}

# Function to deploy PostgreSQL
deploy_postgres() {
    log_info "Deploying PostgreSQL database..."
    
    northflank database create \
        --project $PROJECT_NAME \
        --name postgres \
        --type postgresql \
        --version 15 \
        --plan nf-compute-10 \
        --storage-size 20Gi \
        --environment POSTGRES_DB=secret:mpt-secrets:POSTGRES_DB \
        --environment POSTGRES_USER=secret:mpt-secrets:POSTGRES_USER \
        --environment POSTGRES_PASSWORD=secret:mpt-secrets:POSTGRES_PASSWORD
        
    log_success "PostgreSQL deployment initiated"
}

# Function to deploy Redis
deploy_redis() {
    log_info "Deploying Redis cache..."
    
    northflank database create \
        --project $PROJECT_NAME \
        --name redis \
        --type redis \
        --version 7 \
        --plan nf-compute-5 \
        --environment REDIS_PASSWORD=secret:mpt-secrets:REDIS_PASSWORD \
        --configuration maxmemory=512mb \
        --configuration maxmemory-policy=allkeys-lru \
        --configuration appendonly=yes
        
    log_success "Redis deployment initiated"
}

# Function to deploy API service
deploy_api() {
    log_info "Deploying API service..."
    
    if [ -z "$GITHUB_USERNAME" ]; then
        log_error "Please set GITHUB_USERNAME in this script"
        exit 1
    fi
    
    northflank service create \
        --project $PROJECT_NAME \
        --name api \
        --type deployment \
        --plan nf-compute-20 \
        --instances 2 \
        --build-source github \
        --build-repo "https://github.com/$GITHUB_USERNAME/$GITHUB_REPO" \
        --build-branch main \
        --build-dockerfile ./app/Dockerfile \
        --port 8080:8080:HTTP:public \
        --environment HOST=0.0.0.0 \
        --environment PORT=8080 \
        --environment ENVIRONMENT=production \
        --environment PYTHONPATH=/MoneyPrinterTurbo \
        --environment DATABASE_URL=secret:mpt-secrets:DATABASE_URL \
        --environment REDIS_HOST=secret:mpt-secrets:REDIS_HOST \
        --environment REDIS_PORT=secret:mpt-secrets:REDIS_PORT \
        --environment REDIS_PASSWORD=secret:mpt-secrets:REDIS_PASSWORD \
        --environment ENABLE_REDIS=true \
        --environment JWT_SECRET=secret:mpt-secrets:JWT_SECRET \
        --environment PEXELS_API_KEY=secret:mpt-secrets:PEXELS_API_KEY \
        --environment PIXABAY_API_KEY=secret:mpt-secrets:PIXABAY_API_KEY \
        --environment OPENAI_API_KEY=secret:mpt-secrets:OPENAI_API_KEY \
        --environment TRUSTED_HOSTS="*.northflank.app,localhost,127.0.0.1" \
        --environment ALLOWED_ORIGINS="https://*.northflank.app" \
        --depends-on postgres,redis \
        --health-check-path /health
        
    log_success "API service deployment initiated"
}

# Function to deploy WebUI service
deploy_webui() {
    log_info "Deploying WebUI service..."
    
    northflank service create \
        --project $PROJECT_NAME \
        --name webui \
        --type deployment \
        --plan nf-compute-10 \
        --instances 2 \
        --build-source github \
        --build-repo "https://github.com/$GITHUB_USERNAME/$GITHUB_REPO" \
        --build-branch main \
        --build-dockerfile ./app/Dockerfile \
        --build-command "streamlit run ./webui/Main.py --browser.serverAddress=0.0.0.0 --server.enableCORS=True --browser.gatherUsageStats=False --server.port=8501" \
        --port 8501:8501:HTTP:public \
        --environment HOST=0.0.0.0 \
        --environment PORT=8501 \
        --environment PYTHONPATH=/MoneyPrinterTurbo \
        --environment STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
        --environment DATABASE_URL=secret:mpt-secrets:DATABASE_URL \
        --environment SUPABASE_URL=secret:mpt-secrets:SUPABASE_URL \
        --environment SUPABASE_ANON_KEY=secret:mpt-secrets:SUPABASE_ANON_KEY \
        --depends-on api \
        --health-check-path /
        
    log_success "WebUI service deployment initiated"
}

# Function to deploy MCP server
deploy_mcp() {
    log_info "Deploying MCP server..."
    
    northflank service create \
        --project $PROJECT_NAME \
        --name mcp-server \
        --type deployment \
        --plan nf-compute-15 \
        --instances 2 \
        --build-source github \
        --build-repo "https://github.com/$GITHUB_USERNAME/$GITHUB_REPO" \
        --build-branch main \
        --build-dockerfile ./app/Dockerfile \
        --build-command "python run_mcp_server.py" \
        --port 8081:8081:HTTP:public \
        --environment MCP_HOST=0.0.0.0 \
        --environment MCP_PORT=8081 \
        --environment PYTHONPATH=/MoneyPrinterTurbo \
        --environment PYTHONUNBUFFERED=1 \
        --environment ENVIRONMENT=production \
        --environment REDIS_HOST=secret:mpt-secrets:REDIS_HOST \
        --environment REDIS_PORT=secret:mpt-secrets:REDIS_PORT \
        --environment REDIS_PASSWORD=secret:mpt-secrets:REDIS_PASSWORD \
        --environment ENABLE_REDIS=true \
        --environment MCP_JWT_SECRET=secret:mpt-secrets:MCP_JWT_SECRET \
        --depends-on redis
        
    log_success "MCP server deployment initiated"
}

# Function to check deployment status
check_status() {
    log_info "Checking deployment status..."
    
    echo ""
    echo "Service Status:"
    northflank service list --project $PROJECT_NAME
    
    echo ""
    log_info "Service URLs:"
    northflank service describe --project $PROJECT_NAME --service api --output json | jq -r '.ports[] | select(.public == true) | .url' | head -1 | xargs -I {} echo "API: {}"
    northflank service describe --project $PROJECT_NAME --service webui --output json | jq -r '.ports[] | select(.public == true) | .url' | head -1 | xargs -I {} echo "WebUI: {}"
    northflank service describe --project $PROJECT_NAME --service mcp-server --output json | jq -r '.ports[] | select(.public == true) | .url' | head -1 | xargs -I {} echo "MCP: {}"
}

# Function to show logs
show_logs() {
    local service=$1
    if [ -z "$service" ]; then
        log_error "Please specify a service: api, webui, mcp-server, postgres, redis"
        return 1
    fi
    
    log_info "Showing logs for $service..."
    northflank logs --project $PROJECT_NAME --service $service --follow
}

# Function to configure auto-scaling
configure_scaling() {
    log_info "Configuring auto-scaling..."
    
    # Enable auto-scaling for API service
    northflank service update \
        --project $PROJECT_NAME \
        --service api \
        --autoscaling-enabled \
        --autoscaling-min-instances 2 \
        --autoscaling-max-instances 10 \
        --autoscaling-cpu-target 70 \
        --autoscaling-memory-target 80
        
    log_success "Auto-scaling configured for API service"
}

# Main deployment function
deploy_all() {
    log_info "🚀 Starting MoneyPrinterTurbo deployment to Northflank..."
    
    check_cli
    check_auth
    create_project
    create_secrets
    
    log_info "Deploying database services first..."
    deploy_postgres
    deploy_redis
    
    log_info "Waiting for databases to be ready..."
    sleep 60
    
    log_info "Deploying application services..."
    deploy_api
    deploy_webui
    deploy_mcp
    
    log_info "Configuring scaling..."
    configure_scaling
    
    log_info "Waiting for services to start..."
    sleep 120
    
    check_status
    
    log_success "🎉 Deployment complete!"
    echo ""
    log_info "Next steps:"
    echo "1. Check service health in Northflank dashboard"
    echo "2. Configure custom domains (optional)"
    echo "3. Set up monitoring and alerts"
    echo "4. Test all endpoints"
}

# Script usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  deploy         Deploy all services"
    echo "  status         Check deployment status"
    echo "  logs <service> Show logs for a service"
    echo "  help           Show this help message"
    echo ""
    echo "Before running, set GITHUB_USERNAME in this script!"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        if [ -z "$GITHUB_USERNAME" ]; then
            log_error "Please set GITHUB_USERNAME at the top of this script"
            exit 1
        fi
        deploy_all
        ;;
    "status")
        check_status
        ;;
    "logs")
        show_logs $2
        ;;
    "help")
        usage
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
