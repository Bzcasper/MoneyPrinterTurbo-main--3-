# Docker Setup Guide for MoneyPrinterTurbo

## 🐛 Docker Issues Resolution

This guide provides solutions for common Docker issues encountered in the dev container environment.

## Fixed Issues

### ✅ Docker-in-Docker Configuration
The `.devcontainer/devcontainer.json` has been updated with:
- Docker-in-Docker feature enabled
- All necessary ports forwarded (8501, 8080, 8081, 5432, 6379)
- Privileged mode for Docker access
- Docker socket mount

## Available Docker Configurations

### 1. Main Application Stack (`app/docker-compose.yml`)
**Services:**
- **WebUI** (port 8501) - Streamlit web interface
- **API** (port 8080) - FastAPI backend
- **Redis** (port 6379) - Cache and session storage
- **PostgreSQL** (port 5432) - Database
- **MCP Server** (port 8081) - Model Context Protocol server

**Command:**
```bash
# From project root (/workspaces/bobby/Downloads/moneyp)
docker compose -f app/docker-compose.yml up --build
```

### 2. Microservices Deployment (`deployment/docker/docker-compose.microservices.yml`)
**Command:**
```bash
# From project root
docker compose -f deployment/docker/docker-compose.microservices.yml up --build
```

## Quick Commands

### Check Docker Status
```bash
# Verify Docker is running
docker ps

# Check Docker version
docker --version
docker compose version
```

### Navigate to Correct Directory
```bash
# Ensure you're in the project root
cd /workspaces/bobby/Downloads/moneyp
pwd  # Should show: /workspaces/bobby/Downloads/moneyp
```

### Start Services
```bash
# Full stack (recommended for development)
docker compose -f app/docker-compose.yml up --build

# Run in background
docker compose -f app/docker-compose.yml up --build -d

# Stop services
docker compose -f app/docker-compose.yml down

# View logs
docker compose -f app/docker-compose.yml logs -f
```

## Environment Variables

Ensure you have a `.env` file in the project root with:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

## Troubleshooting

### Docker Daemon Not Running
1. **In Dev Container:** The container should auto-start Docker with the new configuration
2. **On Host:** Ensure Docker Desktop is running

### Port Conflicts
If ports are already in use, modify the port mappings in the docker-compose files:
```yaml
ports:
  - "8502:8501"  # Change host port from 8501 to 8502
```

### Permission Issues
```bash
# Fix Docker socket permissions (if needed)
sudo chmod 666 /var/run/docker.sock
```

### Container Restart Required
After updating `.devcontainer/devcontainer.json`, you need to:
1. Close VS Code
2. Rebuild the dev container
3. Reopen the project

## Health Checks

The services include built-in health checks:
- **API:** `curl -f http://localhost:8080/health`
- **Redis:** `redis-cli --raw incr ping`
- **PostgreSQL:** `pg_isready -U postgres`
- **MCP Server:** Socket connection test on port 8081

## Next Steps

1. Restart your dev container to apply the Docker-in-Docker configuration
2. Navigate to the project root directory
3. Run the Docker Compose command for your desired setup
4. Access the web interface at http://localhost:8501

## Support

If you continue to experience Docker issues:
1. Check the container logs: `docker compose logs`
2. Verify all environment variables are set
3. Ensure no port conflicts exist
4. Try rebuilding containers: `docker compose build --no-cache`