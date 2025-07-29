# Docker Deployment Troubleshooting Guide
**Testing & Troubleshooting Expert - Comprehensive Guide**

## 🚨 Quick Diagnosis Commands

### 1. Check Overall System Status
```bash
# Check all containers
docker-compose -f app/docker-compose.yml ps

# Check system resources
docker system df
docker system info

# Check Docker daemon
systemctl status docker
```

### 2. Service-Specific Health Checks
```bash
# PostgreSQL Health
docker exec moneyprinterturbo-postgres-new pg_isready -U postgres

# Redis Health  
docker exec moneyprinterturbo-redis-new redis-cli ping

# API Health
curl -f http://localhost:8080/health

# MCP Health
curl -f http://localhost:8081/health

# WebUI Health
curl -f http://localhost:8501
```

## 🔧 Common Issues & Solutions

### Build Issues

#### Problem: Docker build fails with package installation errors
```bash
# Symptoms
ERROR: failed to solve: process "/bin/sh -c apt-get update && apt-get install -y ..." didn't complete successfully

# Solutions
1. Clear Docker build cache:
   docker system prune -a
   docker builder prune -a

2. Rebuild with no cache:
   docker-compose -f app/docker-compose.yml build --no-cache

3. Check network connectivity:
   docker run --rm alpine/curl curl -I https://archive.ubuntu.com
```

#### Problem: Python package installation fails
```bash
# Symptoms
ERROR: Could not find a version that satisfies the requirement [package]

# Solutions
1. Check requirements.txt syntax
2. Update pip in Dockerfile:
   RUN pip install --upgrade pip==23.3.1
   
3. Use specific package versions:
   fastapi==0.104.1
   uvicorn==0.24.0
```

### Service Startup Issues

#### Problem: API service fails to start
```bash
# Check logs
docker-compose -f app/docker-compose.yml logs api

# Common causes & solutions:
1. Database not ready:
   - Wait for postgres health check
   - Add depends_on with condition: service_healthy

2. Port conflicts:
   - Check: netstat -tulpn | grep 8080
   - Change port in docker-compose.yml

3. Environment variables missing:
   - Check .env file
   - Verify DATABASE_URL format
```

#### Problem: WebUI service crashes
```bash
# Check Streamlit logs
docker-compose -f app/docker-compose.yml logs webui

# Solutions:
1. Memory issues:
   - Increase container memory limit
   - Check: docker stats

2. Missing files:
   - Verify webui/Main.py exists
   - Check file permissions

3. Port binding issues:
   - Check if port 8501 is available
   - Verify firewall rules
```

#### Problem: MCP server won't start
```bash
# Check MCP server logs
docker-compose -f app/docker-compose.yml logs mcp-server

# Solutions:
1. Redis dependency:
   - Ensure Redis is healthy first
   - Check Redis connection string

2. Python module issues:
   - Verify mcp.server module exists
   - Check PYTHONPATH environment variable

3. Port conflicts:
   - Check port 8081 availability
   - Verify no other MCP processes running
```

### Database Issues

#### Problem: PostgreSQL connection refused
```bash
# Check PostgreSQL status
docker exec moneyprinterturbo-postgres-new pg_isready -U postgres

# Solutions:
1. Check container is running:
   docker ps | grep postgres

2. Verify environment variables:
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DB=mpt

3. Check data directory permissions:
   docker exec moneyprinterturbo-postgres-new ls -la /var/lib/postgresql/data

4. Reset PostgreSQL data:
   docker-compose -f app/docker-compose.yml down -v
   docker volume rm moneyprinterturbo_postgres_data
```

#### Problem: Redis connection issues
```bash
# Test Redis connection
docker exec moneyprinterturbo-redis-new redis-cli ping

# Solutions:
1. Check Redis configuration:
   docker exec moneyprinterturbo-redis-new redis-cli CONFIG GET "*"

2. Verify memory settings:
   - Check maxmemory policy
   - Monitor memory usage: docker stats

3. Check Redis logs:
   docker-compose -f app/docker-compose.yml logs redis

4. Reset Redis data:
   docker exec moneyprinterturbo-redis-new redis-cli FLUSHALL
```

### Network & Communication Issues

#### Problem: Services can't communicate
```bash
# Test inter-service connectivity
docker exec moneyprinterturbo-api-new ping redis
docker exec moneyprinterturbo-api-new ping db

# Solutions:
1. Check Docker network:
   docker network ls
   docker network inspect moneyprinterturbo_default

2. Verify service names in connection strings:
   - Use 'redis' not 'localhost' in containers
   - Use 'db' for PostgreSQL connection

3. Check firewall rules:
   sudo ufw status
   iptables -L DOCKER
```

#### Problem: Health checks failing
```bash
# Debug health check commands
docker exec moneyprinterturbo-api-new curl -f http://localhost:8080/health

# Solutions:
1. Increase health check timeouts:
   interval: 30s
   timeout: 10s
   retries: 5
   start_period: 60s

2. Verify health endpoints exist:
   - Check API routes
   - Test endpoints manually

3. Fix health check commands:
   # Wrong: CMD curl -f http://localhost:8080/health
   # Right: CMD ["curl", "-f", "http://localhost:8080/health"]
```

### Performance Issues

#### Problem: Slow container startup
```bash
# Monitor startup process
docker-compose -f app/docker-compose.yml up

# Solutions:
1. Optimize Dockerfile:
   - Use multi-stage builds
   - Combine RUN commands
   - Order layers by change frequency

2. Pre-pull base images:
   docker pull python:3.11-slim-bullseye
   docker pull postgres:15-alpine
   docker pull redis:7-alpine

3. Increase system resources:
   # Docker Desktop: Settings > Resources
   # Linux: Check available RAM/CPU
```

#### Problem: High memory usage
```bash
# Monitor resource usage
docker stats
docker system df

# Solutions:
1. Set memory limits:
   deploy:
     resources:
       limits:
         memory: 512M
         
2. Optimize application:
   - Review memory leaks
   - Use connection pooling
   - Implement caching strategies

3. Clean up unused resources:
   docker system prune -a
   docker volume prune
```

## 🔍 Debugging Techniques

### 1. Container Inspection
```bash
# Inspect running container
docker inspect moneyprinterturbo-api-new

# Check container processes
docker exec moneyprinterturbo-api-new ps aux

# View container filesystem
docker exec -it moneyprinterturbo-api-new /bin/bash
```

### 2. Log Analysis
```bash
# Follow logs in real-time
docker-compose -f app/docker-compose.yml logs -f

# Filter logs by service
docker-compose -f app/docker-compose.yml logs api

# Search logs for errors
docker-compose -f app/docker-compose.yml logs | grep -i error

# Export logs for analysis
docker-compose -f app/docker-compose.yml logs > deployment_logs.txt
```

### 3. Network Debugging
```bash
# Test connectivity between containers
docker exec moneyprinterturbo-api-new nc -zv redis 6379
docker exec moneyprinterturbo-api-new nc -zv db 5432

# Check DNS resolution
docker exec moneyprinterturbo-api-new nslookup redis
docker exec moneyprinterturbo-api-new nslookup db

# Monitor network traffic
docker exec moneyprinterturbo-api-new netstat -tulpn
```

### 4. Application-Level Debugging
```bash
# Check Python environment
docker exec moneyprinterturbo-api-new python3 -c "import sys; print(sys.path)"

# Test imports
docker exec moneyprinterturbo-api-new python3 -c "import fastapi; print('FastAPI OK')"

# Check file permissions
docker exec moneyprinterturbo-api-new ls -la /MoneyPrinterTurbo/

# Test startup script
docker exec moneyprinterturbo-api-new /MoneyPrinterTurbo/start_api.sh
```

## 🛠️ Recovery Procedures

### 1. Complete Reset
```bash
# Stop all services
docker-compose -f app/docker-compose.yml down

# Remove all containers and volumes
docker-compose -f app/docker-compose.yml down -v

# Clean up images
docker system prune -a

# Rebuild and restart
docker-compose -f app/docker-compose.yml build --no-cache
docker-compose -f app/docker-compose.yml up -d
```

### 2. Individual Service Reset
```bash
# Reset specific service
docker-compose -f app/docker-compose.yml stop api
docker-compose -f app/docker-compose.yml rm api
docker-compose -f app/docker-compose.yml build api
docker-compose -f app/docker-compose.yml up -d api
```

### 3. Database Reset
```bash
# Backup data first (if needed)
docker exec moneyprinterturbo-postgres-new pg_dump -U postgres mpt > backup.sql

# Reset PostgreSQL
docker-compose -f app/docker-compose.yml stop db
docker volume rm moneyprinterturbo_postgres_data
docker-compose -f app/docker-compose.yml up -d db

# Reset Redis
docker exec moneyprinterturbo-redis-new redis-cli FLUSHALL
```

## 📊 Monitoring & Maintenance

### 1. Health Monitoring Script
```bash
#!/bin/bash
# health_check.sh

services=("api" "webui" "mcp-server" "redis" "db")

for service in "${services[@]}"; do
    health=$(docker inspect --format='{{.State.Health.Status}}' moneyprinterturbo-${service}-new 2>/dev/null || echo "none")
    echo "$service: $health"
done
```

### 2. Log Rotation
```bash
# Configure log rotation in docker-compose.yml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 3. Backup Strategy
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)

# Backup PostgreSQL
docker exec moneyprinterturbo-postgres-new pg_dump -U postgres mpt > "backup_${DATE}.sql"

# Backup volumes
docker run --rm -v moneyprinterturbo_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_data_${DATE}.tar.gz -C /data .
```

## 🚨 Emergency Procedures

### 1. Service Down Emergency
```bash
# Quick restart all services
docker-compose -f app/docker-compose.yml restart

# Check critical services first
docker-compose -f app/docker-compose.yml up -d db redis
sleep 30
docker-compose -f app/docker-compose.yml up -d api mcp-server webui
```

### 2. Data Recovery
```bash
# If data corruption occurs:
1. Stop all services
2. Backup current state
3. Restore from last known good backup
4. Verify data integrity
5. Restart services
```

### 3. Performance Emergency
```bash
# If system is unresponsive:
1. Check system resources: htop, df -h
2. Identify resource-hungry containers: docker stats
3. Scale down non-critical services
4. Clear logs and temporary files
5. Restart services one by one
```

## 📋 Maintenance Checklist

### Daily
- [ ] Check service health status
- [ ] Monitor resource usage
- [ ] Review error logs
- [ ] Verify backup completion

### Weekly
- [ ] Update Docker images
- [ ] Clean unused resources
- [ ] Performance analysis
- [ ] Security updates

### Monthly
- [ ] Full system backup
- [ ] Disaster recovery test
- [ ] Configuration review
- [ ] Capacity planning

---

## 🔗 Additional Resources

- **Docker Documentation**: https://docs.docker.com/
- **Docker Compose Reference**: https://docs.docker.com/compose/
- **PostgreSQL Docker**: https://hub.docker.com/_/postgres
- **Redis Docker**: https://hub.docker.com/_/redis
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Streamlit Documentation**: https://docs.streamlit.io/

---

*Generated by Testing & Troubleshooting Expert*
*Last Updated: 2025-07-29*