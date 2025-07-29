# MoneyPrinterTurbo Northflank Deployment Guide

## 🚀 Complete Northflank Deployment Guide

This guide provides step-by-step instructions for deploying MoneyPrinterTurbo to Northflank cloud platform with proper configuration, security, and scaling.

## 📋 Prerequisites

### 1. Northflank Account Setup
- Sign up at [Northflank](https://northflank.com)
- Verify your account and set up billing
- Install Northflank CLI (optional but recommended)

### 2. Repository Setup
- Fork or clone MoneyPrinterTurbo to your GitHub account
- Ensure your repository is public or configure private repo access
- Note your GitHub username and repository URL

### 3. Required API Keys
Gather the following API keys before deployment:
- **Pexels API Key**: Register at https://www.pexels.com/api/
- **Pixabay API Key**: Register at https://pixabay.com/api/docs/
- **OpenAI API Key**: Get from https://platform.openai.com/api-keys
- **Supabase Keys** (optional): From your Supabase project

## 🏗️ Deployment Steps

### Step 1: Create Project

1. Log into Northflank dashboard
2. Click "Create Project"
3. Name: `moneyprinterturbo`
4. Description: "AI-powered video generation platform"
5. Region: Choose closest to your users (e.g., `europe-west-1`, `us-east-1`)

### Step 2: Configure Secrets

Create secrets in your project:

1. Go to **Project Settings > Secrets**
2. Create a new secret group: `mpt-secrets`
3. Add the following secrets:

```env
# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-secure-password-here
POSTGRES_DB=mpt
DATABASE_URL=postgresql://postgres:your-secure-password-here@postgres-service:5432/mpt

# Redis Configuration  
REDIS_HOST=redis-service
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password-here

# Application Secrets
JWT_SECRET=your-jwt-secret-key-256-bit
MCP_JWT_SECRET=your-mcp-secret-key-256-bit

# External API Keys
PEXELS_API_KEY=your-pexels-api-key
PIXABAY_API_KEY=your-pixabay-api-key
OPENAI_API_KEY=your-openai-api-key

# Optional: Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key
```

### Step 3: Deploy Database Services

#### PostgreSQL Database
1. Go to **Services > Create Service**
2. Choose **Database > PostgreSQL**
3. Configuration:
   - Name: `postgres`
   - Version: `15`
   - Plan: `nf-compute-10` (adjust based on needs)
   - Storage: `20GB`
4. Environment variables:
   - Link to your `mpt-secrets` for database credentials
5. Deploy and wait for "Running" status

#### Redis Cache
1. Create new service: **Database > Redis**
2. Configuration:
   - Name: `redis`
   - Version: `7`
   - Plan: `nf-compute-5`
   - Memory: `512MB`
3. Advanced settings:
   - Max memory policy: `allkeys-lru`
   - Persistence: `AOF enabled`
4. Link Redis password from secrets
5. Deploy service

### Step 4: Deploy Application Services

#### API Backend Service
1. Create **Deployment Service**
2. Configuration:
   - Name: `api`
   - Plan: `nf-compute-20` (2 CPU, 4GB RAM)
   - Instances: `2` (for high availability)
3. **Build Settings**:
   - Source: GitHub repository
   - Branch: `main`
   - Dockerfile path: `./app/Dockerfile`
   - Build context: `/`
4. **Environment Variables**: Link all required secrets
5. **Ports**: Expose port `8080` as HTTP (public)
6. **Dependencies**: Add `postgres` and `redis`
7. **Health Check**: Path `/health`
8. Deploy service

#### WebUI Frontend Service  
1. Create **Deployment Service**
2. Configuration:
   - Name: `webui`
   - Plan: `nf-compute-10` (1 CPU, 2GB RAM)
   - Instances: `2`
3. **Build Settings**: Same as API
4. **Environment Variables**: Link database and Supabase secrets
5. **Ports**: Expose port `8501` as HTTP (public)
6. **Dependencies**: Add `api`
7. Deploy service

#### MCP Server Service
1. Create **Deployment Service**
2. Configuration:
   - Name: `mcp-server`
   - Plan: `nf-compute-15` (1.5 CPU, 3GB RAM)
   - Instances: `2`
3. **Build Settings**: Same repository
4. **Environment Variables**: Link Redis and MCP secrets
5. **Ports**: Expose port `8081` as HTTP (public, for WebSocket)
6. **Dependencies**: Add `redis`
7. **Command Override**: `python run_mcp_server.py`
8. Deploy service

### Step 5: Configure Domain and SSL

1. Go to **Project Settings > Domains**
2. Add custom domains for each service:
   - API: `api.yourdomain.com`
   - WebUI: `app.yourdomain.com`
   - MCP: `mcp.yourdomain.com`
3. Configure DNS CNAME records to point to Northflank URLs
4. SSL certificates are automatically provisioned

## 🔧 Advanced Configuration

### Auto-Scaling Setup

For production workloads, configure auto-scaling:

```yaml
scaling:
  autoscaling:
    enabled: true
    minInstances: 2
    maxInstances: 10
    targetCPU: 70
    targetMemory: 80
```

### Load Balancing

Northflank automatically load balances between instances. For custom load balancing:

1. Use Northflank's built-in load balancer
2. Configure health checks for each service
3. Set appropriate timeouts and retry policies

### Monitoring and Logging

1. **Built-in Monitoring**: Available in Northflank dashboard
2. **Custom Metrics**: Configure application metrics
3. **Log Aggregation**: Centralized logging available
4. **Alerting**: Set up alerts for service health

### Environment-Specific Configuration

Create separate projects for different environments:

- **Development**: `moneyprinterturbo-dev`
- **Staging**: `moneyprinterturbo-staging`  
- **Production**: `moneyprinterturbo-prod`

## 🔐 Security Best Practices

### Secret Management
- Use Northflank's secret management
- Rotate secrets regularly
- Never commit secrets to repository
- Use least-privilege access principles

### Network Security
- Services communicate through internal network
- Only expose necessary ports publicly
- Use HTTPS for all external communication
- Configure CORS policies appropriately

### Database Security
- Use strong passwords (generated)
- Enable connection encryption
- Regular security updates
- Backup policies configured

## 📊 Monitoring and Maintenance

### Health Monitoring
Each service includes health checks:
- **API**: `GET /health`
- **WebUI**: `GET /`
- **MCP**: TCP check on port 8081
- **Databases**: Built-in health checks

### Performance Monitoring
Monitor these key metrics:
- **CPU Usage**: Keep below 70%
- **Memory Usage**: Keep below 80%
- **Response Times**: API < 500ms
- **Error Rates**: < 1%

### Backup Strategy
1. **Database Backups**: Daily automated backups
2. **Redis Persistence**: AOF enabled
3. **Application Data**: Stored in persistent volumes

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
1. Check logs in Northflank dashboard
2. Verify all environment variables are set
3. Check service dependencies are running
4. Validate Docker build process

#### Database Connection Issues
1. Verify database service is running
2. Check connection string format
3. Confirm network connectivity
4. Validate credentials

#### Performance Issues
1. Check resource utilization
2. Scale up instances if needed
3. Optimize database queries
4. Review application logs

### Debug Commands

Access service logs:
```bash
# Via Northflank CLI
northflank logs service --project moneyprinterturbo --service api

# Via dashboard
Go to Service > Logs tab
```

## 💰 Cost Optimization

### Right-Sizing Services
- Start with smaller plans and scale up
- Monitor resource usage regularly
- Use auto-scaling to handle peaks
- Consider spot instances for non-critical workloads

### Database Optimization
- Choose appropriate database sizes
- Use read replicas for read-heavy workloads
- Implement connection pooling
- Optimize queries and indexes

## 🔄 CI/CD Integration

### Automated Deployments
1. Configure webhook deployment triggers
2. Set up staging environment for testing
3. Implement blue-green deployments
4. Use feature flags for gradual rollouts

### Testing Pipeline
1. Run tests before deployment
2. Perform security scanning
3. Validate environment configuration
4. Monitor deployment health

## 📞 Support and Resources

- **Northflank Documentation**: https://northflank.com/docs
- **Community Support**: Northflank Discord/Slack
- **Technical Support**: Available with paid plans
- **Status Page**: https://status.northflank.com

## 🎯 Next Steps

After successful deployment:

1. **Configure monitoring alerts**
2. **Set up backup schedules**
3. **Implement CI/CD pipeline**
4. **Performance testing and optimization**
5. **Security audit and hardening**

---

## 📋 Deployment Checklist

- [ ] Northflank account created and verified
- [ ] GitHub repository configured
- [ ] All API keys gathered
- [ ] Project created in Northflank
- [ ] Secrets configured
- [ ] PostgreSQL service deployed
- [ ] Redis service deployed
- [ ] API service deployed and healthy
- [ ] WebUI service deployed and accessible
- [ ] MCP server deployed and functional
- [ ] Custom domains configured (optional)
- [ ] SSL certificates active
- [ ] Auto-scaling configured
- [ ] Monitoring and alerts set up
- [ ] Backup strategy implemented

**Estimated Deployment Time**: 30-60 minutes

**Monthly Cost Estimate**: $50-200 depending on usage and plan selection

---

**Last Updated**: July 28, 2025
**Version**: 1.0.0
