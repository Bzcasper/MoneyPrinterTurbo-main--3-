# Quick Start Guide for Northflank Deployment

## 🚀 5-Minute Quick Start

### Prerequisites
1. Northflank account: [Sign up here](https://northflank.com)
2. GitHub repository with MoneyPrinterTurbo code
3. API keys for external services

### Step 1: Prepare Your Environment
```bash
# Clone or fork the repository
git clone https://github.com/your-username/MoneyPrinterTurbo.git
cd MoneyPrinterTurbo

# Copy environment template
cp .env.northflank.example .env.northflank

# Edit with your values
nano .env.northflank
```

### Step 2: Configure API Keys
Get these API keys before deployment:
- **Pexels**: https://www.pexels.com/api/
- **Pixabay**: https://pixabay.com/api/docs/
- **OpenAI**: https://platform.openai.com/api-keys

### Step 3: Deploy to Northflank

#### Option A: Automated Script
```bash
# Edit the script with your GitHub username
nano deploy-northflank.sh
# Set: GITHUB_USERNAME="your-username"

# Run deployment
./deploy-northflank.sh deploy
```

#### Option B: Manual Deployment
1. Upload `northflank-deploy.yaml` to Northflank
2. Follow the detailed guide in `docs/NORTHFLANK_DEPLOYMENT_GUIDE.md`

### Step 4: Access Your Services
After deployment (5-10 minutes):
- **WebUI**: `https://webui-xxx.northflank.app`
- **API**: `https://api-xxx.northflank.app`
- **MCP Server**: `wss://mcp-server-xxx.northflank.app`

### Step 5: Verify Deployment
```bash
# Check status
./deploy-northflank.sh status

# View logs if needed
./deploy-northflank.sh logs api
```

## 🔧 Key Configuration Points

### Required Secrets
Set these in Northflank dashboard under Project > Secrets:
```
POSTGRES_PASSWORD=secure-password
REDIS_PASSWORD=secure-password
JWT_SECRET=64-character-hex-string
OPENAI_API_KEY=your-openai-key
PEXELS_API_KEY=your-pexels-key
PIXABAY_API_KEY=your-pixabay-key
```

### Service Ports
- **API**: 8080 (FastAPI backend)
- **WebUI**: 8501 (Streamlit frontend)
- **MCP Server**: 8081 (WebSocket server)
- **PostgreSQL**: 5432 (internal)
- **Redis**: 6379 (internal)

### Resource Requirements (Minimum)
- **Total vCPUs**: 6.5
- **Total RAM**: 13GB
- **Storage**: 20GB PostgreSQL
- **Estimated Cost**: $50-150/month

## 🛟 Troubleshooting

### Common Issues
1. **Build Fails**: Check Dockerfile and dependencies
2. **Service Won't Start**: Verify environment variables
3. **Database Connection**: Ensure PostgreSQL is running
4. **Performance Issues**: Scale up compute plans

### Getting Help
- Check service logs in Northflank dashboard
- Review detailed guide: `docs/NORTHFLANK_DEPLOYMENT_GUIDE.md`
- Northflank support: support@northflank.com

## 📊 Monitoring
Once deployed, monitor:
- Service health in Northflank dashboard
- Resource utilization (CPU/Memory)
- Application logs for errors
- Response times and performance

## 🔄 Updates
To update your deployment:
```bash
# Push code changes to GitHub
git push origin main

# Northflank auto-deploys from main branch
# Or trigger manual deployment in dashboard
```

---

**Total Deployment Time**: 10-20 minutes  
**Difficulty**: Beginner-friendly  
**Support**: Northflank documentation + community
