# MoneyPrinterTurbo Modal Deployment Guide

🚀 **Deploy MoneyPrinterTurbo to Modal's serverless cloud infrastructure with GPU support**

## Overview

This guide will help you deploy MoneyPrinterTurbo on Modal, a serverless platform that provides:
- **GPU Access**: Tesla T4 GPUs for video processing
- **Auto-scaling**: Automatic scaling based on demand
- **Pay-per-use**: Only pay for actual computation time
- **Global CDN**: Fast access worldwide
- **Zero DevOps**: No server management required

## Prerequisites

### 1. System Requirements
- Python 3.11+
- Git
- Modal CLI

### 2. Required API Keys
You'll need the following API keys for full functionality:

| Service | Purpose | Required |
|---------|---------|----------|
| OpenAI API | AI script generation | Yes |
| Azure Speech | Text-to-speech | Yes |
| Pexels API | Stock video clips | Yes |
| Google AI | Alternative AI provider | Optional |

## Step-by-Step Deployment

### 1. Install Modal CLI

```bash
# Install Modal
pip install modal

# Authenticate with Modal
modal token new
```

### 2. Clone and Prepare MoneyPrinterTurbo

```bash
# Clone the repository
git clone https://github.com/harry0703/MoneyPrinterTurbo.git
cd MoneyPrinterTurbo

# Copy the Modal deployment script
curl -O https://raw.githubusercontent.com/your-repo/modal_deployment.py
```

### 3. Set Up API Keys as Modal Secrets

Create Modal secrets for your API keys:

```bash
# OpenAI API Key
modal secret create openai-secret OPENAI_API_KEY=sk-your-openai-key-here

# Azure Speech Services
modal secret create azure-speech \
  AZURE_SPEECH_KEY=your-azure-key \
  AZURE_SPEECH_REGION=eastus

# Pexels API Key
modal secret create pexels-api PEXELS_API_KEY=your-pexels-key

# Optional: Google AI
modal secret create google-ai GOOGLE_AI_KEY=your-google-key
```

### 4. Deploy to Modal

```bash
# Deploy the application
modal deploy modal_deployment.py
```

### 5. Access Your Deployed App

After deployment, Modal will provide URLs for:
- **Web Interface**: `https://your-app--streamlit-app.modal.run`
- **API Endpoint**: `https://your-app--api-generate-video.modal.run`
- **Health Check**: `https://your-app--health-check.modal.run`

## Using the Deployed Application

### Web Interface

1. Visit the Streamlit web interface URL
2. Enter your video topic/subject
3. Optionally customize script and keywords
4. Click "Generate Video on Modal"
5. Download your generated video

### API Usage

You can also use the REST API directly:

```python
import requests

# API endpoint
url = "https://your-app--api-generate-video.modal.run"

# Request payload
payload = {
    "video_subject": "The future of artificial intelligence",
    "video_aspect": "9:16",
    "voice_name": "en-US-JennyNeural",
    "subtitle_enabled": True
}

# Make request
response = requests.post(url, json=payload)
result = response.json()

print(f"Success: {result['success']}")
print(f"Task ID: {result['task_id']}")
```

### Download Generated Videos

```python
import requests

# Download endpoint
download_url = f"https://your-app--download-video.modal.run?task_id={task_id}"

response = requests.get(download_url)
with open("generated_video.mp4", "wb") as f:
    f.write(response.content)
```

## Configuration Options

### GPU Types

You can modify the GPU type in `modal_deployment.py`:

```python
# Available GPU options
gpu="T4"      # Tesla T4 (recommended for most use cases)
gpu="A10G"    # A10G (faster processing)
gpu="A100"    # A100 (highest performance)
```

### Memory and Timeout

Adjust memory and timeout settings:

```python
@app.function(
    memory=8192,    # RAM in MB (4096-32768)
    timeout=3600,   # Timeout in seconds
    # ... other settings
)
```

### Concurrent Users

Control how many users can generate videos simultaneously:

```python
allow_concurrent_inputs=10  # Max concurrent video generations
```

## Monitoring and Logs

### View Logs

```bash
# View real-time logs
modal logs moneyprinter-turbo

# View specific function logs
modal logs moneyprinter-turbo.generate_video
```

### Monitor Usage

```bash
# Check app status
modal app list

# View volume usage
modal volume list
```

## Cost Optimization

### Estimated Costs

| Resource | Cost per Hour | Typical Video (3 min) |
|----------|---------------|----------------------|
| T4 GPU | ~$0.60/hour | ~$0.05 |
| CPU (4 cores) | ~$0.10/hour | ~$0.01 |
| Storage | ~$0.15/GB/month | ~$0.001 |

### Tips to Reduce Costs

1. **Use T4 GPUs** for most workloads (cheapest option)
2. **Set appropriate timeouts** to avoid runaway processes
3. **Use keep_warm=0** for infrequent usage
4. **Monitor concurrent limits** to control peak costs

## Troubleshooting

### Common Issues

#### 1. API Key Errors
```bash
# Check if secrets are properly set
modal secret list

# Update a secret
modal secret create openai-secret OPENAI_API_KEY=new-key --force
```

#### 2. Out of Memory Errors
```python
# Increase memory allocation
memory=16384  # 16GB instead of 8GB
```

#### 3. Timeout Errors
```python
# Increase timeout for longer videos
timeout=7200  # 2 hours instead of 1 hour
```

#### 4. Volume Issues
```bash
# Clear old videos to free space
modal volume ls moneyprinter-storage
modal shell  # Access volume and clean up files
```

### Debug Mode

Enable debug logging in your deployment:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Configuration

### Custom Domain

Set up a custom domain for your app:

```bash
# In Modal dashboard, configure custom domain
# Point your domain to the provided CNAME
```

### Environment Variables

Add additional environment variables:

```python
@app.function(
    secrets=[
        Secret.from_name("openai-secret"),
        Secret.from_name("custom-config")
    ]
)
```

### Multiple Environments

Deploy to different environments:

```bash
# Production
modal deploy modal_deployment.py --env prod

# Staging
modal deploy modal_deployment.py --env staging
```

## Security Best Practices

1. **Never hardcode API keys** in your code
2. **Use Modal secrets** for all sensitive data
3. **Regularly rotate API keys**
4. **Monitor usage** for unexpected activity
5. **Set appropriate timeouts** to prevent abuse

## Scaling Considerations

### High Traffic

For high-traffic scenarios:

```python
@app.function(
    allow_concurrent_inputs=50,  # Increase concurrent limit
    keep_warm=5,                 # Keep instances warm
    container_idle_timeout=300,  # Keep containers alive longer
)
```

### Multi-Region Deployment

Deploy to multiple regions for global coverage:

```python
# Configure region in Modal dashboard
# Or use Modal's global load balancing
```

## Support and Resources

### Getting Help

- **Modal Documentation**: https://modal.com/docs
- **Modal Discord**: https://discord.gg/modal
- **GitHub Issues**: Create issues for bugs or feature requests

### Useful Commands

```bash
# Quick reference
modal --help                    # Show all commands
modal app list                  # List deployed apps
modal logs <app-name>          # View logs
modal shell                    # Access container shell
modal volume ls <volume-name>  # List volume contents
modal secret list              # List secrets
```

## Migration from Local Setup

If you're migrating from a local MoneyPrinterTurbo setup:

1. **Export your settings** from local config files
2. **Create Modal secrets** with your API keys
3. **Test with a simple video** before full migration
4. **Update any custom scripts** to use Modal APIs

## Conclusion

You now have MoneyPrinterTurbo running on Modal's serverless infrastructure! Your application will:

- ✅ **Scale automatically** based on demand
- ✅ **Only cost money** when generating videos
- ✅ **Provide global access** via Modal's CDN
- ✅ **Handle high concurrent load** without server management
- ✅ **Maintain persistent storage** for generated videos

Happy video generating! 🎬✨

---

**Need help?** Check the troubleshooting section above or reach out on the Modal community Discord.