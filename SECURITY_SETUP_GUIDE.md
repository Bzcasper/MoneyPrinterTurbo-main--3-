# Security Setup Guide for MoneyPrinter Turbo

This guide helps you securely configure MoneyPrinter Turbo after the security remediation.

## 🚀 Quick Setup (5 minutes)

### 1. Create Your Environment File
```bash
# Copy the template
cp .env.template .env

# Edit with your actual values
nano .env
```

### 2. Generate Secure Secrets
```bash
# Generate JWT secret (Linux/macOS)
echo "JWT_SECRET=$(openssl rand -hex 32)" >> .env

# Generate session secret
echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env
```

### 3. Get New API Keys

#### 🔑 Google Gemini API Key (CRITICAL - REPLACE IMMEDIATELY)
1. Visit [Google AI Studio](https://ai.google.dev/)
2. Create a new project or select existing
3. Generate a new API key
4. Add to `.env`: `GEMINI_API_KEY=your-new-key-here`

**⚠️ IMPORTANT:** The old key `AIzaSyAp9IqLHulnWCCGIv8umVmobJqR917vTEw` was exposed and must be replaced!

#### 🎥 Pixabay API Key (REPLACE RECOMMENDED)
1. Visit [Pixabay API](https://pixabay.com/api/docs/)
2. Create account and generate API key
3. Add to `.env`: `PIXABAY_API_KEY=your-new-key-here`

**⚠️ SECURITY NOTE:** The old key `46693365-cf8143b99556595ef68972852` may be compromised.

### 4. Configure Database
```bash
# PostgreSQL example
DATABASE_URL=postgresql://username:password@localhost:5432/moneyprinter

# SQLite example
DATABASE_URL=sqlite:///./data/moneyprinter.db
```

### 5. Verify Setup
```bash
# Test configuration loading
python -c "import toml; print('Config loads successfully')"

# Verify no hardcoded secrets remain
grep -r "AIzaSyAp9IqLHulnWCCGIv8umVmobJqR917vTEw" . || echo "✅ No exposed keys found"
```

## 🔧 Full Configuration Guide

### Required Environment Variables

```bash
# === CRITICAL SECURITY ===
JWT_SECRET=your-jwt-secret-here                    # Generate with: openssl rand -hex 32
SECRET_KEY=your-session-secret-here                # Generate with: openssl rand -hex 32
GEMINI_API_KEY=your-new-gemini-key-here           # From Google AI Studio
PIXABAY_API_KEY=your-new-pixabay-key-here         # From Pixabay API

# === DATABASE ===
DATABASE_URL=postgresql://user:pass@host:port/db   # Your database connection

# === SUPABASE (if used) ===
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key    # Server-side only!

# === OPTIONAL APIs ===
OPENAI_API_KEY=your-openai-key-here               # For OpenAI models
PEXELS_API_KEY=your-pexels-key-here               # For Pexels videos
AZURE_SPEECH_KEY=your-azure-speech-key            # For TTS
```

### Environment-Specific Configurations

#### Development
```bash
NODE_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG
```

#### Production
```bash
NODE_ENV=production
DEBUG=false
LOG_LEVEL=INFO
# Add production-specific overrides
```

## 🛡️ Security Best Practices

### 1. File Permissions
```bash
# Secure your .env file
chmod 600 .env
chown $(whoami):$(whoami) .env
```

### 2. Git Security
```bash
# Verify .env is ignored
git check-ignore .env
# Should output: .env

# Never add .env to git
git add .env  # DON'T DO THIS!
```

### 3. Key Rotation Schedule
- **Monthly:** Rotate API keys
- **Quarterly:** Review access permissions
- **Immediately:** If keys are suspected to be compromised

### 4. Monitoring
```bash
# Check for accidental credential commits
git log --grep="api_key\|secret\|password" --oneline
```

## 🚨 Incident Response

### If Keys Are Compromised:

1. **Immediate Actions**
   ```bash
   # Rotate all affected keys immediately
   # Update .env with new keys
   # Restart all services
   ```

2. **Investigation**
   ```bash
   # Check git history for exposure
   git log -p --all -S "your-api-key"
   
   # Check for usage in logs
   grep -r "your-api-key" logs/
   ```

3. **Remediation**
   - Generate new keys from providers
   - Update environment variables
   - Review access logs from API providers
   - Document incident

## 🔍 Troubleshooting

### Common Issues

#### Configuration Not Loading
```bash
# Check file exists
ls -la .env

# Check file format
cat .env | head -5

# Check for BOM or encoding issues
file .env
```

#### API Keys Not Working
```bash
# Verify key format
echo $GEMINI_API_KEY | wc -c  # Should be reasonable length
echo $PIXABAY_API_KEY | wc -c

# Test API key manually
curl -H "Authorization: Bearer $GEMINI_API_KEY" https://generativelanguage.googleapis.com/v1/models
```

#### Database Connection Issues
```bash
# Test database connection
python -c "
import os
from sqlalchemy import create_engine
engine = create_engine(os.getenv('DATABASE_URL'))
print('Database connection successful')
"
```

## 📱 Docker Setup

### Docker Compose with Secrets
```yaml
services:
  app:
    build: .
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - DATABASE_URL=${DATABASE_URL}
    env_file:
      - .env
```

### Kubernetes Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: moneyprinter-secrets
type: Opaque
stringData:
  gemini-api-key: ${GEMINI_API_KEY}
  database-url: ${DATABASE_URL}
```

## 🎯 Testing Your Setup

### 1. Configuration Test
```python
import os
import toml

# Load and verify configuration
config = toml.load('config.toml')
print("✅ Configuration loaded successfully")

# Check environment variables are being used
if '${GEMINI_API_KEY' in str(config):
    print("✅ Environment variables properly configured")
else:
    print("❌ Configuration may have hardcoded values")
```

### 2. API Connectivity Test
```python
import os
import requests

# Test Gemini API (example)
api_key = os.getenv('GEMINI_API_KEY')
if api_key and api_key != 'your-new-gemini-key-here':
    print("✅ Gemini API key configured")
else:
    print("❌ Gemini API key not set")
```

## 📞 Support

- **Documentation Issues:** Check this guide first
- **Security Concerns:** Follow incident response procedures
- **API Key Problems:** Contact respective API providers
- **Configuration Help:** Refer to `.env.template` comments

---

**Remember:** Security is an ongoing process. Regularly review and update your configuration following these guidelines.