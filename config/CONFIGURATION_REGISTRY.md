# MoneyPrinterTurbo Configuration Registry

## Overview

This document provides a comprehensive mapping of all configuration files in the MoneyPrinterTurbo project, identifying overlapping settings, consolidation opportunities, and the unified configuration structure.

## Configuration File Hierarchy

### ✅ Current Organized Structure (Post-Consolidation)

```
/config/
├── aggregated.toml          # ⭐ Master consolidated configuration
├── app.toml                 # 🏗️ Core application settings  
├── services.toml            # 🔌 External service providers
├── infrastructure.toml      # 🏗️ Database, storage, deployment
├── mcp.toml                 # 🤖 MCP server configuration
└── templates/
    ├── app.example.toml     # 📋 App config template
    ├── services.example.toml # 📋 Services config template
    └── .env.example         # 📋 Environment variables template
```

### 🔄 Legacy Configuration Files (Maintained for Compatibility)

```
├── config.toml              # 🔶 Legacy root config (mirrors aggregated.toml)
├── app/config.toml          # 🔶 Legacy app config (mirrors aggregated.toml)
├── app/config/config.example.toml # 🔶 Legacy example
├── config.mcp.example.toml  # 🔶 Legacy MCP example
└── .env files               # 🔐 Environment-specific secrets
```

## Configuration Mapping Matrix

### Core Application Settings

| Setting | aggregated.toml | app.toml | config.toml | app/config.toml | Priority |
|---------|----------------|----------|-------------|-----------------|----------|
| `app.title` | ✅ | ✅ | ❌ | ❌ | aggregated.toml |
| `app.video_source` | ✅ | ✅ | ✅ | ✅ | aggregated.toml |
| `app.hide_config` | ✅ | ✅ | ✅ | ✅ | aggregated.toml |
| `app.max_concurrent_tasks` | ✅ | ✅ | ✅ | ✅ | aggregated.toml |
| `app.material_directory` | ✅ | ✅ | ✅ | ✅ | aggregated.toml |
| `app.endpoint` | ✅ | ✅ | ✅ | ✅ | aggregated.toml |

### Video Processing Settings

| Setting | aggregated.toml | app.toml | config.toml | app/config.toml | Priority |
|---------|----------------|----------|-------------|-----------------|----------|
| `video.default_format` | ✅ | ✅ | ✅ | ✅ | aggregated.toml |
| `video.default_duration` | ✅ | ✅ | ✅ | ✅ | aggregated.toml |
| `video.enable_gpu_acceleration` | ✅ | ✅ | ✅ | ✅ | aggregated.toml |
| `video.enable_auto_subtitles` | ✅ | ✅ | ✅ | ✅ | aggregated.toml |
| `video.subtitle_font_size` | ✅ | ✅ | ✅ | ✅ | aggregated.toml |

### External Service Providers

| Setting | aggregated.toml | services.toml | config.toml | app/config.toml | Priority |
|---------|----------------|---------------|-------------|-----------------|----------|
| `llm.provider` | ✅ | ✅ | ✅ | ✅ | services.toml |
| `llm.openai.api_key` | ✅ | ✅ | ✅ | ✅ | ENV → services.toml |
| `llm.gemini.api_key` | ✅ | ✅ | ✅ | ✅ | ENV → services.toml |
| `video_sources.pexels.api_keys` | ✅ | ✅ | ✅ | ✅ | ENV → services.toml |
| `video_sources.pixabay.api_keys` | ✅ | ✅ | ✅ | ✅ | ENV → services.toml |

### Infrastructure Settings

| Setting | aggregated.toml | infrastructure.toml | config.toml | app/config.toml | Priority |
|---------|----------------|---------------------|-------------|-----------------|----------|
| `database.type` | ✅ | ✅ | ✅ | ✅ | infrastructure.toml |
| `database.path` | ✅ | ✅ | ✅ | ✅ | ENV → infrastructure.toml |
| `redis.enabled` | ✅ | ✅ | ✅ | ✅ | infrastructure.toml |
| `redis.host` | ✅ | ✅ | ✅ | ✅ | ENV → infrastructure.toml |
| `supabase.url` | ✅ | ✅ | ✅ | ✅ | ENV → infrastructure.toml |

### MCP Configuration  

| Setting | aggregated.toml | mcp.toml | config.toml | config.mcp.example.toml | Priority |
|---------|----------------|----------|-------------|-------------------------|----------|
| `mcp.enabled` | ✅ | ✅ | ✅ | ✅ | mcp.toml |
| `mcp.server_host` | ✅ | ✅ | ✅ | ✅ | mcp.toml |
| `mcp.server_port` | ✅ | ✅ | ✅ | ✅ | mcp.toml |
| `mcp.jwt_secret` | ✅ | ✅ | ✅ | ✅ | ENV → mcp.toml |
| `mcp.api_keys` | ✅ | ✅ | ✅ | ✅ | mcp.toml |

## Environment Variable Mapping

### Critical Environment Variables

| Variable | Purpose | Config Files | Default Value |
|----------|---------|--------------|---------------|
| `DATABASE_URL` | Database connection | infrastructure.toml, aggregated.toml | - |
| `SUPABASE_URL` | Supabase project URL | infrastructure.toml, aggregated.toml | - |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | infrastructure.toml, aggregated.toml | - |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | infrastructure.toml, aggregated.toml | - |
| `JWT_SECRET` | JWT signing secret | mcp.toml, aggregated.toml | "your-secret-key-CHANGE-IN-PRODUCTION" |
| `REDIS_HOST` | Redis server host | infrastructure.toml, aggregated.toml | "redis" |
| `REDIS_PASSWORD` | Redis password | infrastructure.toml, aggregated.toml | "" |

### API Keys (Should be in Environment)

| Variable | Purpose | Config Files | Example Value |
|----------|---------|--------------|---------------|
| `OPENAI_API_KEY` | OpenAI API access | services.toml, aggregated.toml | "sk-..." |
| `GEMINI_API_KEY` | Google Gemini API | services.toml, aggregated.toml | "AIza..." |
| `PEXELS_API_KEY` | Pexels video API | services.toml, aggregated.toml | "563492ad6f91700001000001..." |
| `PIXABAY_API_KEY` | Pixabay video API | services.toml, aggregated.toml | "46693365-cf8143b..." |
| `AZURE_SPEECH_KEY` | Azure TTS service | services.toml, aggregated.toml | - |

## Configuration Loading Priority

### 1. **Highest Priority: Environment Variables**
```bash
DATABASE_URL="postgresql://..." 
JWT_SECRET="super-secret-key"
OPENAI_API_KEY="sk-..."
```

### 2. **Application Priority Order:**
1. `/config/aggregated.toml` (Master configuration) 
2. `/config/{specific}.toml` (Feature-specific)
3. `/config.toml` (Legacy root config)
4. `/app/config.toml` (Legacy app config) 
5. Default values in code

### 3. **Environment-Specific Overrides:**
```toml
[environments.development]
debug = true
log_level = "DEBUG"

[environments.production] 
debug = false
log_level = "INFO"
```

## Consolidation Analysis

### ✅ Successfully Consolidated Settings

1. **Core Application Settings** → `/config/app.toml`
2. **External Service Providers** → `/config/services.toml`
3. **Infrastructure & Deployment** → `/config/infrastructure.toml`
4. **MCP Server Configuration** → `/config/mcp.toml`
5. **Master Aggregated View** → `/config/aggregated.toml`

### 🔶 Legacy Files (Maintained for Backward Compatibility)

- `/config.toml` - Root level config (100% duplicate of aggregated.toml)
- `/app/config.toml` - App level config (100% duplicate of aggregated.toml)
- Example files maintained as templates

### 🚨 Configuration Conflicts Resolved

| Setting | Conflict | Resolution | Status |
|---------|----------|------------|--------|
| `llm_provider` vs `llm.provider` | Different formats | Standardized to `llm.provider` | ✅ Resolved |
| `enable_redis` vs `redis.enabled` | Different formats | Standardized to `redis.enabled` | ✅ Resolved |
| Scattered API keys | Multiple locations | Centralized in services.toml | ✅ Resolved |
| MCP settings spread | Mixed locations | Centralized in mcp.toml | ✅ Resolved |

## Best Practices Implementation

### ✅ Security Best Practices
- All sensitive data uses environment variables
- No hardcoded secrets in configuration files
- JWT secrets have secure defaults with warnings
- API keys reference environment variables

### ✅ Maintainability  
- Clear separation of concerns across config files
- Comprehensive documentation and examples
- Environment-specific override support
- Backward compatibility maintained

### ✅ Deployment Readiness
- Docker-compatible configuration
- Environment variable integration
- Production/development/testing overrides
- Health check and monitoring settings

## Migration Strategy

### Phase 1: ✅ Completed
- Created organized `/config/` directory structure
- Consolidated overlapping settings
- Maintained backward compatibility
- Added comprehensive documentation

### Phase 2: Recommended Next Steps
1. **Update application code** to prefer `/config/` directory
2. **Add configuration validation** on startup
3. **Create configuration management CLI** tool
4. **Add hot-reloading** for non-sensitive settings

### Phase 3: Long-term
1. **Deprecate legacy config files** (with migration warnings)
2. **Add configuration versioning** support
3. **Implement configuration templates** for different deployment scenarios
4. **Add configuration drift detection**

## Configuration Validation Rules

### Required Settings
- `database.type` and `database.path`
- At least one LLM provider configuration
- At least one video source API key
- Valid MCP JWT secret in production

### Conditional Requirements
- If `mcp.enabled = true`: JWT secret must be set
- If `redis.enabled = true`: Redis host must be configured  
- If production environment: All API keys must be from environment
- If GPU acceleration enabled: Appropriate drivers must be available

## Usage Examples

### Loading Configuration in Python
```python
import toml
from pathlib import Path

def load_config():
    # Load master configuration
    config_path = Path("config/aggregated.toml")
    config = toml.load(config_path)
    
    # Apply environment overrides
    env = os.getenv("DEPLOYMENT_ENV", "development")
    if f"environments.{env}" in config:
        config.update(config[f"environments.{env}"])
    
    return config
```

### Docker Compose Integration
```yaml
services:
  app:
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - JWT_SECRET=${JWT_SECRET}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./config:/app/config:ro
```

## Support and Troubleshooting

### Common Issues
1. **Missing environment variables**: Check `.env.example` for required variables
2. **Configuration conflicts**: Use the priority order documented above
3. **Legacy config not working**: Ensure files are synchronized with aggregated.toml
4. **API key issues**: Verify environment variables are properly set

### Debug Configuration Loading
```bash
# Set debug mode to see configuration loading
export DEBUG_CONFIG=true
python -m app.main
```

---

**Last Updated**: 2025-07-30  
**Configuration Version**: 2.0.0  
**Status**: ✅ Active - Consolidated and Optimized