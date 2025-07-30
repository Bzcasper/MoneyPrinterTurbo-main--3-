# MoneyPrinterTurbo Configuration Guide

## Overview

This guide explains the new unified configuration system for MoneyPrinterTurbo. All configuration files have been organized into a clear, maintainable structure that separates concerns and improves security.

## Configuration Structure

```
config/
├── aggregated.toml           # Complete consolidated configuration (reference)
├── app.toml                  # Core application settings
├── services.toml             # External service configurations
├── infrastructure.toml       # Database, storage, deployment settings
├── mcp.toml                  # Model Context Protocol configuration
└── templates/                # Configuration templates
    ├── app.example.toml      # Application config template
    ├── services.example.toml # Services config template
    └── .env.example          # Environment variables template
```

## Quick Start

### 1. Copy Template Files

```bash
# Copy configuration templates
cp config/templates/app.example.toml config/app.toml
cp config/templates/services.example.toml config/services.toml
cp config/templates/.env.example .env
```

### 2. Set Environment Variables

Edit `.env` file with your actual API keys and credentials:

```bash
# Required: Database connection
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_actual_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_actual_service_role_key

# Required: At least one video source
PEXELS_API_KEY=your_pexels_key  # OR
PIXABAY_API_KEY=your_pixabay_key

# Required: At least one LLM provider
OPENAI_API_KEY=your_openai_key  # OR
GEMINI_API_KEY=your_gemini_key

# Generate secure secrets
JWT_SECRET=$(openssl rand -hex 32)
SECRET_KEY=$(openssl rand -hex 32)
```

### 3. Configure Application Settings

Edit `config/app.toml` for your basic application preferences:

```toml
[app]
video_source = "pixabay"  # or "pexels"
default_format = "youtube_shorts"
max_concurrent_tasks = 5  # Adjust based on system resources

[llm]
provider = "openai"  # or "gemini", "azure", etc.
```

## Configuration Files Explained

### `config/app.toml` - Core Application Settings

Contains fundamental application behavior settings:

- **Video Generation**: Source providers, formats, duration limits
- **UI Preferences**: Hide/show panels, logging preferences
- **Performance**: Concurrent tasks, GPU acceleration
- **Feature Flags**: Enable/disable experimental features

**Key Settings:**
```toml
[app]
video_source = "pixabay"          # Primary video source
default_format = "youtube_shorts"  # Video format
max_concurrent_tasks = 5          # Concurrent video generation limit

[features]
experimental_ai_avatars = false   # AI avatar integration
batch_processing = true           # Batch video processing
multilingual_subtitles = true     # Multi-language support
```

### `config/services.toml` - External Service Configuration

Manages all external API integrations:

- **Video Sources**: Pexels, Pixabay API configurations
- **LLM Providers**: OpenAI, Gemini, Azure OpenAI, etc.
- **TTS Services**: Azure Speech, ElevenLabs, GPT-SoVITS
- **Rate Limits**: API call limits and fallback strategies

**Key Settings:**
```toml
[llm]
provider = "openai"  # Primary LLM provider

[llm.openai]
api_key = "${OPENAI_API_KEY}"
model_name = "gpt-4o-mini"

[fallbacks]
llm_providers = ["openai", "gemini", "azure"]  # Fallback order
```

### `config/infrastructure.toml` - Infrastructure Settings

Handles infrastructure and deployment configurations:

- **Database**: PostgreSQL, SQLite, connection pooling
- **Storage**: Local, S3, Google Cloud, Azure Blob
- **Caching**: Redis configuration, cache strategies
- **Monitoring**: Logging, metrics, health checks
- **Security**: TLS, CORS, rate limiting

**Key Settings:**
```toml
[database]
type = "postgresql"
pool_size = 10

[storage]
backend = "local"  # or "s3", "gcs", "azure_blob"

[redis]
enabled = true
host = "localhost"
```

### `config/mcp.toml` - Model Context Protocol Settings

Advanced MCP server configuration:

- **Authentication**: JWT, API keys, HMAC signing
- **Authorization**: Role-based permissions
- **Performance**: Circuit breakers, rate limiting
- **Monitoring**: Metrics, health checks, logging

**Key Settings:**
```toml
[mcp]
enabled = true
server_port = 8081

[mcp.auth]
jwt_enabled = true
api_key_enabled = true

[mcp.rate_limiting]
requests_per_minute = 100
```

## Environment Variables

All sensitive data should be stored in environment variables:

### Required Variables

```bash
# Database (choose one)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
DATABASE_URL=postgresql://user:pass@host:port/db

# Video Sources (at least one)
PEXELS_API_KEY=your_pexels_key
PIXABAY_API_KEY=your_pixabay_key

# LLM Provider (at least one)
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key

# Security
JWT_SECRET=your_jwt_secret
SECRET_KEY=your_session_secret
```

### Optional Variables

```bash
# TTS Services
AZURE_SPEECH_KEY=your_azure_speech_key
ELEVENLABS_API_KEY=your_elevenlabs_key

# Storage
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_BUCKET_NAME=your_bucket

# Monitoring
SENTRY_DSN=your_sentry_dsn
```

## Configuration Loading Order

The application loads configuration in this order (later values override earlier ones):

1. **Default Values**: Built-in application defaults
2. **Configuration Files**: TOML files in `config/` directory
3. **Environment Variables**: Values from `.env` file and system environment
4. **Command Line Arguments**: Runtime parameters (if applicable)

## Environment-Specific Configuration

### Development

```toml
[environments.development]
debug = true
log_level = "DEBUG"
max_concurrent_tasks = 2
```

### Production

```toml
[environments.production]
debug = false
log_level = "INFO"
max_concurrent_tasks = 10
```

### Testing

```toml
[environments.testing]
debug = true
log_level = "DEBUG"
max_concurrent_tasks = 1
```

## Security Best Practices

### 1. Environment Variables
- Store all secrets in environment variables
- Never commit `.env` files to version control
- Use strong, unique secrets for JWT and session keys

### 2. API Keys
- Rotate API keys regularly
- Use separate keys for development and production
- Monitor API usage and set up alerts

### 3. Database Security
- Use connection pooling to prevent connection exhaustion
- Enable SSL/TLS for database connections in production
- Regularly backup your database

### 4. File Permissions
- Set restrictive permissions on configuration files
- Protect `.env` files with `chmod 600 .env`

## Configuration Validation

The application validates configuration on startup:

- **Required Fields**: Ensures all mandatory settings are present
- **Data Types**: Validates that values match expected types
- **Ranges**: Checks that numeric values are within acceptable ranges
- **Dependencies**: Verifies that dependent services are properly configured

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```
   Error: No valid API key found for provider 'openai'
   Solution: Set OPENAI_API_KEY in .env file
   ```

2. **Database Connection Failed**
   ```
   Error: Could not connect to database
   Solution: Check DATABASE_URL or Supabase credentials
   ```

3. **Port Already in Use**
   ```
   Error: Port 8081 is already in use
   Solution: Change mcp.server_port in config/mcp.toml
   ```

### Configuration Debugging

Enable debug mode to see configuration loading:

```bash
# Set environment variable
export DEBUG=true
export LOG_LEVEL=DEBUG

# Or in .env file
DEBUG=true
LOG_LEVEL=DEBUG
```

## Migration from Old Configuration

If you're migrating from the old configuration system:

1. **Backup Old Files**: Keep copies of your existing `config.toml` files
2. **Extract Settings**: Copy your API keys and custom settings
3. **Use Templates**: Start with the provided templates
4. **Test Gradually**: Test each service integration separately
5. **Validate**: Ensure all functionality works with new configuration

## Configuration Examples

### Minimal Setup

```toml
# config/app.toml
[app]
video_source = "pixabay"
default_format = "youtube_shorts"

# .env
PIXABAY_API_KEY=your_pixabay_key
GEMINI_API_KEY=your_gemini_key
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
JWT_SECRET=your_jwt_secret
```

### Production Setup

```toml
# config/app.toml
[app]
video_source = "pexels"
max_concurrent_tasks = 10
debug = false

[features]
batch_processing = true
caching_enabled = true

# config/services.toml
[llm]
provider = "openai"

[fallbacks]
llm_providers = ["openai", "azure", "gemini"]
video_sources = ["pexels", "pixabay"]

# .env
# Multiple API keys for high availability
PEXELS_API_KEY=key1,key2,key3
OPENAI_API_KEY=your_production_key
REDIS_URL=redis://redis-server:6379
DATABASE_URL=postgresql://user:pass@prod-db:5432/moneyprinter
```

## Support

For configuration help:

1. Check the template files in `config/templates/`
2. Review this documentation
3. Enable debug logging to see configuration loading details
4. Check the application logs for configuration errors
5. Refer to the aggregated configuration file for all available options

## Schema Reference

For a complete list of all configuration options, see:
- `config/aggregated.toml` - Complete configuration reference
- Individual TOML files for specific sections
- Template files for examples and defaults
