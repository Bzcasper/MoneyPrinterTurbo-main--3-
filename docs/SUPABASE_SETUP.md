# Supabase Configuration Guide

This document explains how to configure Supabase credentials for the MoneyPrinterTurbo application.

## Required Environment Variables

The application requires the following Supabase environment variables to be set:

### 1. SUPABASE_URL
- **Description**: Your Supabase project URL
- **Format**: `https://your-project-ref.supabase.co`
- **How to get**: 
  1. Go to your [Supabase Dashboard](https://supabase.com/dashboard)
  2. Select your project
  3. Go to Settings → API
  4. Copy the "Project URL"

### 2. SUPABASE_ANON_KEY
- **Description**: Anonymous/Public API key for client-side operations
- **Security**: Safe to use in frontend applications (respects RLS policies)
- **How to get**:
  1. Go to your Supabase Dashboard
  2. Select your project
  3. Go to Settings → API
  4. Copy the "anon public" key

### 3. SUPABASE_SERVICE_ROLE_KEY (Optional)
- **Description**: Service role key for server-side admin operations
- **Security**: ⚠️ **CRITICAL** - This key bypasses Row Level Security (RLS)
- **Usage**: Only use if you need server-side admin operations
- **How to get**:
  1. Go to your Supabase Dashboard
  2. Select your project
  3. Go to Settings → API
  4. Copy the "service_role" key

## Setting Environment Variables

### Option 1: Using .env file (Development)
Create a `.env` file in your project root:

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here
```

### Option 2: System Environment Variables
```bash
export SUPABASE_URL="https://your-project-ref.supabase.co"
export SUPABASE_ANON_KEY="your-anon-key-here"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key-here"
```

### Option 3: Docker Environment
```dockerfile
ENV SUPABASE_URL=https://your-project-ref.supabase.co
ENV SUPABASE_ANON_KEY=your-anon-key-here
ENV SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here
```

## Configuration Verification

After setting the environment variables, you can verify the configuration by:

1. **Check config loading**: The application will log connection status on startup
2. **Test database connection**: Use the health check endpoint if available
3. **Monitor logs**: Check for any Supabase connection errors

## Security Best Practices

### ✅ DO:
- Use environment variables for all credentials
- Keep service role key secure and server-side only
- Implement proper Row Level Security (RLS) policies
- Rotate keys periodically
- Use different keys for development and production

### ❌ DON'T:
- Hardcode credentials in source code
- Commit `.env` files to version control
- Use service role key in client-side applications
- Share credentials in plain text

## Troubleshooting

### Common Issues:

1. **"Missing Supabase URL and key in config.toml"**
   - Ensure environment variables are set correctly
   - Check variable names match exactly
   - Verify `.env` file is in the correct location

2. **"Supabase URL and key are required in config.toml"**
   - Both `SUPABASE_URL` and `SUPABASE_ANON_KEY` must be set
   - Check for typos in environment variable names

3. **Connection timeout errors**
   - Verify the Supabase URL is correct
   - Check network connectivity
   - Ensure project is not paused

4. **Permission denied errors**
   - Check RLS policies are configured correctly
   - Verify you're using the correct API key
   - Ensure user has proper permissions

## Database URL Configuration

The application also supports direct PostgreSQL connections via the `DATABASE_URL` environment variable:

```bash
DATABASE_URL=postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
```

This is optional and only needed for direct database operations.

## Support

For additional help:
- [Supabase Documentation](https://supabase.com/docs)
- [Supabase API Reference](https://supabase.com/docs/reference)
- [Supabase Discord Community](https://discord.supabase.com/)