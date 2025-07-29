# MoneyPrinterTurbo Fork - Custom Changes Documentation

## Supabase Integration (Custom Addition)

This fork includes custom Supabase integration not present in the original MoneyPrinterTurbo repository.

### Required Environment Variables

```bash
# Supabase Configuration (Required)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key  # Optional, for admin operations
```

### Configuration Files Modified

- `config.toml`: Added `[supabase]` section with environment variable placeholders
- `app/database/connection.py`: Custom Supabase connection manager
- `app/middleware/supabase_middleware.py`: Request middleware for Supabase integration
- `app/models/exception.py`: Added `SupabaseConnectionError` exception class

### Setup Instructions

1. **Create Supabase Project** (if needed):
   - Go to [supabase.com](https://supabase.com)
   - Create new project
   - Copy URL and API keys from Settings > API

2. **Environment Configuration**:
   - Copy `.env.example` to `.env`
   - Update with your Supabase credentials
   - For Docker: Ensure `env_file` is configured in docker-compose.yml

3. **Database Migration**:
   - Application automatically runs migrations on startup
   - Check logs for migration status
   - Use service role key for admin operations

### Dependencies Added

- `supabase`: Python client for Supabase
- `asyncpg`: PostgreSQL async driver
- Additional middleware and error handling

### Custom Components

- `SupabaseConnection`: Database connection manager
- `SupabaseMiddleware`: Request middleware
- `SupabaseConnectionError`: Custom exception
- Database migration system
- Health checks

### Differences from Original

The original MoneyPrinterTurbo repository does not include:
- Supabase integration
- Database middleware
- Custom exception handling for database errors
- Environment variable configuration for external databases

### Troubleshooting

1. **"Invalid URL" errors**: Check SUPABASE_URL environment variable
2. **"SupabaseConnectionError not defined"**: Ensure proper imports in middleware
3. **Connection timeouts**: Verify Supabase project is active and accessible
4. **Migration failures**: Check service role key permissions

### Maintenance Notes

- Keep Supabase client library updated
- Monitor database connection pool usage
- Regular health checks recommended
- Log rotation for database operations

Last Updated: July 29, 2025
