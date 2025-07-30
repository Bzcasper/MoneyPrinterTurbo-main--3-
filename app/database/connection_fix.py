# Add this to your database connection configuration
ASYNCPG_CONFIG = {
    'statement_cache_size': 0,  # Disable prepared statement cache
    'command_timeout': 60,
    'server_settings': {
        'application_name': 'moneyprinter_api'
    }
}
