"""
Centralized Redis Configuration Manager

This module provides a unified Redis configuration system that consolidates
all Redis settings from various sources and provides a single source of truth
for Redis connectivity across the application.
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import toml

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Redis configuration dataclass with validation and defaults"""
    
    # Connection settings
    host: str = "redis"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    # Connection pool settings
    max_connections: int = 20
    connection_pool_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Retry and timeout settings
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, Any] = field(default_factory=dict)
    
    # Health check settings
    health_check_interval: float = 30.0
    retry_on_timeout: bool = True
    retry_on_error: list = field(default_factory=lambda: [ConnectionError, TimeoutError])
    
    # Application-specific settings
    enabled: bool = True
    cluster_mode: bool = False
    ssl_enabled: bool = False
    ssl_cert_reqs: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    
    # State management settings
    state_db: int = 0
    cache_db: int = 1
    session_db: int = 2
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid Redis port: {self.port}")
        
        if self.db < 0 or self.db > 15:
            raise ValueError(f"Invalid Redis database number: {self.db}")
        
        if self.max_connections < 1:
            raise ValueError(f"Invalid max_connections: {self.max_connections}")
    
    @property
    def connection_url(self) -> str:
        """Generate Redis connection URL"""
        auth_part = f":{self.password}@" if self.password else ""
        protocol = "rediss" if self.ssl_enabled else "redis"
        return f"{protocol}://{auth_part}{self.host}:{self.port}/{self.db}"
    
    @property
    def connection_kwargs(self) -> Dict[str, Any]:
        """Get connection keyword arguments for redis-py"""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "socket_keepalive": self.socket_keepalive,
            "socket_keepalive_options": self.socket_keepalive_options,
            "retry_on_timeout": self.retry_on_timeout,
            "retry_on_error": self.retry_on_error,
            "health_check_interval": self.health_check_interval,
        }
        
        if self.password:
            kwargs["password"] = self.password
        
        if self.ssl_enabled:
            kwargs.update({
                "ssl": True,
                "ssl_cert_reqs": self.ssl_cert_reqs,
                "ssl_ca_certs": self.ssl_ca_certs,
                "ssl_certfile": self.ssl_certfile,
                "ssl_keyfile": self.ssl_keyfile,
            })
        
        # Add custom connection pool kwargs
        kwargs.update(self.connection_pool_kwargs)
        
        return kwargs


class RedisConfigManager:
    """
    Redis configuration manager that loads settings from multiple sources
    with proper precedence order:
    1. Environment variables (highest priority)
    2. TOML configuration file
    3. Default values (lowest priority)
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize Redis configuration manager
        
        Args:
            config_file: Path to TOML configuration file
        """
        self.config_file = config_file or self._find_config_file()
        self._config_cache: Optional[RedisConfig] = None
        
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in common locations"""
        search_paths = [
            Path("config.toml"),
            Path("app/config.toml"),
            Path("app/config/config.toml"),
            Path.cwd() / "config.toml",
            Path.cwd() / "app" / "config.toml",
        ]
        
        for path in search_paths:
            if path.exists():
                logger.info(f"Found Redis config file at: {path}")
                return path
        
        logger.warning("No Redis config file found, using environment variables and defaults")
        return None
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load Redis configuration from environment variables"""
        env_config = {}
        
        # Connection settings
        if host := os.getenv("REDIS_HOST"):
            env_config["host"] = host
        if port := os.getenv("REDIS_PORT"):
            env_config["port"] = int(port)
        if db := os.getenv("REDIS_DB"):
            env_config["db"] = int(db)
        if password := os.getenv("REDIS_PASSWORD"):
            env_config["password"] = password if password else None
        
        # URL-based configuration (takes precedence)
        if redis_url := os.getenv("REDIS_URL"):
            env_config.update(self._parse_redis_url(redis_url))
        
        # Pool settings
        if max_conn := os.getenv("REDIS_MAX_CONNECTIONS"):
            env_config["max_connections"] = int(max_conn)
        
        # Timeout settings
        if socket_timeout := os.getenv("REDIS_SOCKET_TIMEOUT"):
            env_config["socket_timeout"] = float(socket_timeout)
        if connect_timeout := os.getenv("REDIS_CONNECT_TIMEOUT"):
            env_config["socket_connect_timeout"] = float(connect_timeout)
        
        # SSL settings
        if ssl_enabled := os.getenv("REDIS_SSL"):
            env_config["ssl_enabled"] = ssl_enabled.lower() in ("true", "1", "yes")
        if ssl_cert_reqs := os.getenv("REDIS_SSL_CERT_REQS"):
            env_config["ssl_cert_reqs"] = ssl_cert_reqs
        if ssl_ca_certs := os.getenv("REDIS_SSL_CA_CERTS"):
            env_config["ssl_ca_certs"] = ssl_ca_certs
        
        # Application settings
        if enabled := os.getenv("REDIS_ENABLED"):
            env_config["enabled"] = enabled.lower() in ("true", "1", "yes")
        
        return env_config
    
    def _load_from_toml(self) -> Dict[str, Any]:
        """Load Redis configuration from TOML file"""
        if not self.config_file or not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = toml.load(f)
            
            # Extract Redis settings from different sections
            redis_config = {}
            
            # Main app section
            if app_config := config_data.get("app", {}):
                if "enable_redis" in app_config:
                    redis_config["enabled"] = app_config["enable_redis"]
                if "redis_host" in app_config:
                    redis_config["host"] = app_config["redis_host"]
                if "redis_port" in app_config:
                    redis_config["port"] = app_config["redis_port"]
                if "redis_db" in app_config:
                    redis_config["db"] = app_config["redis_db"]
                if "redis_password" in app_config:
                    redis_config["password"] = app_config["redis_password"] or None
            
            # Dedicated redis section (if exists)
            if redis_section := config_data.get("redis", {}):
                redis_config.update(redis_section)
            
            return redis_config
            
        except Exception as e:
            logger.error(f"Error loading Redis config from {self.config_file}: {e}")
            return {}
    
    def _parse_redis_url(self, redis_url: str) -> Dict[str, Any]:
        """Parse Redis URL into configuration parameters"""
        try:
            from urllib.parse import urlparse
            
            parsed = urlparse(redis_url)
            config = {}
            
            if parsed.hostname:
                config["host"] = parsed.hostname
            if parsed.port:
                config["port"] = parsed.port
            if parsed.password:
                config["password"] = parsed.password
            if parsed.path and len(parsed.path) > 1:
                # Remove leading slash and parse db number
                db_str = parsed.path[1:]
                if db_str.isdigit():
                    config["db"] = int(db_str)
            
            # SSL based on scheme
            if parsed.scheme == "rediss":
                config["ssl_enabled"] = True
            
            return config
            
        except Exception as e:
            logger.error(f"Error parsing Redis URL '{redis_url}': {e}")
            return {}
    
    def get_config(self, force_reload: bool = False) -> RedisConfig:
        """
        Get Redis configuration with caching
        
        Args:
            force_reload: Force reload configuration from sources
            
        Returns:
            RedisConfig: Consolidated Redis configuration
        """
        if self._config_cache is not None and not force_reload:
            return self._config_cache
        
        # Load configuration from all sources
        toml_config = self._load_from_toml()
        env_config = self._load_from_env()
        
        # Merge configurations (env takes precedence)
        merged_config = {**toml_config, **env_config}
        
        # Create RedisConfig with merged settings
        try:
            self._config_cache = RedisConfig(**merged_config)
            logger.info(f"Redis configuration loaded: {self._config_cache.host}:{self._config_cache.port}")
            return self._config_cache
            
        except Exception as e:
            logger.error(f"Error creating Redis configuration: {e}")
            # Return default configuration
            self._config_cache = RedisConfig()
            return self._config_cache
    
    def validate_connection(self, config: Optional[RedisConfig] = None) -> bool:
        """
        Validate Redis connection with given or default configuration
        
        Args:
            config: Redis configuration to validate
            
        Returns:
            bool: True if connection is valid
        """
        if config is None:
            config = self.get_config()
        
        if not config.enabled:
            logger.info("Redis is disabled in configuration")
            return False
        
        try:
            import redis
            
            # Create connection
            client = redis.Redis(**config.connection_kwargs)
            
            # Test connection
            client.ping()
            client.close()
            
            logger.info("Redis connection validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Redis connection validation failed: {e}")
            return False
    
    def get_database_config(self, db_type: str) -> RedisConfig:
        """
        Get Redis configuration for specific database type
        
        Args:
            db_type: Database type ('state', 'cache', 'session')
            
        Returns:
            RedisConfig: Configuration for specific database
        """
        base_config = self.get_config()
        
        # Create a copy with specific database number
        config_dict = base_config.__dict__.copy()
        
        if db_type == "state":
            config_dict["db"] = base_config.state_db
        elif db_type == "cache":
            config_dict["db"] = base_config.cache_db
        elif db_type == "session":
            config_dict["db"] = base_config.session_db
        else:
            logger.warning(f"Unknown database type: {db_type}, using default")
        
        return RedisConfig(**config_dict)


# Global configuration manager instance
_config_manager: Optional[RedisConfigManager] = None


def get_redis_config_manager() -> RedisConfigManager:
    """Get global Redis configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = RedisConfigManager()
    
    return _config_manager


def get_redis_config(force_reload: bool = False) -> RedisConfig:
    """Get Redis configuration"""
    return get_redis_config_manager().get_config(force_reload=force_reload)


def validate_redis_connection() -> bool:
    """Validate Redis connection with current configuration"""
    return get_redis_config_manager().validate_connection()


# Export main components
__all__ = [
    "RedisConfig",
    "RedisConfigManager", 
    "get_redis_config_manager",
    "get_redis_config",
    "validate_redis_connection"
]