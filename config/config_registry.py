#!/usr/bin/env python3
"""
MoneyPrinterTurbo Configuration Registry
=======================================

This module provides a centralized registry for all configuration files,
handles overlapping settings resolution, and provides unified access to
configuration across the application.

Usage:
    from config.config_registry import ConfigRegistry
    
    registry = ConfigRegistry()
    config = registry.get_unified_config()
    
    # Access specific sections
    mcp_config = registry.get_mcp_config()
    database_config = registry.get_database_config()
"""

import os
import json
import toml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConfigPriority(Enum):
    """Configuration loading priority levels"""
    ENVIRONMENT = 1      # Highest priority
    AGGREGATED = 2       # Master config file
    SPECIALIZED = 3      # Feature-specific configs
    LEGACY = 4          # Legacy config files
    DEFAULTS = 5        # Lowest priority


@dataclass
class ConfigSource:
    """Represents a configuration source with metadata"""
    path: Path
    priority: ConfigPriority
    section: Optional[str] = None
    description: str = ""
    is_template: bool = False
    is_legacy: bool = False


@dataclass
class ConfigMapping:
    """Maps configuration keys across different files"""
    key: str
    sources: List[ConfigSource] = field(default_factory=list)
    env_var: Optional[str] = None
    default_value: Any = None
    required: bool = False
    description: str = ""


class ConfigRegistry:
    """
    Centralized configuration registry that manages all config files
    and resolves overlapping settings according to priority rules.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the configuration registry.
        
        Args:
            config_dir: Path to configuration directory (default: ./config)
        """
        self.config_dir = config_dir or Path(__file__).parent
        self.project_root = self.config_dir.parent
        
        # Configuration sources registry
        self.sources: Dict[str, ConfigSource] = {}
        self.mappings: Dict[str, ConfigMapping] = {}
        
        # Cached configurations
        self._unified_config = None
        self._config_cache = {}
        
        self._initialize_sources()
        self._initialize_mappings()
    
    def _initialize_sources(self):
        """Initialize all known configuration sources"""
        
        # Primary configuration files (organized structure)
        self.sources.update({
            "aggregated": ConfigSource(
                path=self.config_dir / "aggregated.toml",
                priority=ConfigPriority.AGGREGATED,
                description="Master consolidated configuration"
            ),
            "app": ConfigSource(
                path=self.config_dir / "app.toml",
                priority=ConfigPriority.SPECIALIZED,
                section="app",
                description="Core application settings"
            ),
            "services": ConfigSource(
                path=self.config_dir / "services.toml",
                priority=ConfigPriority.SPECIALIZED,
                section="external_services",
                description="External service providers"
            ),
            "infrastructure": ConfigSource(
                path=self.config_dir / "infrastructure.toml",
                priority=ConfigPriority.SPECIALIZED,
                section="infrastructure",
                description="Database, storage, deployment"
            ),
            "mcp": ConfigSource(
                path=self.config_dir / "mcp.toml",
                priority=ConfigPriority.SPECIALIZED,
                section="mcp",
                description="MCP server configuration"
            )
        })
        
        # Legacy configuration files (backward compatibility)
        self.sources.update({
            "legacy_root": ConfigSource(
                path=self.project_root / "config.toml",
                priority=ConfigPriority.LEGACY,
                description="Legacy root configuration",
                is_legacy=True
            ),
            "legacy_app": ConfigSource(
                path=self.project_root / "app" / "config.toml",
                priority=ConfigPriority.LEGACY,
                description="Legacy app configuration",
                is_legacy=True
            ),
            "legacy_example": ConfigSource(
                path=self.project_root / "app" / "config" / "config.example.toml",
                priority=ConfigPriority.LEGACY,
                description="Legacy configuration example",
                is_legacy=True,
                is_template=True
            ),
            "legacy_mcp_example": ConfigSource(
                path=self.project_root / "config.mcp.example.toml",
                priority=ConfigPriority.LEGACY,
                description="Legacy MCP configuration example",
                is_legacy=True,
                is_template=True
            )
        })
        
        # Template files
        self.sources.update({
            "app_template": ConfigSource(
                path=self.config_dir / "templates" / "app.example.toml",
                priority=ConfigPriority.DEFAULTS,
                description="App configuration template",
                is_template=True
            ),
            "services_template": ConfigSource(
                path=self.config_dir / "templates" / "services.example.toml",
                priority=ConfigPriority.DEFAULTS,
                description="Services configuration template",
                is_template=True
            )
        })
    
    def _initialize_mappings(self):
        """Initialize configuration key mappings"""
        
        # Core application settings
        self.mappings.update({
            "app.title": ConfigMapping(
                key="app.title",
                sources=[self.sources["aggregated"], self.sources["app"]],
                default_value="MoneyPrinterTurbo",
                description="Application title"
            ),
            "app.video_source": ConfigMapping(
                key="app.video_source",
                sources=[self.sources["aggregated"], self.sources["app"], 
                        self.sources["legacy_root"], self.sources["legacy_app"]],
                default_value="pixabay",
                required=True,
                description="Primary video source provider"
            ),
            "app.max_concurrent_tasks": ConfigMapping(
                key="app.max_concurrent_tasks",
                sources=[self.sources["aggregated"], self.sources["app"],
                        self.sources["legacy_root"], self.sources["legacy_app"]],
                default_value=5,
                description="Maximum concurrent video generation tasks"
            )
        })
        
        # Database configuration
        self.mappings.update({
            "database.type": ConfigMapping(
                key="database.type",
                sources=[self.sources["aggregated"], self.sources["infrastructure"]],
                env_var="DATABASE_TYPE",
                default_value="postgresql",
                required=True,
                description="Database type (postgresql, sqlite, memory)"
            ),
            "database.path": ConfigMapping(
                key="database.path",
                sources=[self.sources["aggregated"], self.sources["infrastructure"]],
                env_var="DATABASE_URL",
                required=True,
                description="Database connection string or path"
            )
        })
        
        # MCP configuration
        self.mappings.update({
            "mcp.enabled": ConfigMapping(
                key="mcp.enabled",
                sources=[self.sources["aggregated"], self.sources["mcp"]],
                env_var="MCP_ENABLED",
                default_value=True,
                description="Enable MCP server functionality"
            ),
            "mcp.server_port": ConfigMapping(
                key="mcp.server_port",
                sources=[self.sources["aggregated"], self.sources["mcp"]],
                env_var="MCP_SERVER_PORT",
                default_value=8081,
                description="MCP server port"
            ),
            "mcp.jwt_secret": ConfigMapping(
                key="mcp.jwt_secret",
                sources=[self.sources["aggregated"], self.sources["mcp"]],
                env_var="JWT_SECRET",
                default_value="your-secret-key-CHANGE-IN-PRODUCTION",
                required=True,
                description="JWT signing secret for MCP authentication"
            )
        })
        
        # External services
        self.mappings.update({
            "llm.provider": ConfigMapping(
                key="llm.provider",
                sources=[self.sources["aggregated"], self.sources["services"]],
                env_var="LLM_PROVIDER",
                default_value="gemini",
                description="Primary LLM provider"
            ),
            "llm.openai.api_key": ConfigMapping(
                key="llm.openai.api_key",
                sources=[self.sources["aggregated"], self.sources["services"]],
                env_var="OPENAI_API_KEY",
                default_value="your_openai_key_here",
                description="OpenAI API key"
            ),
            "llm.gemini.api_key": ConfigMapping(
                key="llm.gemini.api_key",
                sources=[self.sources["aggregated"], self.sources["services"]],
                env_var="GEMINI_API_KEY",
                description="Google Gemini API key"
            )
        })
    
    def load_config_file(self, source: ConfigSource) -> Dict[str, Any]:
        """
        Load a configuration file with error handling.
        
        Args:
            source: Configuration source to load
            
        Returns:
            Configuration dictionary or empty dict if file doesn't exist
        """
        try:
            if not source.path.exists():
                if not source.is_template:
                    logger.warning(f"Configuration file not found: {source.path}")
                return {}
            
            with open(source.path, 'r', encoding='utf-8') as f:
                config = toml.load(f)
                logger.debug(f"Loaded configuration from {source.path}")
                return config
                
        except Exception as e:
            logger.error(f"Error loading configuration from {source.path}: {e}")
            return {}
    
    def get_value_with_priority(self, mapping: ConfigMapping) -> Any:
        """
        Get configuration value following priority rules.
        
        Args:
            mapping: Configuration mapping definition
            
        Returns:
            Configuration value from highest priority source
        """
        # 1. Check environment variable first (highest priority)
        if mapping.env_var:
            env_value = os.getenv(mapping.env_var)
            if env_value is not None:
                # Convert string environment variables to appropriate types
                if isinstance(mapping.default_value, bool):
                    return env_value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(mapping.default_value, int):
                    try:
                        return int(env_value)
                    except ValueError:
                        pass
                elif isinstance(mapping.default_value, float):
                    try:
                        return float(env_value)
                    except ValueError:
                        pass
                return env_value
        
        # 2. Check configuration files by priority
        for source in sorted(mapping.sources, key=lambda s: s.priority.value):
            config = self.load_config_file(source)
            if not config:
                continue
                
            # Navigate to nested key
            keys = mapping.key.split('.')
            value = config
            
            try:
                for key in keys:
                    value = value[key]
                
                if value is not None:
                    logger.debug(f"Found {mapping.key} in {source.path}")
                    return value
                    
            except (KeyError, TypeError):
                continue
        
        # 3. Return default value
        return mapping.default_value
    
    def get_unified_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Get unified configuration from all sources with proper priority handling.
        
        Args:
            force_reload: Force reload from files (ignore cache)
            
        Returns:
            Unified configuration dictionary
        """
        if self._unified_config is not None and not force_reload:
            return self._unified_config
        
        logger.info("Building unified configuration...")
        unified = {}
        
        # Build configuration using mappings with priority
        for key, mapping in self.mappings.items():
            value = self.get_value_with_priority(mapping)
            
            # Set nested key in unified config
            keys = key.split('.')
            current = unified
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
        
        # Load complete sections from specialized configs
        specialized_configs = ["app", "services", "infrastructure", "mcp"]
        for config_name in specialized_configs:
            source = self.sources[config_name]
            config = self.load_config_file(source)
            
            if config:
                # Merge specialized config into unified, respecting existing values
                self._deep_merge(unified, config, overwrite=False)
        
        # Apply environment-specific overrides
        env = os.getenv("DEPLOYMENT_ENV", "development")
        if f"environments.{env}" in unified:
            env_overrides = unified[f"environments.{env}"]
            self._deep_merge(unified, env_overrides, overwrite=True)
            logger.info(f"Applied {env} environment overrides")
        
        self._unified_config = unified
        logger.info("Unified configuration built successfully")
        return unified
    
    def _deep_merge(self, target: Dict, source: Dict, overwrite: bool = True):
        """
        Deep merge source dictionary into target dictionary.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
            overwrite: Whether to overwrite existing values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value, overwrite)
            elif overwrite or key not in target:
                target[key] = value
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application-specific configuration"""
        unified = self.get_unified_config()
        return unified.get("app", {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        unified = self.get_unified_config()
        return unified.get("database", {})
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """Get MCP server configuration"""
        unified = self.get_unified_config()
        return unified.get("mcp", {})
    
    def get_services_config(self) -> Dict[str, Any]:
        """Get external services configuration"""
        unified = self.get_unified_config()
        return {
            "llm": unified.get("llm", {}),
            "video_sources": unified.get("video_sources", {}),
            "tts": unified.get("tts", {}),
            "ai_services": unified.get("ai_services", {})
        }
    
    def get_infrastructure_config(self) -> Dict[str, Any]:
        """Get infrastructure configuration"""
        unified = self.get_unified_config()
        return {
            "database": unified.get("database", {}),
            "redis": unified.get("redis", {}),
            "storage": unified.get("storage", {}),
            "monitoring": unified.get("monitoring", {}),
            "logging": unified.get("logging", {})
        }
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Check required mappings
        for key, mapping in self.mappings.items():
            if mapping.required:
                value = self.get_value_with_priority(mapping)
                if value is None or (isinstance(value, str) and not value.strip()):
                    issues.append(f"Required configuration missing: {key}")
        
        # Check critical environment variables in production
        env = os.getenv("DEPLOYMENT_ENV", "development")
        if env == "production":
            critical_env_vars = [
                "DATABASE_URL", "JWT_SECRET", "SUPABASE_URL", "SUPABASE_ANON_KEY"
            ]
            for env_var in critical_env_vars:
                if not os.getenv(env_var):
                    issues.append(f"Production environment variable missing: {env_var}")
        
        # Check configuration file existence
        critical_files = ["aggregated"]
        for file_key in critical_files:
            source = self.sources[file_key]
            if not source.path.exists():
                issues.append(f"Critical configuration file missing: {source.path}")
        
        return issues
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive configuration status report.
        
        Returns:
            Configuration status information
        """
        status = {
            "sources": {},
            "mappings_count": len(self.mappings),
            "validation_issues": self.validate_configuration(),
            "environment": os.getenv("DEPLOYMENT_ENV", "development"),
            "config_dir": str(self.config_dir),
            "project_root": str(self.project_root)
        }
        
        # Check each source file
        for name, source in self.sources.items():
            status["sources"][name] = {
                "path": str(source.path),
                "exists": source.path.exists(),
                "priority": source.priority.name,
                "is_legacy": source.is_legacy,
                "is_template": source.is_template,
                "description": source.description
            }
            
            if source.path.exists():
                try:
                    stat = source.path.stat()
                    status["sources"][name]["size"] = stat.st_size
                    status["sources"][name]["modified"] = stat.st_mtime
                except Exception as e:
                    status["sources"][name]["error"] = str(e)
        
        return status
    
    def export_unified_config(self, output_path: Path, format: str = "toml") -> bool:
        """
        Export unified configuration to file.
        
        Args:
            output_path: Path to output file
            format: Export format ('toml', 'json', 'yaml')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            unified = self.get_unified_config()
            
            if format.lower() == "toml":
                with open(output_path, 'w', encoding='utf-8') as f:
                    toml.dump(unified, f)
            elif format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(unified, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported unified configuration to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False


def main():
    """CLI interface for configuration registry"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MoneyPrinterTurbo Configuration Registry")
    parser.add_argument("--status", action="store_true", help="Show configuration status")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    parser.add_argument("--export", help="Export unified config to file")
    parser.add_argument("--format", default="toml", choices=["toml", "json"], 
                       help="Export format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    registry = ConfigRegistry()
    
    if args.status:
        status = registry.get_configuration_status()
        print(json.dumps(status, indent=2, default=str))
    
    if args.validate:
        issues = registry.validate_configuration()
        if issues:
            print("Configuration validation issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration validation passed!")
    
    if args.export:
        output_path = Path(args.export)
        success = registry.export_unified_config(output_path, args.format)
        if success:
            print(f"Configuration exported to {output_path}")
        else:
            print("Export failed!")


if __name__ == "__main__":
    main()