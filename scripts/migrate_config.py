#!/usr/bin/env python3
"""
MoneyPrinterTurbo Configuration Migration Script

This script helps migrate from the old scattered configuration files
to the new unified configuration system.

Usage:
    python migrate_config.py [--backup] [--dry-run]

Options:
    --backup    Create backup of existing configuration files
    --dry-run   Show what would be done without making changes
"""

import os
import sys
import shutil
import argparse
import tomllib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

class ConfigMigrator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.backup_dir = project_root / f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.new_config_dir = project_root / "config"
        
        # Old configuration file locations
        self.old_files = {
            "root_config": project_root / "config.toml",
            "app_config": project_root / "app" / "config.toml", 
            "example_config": project_root / "app" / "config" / "config.example.toml",
            "mcp_config": project_root / "config.mcp.example.toml",
            "postgres_config": project_root / "app" / "config" / "postgres.toml",
            "env_example": project_root / ".env.example",
            "docker_compose": project_root / "app" / "docker-compose.yml"
        }
        
    def create_backup(self):
        """Create backup of existing configuration files."""
        print(f"Creating backup in {self.backup_dir}")
        self.backup_dir.mkdir(exist_ok=True)
        
        for name, file_path in self.old_files.items():
            if file_path.exists():
                backup_path = self.backup_dir / f"{name}_{file_path.name}"
                shutil.copy2(file_path, backup_path)
                print(f"  Backed up: {file_path} -> {backup_path}")
    
    def load_toml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load TOML file safely."""
        try:
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return tomllib.load(f)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
        return {}
    
    def extract_app_settings(self) -> Dict[str, Any]:
        """Extract application settings from old configuration files."""
        settings = {}
        
        # Load from root config.toml
        root_config = self.load_toml_file(self.old_files["root_config"])
        if root_config:
            # App section
            if "app" in root_config:
                settings.update(root_config["app"])
            
            # Browser section
            if "browser" in root_config:
                settings["browser"] = root_config["browser"]
            
            # Video section
            if "video" in root_config:
                settings["video"] = root_config["video"]
            
            # UI section
            if "ui" in root_config:
                settings["ui"] = root_config["ui"]
            
            # Feature flags
            if "features" in root_config:
                settings["features"] = root_config["features"]
        
        return settings
    
    def extract_service_settings(self) -> Dict[str, Any]:
        """Extract external service settings."""
        settings = {}
        
        root_config = self.load_toml_file(self.old_files["root_config"])
        if root_config:
            # Video source APIs
            video_sources = {}
            if "pexels_api_keys" in root_config:
                video_sources["pexels"] = {"api_keys": root_config["pexels_api_keys"]}
            if "pixabay_api_keys" in root_config:
                video_sources["pixabay"] = {"api_keys": root_config["pixabay_api_keys"]}
            
            if video_sources:
                settings["video_sources"] = video_sources
            
            # LLM providers
            llm_settings = {"provider": root_config.get("llm_provider", "openai")}
            
            # Extract all LLM provider configurations
            llm_providers = {}
            for key, value in root_config.items():
                if key.endswith("_api_key") or key.endswith("_base_url") or key.endswith("_model_name"):
                    provider = key.split("_")[0]
                    setting = "_".join(key.split("_")[1:])
                    
                    if provider not in llm_providers:
                        llm_providers[provider] = {}
                    llm_providers[provider][setting] = value
            
            if llm_providers:
                for provider, config in llm_providers.items():
                    llm_settings[provider] = config
            
            if llm_settings:
                settings["llm"] = llm_settings
            
            # TTS settings
            if "subtitle_provider" in root_config:
                settings["tts"] = {"subtitle_provider": root_config["subtitle_provider"]}
            
            # Azure TTS
            if "azure" in root_config:
                if "tts" not in settings:
                    settings["tts"] = {}
                settings["tts"]["azure"] = root_config["azure"]
            
            # Other sections
            for section in ["whisper", "proxy", "siliconflow", "gpt_sovits"]:
                if section in root_config:
                    settings[section] = root_config[section]
        
        return settings
    
    def extract_infrastructure_settings(self) -> Dict[str, Any]:
        """Extract infrastructure and deployment settings."""
        settings = {}
        
        root_config = self.load_toml_file(self.old_files["root_config"])
        if root_config:
            # Database settings
            if "database" in root_config:
                settings["database"] = root_config["database"]
            
            # Supabase settings
            if "supabase" in root_config:
                settings["supabase"] = root_config["supabase"]
            
            # Redis settings
            redis_config = {}
            for key in ["enable_redis", "redis_host", "redis_port", "redis_db", "redis_password"]:
                if key in root_config:
                    redis_key = key.replace("redis_", "").replace("enable_redis", "enabled")
                    redis_config[redis_key] = root_config[key]
            
            if redis_config:
                settings["redis"] = redis_config
            
            # Storage settings (if any)
            storage_keys = ["material_directory", "endpoint"]
            storage_config = {}
            for key in storage_keys:
                if key in root_config:
                    storage_config[key] = root_config[key]
            
            if storage_config:
                settings["storage"] = storage_config
        
        return settings
    
    def extract_mcp_settings(self) -> Dict[str, Any]:
        """Extract MCP configuration."""
        settings = {}
        
        # From root config
        root_config = self.load_toml_file(self.old_files["root_config"])
        if "mcp" in root_config:
            settings.update(root_config["mcp"])
        
        # From MCP example config
        mcp_config = self.load_toml_file(self.old_files["mcp_config"])
        if mcp_config:
            # Merge app-level MCP settings
            if "app" in mcp_config:
                app_mcp = mcp_config["app"]
                for key, value in app_mcp.items():
                    if key.startswith("mcp_"):
                        mcp_key = key.replace("mcp_", "")
                        settings[mcp_key] = value
        
        return settings
    
    def extract_environment_variables(self) -> List[str]:
        """Extract environment variables that should be in .env file."""
        env_vars = []
        
        # Load .env.example
        env_file = self.old_files["env_example"]
        if env_file.exists():
            with open(env_file, 'r') as f:
                content = f.read()
                # Copy the entire content as it's already well-structured
                return [content]
        
        # Default env vars if no example file
        return [
            "# Essential environment variables",
            "SUPABASE_URL=https://your-project.supabase.co", 
            "SUPABASE_ANON_KEY=your_anon_key_here",
            "SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here",
            "OPENAI_API_KEY=your_openai_key_here",
            "GEMINI_API_KEY=your_gemini_key_here",
            "JWT_SECRET=your_jwt_secret_here",
            "SECRET_KEY=your_session_secret_here"
        ]
    
    def write_toml_file(self, file_path: Path, data: Dict[str, Any]):
        """Write data to TOML file with proper formatting."""
        try:
            import tomli_w
            with open(file_path, 'wb') as f:
                tomli_w.dump(data, f)
        except ImportError:
            # Fallback to manual TOML writing
            self._write_toml_manual(file_path, data)
    
    def _write_toml_manual(self, file_path: Path, data: Dict[str, Any]):
        """Manual TOML writing as fallback."""
        def format_value(value):
            if isinstance(value, str):
                return f'"{value}"'
            elif isinstance(value, bool):
                return str(value).lower()
            elif isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, list):
                formatted_items = [format_value(item) for item in value]
                return f'[{", ".join(formatted_items)}]'
            else:
                return f'"{str(value)}"'
        
        with open(file_path, 'w') as f:
            for section_name, section_data in data.items():
                if isinstance(section_data, dict):
                    f.write(f"[{section_name}]\n")
                    for key, value in section_data.items():
                        if isinstance(value, dict):
                            f.write(f"\n[{section_name}.{key}]\n")
                            for subkey, subvalue in value.items():
                                f.write(f"{subkey} = {format_value(subvalue)}\n")
                        else:
                            f.write(f"{key} = {format_value(value)}\n")
                    f.write("\n")
                else:
                    f.write(f"{section_name} = {format_value(section_data)}\n")
    
    def migrate(self, dry_run: bool = False):
        """Perform the migration."""
        print("Starting configuration migration...")
        
        if dry_run:
            print("DRY RUN MODE - No files will be changed")
        
        # Extract settings from old files
        app_settings = self.extract_app_settings()
        service_settings = self.extract_service_settings()
        infrastructure_settings = self.extract_infrastructure_settings()
        mcp_settings = self.extract_mcp_settings()
        env_vars = self.extract_environment_variables()
        
        # Create new config directory
        if not dry_run:
            self.new_config_dir.mkdir(exist_ok=True)
            (self.new_config_dir / "templates").mkdir(exist_ok=True)
        
        # Generate new configuration files
        files_to_create = {
            "app.toml": app_settings,
            "services.toml": service_settings,
            "infrastructure.toml": infrastructure_settings,
            "mcp.toml": {"mcp": mcp_settings} if mcp_settings else {}
        }
        
        for filename, data in files_to_create.items():
            file_path = self.new_config_dir / filename
            
            if dry_run:
                print(f"Would create: {file_path}")
                if data:
                    print(f"  Sections: {list(data.keys())}")
            else:
                if data:  # Only create if there's data
                    self.write_toml_file(file_path, data)
                    print(f"Created: {file_path}")
        
        # Create .env file
        env_path = self.project_root / ".env.new"
        if not dry_run:
            with open(env_path, 'w') as f:
                f.write("\n".join(env_vars))
            print(f"Created: {env_path} (rename to .env after reviewing)")
        else:
            print(f"Would create: {env_path}")
        
        print("\nMigration completed!")
        if not dry_run:
            print(f"Backup created in: {self.backup_dir}")
            print("Please review the new configuration files and update with your actual values.")

def main():
    parser = argparse.ArgumentParser(description="Migrate MoneyPrinterTurbo configuration files")
    parser.add_argument("--backup", action="store_true", help="Create backup of existing files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()
    
    # Find project root
    current_dir = Path.cwd()
    project_root = current_dir
    
    # Look for project indicators
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "config.toml").exists() or (parent / "pyproject.toml").exists():
            project_root = parent
            break
    
    print(f"Project root: {project_root}")
    
    migrator = ConfigMigrator(project_root)
    
    if args.backup and not args.dry_run:
        migrator.create_backup()
    
    try:
        migrator.migrate(dry_run=args.dry_run)
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
