"""
Secure Configuration Manager for MoneyPrinterTurbo

Provides encrypted configuration management with HashiCorp Vault integration,
environment variable validation, and secure secret handling.

Complies with SPARC principles: ≤500 lines, modular, testable, secure.
"""

import os
import json
import base64
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger
import hvac  # HashiCorp Vault client


class SecureConfigManager:
    """
    Secure configuration management with encryption and validation.
    
    Features:
    - Environment variable validation and sanitization
    - Configuration encryption at rest
    - HashiCorp Vault integration
    - Schema validation
    - Audit logging for configuration access
    """
    
    def __init__(self, config_file: str = "config.toml", vault_url: Optional[str] = None):
        self.config_file = config_file
        self.vault_url = vault_url or os.getenv("VAULT_URL")
        self.vault_token = os.getenv("VAULT_TOKEN")
        self.encryption_key = self._get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.vault_client = None
        self._config_cache = {}
        self._last_cache_update = None
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize Vault client if available
        if self.vault_url and self.vault_token:
            self._init_vault_client()
    
    def _get_encryption_key(self) -> bytes:
        """Generate or retrieve encryption key for configuration data."""
        key_file = ".config_key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        
        # Generate new key from environment or create random
        master_key = os.getenv("CONFIG_MASTER_KEY", "").encode()
        if not master_key:
            # Generate random key and save securely
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Owner read/write only
            logger.warning("Generated new encryption key. Store CONFIG_MASTER_KEY securely!")
            return key
        
        # Derive key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'moneyprinter_salt',  # In production, use random salt
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(master_key))
    
    def _init_vault_client(self):
        """Initialize HashiCorp Vault client."""
        try:
            self.vault_client = hvac.Client(url=self.vault_url, token=self.vault_token)
            if not self.vault_client.is_authenticated():
                logger.error("Vault authentication failed")
                self.vault_client = None
            else:
                logger.info("Successfully connected to HashiCorp Vault")
        except Exception as e:
            logger.error(f"Failed to initialize Vault client: {str(e)}")
            self.vault_client = None
    
    def get_secret(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        Retrieve secret with priority: Vault > Environment > Config file > Default
        
        Args:
            key: Secret key name
            default: Default value if not found
            required: Raise exception if secret not found
            
        Returns:
            Secret value
            
        Raises:
            ValueError: If required secret is not found
        """
        # Log access attempt (without value)
        self._log_config_access(key, "read")
        
        # Try Vault first
        if self.vault_client:
            vault_value = self._get_vault_secret(key)
            if vault_value is not None:
                return vault_value
        
        # Try environment variables
        env_key = key.upper().replace('.', '_')
        env_value = os.getenv(env_key)
        if env_value is not None:
            return self._parse_env_value(env_value)
        
        # Try encrypted config file
        config_value = self._get_config_value(key)
        if config_value is not None:
            return config_value
        
        # Return default or raise error
        if required and default is None:
            raise ValueError(f"Required configuration '{key}' not found")
        
        return default
    
    def set_secret(self, key: str, value: Any, store_in_vault: bool = True) -> bool:
        """
        Store secret securely.
        
        Args:
            key: Secret key name
            value: Secret value
            store_in_vault: Whether to store in Vault (preferred)
            
        Returns:
            Success status
        """
        self._log_config_access(key, "write")
        
        try:
            # Store in Vault if available
            if store_in_vault and self.vault_client:
                return self._set_vault_secret(key, value)
            
            # Fallback to encrypted local storage
            return self._set_config_value(key, value)
            
        except Exception as e:
            logger.error(f"Failed to store secret '{key}': {str(e)}")
            return False
    
    def _get_vault_secret(self, key: str) -> Optional[Any]:
        """Retrieve secret from HashiCorp Vault."""
        try:
            response = self.vault_client.secrets.kv.v2.read_secret_version(
                path=f"moneyprinter/{key}"
            )
            return response['data']['data'].get('value')
        except Exception as e:
            logger.debug(f"Vault secret '{key}' not found: {str(e)}")
            return None
    
    def _set_vault_secret(self, key: str, value: Any) -> bool:
        """Store secret in HashiCorp Vault."""
        try:
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=f"moneyprinter/{key}",
                secret={'value': value}
            )
            logger.info(f"Successfully stored secret '{key}' in Vault")
            return True
        except Exception as e:
            logger.error(f"Failed to store secret in Vault: {str(e)}")
            return False
    
    def _get_config_value(self, key: str) -> Optional[Any]:
        """Retrieve value from encrypted config file."""
        try:
            config_data = self._load_encrypted_config()
            keys = key.split('.')
            value = config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return None
            
            return value
            
        except Exception as e:
            logger.debug(f"Config value '{key}' not found: {str(e)}")
            return None
    
    def _set_config_value(self, key: str, value: Any) -> bool:
        """Store value in encrypted config file."""
        try:
            config_data = self._load_encrypted_config()
            keys = key.split('.')
            
            # Navigate to the correct nested location
            current = config_data
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
            
            return self._save_encrypted_config(config_data)
            
        except Exception as e:
            logger.error(f"Failed to store config value: {str(e)}")
            return False
    
    def _load_encrypted_config(self) -> Dict[str, Any]:
        """Load and decrypt configuration file."""
        encrypted_file = f"{self.config_file}.enc"
        
        if not os.path.exists(encrypted_file):
            return {}
        
        try:
            with open(encrypted_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"Failed to load encrypted config: {str(e)}")
            return {}
    
    def _save_encrypted_config(self, config_data: Dict[str, Any]) -> bool:
        """Encrypt and save configuration file."""
        encrypted_file = f"{self.config_file}.enc"
        
        try:
            json_data = json.dumps(config_data).encode('utf-8')
            encrypted_data = self.cipher_suite.encrypt(json_data)
            
            with open(encrypted_file, 'wb') as f:
                f.write(encrypted_data)
            
            os.chmod(encrypted_file, 0o600)  # Owner read/write only
            return True
            
        except Exception as e:
            logger.error(f"Failed to save encrypted config: {str(e)}")
            return False
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value with type conversion."""
        # Handle boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Handle numeric values
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Handle JSON values
        if value.startswith(('{', '[')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        return value
    
    def _log_config_access(self, key: str, operation: str):
        """Log configuration access for security auditing."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "config_key": key,
            "source": "config_manager"
        }
        
        # In production, send to centralized logging system
        logger.info(f"Config access: {operation} {key}")
    
    def validate_config_schema(self, schema: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against schema.
        
        Args:
            schema: Configuration schema with required fields and types
            
        Returns:
            List of validation errors
        """
        errors = []
        
        for key, requirements in schema.items():
            value = self.get_secret(key)
            
            # Check if required
            if requirements.get('required', False) and value is None:
                errors.append(f"Required configuration '{key}' is missing")
                continue
            
            # Check type
            expected_type = requirements.get('type')
            if expected_type and value is not None:
                if not isinstance(value, expected_type):
                    errors.append(f"Configuration '{key}' must be of type {expected_type.__name__}")
            
            # Check format (for strings)
            format_pattern = requirements.get('format')
            if format_pattern and isinstance(value, str):
                import re
                if not re.match(format_pattern, value):
                    errors.append(f"Configuration '{key}' format is invalid")
        
        return errors
    
    def rotate_encryption_key(self) -> bool:
        """Rotate the encryption key and re-encrypt all stored data."""
        try:
            # Load existing data
            old_config = self._load_encrypted_config()
            
            # Generate new key
            new_key = Fernet.generate_key()
            old_cipher = self.cipher_suite
            self.cipher_suite = Fernet(new_key)
            
            # Re-encrypt data
            if self._save_encrypted_config(old_config):
                # Update key file
                with open(".config_key", 'wb') as f:
                    f.write(new_key)
                os.chmod(".config_key", 0o600)
                
                logger.info("Successfully rotated encryption key")
                return True
            
            # Rollback on failure
            self.cipher_suite = old_cipher
            return False
            
        except Exception as e:
            logger.error(f"Key rotation failed: {str(e)}")
            return False


# Configuration schema for validation
CONFIG_SCHEMA = {
    'mcp_jwt_secret': {
        'required': True,
        'type': str,
        'format': r'^.{32,}$'  # At least 32 characters
    },
    'openai_api_key': {
        'required': False,
        'type': str,
        'format': r'^sk-[a-zA-Z0-9]{48}$'  # OpenAI API key format
    },
    'redis_password': {
        'required': False,
        'type': str
    },
    'supabase_url': {
        'required': False,
        'type': str,
        'format': r'^https://[a-zA-Z0-9-]+\.supabase\.co$'
    }
}


# Global instance for easy access
secure_config = SecureConfigManager()


def get_secure_config(key: str, default: Any = None, required: bool = False) -> Any:
    """Convenience function to get configuration values securely."""
    return secure_config.get_secret(key, default, required)


def validate_security_config() -> List[str]:
    """Validate all security-related configuration."""
    return secure_config.validate_config_schema(CONFIG_SCHEMA)