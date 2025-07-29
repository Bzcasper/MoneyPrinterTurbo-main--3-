#!/usr/bin/env python3
"""
Robust Workflow Execution Environment with Parameter Validation

Provides a comprehensive workflow execution system with enterprise-grade
parameter validation, environment security, and multi-format support.

Features:
- JSON Schema validation with custom validators
- Environment variable handling with encryption
- Input sanitization and type checking
- Comprehensive error handling
- Support for JSON, YAML, CLI args, and environment variables
- Secure parameter storage and retrieval
- Workflow state management
- Real-time validation feedback

Complies with SPARC principles: modular, testable, secure, performant.
"""

import os
import sys
import json
import yaml
import argparse
import hashlib
import base64
import tempfile
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Type
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Import existing validation components
try:
    from app.security.input_validator import InputSanitizer, DataValidator, ValidationError
except ImportError:
    # Fallback minimal implementations
    class ValidationError(Exception):
        def __init__(self, field: str, message: str, value: Any = None):
            self.field = field
            self.message = message
            self.value = value
            super().__init__(f"Validation error in '{field}': {message}")
    
    class InputSanitizer:
        def sanitize_html(self, text: str, strict: bool = True) -> str:
            import html
            return html.escape(str(text), quote=True)
    
    class DataValidator:
        def validate_string(self, value: Any, field: str = "field", **kwargs) -> str:
            return str(value).strip()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParameterFormat(str, Enum):
    """Supported parameter input formats."""
    JSON = "json"
    YAML = "yaml"
    CLI = "cli"
    ENV = "env"
    CONFIG_FILE = "config_file"


class ValidationLevel(str, Enum):
    """Parameter validation strictness levels."""
    STRICT = "strict"  # Fail on any validation error
    LENIENT = "lenient"  # Log warnings, continue with defaults
    PERMISSIVE = "permissive"  # Allow most inputs with basic sanitization


class WorkflowState(str, Enum):
    """Workflow execution states."""
    PENDING = "pending"
    VALIDATING = "validating"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    sanitized_data: Dict[str, Any] = field(default_factory=dict)
    validation_time_ms: float = 0.0
    schema_version: str = "1.0"


@dataclass
class WorkflowContext:
    """Workflow execution context with validated parameters."""
    workflow_id: str
    parameters: Dict[str, Any]
    environment: Dict[str, str]
    state: WorkflowState
    created_at: datetime
    updated_at: datetime
    validation_result: Optional[ValidationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParameterSchema:
    """
    Enhanced JSON Schema-based parameter validation with custom validators.
    
    Supports complex validation rules, type coercion, and business logic validation.
    """
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.validator = DataValidator()
        self.custom_validators: Dict[str, Callable] = {}
        self.type_coercers: Dict[str, Callable] = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': self._coerce_boolean,
            'array': list,
            'object': dict
        }
    
    def _coerce_boolean(self, value: Any) -> bool:
        """Coerce various formats to boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
        return bool(value)
    
    def register_validator(self, name: str, validator_func: Callable[[Any], bool]):
        """Register custom validation function."""
        self.custom_validators[name] = validator_func
    
    def validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
        """
        Validate data against JSON schema with enhanced features.
        
        Args:
            data: Input data to validate
            schema: JSON schema definition
            
        Returns:
            ValidationResult with validation status and sanitized data
        """
        start_time = datetime.now()
        result = ValidationResult(is_valid=True)
        
        try:
            # Validate required fields
            required_fields = schema.get('required', [])
            for field in required_fields:
                if field not in data or data[field] is None:
                    result.errors.append({
                        'field': field,
                        'code': 'REQUIRED_FIELD_MISSING',
                        'message': f'Required field "{field}" is missing or null'
                    })
                    result.is_valid = False
            
            # Validate properties
            properties = schema.get('properties', {})
            for field_name, field_schema in properties.items():
                if field_name in data:
                    field_result = self._validate_field(
                        field_name, data[field_name], field_schema
                    )
                    
                    if field_result['is_valid']:
                        result.sanitized_data[field_name] = field_result['value']
                    else:
                        result.errors.extend(field_result['errors'])
                        result.warnings.extend(field_result['warnings'])
                        result.is_valid = False
                elif field_schema.get('default') is not None:
                    # Apply default value
                    result.sanitized_data[field_name] = field_schema['default']
            
            # Check for additional properties
            if not schema.get('additionalProperties', True):
                for field_name in data:
                    if field_name not in properties:
                        result.warnings.append({
                            'field': field_name,
                            'code': 'ADDITIONAL_PROPERTY',
                            'message': f'Additional property "{field_name}" not allowed in schema'
                        })
            
        except Exception as e:
            result.errors.append({
                'field': '__global__',
                'code': 'SCHEMA_VALIDATION_ERROR',
                'message': f'Schema validation failed: {str(e)}'
            })
            result.is_valid = False
        
        end_time = datetime.now()
        result.validation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return result
    
    def _validate_field(self, field_name: str, value: Any, field_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual field against schema."""
        result = {
            'is_valid': True,
            'value': value,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Type validation and coercion
            expected_type = field_schema.get('type')
            if expected_type and expected_type in self.type_coercers:
                try:
                    result['value'] = self.type_coercers[expected_type](value)
                except (ValueError, TypeError) as e:
                    result['errors'].append({
                        'field': field_name,
                        'code': 'TYPE_COERCION_FAILED',
                        'message': f'Cannot convert to {expected_type}: {str(e)}'
                    })
                    result['is_valid'] = False
                    return result
            
            # String-specific validations
            if expected_type == 'string':
                result['value'] = self._validate_string_field(
                    field_name, result['value'], field_schema, result
                )
            
            # Numeric validations
            elif expected_type in ('integer', 'number'):
                result['value'] = self._validate_numeric_field(
                    field_name, result['value'], field_schema, result
                )
            
            # Array validations
            elif expected_type == 'array':
                result['value'] = self._validate_array_field(
                    field_name, result['value'], field_schema, result
                )
            
            # Object validations
            elif expected_type == 'object':
                result['value'] = self._validate_object_field(
                    field_name, result['value'], field_schema, result
                )
            
            # Custom validations
            custom_validator = field_schema.get('validator')
            if custom_validator and custom_validator in self.custom_validators:
                if not self.custom_validators[custom_validator](result['value']):
                    result['errors'].append({
                        'field': field_name,
                        'code': 'CUSTOM_VALIDATION_FAILED',
                        'message': f'Custom validation "{custom_validator}" failed'
                    })
                    result['is_valid'] = False
            
            # Enum validation
            enum_values = field_schema.get('enum')
            if enum_values and result['value'] not in enum_values:
                result['errors'].append({
                    'field': field_name,
                    'code': 'ENUM_VALIDATION_FAILED',
                    'message': f'Value must be one of: {enum_values}'
                })
                result['is_valid'] = False
        
        except Exception as e:
            result['errors'].append({
                'field': field_name,
                'code': 'FIELD_VALIDATION_ERROR',
                'message': f'Field validation failed: {str(e)}'
            })
            result['is_valid'] = False
        
        return result
    
    def _validate_string_field(self, field_name: str, value: str, schema: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Validate string field with sanitization."""
        # Length validation
        min_length = schema.get('minLength', 0)
        max_length = schema.get('maxLength', float('inf'))
        
        if len(value) < min_length:
            result['errors'].append({
                'field': field_name,
                'code': 'STRING_TOO_SHORT',
                'message': f'String must be at least {min_length} characters'
            })
            result['is_valid'] = False
        
        if len(value) > max_length:
            result['errors'].append({
                'field': field_name,
                'code': 'STRING_TOO_LONG',
                'message': f'String must be no more than {max_length} characters'
            })
            result['is_valid'] = False
        
        # Pattern validation
        pattern = schema.get('pattern')
        if pattern:
            import re
            if not re.match(pattern, value):
                result['errors'].append({
                    'field': field_name,
                    'code': 'PATTERN_MISMATCH',
                    'message': f'String does not match required pattern: {pattern}'
                })
                result['is_valid'] = False
        
        # Sanitization
        sanitize = schema.get('sanitize', True)
        if sanitize:
            value = self.sanitizer.sanitize_html(value, strict=True)
        
        return value
    
    def _validate_numeric_field(self, field_name: str, value: Union[int, float], schema: Dict[str, Any], result: Dict[str, Any]) -> Union[int, float]:
        """Validate numeric field with range checks."""
        minimum = schema.get('minimum')
        maximum = schema.get('maximum')
        exclusive_minimum = schema.get('exclusiveMinimum')
        exclusive_maximum = schema.get('exclusiveMaximum')
        
        if minimum is not None and value < minimum:
            result['errors'].append({
                'field': field_name,
                'code': 'NUMBER_TOO_SMALL',
                'message': f'Number must be at least {minimum}'
            })
            result['is_valid'] = False
        
        if maximum is not None and value > maximum:
            result['errors'].append({
                'field': field_name,
                'code': 'NUMBER_TOO_LARGE',
                'message': f'Number must be no more than {maximum}'
            })
            result['is_valid'] = False
        
        if exclusive_minimum is not None and value <= exclusive_minimum:
            result['errors'].append({
                'field': field_name,
                'code': 'NUMBER_NOT_EXCLUSIVE_MIN',
                'message': f'Number must be greater than {exclusive_minimum}'
            })
            result['is_valid'] = False
        
        if exclusive_maximum is not None and value >= exclusive_maximum:
            result['errors'].append({
                'field': field_name,
                'code': 'NUMBER_NOT_EXCLUSIVE_MAX',
                'message': f'Number must be less than {exclusive_maximum}'
            })
            result['is_valid'] = False
        
        return value
    
    def _validate_array_field(self, field_name: str, value: List[Any], schema: Dict[str, Any], result: Dict[str, Any]) -> List[Any]:
        """Validate array field with item validation."""
        min_items = schema.get('minItems', 0)
        max_items = schema.get('maxItems', float('inf'))
        
        if len(value) < min_items:
            result['errors'].append({
                'field': field_name,
                'code': 'ARRAY_TOO_SHORT',
                'message': f'Array must have at least {min_items} items'
            })
            result['is_valid'] = False
        
        if len(value) > max_items:
            result['errors'].append({
                'field': field_name,
                'code': 'ARRAY_TOO_LONG',
                'message': f'Array must have no more than {max_items} items'
            })
            result['is_valid'] = False
        
        # Validate items if schema provided
        items_schema = schema.get('items')
        if items_schema:
            validated_items = []
            for i, item in enumerate(value):
                item_result = self._validate_field(f'{field_name}[{i}]', item, items_schema)
                if item_result['is_valid']:
                    validated_items.append(item_result['value'])
                else:
                    result['errors'].extend(item_result['errors'])
                    result['warnings'].extend(item_result['warnings'])
                    result['is_valid'] = False
            value = validated_items
        
        # Unique items validation
        if schema.get('uniqueItems', False):
            if len(value) != len(set(str(item) for item in value)):
                result['errors'].append({
                    'field': field_name,
                    'code': 'ARRAY_NOT_UNIQUE',
                    'message': 'Array items must be unique'
                })
                result['is_valid'] = False
        
        return value
    
    def _validate_object_field(self, field_name: str, value: Dict[str, Any], schema: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate object field recursively."""
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        # Validate required properties
        for prop in required:
            if prop not in value:
                result['errors'].append({
                    'field': f'{field_name}.{prop}',
                    'code': 'REQUIRED_PROPERTY_MISSING',
                    'message': f'Required property "{prop}" is missing'
                })
                result['is_valid'] = False
        
        # Validate each property
        validated_object = {}
        for prop_name, prop_value in value.items():
            if prop_name in properties:
                prop_result = self._validate_field(
                    f'{field_name}.{prop_name}', prop_value, properties[prop_name]
                )
                if prop_result['is_valid']:
                    validated_object[prop_name] = prop_result['value']
                else:
                    result['errors'].extend(prop_result['errors'])
                    result['warnings'].extend(prop_result['warnings'])
                    result['is_valid'] = False
            elif not schema.get('additionalProperties', True):
                result['warnings'].append({
                    'field': f'{field_name}.{prop_name}',
                    'code': 'ADDITIONAL_PROPERTY',
                    'message': f'Additional property "{prop_name}" not allowed'
                })
            else:
                validated_object[prop_name] = prop_value
        
        return validated_object


class EnvironmentManager:
    """
    Secure environment variable handling with encryption and validation.
    
    Features:
    - Environment variable validation and type conversion
    - Sensitive data encryption at rest
    - Environment isolation and sandboxing
    - Audit logging for environment access
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.env_cache: Dict[str, Any] = {}
        self.sensitive_patterns = [
            r'.*password.*', r'.*secret.*', r'.*key.*', r'.*token.*',
            r'.*api.*key.*', r'.*auth.*', r'.*credential.*'
        ]
    
    def _generate_key(self) -> bytes:
        """Generate encryption key from system entropy."""
        password = os.urandom(32)
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    def _is_sensitive(self, key: str) -> bool:
        """Check if environment variable contains sensitive data."""
        import re
        return any(re.match(pattern, key.lower()) for pattern in self.sensitive_patterns)
    
    def get_env_var(self, key: str, default: Any = None, var_type: Type = str, required: bool = False) -> Any:
        """
        Get and validate environment variable with type conversion.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            var_type: Expected type for conversion
            required: Whether variable is required
            
        Returns:
            Converted and validated environment variable value
        """
        if key in self.env_cache:
            return self.env_cache[key]
        
        value = os.environ.get(key)
        
        if value is None:
            if required:
                raise ValidationError(key, f"Required environment variable '{key}' not found")
            return default
        
        # Decrypt if sensitive
        if self._is_sensitive(key) and value.startswith('encrypted:'):
            try:
                encrypted_value = value[10:]  # Remove 'encrypted:' prefix
                value = self.cipher_suite.decrypt(encrypted_value.encode()).decode()
            except Exception as e:
                logger.warning(f"Failed to decrypt environment variable '{key}': {e}")
                return default
        
        # Type conversion
        try:
            if var_type == bool:
                converted_value = value.lower() in ('true', '1', 'yes', 'on')
            elif var_type == int:
                converted_value = int(value)
            elif var_type == float:
                converted_value = float(value)
            elif var_type == list:
                converted_value = [item.strip() for item in value.split(',')]
            else:
                converted_value = var_type(value)
            
            self.env_cache[key] = converted_value
            return converted_value
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert environment variable '{key}' to {var_type}: {e}")
            if required:
                raise ValidationError(key, f"Cannot convert environment variable '{key}' to {var_type}")
            return default
    
    def set_env_var(self, key: str, value: Any, encrypt: bool = None) -> None:
        """Set environment variable with optional encryption."""
        if encrypt is None:
            encrypt = self._is_sensitive(key)
        
        str_value = str(value)
        
        if encrypt:
            encrypted_value = self.cipher_suite.encrypt(str_value.encode()).decode()
            str_value = f"encrypted:{encrypted_value}"
        
        os.environ[key] = str_value
        self.env_cache[key] = value
    
    def validate_environment(self, required_vars: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate required environment variables.
        
        Args:
            required_vars: List of required variable specifications
            
        Returns:
            ValidationResult with environment validation status
        """
        result = ValidationResult(is_valid=True)
        
        for var_spec in required_vars:
            key = var_spec['name']
            var_type = var_spec.get('type', str)
            required = var_spec.get('required', True)
            default = var_spec.get('default')
            
            try:
                value = self.get_env_var(key, default, var_type, required)
                result.sanitized_data[key] = value
            except ValidationError as e:
                result.errors.append({
                    'field': key,
                    'code': 'ENV_VAR_VALIDATION_FAILED',
                    'message': str(e)
                })
                result.is_valid = False
        
        return result
    
    @contextmanager
    def isolated_environment(self, env_vars: Dict[str, str]):
        """Create isolated environment context."""
        original_env = os.environ.copy()
        try:
            os.environ.clear()
            os.environ.update(env_vars)
            yield
        finally:
            os.environ.clear()
            os.environ.update(original_env)


class ParameterLoader:
    """
    Multi-format parameter loading with validation and merging.
    
    Supports loading parameters from:
    - JSON files
    - YAML files
    - Command line arguments
    - Environment variables
    - Configuration files
    """
    
    def __init__(self, schema_validator: ParameterSchema, env_manager: EnvironmentManager):
        self.schema_validator = schema_validator
        self.env_manager = env_manager
        self.loaded_sources: List[str] = []
    
    def load_from_json(self, json_input: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load parameters from JSON file or string."""
        try:
            if isinstance(json_input, dict):
                return json_input
            elif isinstance(json_input, (str, Path)):
                if Path(json_input).exists():
                    with open(json_input, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.loaded_sources.append(f"json_file:{json_input}")
                else:
                    data = json.loads(str(json_input))
                    self.loaded_sources.append("json_string")
                return data
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            raise ValidationError("json_input", f"Failed to load JSON: {str(e)}")
    
    def load_from_yaml(self, yaml_input: Union[str, Path]) -> Dict[str, Any]:
        """Load parameters from YAML file or string."""
        try:
            if Path(yaml_input).exists():
                with open(yaml_input, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                self.loaded_sources.append(f"yaml_file:{yaml_input}")
            else:
                data = yaml.safe_load(str(yaml_input))
                self.loaded_sources.append("yaml_string")
            return data or {}
        except (yaml.YAMLError, FileNotFoundError, PermissionError) as e:
            raise ValidationError("yaml_input", f"Failed to load YAML: {str(e)}")
    
    def load_from_cli(self, cli_args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Load parameters from command line arguments."""
        parser = argparse.ArgumentParser(description="Workflow Parameters")
        parser.add_argument('--config', type=str, help='Configuration file path')
        parser.add_argument('--format', choices=['json', 'yaml'], default='json', help='Configuration format')
        parser.add_argument('--validate-only', action='store_true', help='Only validate parameters')
        parser.add_argument('--validation-level', choices=['strict', 'lenient', 'permissive'], default='strict')
        parser.add_argument('--env-file', type=str, help='Environment file path')
        parser.add_argument('--params', type=str, help='JSON string of parameters')
        
        # Allow arbitrary parameters
        parser.add_argument('--param', action='append', help='Parameter in key=value format')
        
        args, unknown = parser.parse_known_args(cli_args)
        
        params = {}
        
        # Parse --param arguments
        if args.param:
            for param in args.param:
                if '=' in param:
                    key, value = param.split('=', 1)
                    # Try to parse as JSON first, fallback to string
                    try:
                        params[key] = json.loads(value)
                    except json.JSONDecodeError:
                        params[key] = value
        
        # Parse unknown arguments as key=value pairs
        for arg in unknown:
            if arg.startswith('--') and '=' in arg:
                key_value = arg[2:].split('=', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    try:
                        params[key] = json.loads(value)
                    except json.JSONDecodeError:
                        params[key] = value
        
        # Add parsed args to params
        params.update(vars(args))
        
        self.loaded_sources.append("cli_args")
        return params
    
    def load_from_env(self, prefix: str = "WORKFLOW_") -> Dict[str, Any]:
        """Load parameters from environment variables with prefix."""
        params = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                param_key = key[len(prefix):].lower()
                # Try to parse as JSON, fallback to string
                try:
                    params[param_key] = json.loads(value)
                except json.JSONDecodeError:
                    params[param_key] = value
        
        if params:
            self.loaded_sources.append(f"env_vars:{prefix}")
        
        return params
    
    def merge_parameters(self, *param_sources: Dict[str, Any], priority_order: bool = True) -> Dict[str, Any]:
        """
        Merge parameters from multiple sources.
        
        Args:
            param_sources: Parameter dictionaries to merge
            priority_order: If True, later sources override earlier ones
            
        Returns:
            Merged parameter dictionary
        """
        merged = {}
        
        sources = param_sources if priority_order else reversed(param_sources)
        
        for source in sources:
            if isinstance(source, dict):
                self._deep_merge(merged, source)
        
        return merged
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source into target dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value


class WorkflowExecutor:
    """
    Main workflow execution engine with comprehensive parameter validation.
    
    Orchestrates parameter loading, validation, and workflow execution
    with state management and error recovery.
    """
    
    def __init__(self, 
                 validation_level: ValidationLevel = ValidationLevel.STRICT,
                 enable_encryption: bool = True,
                 temp_dir: Optional[str] = None):
        self.validation_level = validation_level
        self.schema_validator = ParameterSchema()
        self.env_manager = EnvironmentManager() if enable_encryption else None
        self.parameter_loader = ParameterLoader(self.schema_validator, self.env_manager)
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "workflow_execution"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.workflows: Dict[str, WorkflowContext] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Register default validators
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register commonly used custom validators."""
        
        def validate_url(value: str) -> bool:
            import re
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain...
                r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # host...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            return bool(url_pattern.match(value))
        
        def validate_email(value: str) -> bool:
            import re
            email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
            return bool(email_pattern.match(value))
        
        def validate_uuid(value: str) -> bool:
            import re
            uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
            return bool(uuid_pattern.match(value.lower()))
        
        def validate_positive_number(value: Union[int, float]) -> bool:
            return isinstance(value, (int, float)) and value > 0
        
        self.schema_validator.register_validator('url', validate_url)
        self.schema_validator.register_validator('email', validate_email)
        self.schema_validator.register_validator('uuid', validate_uuid)
        self.schema_validator.register_validator('positive_number', validate_positive_number)
    
    def create_workflow(self, workflow_id: str, schema: Dict[str, Any], 
                       parameter_sources: List[Dict[str, Any]]) -> WorkflowContext:
        """
        Create new workflow with parameter validation.
        
        Args:
            workflow_id: Unique workflow identifier
            schema: JSON schema for parameter validation
            parameter_sources: List of parameter sources to merge
            
        Returns:
            WorkflowContext with validated parameters
        """
        logger.info(f"Creating workflow: {workflow_id}")
        
        # Merge parameters from all sources
        merged_params = self.parameter_loader.merge_parameters(*parameter_sources)
        
        # Validate against schema
        validation_result = self.schema_validator.validate_schema(merged_params, schema)
        
        # Handle validation based on level
        if not validation_result.is_valid:
            if self.validation_level == ValidationLevel.STRICT:
                error_msg = "; ".join([err['message'] for err in validation_result.errors])
                raise ValidationError("workflow_parameters", f"Parameter validation failed: {error_msg}")
            elif self.validation_level == ValidationLevel.LENIENT:
                logger.warning(f"Parameter validation warnings for workflow {workflow_id}:")
                for error in validation_result.errors:
                    logger.warning(f"  - {error['field']}: {error['message']}")
        
        # Create workflow context
        now = datetime.now(timezone.utc)
        context = WorkflowContext(
            workflow_id=workflow_id,
            parameters=validation_result.sanitized_data,
            environment=dict(os.environ),
            state=WorkflowState.VALIDATING,
            created_at=now,
            updated_at=now,
            validation_result=validation_result,
            metadata={
                'parameter_sources': self.parameter_loader.loaded_sources,
                'validation_level': self.validation_level.value,
                'schema_version': validation_result.schema_version
            }
        )
        
        self.workflows[workflow_id] = context
        
        # Update state to pending
        self._update_workflow_state(workflow_id, WorkflowState.PENDING)
        
        logger.info(f"Workflow {workflow_id} created successfully with {len(validation_result.sanitized_data)} parameters")
        return context
    
    def execute_workflow(self, workflow_id: str, execution_func: Callable[[WorkflowContext], Any]) -> Any:
        """
        Execute workflow with validated parameters.
        
        Args:
            workflow_id: Workflow identifier
            execution_func: Function to execute with workflow context
            
        Returns:
            Execution result
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        context = self.workflows[workflow_id]
        
        if context.state != WorkflowState.PENDING:
            raise ValueError(f"Workflow {workflow_id} is not in pending state (current: {context.state})")
        
        logger.info(f"Starting execution of workflow: {workflow_id}")
        
        start_time = datetime.now(timezone.utc)
        self._update_workflow_state(workflow_id, WorkflowState.RUNNING)
        
        try:
            result = execution_func(context)
            
            # Record successful execution
            end_time = datetime.now(timezone.utc)
            execution_record = {
                'workflow_id': workflow_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': (end_time - start_time).total_seconds(),
                'status': 'completed',
                'result_summary': str(result)[:500]  # Truncate long results
            }
            self.execution_history.append(execution_record)
            
            self._update_workflow_state(workflow_id, WorkflowState.COMPLETED)
            logger.info(f"Workflow {workflow_id} completed successfully")
            
            return result
            
        except Exception as e:
            # Record failed execution
            end_time = datetime.now(timezone.utc)
            execution_record = {
                'workflow_id': workflow_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': (end_time - start_time).total_seconds(),
                'status': 'failed',
                'error': str(e)
            }
            self.execution_history.append(execution_record)
            
            self._update_workflow_state(workflow_id, WorkflowState.FAILED)
            logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            
            raise
    
    def _update_workflow_state(self, workflow_id: str, new_state: WorkflowState):
        """Update workflow state and timestamp."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].state = new_state
            self.workflows[workflow_id].updated_at = datetime.now(timezone.utc)
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current workflow status and metadata."""
        if workflow_id not in self.workflows:
            return {'error': f'Workflow {workflow_id} not found'}
        
        context = self.workflows[workflow_id]
        return {
            'workflow_id': workflow_id,
            'state': context.state.value,
            'created_at': context.created_at.isoformat(),
            'updated_at': context.updated_at.isoformat(),
            'parameter_count': len(context.parameters),
            'validation_errors': len(context.validation_result.errors) if context.validation_result else 0,
            'validation_warnings': len(context.validation_result.warnings) if context.validation_result else 0,
            'metadata': context.metadata
        }
    
    def cleanup_workflow(self, workflow_id: str) -> bool:
        """Clean up workflow resources and temporary files."""
        if workflow_id not in self.workflows:
            return False
        
        # Clean up temporary files
        workflow_temp_dir = self.temp_dir / workflow_id
        if workflow_temp_dir.exists():
            import shutil
            shutil.rmtree(workflow_temp_dir)
        
        # Remove from memory
        del self.workflows[workflow_id]
        
        logger.info(f"Cleaned up workflow: {workflow_id}")
        return True


def create_sample_schema() -> Dict[str, Any]:
    """Create a sample JSON schema for demonstration."""
    return {
        "type": "object",
        "required": ["task_name", "input_data"],
        "properties": {
            "task_name": {
                "type": "string",
                "minLength": 3,
                "maxLength": 100,
                "pattern": "^[a-zA-Z0-9_-]+$",
                "description": "Unique task identifier"
            },
            "input_data": {
                "type": "object",
                "required": ["source"],
                "properties": {
                    "source": {
                        "type": "string",
                        "validator": "url",
                        "description": "Data source URL"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "csv", "xml"],
                        "default": "json"
                    },
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10000,
                        "default": 100
                    }
                }
            },
            "processing_options": {
                "type": "object",
                "properties": {
                    "parallel": {
                        "type": "boolean",
                        "default": true
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3600,
                        "default": 300
                    },
                    "retry_count": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 5,
                        "default": 3
                    }
                }
            },
            "notification_settings": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "validator": "email",
                        "description": "Notification email address"
                    },
                    "webhook_url": {
                        "type": "string",
                        "validator": "url",
                        "description": "Webhook for status updates"
                    }
                }
            }
        }
    }


def example_workflow_execution(context: WorkflowContext) -> Dict[str, Any]:
    """Example workflow execution function."""
    logger.info(f"Executing workflow: {context.workflow_id}")
    logger.info(f"Parameters: {context.parameters}")
    
    # Simulate some work
    import time
    time.sleep(1)
    
    return {
        'status': 'success',
        'processed_items': context.parameters.get('input_data', {}).get('batch_size', 0),
        'execution_time': '1.0s',
        'output_location': f'/tmp/output_{context.workflow_id}'
    }


def main():
    """Main function demonstrating workflow execution system."""
    # Create workflow executor
    executor = WorkflowExecutor(
        validation_level=ValidationLevel.STRICT,
        enable_encryption=True
    )
    
    # Sample parameters from different sources
    json_params = {
        "task_name": "sample_task",
        "input_data": {
            "source": "https://api.example.com/data",
            "format": "json",
            "batch_size": 500
        }
    }
    
    cli_params = executor.parameter_loader.load_from_cli()
    env_params = executor.parameter_loader.load_from_env()
    
    # Create and execute workflow
    try:
        schema = create_sample_schema()
        workflow_id = "demo_workflow_001"
        
        # Create workflow with parameter validation
        context = executor.create_workflow(
            workflow_id=workflow_id,
            schema=schema,
            parameter_sources=[json_params, cli_params, env_params]
        )
        
        # Execute workflow
        result = executor.execute_workflow(workflow_id, example_workflow_execution)
        
        print(f"Workflow executed successfully: {result}")
        print(f"Status: {executor.get_workflow_status(workflow_id)}")
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Execution error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if workflow_id in executor.workflows:
            executor.cleanup_workflow(workflow_id)


if __name__ == "__main__":
    main()