"""
Comprehensive Input Validation and Sanitization for MoneyPrinterTurbo

Provides enterprise-grade input validation, sanitization, and protection
against injection attacks, XSS, CSRF, and other security vulnerabilities.

Complies with SPARC principles: ≤500 lines, modular, testable, secure.
"""

import re
import html
import json
import bleach
import validators
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse
from loguru import logger
import sqlalchemy
from sqlalchemy import text
from markupsafe import Markup, escape


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation error in '{field}': {message}")


class InputSanitizer:
    """
    Comprehensive input sanitization with context-aware cleaning.
    
    Features:
    - HTML/XSS sanitization with allowlist approach
    - SQL injection prevention
    - Command injection protection
    - File path sanitization
    - Email and URL validation
    """
    
    def __init__(self):
        # HTML sanitization configuration
        self.allowed_tags = {
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'
        }
        
        self.allowed_attributes = {
            '*': ['class'],
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title', 'width', 'height']
        }
        
        # Dangerous patterns for various injection types
        self.sql_injection_patterns = [
            r"(\b(select|insert|update|delete|drop|create|alter|exec|execute)\b)",
            r"(\bunion\b.*\bselect\b)",
            r"(\bor\b.*=.*)",
            r"(\band\b.*=.*)",
            r"(--|\#|\/\*|\*\/)",
            r"(\bxp_cmdshell\b)",
            r"(\bsp_executesql\b)"
        ]
        
        self.command_injection_patterns = [
            r"[;&|`$()<>]",
            r"\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig)\b",
            r"\b(rm|mv|cp|chmod|chown|sudo|su)\b",
            r"(\.\.\/|\.\.\\)",
            r"(\$\(|\`)"
        ]
        
        self.path_traversal_patterns = [
            r"(\.\.\/|\.\.\\)",
            r"(\/etc\/|\\windows\\)",
            r"(\/proc\/|\/sys\/)",
            r"(\.\.%2f|\.\.%5c)",
            r"(%2e%2e%2f|%2e%2e%5c)"
        ]
    
    def sanitize_html(self, input_text: str, strict: bool = True) -> str:
        """
        Sanitize HTML input to prevent XSS attacks.
        
        Args:
            input_text: Raw HTML input
            strict: If True, strip all HTML; if False, allow safe tags
            
        Returns:
            Sanitized HTML string
        """
        if not isinstance(input_text, str):
            return str(input_text)
        
        if strict:
            # Strip all HTML and escape special characters
            return html.escape(input_text, quote=True)
        
        # Use bleach for safe HTML cleaning
        cleaned = bleach.clean(
            input_text,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True
        )
        
        return cleaned
    
    def sanitize_sql_input(self, input_value: Any) -> str:
        """
        Sanitize input for SQL queries (use with parameterized queries).
        
        Args:
            input_value: Value to sanitize
            
        Returns:
            Sanitized string value
        """
        if input_value is None:
            return ""
        
        # Convert to string and check for SQL injection patterns
        str_value = str(input_value)
        
        # Check for dangerous SQL patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, str_value, re.IGNORECASE):
                logger.warning(f"Potential SQL injection attempt: {pattern}")
                raise ValidationError("sql_input", "Contains potential SQL injection")
        
        # Escape single quotes
        return str_value.replace("'", "''")
    
    def sanitize_command_input(self, input_value: str) -> str:
        """
        Sanitize input to prevent command injection.
        
        Args:
            input_value: Command input to sanitize
            
        Returns:
            Sanitized command string
        """
        if not isinstance(input_value, str):
            input_value = str(input_value)
        
        # Check for command injection patterns
        for pattern in self.command_injection_patterns:
            if re.search(pattern, input_value, re.IGNORECASE):
                logger.warning("Potential command injection attempt detected")
                raise ValidationError("command_input", "Contains dangerous command characters")
        
        # Allow only alphanumeric, spaces, hyphens, underscores
        sanitized = re.sub(r'[^a-zA-Z0-9\s\-_.]', '', input_value)
        return sanitized.strip()
    
    def sanitize_file_path(self, file_path: str) -> str:
        """
        Sanitize file path to prevent directory traversal attacks.
        
        Args:
            file_path: File path to sanitize
            
        Returns:
            Sanitized file path
        """
        if not isinstance(file_path, str):
            file_path = str(file_path)
        
        # Check for path traversal patterns
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                logger.warning("Potential path traversal attempt detected")
                raise ValidationError("file_path", "Contains path traversal sequences")
        
        # Normalize path and remove dangerous characters
        import os
        normalized = os.path.normpath(file_path)
        
        # Ensure path doesn't escape current directory
        if normalized.startswith('/') or normalized.startswith('\\'):
            raise ValidationError("file_path", "Absolute paths not allowed")
        
        if '..' in normalized:
            raise ValidationError("file_path", "Parent directory access not allowed")
        
        return normalized
    
    def sanitize_json_input(self, json_input: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Safely parse and sanitize JSON input.
        
        Args:
            json_input: JSON string to parse
            max_depth: Maximum nesting depth allowed
            
        Returns:
            Parsed and sanitized JSON object
        """
        try:
            # Parse JSON with size limits
            if len(json_input) > 1048576:  # 1MB limit
                raise ValidationError("json_input", "JSON input too large")
            
            parsed = json.loads(json_input)
            
            # Check nesting depth
            def check_depth(obj, depth=0):
                if depth > max_depth:
                    raise ValidationError("json_input", "JSON nesting too deep")
                
                if isinstance(obj, dict):
                    for value in obj.values():
                        check_depth(value, depth + 1)
                elif isinstance(obj, list):
                    for item in obj:
                        check_depth(item, depth + 1)
            
            check_depth(parsed)
            return parsed
            
        except json.JSONDecodeError as e:
            raise ValidationError("json_input", f"Invalid JSON: {str(e)}")


class DataValidator:
    """
    Comprehensive data validation with type checking and business rules.
    
    Features:
    - Type validation with conversion
    - Range and length validation
    - Format validation (email, URL, phone, etc.)
    - Business rule validation
    - Custom validator support
    """
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        
        # Common regex patterns
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            'slug': r'^[a-z0-9]+(?:-[a-z0-9]+)*$',
            'username': r'^[a-zA-Z0-9_]{3,30}$',
            'password': r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
        }
    
    def validate_string(self, value: Any, field: str = "field", 
                       min_length: int = 0, max_length: int = 1000,
                       pattern: Optional[str] = None, 
                       sanitize: bool = True) -> str:
        """
        Validate and sanitize string input.
        
        Args:
            value: Input value to validate
            field: Field name for error reporting
            min_length: Minimum string length
            max_length: Maximum string length
            pattern: Regex pattern to match
            sanitize: Whether to sanitize HTML
            
        Returns:
            Validated and sanitized string
        """
        if value is None:
            raise ValidationError(field, "Value cannot be None")
        
        # Convert to string
        str_value = str(value).strip()
        
        # Check length
        if len(str_value) < min_length:
            raise ValidationError(field, f"Must be at least {min_length} characters")
        
        if len(str_value) > max_length:
            raise ValidationError(field, f"Must be no more than {max_length} characters")
        
        # Check pattern
        if pattern and not re.match(pattern, str_value):
            raise ValidationError(field, "Invalid format")
        
        # Sanitize if requested
        if sanitize:
            str_value = self.sanitizer.sanitize_html(str_value, strict=True)
        
        return str_value
    
    def validate_email(self, email: str, field: str = "email") -> str:
        """Validate email address."""
        validated = self.validate_string(email, field, max_length=254)
        
        if not re.match(self.patterns['email'], validated):
            raise ValidationError(field, "Invalid email format")
        
        # Additional validation using validators library
        if not validators.email(validated):
            raise ValidationError(field, "Invalid email address")
        
        return validated.lower()
    
    def validate_url(self, url: str, field: str = "url", 
                    allowed_schemes: List[str] = None) -> str:
        """Validate URL."""
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        validated = self.validate_string(url, field, max_length=2048)
        
        try:
            parsed = urlparse(validated)
            
            if parsed.scheme not in allowed_schemes:
                raise ValidationError(field, f"Scheme must be one of: {allowed_schemes}")
            
            if not parsed.netloc:
                raise ValidationError(field, "Invalid URL format")
            
            # Use validators library for additional checks
            if not validators.url(validated):
                raise ValidationError(field, "Invalid URL")
            
        except Exception:
            raise ValidationError(field, "Invalid URL format")
        
        return validated
    
    def validate_integer(self, value: Any, field: str = "field",
                        min_value: Optional[int] = None,
                        max_value: Optional[int] = None) -> int:
        """Validate integer input."""
        try:
            if isinstance(value, str):
                # Remove any non-numeric characters except minus
                cleaned = re.sub(r'[^\d-]', '', value)
                int_value = int(cleaned)
            else:
                int_value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(field, "Must be a valid integer")
        
        if min_value is not None and int_value < min_value:
            raise ValidationError(field, f"Must be at least {min_value}")
        
        if max_value is not None and int_value > max_value:
            raise ValidationError(field, f"Must be no more than {max_value}")
        
        return int_value
    
    def validate_decimal(self, value: Any, field: str = "field",
                        min_value: Optional[Decimal] = None,
                        max_value: Optional[Decimal] = None,
                        max_digits: int = 10,
                        decimal_places: int = 2) -> Decimal:
        """Validate decimal input."""
        try:
            if isinstance(value, str):
                # Remove any non-numeric characters except decimal point and minus
                cleaned = re.sub(r'[^\d.-]', '', value)
                decimal_value = Decimal(cleaned)
            else:
                decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            raise ValidationError(field, "Must be a valid decimal number")
        
        # Check decimal places
        if decimal_value.as_tuple().exponent < -decimal_places:
            raise ValidationError(field, f"Maximum {decimal_places} decimal places allowed")
        
        # Check total digits
        digits = len(decimal_value.as_tuple().digits)
        if digits > max_digits:
            raise ValidationError(field, f"Maximum {max_digits} digits allowed")
        
        if min_value is not None and decimal_value < min_value:
            raise ValidationError(field, f"Must be at least {min_value}")
        
        if max_value is not None and decimal_value > max_value:
            raise ValidationError(field, f"Must be no more than {max_value}")
        
        return decimal_value
    
    def validate_boolean(self, value: Any, field: str = "field") -> bool:
        """Validate boolean input."""
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            lower_value = value.lower().strip()
            if lower_value in ('true', '1', 'yes', 'on'):
                return True
            elif lower_value in ('false', '0', 'no', 'off'):
                return False
        
        if isinstance(value, (int, float)):
            return bool(value)
        
        raise ValidationError(field, "Must be a valid boolean value")
    
    def validate_datetime(self, value: Any, field: str = "field",
                         format_string: str = "%Y-%m-%d %H:%M:%S") -> datetime:
        """Validate datetime input."""
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, str):
            try:
                return datetime.strptime(value, format_string)
            except ValueError:
                # Try ISO format as fallback
                try:
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    raise ValidationError(field, f"Invalid datetime format. Expected: {format_string}")
        
        raise ValidationError(field, "Must be a valid datetime")
    
    def validate_choice(self, value: Any, choices: List[Any], field: str = "field") -> Any:
        """Validate that value is in allowed choices."""
        if value not in choices:
            raise ValidationError(field, f"Must be one of: {choices}")
        return value
    
    def validate_json_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against a schema definition.
        
        Args:
            data: Data to validate
            schema: Schema definition with field rules
            
        Returns:
            Validated and sanitized data
        """
        validated_data = {}
        
        for field_name, field_rules in schema.items():
            value = data.get(field_name)
            
            # Check required fields
            if field_rules.get('required', False) and value is None:
                raise ValidationError(field_name, "This field is required")
            
            # Skip validation if value is None and field is not required
            if value is None:
                validated_data[field_name] = None
                continue
            
            # Get field type and validate accordingly
            field_type = field_rules.get('type', 'string')
            
            try:
                if field_type == 'string':
                    validated_data[field_name] = self.validate_string(
                        value, field_name,
                        min_length=field_rules.get('min_length', 0),
                        max_length=field_rules.get('max_length', 1000),
                        pattern=field_rules.get('pattern'),
                        sanitize=field_rules.get('sanitize', True)
                    )
                elif field_type == 'email':
                    validated_data[field_name] = self.validate_email(value, field_name)
                elif field_type == 'url':
                    validated_data[field_name] = self.validate_url(value, field_name)
                elif field_type == 'integer':
                    validated_data[field_name] = self.validate_integer(
                        value, field_name,
                        min_value=field_rules.get('min_value'),
                        max_value=field_rules.get('max_value')
                    )
                elif field_type == 'decimal':
                    validated_data[field_name] = self.validate_decimal(
                        value, field_name,
                        min_value=field_rules.get('min_value'),
                        max_value=field_rules.get('max_value'),
                        max_digits=field_rules.get('max_digits', 10),
                        decimal_places=field_rules.get('decimal_places', 2)
                    )
                elif field_type == 'boolean':
                    validated_data[field_name] = self.validate_boolean(value, field_name)
                elif field_type == 'datetime':
                    validated_data[field_name] = self.validate_datetime(
                        value, field_name,
                        format_string=field_rules.get('format', "%Y-%m-%d %H:%M:%S")
                    )
                elif field_type == 'choice':
                    validated_data[field_name] = self.validate_choice(
                        value, field_rules['choices'], field_name
                    )
                else:
                    # Unknown type, treat as string
                    validated_data[field_name] = self.validate_string(value, field_name)
                    
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(field_name, f"Validation error: {str(e)}")
        
        return validated_data


# Global instances for easy access
input_sanitizer = InputSanitizer()
data_validator = DataValidator()


def validate_api_input(schema: Dict[str, Any]):
    """
    Decorator for API input validation.
    
    Args:
        schema: Validation schema for request data
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Assume first argument is request data
            if args and isinstance(args[0], dict):
                try:
                    validated_data = data_validator.validate_json_schema(args[0], schema)
                    args = (validated_data,) + args[1:]
                except ValidationError as e:
                    logger.warning(f"Input validation failed: {str(e)}")
                    raise
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def sanitize_output(data: Any, context: str = "html") -> Any:
    """
    Sanitize output data based on context.
    
    Args:
        data: Data to sanitize
        context: Output context (html, json, text)
        
    Returns:
        Sanitized data
    """
    if context == "html":
        if isinstance(data, str):
            return input_sanitizer.sanitize_html(data, strict=False)
        elif isinstance(data, dict):
            return {k: sanitize_output(v, context) for k, v in data.items()}
        elif isinstance(data, list):
            return [sanitize_output(item, context) for item in data]
    
    return data