"""
MCP Authentication and Authorization

Provides secure authentication and authorization mechanisms for MCP connections,
including API key validation, JWT tokens, and role-based access control.
"""

import hashlib
import hmac
import jwt
import time
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from loguru import logger

from app.config import config


class MCPAuthenticator:
    """Handles authentication and authorization for MCP connections"""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        self.jwt_secret = config.app.get("mcp_jwt_secret", "default_secret_change_in_production")
        self.jwt_algorithm = config.app.get("mcp_jwt_algorithm", "HS256")
        self.jwt_expiration = config.app.get("mcp_jwt_expiration", 3600)  # 1 hour
        
        # Load API keys from config
        self._load_api_keys()
        
        # Default roles and permissions
        self.permissions = {
            "admin": {
                "tools": ["*"],
                "resources": ["*"],
                "methods": ["*"]
            },
            "user": {
                "tools": [
                    "generate_video_script",
                    "generate_video_terms", 
                    "create_video",
                    "synthesize_voice",
                    "get_generation_status"
                ],
                "resources": ["video/*", "audio/*"],
                "methods": [
                    "tools/list",
                    "tools/call",
                    "service/status"
                ]
            },
            "viewer": {
                "tools": [
                    "get_generation_status",
                    "analyze_video_content"
                ],
                "resources": ["video/read", "audio/read"],
                "methods": [
                    "tools/list",
                    "service/status"
                ]
            }
        }
        
    def _load_api_keys(self):
        """Load API keys from configuration"""
        # This would typically load from database or config file
        # For demo purposes, using config
        api_keys_config = config.app.get("mcp_api_keys", {})
        
        for key, info in api_keys_config.items():
            self.api_keys[key] = {
                "name": info.get("name", "Unknown"),
                "role": info.get("role", "user"),
                "created": info.get("created", datetime.utcnow().isoformat()),
                "expires": info.get("expires"),
                "active": info.get("active", True)
            }
            
        # Add default keys if none configured
        if not self.api_keys:
            self.api_keys["default_admin"] = {
                "name": "Default Admin",
                "role": "admin", 
                "created": datetime.utcnow().isoformat(),
                "expires": None,
                "active": True
            }
            
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key and return user info"""
        if not api_key or api_key not in self.api_keys:
            return None
            
        key_info = self.api_keys[api_key]
        
        # Check if key is active
        if not key_info.get("active", True):
            return None
            
        # Check expiration
        expires = key_info.get("expires")
        if expires:
            expiry_date = datetime.fromisoformat(expires)
            if datetime.utcnow() > expiry_date:
                return None
                
        return key_info
        
    def generate_jwt_token(self, user_info: Dict) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            "user": user_info["name"],
            "role": user_info["role"],
            "iat": int(time.time()),
            "exp": int(time.time()) + self.jwt_expiration
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
    def validate_jwt_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token and return user info"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return {
                "name": payload["user"],
                "role": payload["role"]
            }
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {str(e)}")
            return None
            
    def authenticate_connection(self, auth_data: Dict) -> Optional[Dict]:
        """Authenticate a connection using provided credentials"""
        auth_type = auth_data.get("type", "api_key")
        
        if auth_type == "api_key":
            api_key = auth_data.get("api_key")
            return self.validate_api_key(api_key)
            
        elif auth_type == "jwt":
            token = auth_data.get("token")
            return self.validate_jwt_token(token)
            
        elif auth_type == "hmac":
            return self._validate_hmac_auth(auth_data)
            
        return None
        
    def _validate_hmac_auth(self, auth_data: Dict) -> Optional[Dict]:
        """Validate HMAC-based authentication"""
        try:
            client_id = auth_data.get("client_id")
            timestamp = auth_data.get("timestamp")
            signature = auth_data.get("signature")
            
            if not all([client_id, timestamp, signature]):
                return None
                
            # Check timestamp (prevent replay attacks)
            current_time = int(time.time())
            if abs(current_time - int(timestamp)) > 300:  # 5 minute window
                logger.warning("HMAC timestamp too old")
                return None
                
            # Get client secret
            client_secret = config.app.get("mcp_client_secrets", {}).get(client_id)
            if not client_secret:
                return None
                
            # Calculate expected signature
            message = f"{client_id}{timestamp}"
            expected_signature = hmac.new(
                client_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
                
            # Return user info for this client
            return {
                "name": client_id,
                "role": config.app.get("mcp_client_roles", {}).get(client_id, "user")
            }
            
        except Exception as e:
            logger.error(f"HMAC authentication error: {str(e)}")
            return None
            
    def check_permission(self, user_role: str, permission_type: str, resource: str) -> bool:
        """Check if user role has permission for specific resource"""
        if user_role not in self.permissions:
            return False
            
        role_perms = self.permissions[user_role]
        allowed_resources = role_perms.get(permission_type, [])
        
        # Check for wildcard permission
        if "*" in allowed_resources:
            return True
            
        # Check exact match
        if resource in allowed_resources:
            return True
            
        # Check pattern match (simple glob-style)
        for pattern in allowed_resources:
            if "*" in pattern:
                prefix = pattern.replace("*", "")
                if resource.startswith(prefix):
                    return True
                    
        return False
        
    def can_call_tool(self, user_role: str, tool_name: str) -> bool:
        """Check if user can call a specific tool"""
        return self.check_permission(user_role, "tools", tool_name)
        
    def can_access_resource(self, user_role: str, resource_uri: str) -> bool:
        """Check if user can access a specific resource"""
        return self.check_permission(user_role, "resources", resource_uri)
        
    def can_use_method(self, user_role: str, method: str) -> bool:
        """Check if user can use a specific method"""
        return self.check_permission(user_role, "methods", method)
        
    def create_api_key(self, name: str, role: str = "user", 
                      expires: Optional[datetime] = None) -> str:
        """Create a new API key"""
        import secrets
        
        api_key = f"mcp_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "name": name,
            "role": role,
            "created": datetime.utcnow().isoformat(),
            "expires": expires.isoformat() if expires else None,
            "active": True
        }
        
        logger.info(f"Created API key for {name} with role {role}")
        return api_key
        
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            logger.info(f"Revoked API key: {api_key}")
            return True
        return False
        
    def list_api_keys(self) -> List[Dict]:
        """List all API keys (without the actual key values)"""
        return [
            {
                "key_hash": hashlib.sha256(key.encode()).hexdigest()[:16],
                "name": info["name"],
                "role": info["role"],
                "created": info["created"],
                "expires": info["expires"],
                "active": info["active"]
            }
            for key, info in self.api_keys.items()
        ]
        
    def get_user_stats(self, user_name: str) -> Dict:
        """Get usage statistics for a user"""
        # This would typically query from database
        # For demo purposes, returning mock data
        return {
            "requests_today": 45,
            "requests_this_month": 1250,
            "tools_used": ["generate_video_script", "create_video"],
            "last_active": datetime.utcnow().isoformat()
        }


class MCPRateLimiter:
    """Rate limiting for MCP connections"""
    
    def __init__(self):
        self.limits: Dict[str, Dict] = {}
        self.default_limits = {
            "admin": {"requests_per_minute": 1000, "requests_per_hour": 10000},
            "user": {"requests_per_minute": 100, "requests_per_hour": 1000},
            "viewer": {"requests_per_minute": 20, "requests_per_hour": 200}
        }
        
    def check_rate_limit(self, user_id: str, user_role: str) -> bool:
        """Check if user is within rate limits"""
        now = time.time()
        
        if user_id not in self.limits:
            self.limits[user_id] = {
                "requests": [],
                "role": user_role
            }
            
        user_limits = self.limits[user_id]
        role_limits = self.default_limits.get(user_role, self.default_limits["user"])
        
        # Clean old requests
        minute_ago = now - 60
        hour_ago = now - 3600
        
        user_limits["requests"] = [
            req_time for req_time in user_limits["requests"]
            if req_time > hour_ago
        ]
        
        # Check limits
        recent_minute = [t for t in user_limits["requests"] if t > minute_ago]
        recent_hour = user_limits["requests"]
        
        if len(recent_minute) >= role_limits["requests_per_minute"]:
            return False
            
        if len(recent_hour) >= role_limits["requests_per_hour"]:
            return False
            
        # Record this request
        user_limits["requests"].append(now)
        return True