# MoneyPrinterTurbo Security Guidelines

## Overview

This document provides comprehensive security guidelines for the MoneyPrinterTurbo application, covering secure development practices, deployment considerations, and operational security.

## Table of Contents

1. [Security Architecture](#security-architecture)
2. [Authentication & Authorization](#authentication--authorization)
3. [Input Validation & Sanitization](#input-validation--sanitization)
4. [Configuration Security](#configuration-security)
5. [Audit Logging](#audit-logging)
6. [Deployment Security](#deployment-security)
7. [Incident Response](#incident-response)
8. [Compliance](#compliance)

## Security Architecture

### Defense in Depth

Our security architecture implements multiple layers of protection:

- **Application Layer**: Input validation, authentication, authorization
- **Infrastructure Layer**: Network security, access controls, monitoring
- **Data Layer**: Encryption at rest and in transit, backup security
- **Operational Layer**: Audit logging, incident response, security monitoring

### Security Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Security Framework                     │
├─────────────────────────────────────────────────────────────┤
│ app/security/config_manager.py    - Secure Configuration   │
│ app/security/auth_middleware.py   - Authentication/AuthZ   │
│ app/security/input_validator.py   - Input Validation       │
│ app/security/audit_logger.py      - Security Logging       │
│ app/middleware/rate_limiter.py    - Rate Limiting          │
│ app/middleware/supabase_middleware.py - DB Security        │
└─────────────────────────────────────────────────────────────┘
```

## Authentication & Authorization

### JWT Token Management

**Secure Token Generation:**
```python
from app.security.auth_middleware import jwt_manager

# Generate tokens with proper expiration
tokens = jwt_manager.generate_tokens({
    "id": user_id,
    "email": user_email,
    "role": user_role
})
```

**Token Validation:**
```python
# Validate tokens with blacklist checking
payload = jwt_manager.verify_token(token, "access")
if payload:
    # Token is valid
    user_id = payload["sub"]
```

### Role-Based Access Control (RBAC)

**Role Hierarchy:**
- `admin`: Full system access
- `user`: Standard operations (video generation, script creation)
- `viewer`: Read-only access
- `anonymous`: Public endpoints only

**Permission Checking:**
```python
from app.security.auth_middleware import get_current_user, get_admin_user

# Require authentication
@app.route("/api/protected")
async def protected_endpoint(request: Request):
    user = await get_current_user(request)
    # Process request

# Require admin access
@app.route("/api/admin")
async def admin_endpoint(request: Request):
    admin = await get_admin_user(request)
    # Admin operations
```

### Session Management

**Secure Session Configuration:**
- Session timeout: 30 minutes default
- Maximum concurrent sessions: 5 per user
- IP binding (optional for high security)
- Secure session storage with encryption

## Input Validation & Sanitization

### Validation Schema

**Define validation schemas for all inputs:**
```python
from app.security.input_validator import data_validator

SCRIPT_SCHEMA = {
    "title": {
        "required": True,
        "type": "string",
        "min_length": 1,
        "max_length": 200,
        "sanitize": True
    },
    "content": {
        "required": True,
        "type": "string",
        "max_length": 5000,
        "sanitize": True
    },
    "duration": {
        "required": False,
        "type": "integer",
        "min_value": 1,
        "max_value": 600
    }
}

# Validate input
validated_data = data_validator.validate_json_schema(request_data, SCRIPT_SCHEMA)
```

### XSS Prevention

**HTML Sanitization:**
```python
from app.security.input_validator import input_sanitizer

# Sanitize HTML content
safe_content = input_sanitizer.sanitize_html(user_input, strict=True)

# For rich text (allow safe tags)
safe_html = input_sanitizer.sanitize_html(user_input, strict=False)
```

### SQL Injection Prevention

**Always use parameterized queries:**
```python
# SECURE - Parameterized query
cursor.execute("SELECT * FROM videos WHERE user_id = %s", (user_id,))

# INSECURE - String concatenation (NEVER DO THIS)
# cursor.execute(f"SELECT * FROM videos WHERE user_id = {user_id}")
```

### File Upload Security

**Secure file handling:**
```python
from app.security.input_validator import input_sanitizer

# Validate file path
safe_path = input_sanitizer.sanitize_file_path(file_path)

# Validate file type
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}
if not any(safe_path.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
    raise ValidationError("file", "Invalid file type")
```

## Configuration Security

### Environment Variables

**Required security configurations:**
```bash
# JWT Configuration
MCP_JWT_SECRET=your-strong-secret-key-here-min-32-chars
JWT_ACCESS_EXPIRE_MINUTES=15
JWT_REFRESH_EXPIRE_DAYS=7

# Database Security
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
DB_SSL_MODE=require

# API Keys (external services)
OPENAI_API_KEY=sk-your-openai-key
PEXELS_API_KEY=your-pexels-key

# Security Settings
RATE_LIMIT_ENABLED=true
AUDIT_LOGGING_ENABLED=true
ENCRYPTION_ENABLED=true

# HashiCorp Vault (production)
VAULT_URL=https://vault.example.com
VAULT_TOKEN=your-vault-token
```

### Secure Configuration Loading

**Use secure configuration manager:**
```python
from app.security.config_manager import get_secure_config

# Load configurations securely
api_key = get_secure_config("openai_api_key", required=True)
jwt_secret = get_secure_config("mcp_jwt_secret", required=True)

# Validate configuration
errors = validate_security_config()
if errors:
    logger.error(f"Security configuration errors: {errors}")
```

## Audit Logging

### Security Event Logging

**Log all security-relevant events:**
```python
from app.security.audit_logger import (
    log_auth_success, log_auth_failure, 
    log_injection_attempt, log_data_access
)

# Authentication events
log_auth_success(user_id, ip_address, user_agent)
log_auth_failure(user_id, ip_address, user_agent, "invalid_password")

# Data access events
log_data_access(user_id, "user_videos", "read", ip_address, sensitive=False)

# Security violations
log_injection_attempt(ip_address, user_agent, "sql_injection", payload)
```

### Log Analysis

**Monitor for security patterns:**
- Multiple failed login attempts (brute force)
- Privilege escalation attempts
- Unusual data access patterns
- Injection attack attempts
- Rate limit violations

## Deployment Security

### HTTPS Configuration

**Enforce HTTPS in production:**
```python
# In middleware
if not request.url.scheme == "https" and not is_development():
    return RedirectResponse(
        url=request.url.replace(scheme="https"),
        status_code=301
    )
```

### Security Headers

**Essential security headers:**
```python
response.headers.update({
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin"
})
```

### Database Security

**PostgreSQL security settings:**
```sql
-- Enable SSL
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'

-- Connection security
listen_addresses = 'localhost'
port = 5432

-- Authentication
password_encryption = scram-sha-256
```

### Docker Security

**Secure Docker configuration:**
```dockerfile
# Use non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Minimize attack surface
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Security scanning
RUN pip install --no-cache-dir safety
RUN safety check
```

## Incident Response

### Security Incident Classification

**Severity Levels:**
- **Critical**: Active breach, data exfiltration, system compromise
- **High**: Failed security controls, privilege escalation
- **Medium**: Suspicious activity, policy violations
- **Low**: Security configuration issues, minor vulnerabilities

### Response Procedures

**Immediate Actions:**
1. Identify and contain the threat
2. Assess impact and scope
3. Notify stakeholders
4. Preserve evidence
5. Implement remediation
6. Document lessons learned

**Emergency Contacts:**
```python
SECURITY_CONTACTS = {
    "security_team": "security@company.com",
    "incident_response": "+1-555-SECURITY",
    "legal": "legal@company.com"
}
```

### Automated Response

**Implement automated threat response:**
```python
from app.security.audit_logger import SecurityEventType, SecuritySeverity

def handle_security_event(event):
    if event.severity == SecuritySeverity.CRITICAL:
        # Immediate action
        if event.event_type == SecurityEventType.INJECTION_ATTEMPT:
            block_ip_address(event.ip_address)
            send_alert(f"Critical injection attempt from {event.ip_address}")
        
        elif event.event_type == SecurityEventType.BRUTE_FORCE_ATTEMPT:
            lock_user_account(event.user_id)
            send_alert(f"Brute force attack on user {event.user_id}")
```

## Compliance

### GDPR Compliance

**Data Protection Requirements:**
- Explicit consent for data processing
- Right to data portability
- Right to erasure ("right to be forgotten")
- Data protection by design and default
- Regular security assessments

**Implementation:**
```python
# Anonymize PII in logs
if compliance_config["anonymize_pii"]:
    user_id_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
    log_entry["user_id"] = user_id_hash

# Data retention
if compliance_config["gdpr_enabled"]:
    retention_days = compliance_config["retention_days"]
    cleanup_old_data(retention_days)
```

### SOX Compliance

**Financial Controls:**
- Immutable audit logs
- Segregation of duties
- Regular access reviews
- Change management controls

### Security Standards

**Follow industry standards:**
- OWASP Top 10
- NIST Cybersecurity Framework
- ISO 27001/27002
- PCI DSS (if processing payments)

## Security Testing

### Automated Security Testing

**Integration with CI/CD:**
```bash
# Dependency scanning
pip install safety
safety check

# SAST scanning
pip install bandit
bandit -r app/

# Secret scanning
pip install detect-secrets
detect-secrets scan
```

### Penetration Testing

**Regular security assessments:**
- Quarterly internal assessments
- Annual external penetration testing
- Continuous vulnerability scanning
- Security code reviews

### Security Metrics

**Track security KPIs:**
- Mean time to detect (MTTD)
- Mean time to respond (MTTR)
- Number of security incidents
- Vulnerability remediation time
- Security training completion rates

## Emergency Procedures

### Security Breach Response

**Immediate Actions (First 30 minutes):**
1. Disconnect affected systems from network
2. Preserve system state for forensics
3. Notify security team and management
4. Begin evidence collection
5. Implement emergency communication plan

### Recovery Procedures

**System Recovery Steps:**
1. Assess damage and impact
2. Restore from clean backups
3. Apply security patches
4. Strengthen security controls
5. Monitor for continued threats
6. Conduct post-incident review

---

## Security Contacts

- **Security Team**: security@moneyprinterturbo.com
- **Emergency Response**: +1-555-SECURITY
- **Vulnerability Reports**: security-reports@moneyprinterturbo.com

---

*Last Updated: 2024-01-29*
*Version: 1.0*
*Classification: Internal Use*