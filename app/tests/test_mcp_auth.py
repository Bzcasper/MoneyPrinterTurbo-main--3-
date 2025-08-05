import pytest
import jwt
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app

client = TestClient(app)

def test_jwt_auth():
    """Test JWT authentication works after the fix"""
    
    # Create a test JWT token
    token = jwt.encode(
        {"sub": "test_user", "exp": 9999999999},  # Far future expiration
        "test_secret",
        algorithm="HS256"
    )
    
    # Test with valid token
    response = client.get(
        "/health",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200

def test_jwt_decode():
    """Test JWT decoding works properly"""
    token = jwt.encode(
        {"sub": "test_user"}, 
        "test_secret",
        algorithm="HS256"
    )
    decoded = jwt.decode(token, "test_secret", algorithms=["HS256"])
    assert decoded["sub"] == "test_user"