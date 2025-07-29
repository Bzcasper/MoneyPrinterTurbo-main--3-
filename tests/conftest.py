"""
Global pytest configuration and fixtures for microservices testing

This module provides shared fixtures, configuration, and utilities
for comprehensive testing across the MoneyPrinterTurbo ecosystem.

Author: QA Testing Strategist
Version: 1.0.0
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator, Dict, Any
from unittest.mock import AsyncMock, MagicMock
import tempfile
import shutil
from pathlib import Path
import json
import os

# Third-party imports for testing infrastructure
from fastapi.testclient import TestClient
from httpx import AsyncClient
import fakeredis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Local imports
from app.main import app
from app.database.models import Base
from app.services.state import StateManager
from app.services.task import TaskManager
from app.mcp.server import MCPServer


# ==========================================
# PYTEST CONFIGURATION
# ==========================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "chaos: Chaos engineering tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "external: Tests requiring external services")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their location"""
    for item in items:
        # Auto-mark based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "chaos" in str(item.fspath):
            item.add_marker(pytest.mark.chaos)


# ==========================================
# EVENT LOOP FIXTURES
# ==========================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ==========================================
# DATABASE FIXTURES
# ==========================================

@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Provide test database URL"""
    return "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_engine(test_database_url):
    """Create test database engine"""
    engine = create_engine(
        test_database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def test_db_session(test_engine):
    """Create test database session"""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ==========================================
# REDIS FIXTURES
# ==========================================

@pytest.fixture
def mock_redis():
    """Provide fake Redis instance for testing"""
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    yield fake_redis
    fake_redis.flushall()


@pytest.fixture
async def redis_state_manager(mock_redis):
    """Create Redis-based state manager for testing"""
    from app.services.state import RedisStateManager
    manager = RedisStateManager(redis_client=mock_redis)
    yield manager
    await manager.cleanup()


# ==========================================
# HTTP CLIENT FIXTURES
# ==========================================

@pytest.fixture
def test_client() -> TestClient:
    """Provide FastAPI test client"""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Provide async HTTP client for testing"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# ==========================================
# MOCK SERVICE FIXTURES
# ==========================================

@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing"""
    mock = AsyncMock()
    mock.generate_script.return_value = {
        "title": "Test Video Title",
        "script": "This is a test video script for testing purposes.",
        "terms": ["test", "video", "script"]
    }
    return mock


@pytest.fixture
def mock_tts_service():
    """Mock TTS service for testing"""
    mock = AsyncMock()
    mock.synthesize.return_value = "/tmp/test_audio.mp3"
    mock.get_voices.return_value = ["voice1", "voice2"]
    return mock


@pytest.fixture
def mock_material_service():
    """Mock material service for testing"""
    mock = AsyncMock()
    mock.fetch_materials.return_value = [
        {"path": "/tmp/video1.mp4", "type": "video"},
        {"path": "/tmp/video2.mp4", "type": "video"}
    ]
    return mock


@pytest.fixture
def mock_video_service():
    """Mock video processing service"""
    mock = AsyncMock()
    mock.process_video.return_value = {
        "output_path": "/tmp/test_output.mp4",
        "duration": 30.0,
        "success": True
    }
    return mock


# ==========================================
# FILE SYSTEM FIXTURES
# ==========================================

@pytest.fixture
def temp_directory():
    """Provide temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_video_files(temp_directory):
    """Create sample video files for testing"""
    video_files = []
    for i in range(3):
        video_path = temp_directory / f"sample_video_{i}.mp4"
        # Create empty file for testing (real video processing would need actual files)
        video_path.touch()
        video_files.append(str(video_path))
    return video_files


@pytest.fixture
def sample_audio_file(temp_directory):
    """Create sample audio file for testing"""
    audio_path = temp_directory / "sample_audio.mp3"
    audio_path.touch()
    return str(audio_path)


# ==========================================
# CONFIGURATION FIXTURES
# ==========================================

@pytest.fixture
def test_config():
    """Provide test configuration"""
    return {
        "environment": "test",
        "debug": True,
        "database_url": "sqlite:///:memory:",
        "redis_url": "redis://localhost:6379/1",
        "max_workers": 2,
        "video_output_dir": "/tmp/test_videos",
        "audio_output_dir": "/tmp/test_audio",
        "log_level": "DEBUG"
    }


@pytest.fixture
def test_environment_variables(test_config):
    """Set test environment variables"""
    original_env = os.environ.copy()
    
    # Set test environment variables
    for key, value in test_config.items():
        os.environ[key.upper()] = str(value)
    
    yield test_config
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# ==========================================
# TASK MANAGEMENT FIXTURES
# ==========================================

@pytest.fixture
async def task_manager(mock_redis):
    """Create task manager for testing"""
    manager = TaskManager(redis_client=mock_redis)
    yield manager
    await manager.cleanup()


# ==========================================
# MCP SERVER FIXTURES
# ==========================================

@pytest.fixture
async def mcp_server():
    """Create MCP server for testing"""
    server = MCPServer()
    await server.initialize()
    yield server
    await server.shutdown()


# ==========================================
# DATA FACTORY FIXTURES
# ==========================================

@pytest.fixture
def user_factory():
    """Factory for creating test users"""
    def create_user(**kwargs):
        defaults = {
            "id": "test-user-123",
            "email": "test@example.com",
            "name": "Test User",
            "role": "user",
            "active": True
        }
        defaults.update(kwargs)
        return defaults
    return create_user


@pytest.fixture
def project_factory():
    """Factory for creating test projects"""
    def create_project(**kwargs):
        defaults = {
            "id": "test-project-123",
            "name": "Test Project",
            "description": "A test project for unit testing",
            "user_id": "test-user-123",
            "status": "active",
            "settings": {
                "video_format": "mp4",
                "quality": "high",
                "duration": 30
            }
        }
        defaults.update(kwargs)
        return defaults
    return create_project


@pytest.fixture
def video_params_factory():
    """Factory for creating video processing parameters"""
    def create_video_params(**kwargs):
        defaults = {
            "clips": ["/tmp/video1.mp4", "/tmp/video2.mp4"],
            "dimensions": {"width": 1920, "height": 1080},
            "quality": {"bitrate": "2M", "fps": 30},
            "output_path": "/tmp/output.mp4",
            "audio_path": "/tmp/audio.mp3",
            "subtitle_path": "/tmp/subtitles.srt",
            "output_format": "mp4",
            "concat_mode": "progressive"
        }
        defaults.update(kwargs)
        return defaults
    return create_video_params


# ==========================================
# ERROR SIMULATION FIXTURES
# ==========================================

@pytest.fixture
def network_error_simulator():
    """Simulate network errors for chaos testing"""
    class NetworkErrorSimulator:
        def __init__(self):
            self.enabled = False
            self.error_rate = 0.0
        
        def enable(self, rate=0.5):
            self.enabled = True
            self.error_rate = rate
        
        def disable(self):
            self.enabled = False
            self.error_rate = 0.0
        
        def should_fail(self):
            import random
            return self.enabled and random.random() < self.error_rate
    
    return NetworkErrorSimulator()


# ==========================================
# PERFORMANCE MONITORING FIXTURES
# ==========================================

@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests"""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            self.start_time = None
        
        def start(self, test_name):
            import time
            self.start_time = time.time()
            self.metrics[test_name] = {"start": self.start_time}
        
        def stop(self, test_name):
            import time
            if test_name in self.metrics:
                self.metrics[test_name]["end"] = time.time()
                self.metrics[test_name]["duration"] = (
                    self.metrics[test_name]["end"] - 
                    self.metrics[test_name]["start"]
                )
        
        def get_metrics(self):
            return self.metrics
    
    return PerformanceMonitor()


# ==========================================
# CLEANUP FIXTURES
# ==========================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Automatic cleanup after each test"""
    yield
    # Cleanup logic runs after each test
    import gc
    gc.collect()


# ==========================================
# MARKER UTILITIES
# ==========================================

# Utility functions for test markers
def requires_external_service(service_name):
    """Mark test as requiring external service"""
    return pytest.mark.external(reason=f"Requires {service_name} service")


def slow_test(reason="Slow running test"):
    """Mark test as slow"""
    return pytest.mark.slow(reason=reason)


def performance_test(threshold_ms=1000):
    """Mark test as performance test with threshold"""
    return pytest.mark.performance(threshold=threshold_ms)