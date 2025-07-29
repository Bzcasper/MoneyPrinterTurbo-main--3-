"""
Pytest Configuration for MoneyPrinterTurbo Test Suite
QA Validator - Centralized test configuration and fixtures
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from typing import Dict, Any, Generator
from unittest.mock import Mock, patch
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration constants
TEST_CONFIG = {
    "performance_targets": {
        "tts_generation_max_time": 30.0,
        "video_generation_max_time": 120.0,
        "api_response_max_time": 5.0,
        "concurrent_operations_max_time": 60.0
    },
    "quality_targets": {
        "min_success_rate": 0.95,
        "min_test_coverage": 0.90,
        "max_error_rate": 0.05
    },
    "security_config": {
        "max_input_size": 50000,
        "allowed_file_extensions": [".wav", ".mp3", ".mp4", ".avi"],
        "rate_limit_per_minute": 60
    },
    "test_data": {
        "sample_texts": [
            "This is a test for TTS generation.",
            "Complex sentence with punctuation, numbers 123, and symbols!",
            "Long text for performance testing. " * 50
        ],
        "sample_topics": [
            "Introduction to Python Programming",
            "Advanced Machine Learning Concepts",
            "Web Development Best Practices"
        ]
    }
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_config():
    """Provide test configuration to tests"""
    return TEST_CONFIG

@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp(prefix="mpt_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    with patch('app.controllers.tts_controller.openai_client') as mock_client:
        # Configure default mock responses
        mock_response = Mock()
        mock_response.content = b"mock_audio_data" * 1000
        mock_client.audio.speech.create.return_value = mock_response
        
        # Mock embeddings
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        yield mock_client

@pytest.fixture
def sample_tts_request():
    """Sample TTS request for testing"""
    from models.schemas import TTSRequest
    return TTSRequest(
        text="This is a sample text for testing TTS functionality.",
        voice="alloy",
        model="tts-1",
        speed=1.0
    )

@pytest.fixture
def sample_video_request():
    """Sample video generation request for testing"""
    from models.schemas import VideoGenerationRequest
    return VideoGenerationRequest(
        topic="Sample Video Topic for Testing",
        duration=60,
        voice="alloy",
        background_music=False,
        subtitle_style="simple"
    )

@pytest.fixture
def sample_character_request():
    """Sample character creation request for testing"""
    from models.schemas import CharacterRequest
    return CharacterRequest(
        character_type="narrator",
        voice_style="professional",
        personality_traits=["authoritative", "clear"],
        speaking_pace="medium"
    )

@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture"""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
        
        def start_timer(self, operation: str):
            import time
            self.metrics[operation] = {"start": time.time()}
        
        def end_timer(self, operation: str):
            import time
            if operation in self.metrics:
                self.metrics[operation]["end"] = time.time()
                self.metrics[operation]["duration"] = (
                    self.metrics[operation]["end"] - self.metrics[operation]["start"]
                )
        
        def get_duration(self, operation: str) -> float:
            return self.metrics.get(operation, {}).get("duration", 0.0)
        
        def assert_performance_target(self, operation: str, max_duration: float):
            actual_duration = self.get_duration(operation)
            assert actual_duration < max_duration, (
                f"{operation} took {actual_duration:.2f}s, exceeds {max_duration}s target"
            )
    
    return PerformanceMonitor()

@pytest.fixture
def security_validator():
    """Security validation utilities"""
    class SecurityValidator:
        @staticmethod
        def validate_input_sanitization(input_text: str, output_text: str):
            """Validate that input has been properly sanitized"""
            dangerous_patterns = [
                "<script>", "javascript:", "data:", "vbscript:",
                "DROP TABLE", "DELETE FROM", "INSERT INTO",
                "rm -rf", "system(", "exec(",
                "../", "..\\", "\x00"
            ]
            
            for pattern in dangerous_patterns:
                assert pattern not in output_text, f"Dangerous pattern '{pattern}' not sanitized"
        
        @staticmethod
        def validate_file_path_security(file_path: str):
            """Validate file path security"""
            assert not file_path.startswith("../"), "Path traversal attempt detected"
            assert not file_path.startswith("/etc/"), "System file access attempt detected"
            assert "\x00" not in file_path, "Null byte injection detected"
        
        @staticmethod
        def validate_api_response_headers(headers: Dict[str, str]):
            """Validate security headers in API responses"""
            required_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "X-XSS-Protection"
            ]
            
            for header in required_headers:
                assert header in headers, f"Missing security header: {header}"
    
    return SecurityValidator()

@pytest.fixture
def quality_metrics():
    """Quality metrics collector"""
    class QualityMetrics:
        def __init__(self):
            self.test_results = []
            self.error_count = 0
            self.success_count = 0
        
        def record_test_result(self, test_name: str, success: bool, duration: float = 0.0):
            """Record test result"""
            self.test_results.append({
                "test_name": test_name,
                "success": success,
                "duration": duration,
                "timestamp": __import__("time").time()
            })
            
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
        
        def get_success_rate(self) -> float:
            """Calculate success rate"""
            total_tests = len(self.test_results)
            if total_tests == 0:
                return 0.0
            return self.success_count / total_tests
        
        def get_error_rate(self) -> float:
            """Calculate error rate"""
            total_tests = len(self.test_results)
            if total_tests == 0:
                return 0.0
            return self.error_count / total_tests
        
        def assert_quality_targets(self):
            """Assert quality targets are met"""
            success_rate = self.get_success_rate()
            error_rate = self.get_error_rate()
            
            assert success_rate >= TEST_CONFIG["quality_targets"]["min_success_rate"], (
                f"Success rate {success_rate:.2%} below target "
                f"{TEST_CONFIG['quality_targets']['min_success_rate']:.2%}"
            )
            
            assert error_rate <= TEST_CONFIG["quality_targets"]["max_error_rate"], (
                f"Error rate {error_rate:.2%} above target "
                f"{TEST_CONFIG['quality_targets']['max_error_rate']:.2%}"
            )
    
    return QualityMetrics()

@pytest.fixture
def mock_database():
    """Mock database for testing"""
    class MockDatabase:
        def __init__(self):
            self.data = {}
        
        async def store(self, key: str, value: Any):
            """Store data in mock database"""
            self.data[key] = value
        
        async def retrieve(self, key: str) -> Any:
            """Retrieve data from mock database"""
            return self.data.get(key)
        
        async def delete(self, key: str):
            """Delete data from mock database"""
            if key in self.data:
                del self.data[key]
        
        async def list_keys(self) -> list:
            """List all keys in mock database"""
            return list(self.data.keys())
        
        def clear(self):
            """Clear all data"""
            self.data.clear()
    
    return MockDatabase()

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment for each test"""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "INFO"
    
    # Create test directories if they don't exist
    test_dirs = ["tests/fixtures", "tests/output", "tests/logs"]
    for test_dir in test_dirs:
        os.makedirs(test_dir, exist_ok=True)
    
    yield
    
    # Cleanup after test
    # Remove test environment variables
    os.environ.pop("TESTING", None)

@pytest.fixture
def integration_test_data():
    """Test data for integration tests"""
    return {
        "valid_requests": [
            {
                "text": "Hello, this is a test.",
                "voice": "alloy",
                "model": "tts-1"
            },
            {
                "text": "Another test with different parameters.",
                "voice": "echo",
                "model": "tts-1-hd",
                "speed": 1.2
            }
        ],
        "invalid_requests": [
            {
                "text": "",  # Empty text
                "voice": "alloy",
                "model": "tts-1"
            },
            {
                "text": "Valid text",
                "voice": "invalid_voice",  # Invalid voice
                "model": "tts-1"
            }
        ],
        "malicious_requests": [
            {
                "text": "<script>alert('xss')</script>",
                "voice": "alloy",
                "model": "tts-1"
            },
            {
                "text": "'; DROP TABLE users; --",
                "voice": "alloy",
                "model": "tts-1"
            }
        ]
    }

# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add slow marker to tests that might be slow
    for item in items:
        if "performance" in item.keywords or "e2e" in item.keywords:
            item.add_marker(pytest.mark.slow)

# Test reporting hooks
def pytest_runtest_makereport(item, call):
    """Create test report"""
    if call.when == "call":
        # Log test results
        outcome = "PASSED" if call.excinfo is None else "FAILED"
        duration = call.stop - call.start
        
        logger.info(f"Test {item.name}: {outcome} in {duration:.2f}s")
        
        # Record performance data for performance tests
        if "performance" in item.keywords and hasattr(item, "performance_data"):
            logger.info(f"Performance data: {item.performance_data}")

@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Setup for entire test session"""
    logger.info("Starting MoneyPrinterTurbo test suite")
    logger.info(f"Test configuration: {TEST_CONFIG}")
    
    yield
    
    logger.info("Test suite completed")

# Custom assertions
def assert_response_time(duration: float, max_time: float, operation: str):
    """Custom assertion for response time"""
    assert duration < max_time, (
        f"{operation} took {duration:.2f}s, exceeds {max_time}s target"
    )

def assert_quality_score(score: float, min_score: float, metric: str):
    """Custom assertion for quality scores"""
    assert score >= min_score, (
        f"{metric} score {score:.2f} below minimum {min_score:.2f}"
    )

# Export custom assertions
pytest.assert_response_time = assert_response_time
pytest.assert_quality_score = assert_quality_score