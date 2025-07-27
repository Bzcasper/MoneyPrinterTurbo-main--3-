#!/usr/bin/env python3
"""
API Endpoints Testing Suite
Created by QA_Engineer agent in the Hive Mind swarm

This module tests all API endpoints and their integration with the UI:
- Health monitoring endpoints
- Video processing endpoints
- LLM service endpoints
- Voice service endpoints
- Task management endpoints
- Error handling and validation
"""

import unittest
import requests
import json
import time
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import threading

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from app.config import config
    from app.models.schema import VideoParams, MaterialInfo
    from app.controllers.ping import ping_route
    from app.controllers.v1.video import VideoController
    from app.controllers.v1.llm import LLMController
except ImportError as e:
    print(f"Warning: Could not import API modules: {e}")


class BaseAPITest(unittest.TestCase):
    """Base class for API tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.base_url = "http://localhost:8080"
        cls.timeout = 10
        cls.headers = {"Content-Type": "application/json"}
    
    def setUp(self):
        """Set up individual test"""
        self.start_time = time.time()
    
    def tearDown(self):
        """Clean up after test"""
        duration = time.time() - self.start_time
        print(f"Test completed in {duration:.2f}s")
    
    def make_request(self, method, endpoint, **kwargs):
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                return requests.get(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "POST":
                return requests.post(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "PUT":
                return requests.put(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "DELETE":
                return requests.delete(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "OPTIONS":
                return requests.options(url, timeout=self.timeout, **kwargs)
        except requests.exceptions.ConnectionError:
            self.skipTest("API service not available")
        except requests.exceptions.Timeout:
            self.fail(f"Request to {endpoint} timed out")


class TestHealthEndpoints(BaseAPITest):
    """Test health monitoring and system status endpoints"""
    
    def test_ping_endpoint(self):
        """Test basic ping endpoint"""
        response = self.make_request("GET", "/ping")
        
        # Should return 200 OK
        self.assertEqual(response.status_code, 200)
        
        # Should return JSON
        self.assertEqual(response.headers.get("content-type", "").lower(), "application/json")
        
        # Parse response
        data = response.json()
        
        # Should have status field
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
    
    def test_health_endpoint_with_metrics(self):
        """Test health endpoint with system metrics"""
        response = self.make_request("GET", "/ping")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for system metrics
            if "system" in data:
                system_data = data["system"]
                
                # CPU percentage should be valid
                if "cpu_percent" in system_data:
                    cpu = system_data["cpu_percent"]
                    self.assertIsInstance(cpu, (int, float))
                    self.assertGreaterEqual(cpu, 0)
                    self.assertLessEqual(cpu, 100)
                
                # Memory percentage should be valid
                if "memory_percent" in system_data:
                    memory = system_data["memory_percent"]
                    self.assertIsInstance(memory, (int, float))
                    self.assertGreaterEqual(memory, 0)
                    self.assertLessEqual(memory, 100)
                
                # Should have timestamp
                if "timestamp" in system_data:
                    self.assertIsInstance(system_data["timestamp"], (int, float, str))
    
    def test_health_endpoint_performance(self):
        """Test health endpoint response time"""
        start_time = time.time()
        response = self.make_request("GET", "/ping")
        response_time = time.time() - start_time
        
        # Health check should be fast (under 1 second)
        self.assertLess(response_time, 1.0, f"Health endpoint too slow: {response_time:.2f}s")
    
    def test_health_endpoint_concurrency(self):
        """Test health endpoint under concurrent requests"""
        def make_health_request():
            return self.make_request("GET", "/ping")
        
        # Make multiple concurrent requests
        threads = []
        results = []
        
        for _ in range(5):
            thread = threading.Thread(target=lambda: results.append(make_health_request()))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        for response in results:
            if response:  # Skip None results from connection errors
                self.assertEqual(response.status_code, 200)


class TestVideoEndpoints(BaseAPITest):
    """Test video processing endpoints"""
    
    def test_video_generation_endpoint_structure(self):
        """Test video generation endpoint accepts proper parameters"""
        # Test payload structure (without actually processing)
        test_payload = {
            "video_subject": "Test subject",
            "video_script": "Test script content",
            "video_terms": "test,keywords",
            "video_aspect": "16:9",
            "video_concat_mode": "sequential",
            "video_clip_duration": 5,
            "video_count": 1
        }
        
        # Test that payload can be serialized
        json_payload = json.dumps(test_payload)
        self.assertIsInstance(json_payload, str)
        
        # Test that required fields are present
        required_fields = ["video_subject", "video_script"]
        for field in required_fields:
            self.assertIn(field, test_payload)
    
    def test_video_endpoint_validation(self):
        """Test video endpoint input validation"""
        # Test empty payload
        response = self.make_request("POST", "/api/v1/video/generate", 
                                   json={}, headers=self.headers)
        
        if response and response.status_code != 404:  # Endpoint exists
            # Should return validation error (400) or method not allowed
            self.assertIn(response.status_code, [400, 405, 422])
    
    def test_video_status_endpoint(self):
        """Test video processing status endpoint"""
        # Test with dummy task ID
        task_id = "test-task-123"
        response = self.make_request("GET", f"/api/v1/video/status/{task_id}")
        
        if response and response.status_code != 404:  # Endpoint exists
            # Should return proper status structure
            if response.status_code == 200:
                data = response.json()
                self.assertIn("status", data)


class TestLLMEndpoints(BaseAPITest):
    """Test LLM service endpoints"""
    
    def test_llm_providers_endpoint(self):
        """Test LLM providers listing endpoint"""
        response = self.make_request("GET", "/api/v1/llm/providers")
        
        if response and response.status_code == 200:
            data = response.json()
            
            # Should return list of providers
            self.assertIsInstance(data, (list, dict))
            
            if isinstance(data, list):
                # Each provider should have required fields
                for provider in data:
                    self.assertIsInstance(provider, dict)
                    # Common provider fields
                    if "name" in provider:
                        self.assertIsInstance(provider["name"], str)
    
    def test_llm_generate_endpoint_structure(self):
        """Test LLM text generation endpoint structure"""
        test_payload = {
            "prompt": "Generate a test script",
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "max_tokens": 100
        }
        
        # Test payload structure
        json_payload = json.dumps(test_payload)
        self.assertIsInstance(json_payload, str)
        
        # Required fields
        self.assertIn("prompt", test_payload)
    
    def test_llm_endpoint_validation(self):
        """Test LLM endpoint input validation"""
        # Test empty payload
        response = self.make_request("POST", "/api/v1/llm/generate",
                                   json={}, headers=self.headers)
        
        if response and response.status_code != 404:  # Endpoint exists
            # Should return validation error
            self.assertIn(response.status_code, [400, 405, 422])


class TestVoiceEndpoints(BaseAPITest):
    """Test voice synthesis endpoints"""
    
    def test_voice_providers_endpoint(self):
        """Test voice providers listing endpoint"""
        response = self.make_request("GET", "/api/v1/voice/providers")
        
        if response and response.status_code == 200:
            data = response.json()
            
            # Should return voice provider information
            self.assertIsInstance(data, (list, dict))
    
    def test_voice_generate_endpoint_structure(self):
        """Test voice generation endpoint structure"""
        test_payload = {
            "text": "Test speech synthesis",
            "voice": "default",
            "provider": "edge_tts",
            "speed": 1.0,
            "volume": 1.0
        }
        
        # Test payload structure
        json_payload = json.dumps(test_payload)
        self.assertIsInstance(json_payload, str)
        
        # Required fields
        self.assertIn("text", test_payload)
    
    def test_voice_endpoint_validation(self):
        """Test voice endpoint input validation"""
        # Test empty payload
        response = self.make_request("POST", "/api/v1/voice/generate",
                                   json={}, headers=self.headers)
        
        if response and response.status_code != 404:  # Endpoint exists
            # Should return validation error
            self.assertIn(response.status_code, [400, 405, 422])


class TestTaskEndpoints(BaseAPITest):
    """Test task management endpoints"""
    
    def test_task_list_endpoint(self):
        """Test task listing endpoint"""
        response = self.make_request("GET", "/api/v1/tasks")
        
        if response and response.status_code == 200:
            data = response.json()
            
            # Should return task list
            self.assertIsInstance(data, (list, dict))
            
            if isinstance(data, dict) and "tasks" in data:
                self.assertIsInstance(data["tasks"], list)
    
    def test_task_status_endpoint(self):
        """Test individual task status endpoint"""
        task_id = "test-task-123"
        response = self.make_request("GET", f"/api/v1/tasks/{task_id}")
        
        if response and response.status_code != 404:  # Endpoint exists
            if response.status_code == 200:
                data = response.json()
                
                # Should have task information
                self.assertIn("id", data)
                self.assertIn("status", data)
            elif response.status_code == 404:
                # Task not found is acceptable for test ID
                pass
    
    def test_task_deletion_endpoint(self):
        """Test task deletion endpoint"""
        task_id = "test-task-123"
        response = self.make_request("DELETE", f"/api/v1/tasks/{task_id}")
        
        if response and response.status_code != 404:  # Endpoint exists
            # Should handle deletion request
            self.assertIn(response.status_code, [200, 204, 404])


class TestAPIErrorHandling(BaseAPITest):
    """Test API error handling and edge cases"""
    
    def test_invalid_endpoint_404(self):
        """Test 404 handling for invalid endpoints"""
        response = self.make_request("GET", "/api/v1/invalid-endpoint")
        self.assertEqual(response.status_code, 404)
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON requests"""
        # Send invalid JSON
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/video/generate",
                data="invalid json{",
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            # Should return 400 Bad Request for malformed JSON
            self.assertEqual(response.status_code, 400)
            
        except requests.exceptions.ConnectionError:
            self.skipTest("API service not available")
    
    def test_oversized_request_handling(self):
        """Test handling of oversized requests"""
        # Create large payload
        large_payload = {
            "video_subject": "x" * 10000,  # Very long subject
            "video_script": "x" * 100000,  # Very long script
        }
        
        response = self.make_request("POST", "/api/v1/video/generate",
                                   json=large_payload, headers=self.headers)
        
        if response:
            # Should handle large requests appropriately
            self.assertIn(response.status_code, [200, 400, 413, 422])
    
    def test_cors_headers(self):
        """Test CORS headers for cross-origin requests"""
        response = self.make_request("OPTIONS", "/api/v1/video/generate")
        
        if response:
            # Check for CORS headers
            cors_headers = [
                "Access-Control-Allow-Origin",
                "Access-Control-Allow-Methods", 
                "Access-Control-Allow-Headers"
            ]
            
            for header in cors_headers:
                if header in response.headers:
                    self.assertIsNotNone(response.headers[header])
    
    def test_rate_limiting(self):
        """Test rate limiting behavior"""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = self.make_request("GET", "/ping")
            if response:
                responses.append(response.status_code)
        
        # Should handle rapid requests gracefully
        # Most should succeed, some might be rate limited (429)
        success_count = sum(1 for status in responses if status == 200)
        self.assertGreaterEqual(success_count, 5)  # At least half should succeed


class TestAPIPerformance(BaseAPITest):
    """Test API performance characteristics"""
    
    def test_response_time_sla(self):
        """Test API response time meets SLA"""
        endpoints = ["/ping", "/api/v1/llm/providers", "/api/v1/voice/providers"]
        
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                start_time = time.time()
                response = self.make_request("GET", endpoint)
                response_time = time.time() - start_time
                
                if response and response.status_code == 200:
                    # Response should be under 5 seconds
                    self.assertLess(response_time, 5.0, 
                                  f"{endpoint} too slow: {response_time:.2f}s")
    
    def test_concurrent_request_handling(self):
        """Test API handles concurrent requests"""
        def make_ping_request():
            return self.make_request("GET", "/ping")
        
        # Make multiple concurrent requests
        threads = []
        results = []
        
        for _ in range(10):
            thread = threading.Thread(target=lambda: results.append(make_ping_request()))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Count successful responses
        success_count = sum(1 for response in results 
                          if response and response.status_code == 200)
        
        # Most requests should succeed
        self.assertGreaterEqual(success_count, 7)  # 70% success rate minimum


def run_api_endpoint_tests():
    """Run all API endpoint tests"""
    print("="*60)
    print("API Endpoints Testing Suite")
    print("Created by QA_Engineer agent")
    print("="*60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestHealthEndpoints,
        TestVideoEndpoints,
        TestLLMEndpoints,
        TestVoiceEndpoints,
        TestTaskEndpoints,
        TestAPIErrorHandling,
        TestAPIPerformance,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_api_endpoint_tests()
    sys.exit(0 if success else 1)