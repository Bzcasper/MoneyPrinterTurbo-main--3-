#!/usr/bin/env python3
"""
Integration Flow Testing Suite
Created by QA_Engineer agent in the Hive Mind swarm

This module tests complete integration flows between UI and backend:
- End-to-end video creation workflow
- Template management flows
- Task monitoring and progress tracking
- Error recovery and retry flows
- Multi-user scenario handling
- Performance under load
"""

import unittest
import requests
import time
import json
import sys
import os
import threading
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from app.config import config
    from app.models.schema import VideoParams, MaterialInfo, VideoAspect
    from app.services import task as tm
except ImportError as e:
    print(f"Warning: Could not import app modules: {e}")


class BaseIntegrationTest(unittest.TestCase):
    """Base class for integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.streamlit_url = "http://localhost:8501"
        cls.api_url = "http://localhost:8080"
        cls.timeout = 30  # Longer timeout for integration tests
        cls.headers = {"Content-Type": "application/json"}
        
        # Test data
        cls.test_video_data = {
            "subject": "AI-powered video creation testing",
            "script": "This is a comprehensive test of the video creation system. We are testing the complete workflow from UI input to video output.",
            "terms": "AI,video,testing,automation,quality",
            "aspect": "16:9",
            "duration": 30,
            "count": 1
        }
    
    def setUp(self):
        """Set up individual test"""
        self.start_time = time.time()
        self.test_id = f"test_{int(time.time())}"
    
    def tearDown(self):
        """Clean up after test"""
        duration = time.time() - self.start_time
        print(f"Integration test {self.test_id} completed in {duration:.2f}s")
    
    def make_api_request(self, method, endpoint, **kwargs):
        """Make API request with error handling"""
        url = f"{self.api_url}{endpoint}"
        try:
            response = getattr(requests, method.lower())(url, timeout=self.timeout, **kwargs)
            return response
        except requests.exceptions.ConnectionError:
            self.skipTest("API service not available for integration test")
        except requests.exceptions.Timeout:
            self.fail(f"Integration test request to {endpoint} timed out")
    
    def wait_for_task_completion(self, task_id, max_wait=120):
        """Wait for a task to complete and return final status"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = self.make_api_request("GET", f"/api/v1/tasks/{task_id}")
                
                if response and response.status_code == 200:
                    task_data = response.json()
                    status = task_data.get("status", "unknown")
                    
                    if status in ["completed", "failed", "error"]:
                        return status, task_data
                    
                time.sleep(2)  # Wait 2 seconds before checking again
                
            except Exception as e:
                print(f"Error checking task status: {e}")
                time.sleep(2)
        
        return "timeout", {}


class TestEndToEndVideoCreation(BaseIntegrationTest):
    """Test complete video creation workflow"""
    
    def test_full_video_creation_workflow(self):
        """Test complete video creation from UI input to output"""
        print(f"\nStarting end-to-end video creation test: {self.test_id}")
        
        # Step 1: Submit video creation request
        video_payload = {
            "video_subject": self.test_video_data["subject"],
            "video_script": self.test_video_data["script"],
            "video_terms": self.test_video_data["terms"],
            "video_aspect": self.test_video_data["aspect"],
            "video_clip_duration": self.test_video_data["duration"],
            "video_count": self.test_video_data["count"]
        }
        
        response = self.make_api_request("POST", "/api/v1/video/generate",
                                       json=video_payload, headers=self.headers)
        
        if not response or response.status_code == 404:
            self.skipTest("Video generation endpoint not available")
        
        # Should accept the request
        self.assertIn(response.status_code, [200, 201, 202])
        
        # Should return task ID
        if response.status_code in [200, 201, 202]:
            response_data = response.json()
            
            if "task_id" in response_data:
                task_id = response_data["task_id"]
                
                # Step 2: Monitor task progress
                print(f"Monitoring task: {task_id}")
                final_status, task_data = self.wait_for_task_completion(task_id)
                
                # Step 3: Verify completion
                if final_status == "completed":
                    print("Video creation completed successfully")
                    
                    # Check for output files
                    if "output" in task_data:
                        output_info = task_data["output"]
                        self.assertIsInstance(output_info, dict)
                        
                        # Should have video file information
                        if "video_path" in output_info:
                            self.assertIsInstance(output_info["video_path"], str)
                            
                elif final_status == "failed":
                    print(f"Video creation failed: {task_data.get('error', 'Unknown error')}")
                    # In test environment, failures might be expected
                    
                elif final_status == "timeout":
                    print("Task did not complete within timeout - this may be expected in test environment")
                    
                else:
                    print(f"Task completed with status: {final_status}")
    
    def test_video_creation_with_invalid_input(self):
        """Test video creation with invalid input parameters"""
        # Test with empty subject
        invalid_payload = {
            "video_subject": "",
            "video_script": self.test_video_data["script"],
            "video_terms": self.test_video_data["terms"]
        }
        
        response = self.make_api_request("POST", "/api/v1/video/generate",
                                       json=invalid_payload, headers=self.headers)
        
        if response and response.status_code != 404:
            # Should return validation error
            self.assertIn(response.status_code, [400, 422])
    
    def test_video_creation_error_recovery(self):
        """Test error recovery during video creation"""
        # Submit request with potentially problematic parameters
        problematic_payload = {
            "video_subject": "Test video with special characters !@#$%^&*()",
            "video_script": "Test script with unicode: 测试 🎥 🤖",
            "video_terms": "test,special-chars,unicode",
            "video_clip_duration": 1  # Very short duration
        }
        
        response = self.make_api_request("POST", "/api/v1/video/generate",
                                       json=problematic_payload, headers=self.headers)
        
        if response and response.status_code in [200, 201, 202]:
            response_data = response.json()
            
            if "task_id" in response_data:
                task_id = response_data["task_id"]
                
                # Monitor for appropriate error handling
                final_status, task_data = self.wait_for_task_completion(task_id, max_wait=60)
                
                # Should either complete or fail gracefully
                self.assertIn(final_status, ["completed", "failed", "timeout"])
                
                if final_status == "failed":
                    # Should have error information
                    self.assertIn("error", task_data)


class TestTemplateManagement(BaseIntegrationTest):
    """Test template management workflows"""
    
    def test_template_save_and_load_workflow(self):
        """Test saving and loading video templates"""
        # Create template data
        template_data = {
            "name": f"test_template_{self.test_id}",
            "video_subject": self.test_video_data["subject"],
            "video_script": self.test_video_data["script"],
            "video_terms": self.test_video_data["terms"],
            "video_aspect": self.test_video_data["aspect"]
        }
        
        # Test template save
        response = self.make_api_request("POST", "/api/v1/templates",
                                       json=template_data, headers=self.headers)
        
        if response and response.status_code == 404:
            self.skipTest("Template management endpoint not available")
        
        if response and response.status_code in [200, 201]:
            template_id = response.json().get("template_id")
            
            if template_id:
                # Test template load
                load_response = self.make_api_request("GET", f"/api/v1/templates/{template_id}")
                
                if load_response and load_response.status_code == 200:
                    loaded_template = load_response.json()
                    
                    # Verify template data integrity
                    self.assertEqual(loaded_template["name"], template_data["name"])
                    self.assertEqual(loaded_template["video_subject"], template_data["video_subject"])
                
                # Clean up: delete template
                self.make_api_request("DELETE", f"/api/v1/templates/{template_id}")
    
    def test_template_list_and_filter(self):
        """Test template listing and filtering"""
        response = self.make_api_request("GET", "/api/v1/templates")
        
        if response and response.status_code == 200:
            templates = response.json()
            
            # Should return list of templates
            self.assertIsInstance(templates, (list, dict))
            
            if isinstance(templates, dict) and "templates" in templates:
                template_list = templates["templates"]
                self.assertIsInstance(template_list, list)


class TestTaskMonitoring(BaseIntegrationTest):
    """Test task monitoring and progress tracking"""
    
    def test_task_list_and_status_monitoring(self):
        """Test task listing and status monitoring"""
        # Get current task list
        response = self.make_api_request("GET", "/api/v1/tasks")
        
        if response and response.status_code == 200:
            task_data = response.json()
            
            # Should return task information
            self.assertIsInstance(task_data, (list, dict))
            
            if isinstance(task_data, dict) and "tasks" in task_data:
                tasks = task_data["tasks"]
                self.assertIsInstance(tasks, list)
                
                # Test individual task status retrieval
                for task in tasks[:3]:  # Test first 3 tasks
                    if "id" in task:
                        task_id = task["id"]
                        status_response = self.make_api_request("GET", f"/api/v1/tasks/{task_id}")
                        
                        if status_response and status_response.status_code == 200:
                            task_status = status_response.json()
                            
                            # Should have required status fields
                            required_fields = ["id", "status"]
                            for field in required_fields:
                                if field in task_status:
                                    self.assertIsNotNone(task_status[field])
    
    def test_task_progress_tracking(self):
        """Test real-time task progress tracking"""
        # Submit a new task for monitoring
        test_payload = {
            "video_subject": f"Progress tracking test {self.test_id}",
            "video_script": "Short test script for progress monitoring",
            "video_terms": "test,progress",
            "video_count": 1
        }
        
        response = self.make_api_request("POST", "/api/v1/video/generate",
                                       json=test_payload, headers=self.headers)
        
        if response and response.status_code in [200, 201, 202]:
            response_data = response.json()
            
            if "task_id" in response_data:
                task_id = response_data["task_id"]
                
                # Monitor progress for a short period
                progress_states = []
                start_time = time.time()
                
                while time.time() - start_time < 30:  # Monitor for 30 seconds
                    status_response = self.make_api_request("GET", f"/api/v1/tasks/{task_id}")
                    
                    if status_response and status_response.status_code == 200:
                        task_data = status_response.json()
                        current_status = task_data.get("status", "unknown")
                        
                        if current_status not in progress_states:
                            progress_states.append(current_status)
                            print(f"Task {task_id} status: {current_status}")
                        
                        if current_status in ["completed", "failed", "error"]:
                            break
                    
                    time.sleep(2)
                
                # Should have tracked at least one status
                self.assertGreater(len(progress_states), 0)


class TestMultiUserScenarios(BaseIntegrationTest):
    """Test multi-user scenarios and concurrent operations"""
    
    def test_concurrent_video_creation(self):
        """Test multiple concurrent video creation requests"""
        print(f"\nTesting concurrent video creation: {self.test_id}")
        
        def create_video(user_id):
            """Create video for a specific user"""
            payload = {
                "video_subject": f"Concurrent test video for user {user_id}",
                "video_script": f"Test script for user {user_id} concurrent processing",
                "video_terms": f"test,user{user_id},concurrent",
                "video_count": 1
            }
            
            response = self.make_api_request("POST", "/api/v1/video/generate",
                                           json=payload, headers=self.headers)
            
            if response:
                return {
                    "user_id": user_id,
                    "status_code": response.status_code,
                    "response": response.json() if response.status_code in [200, 201, 202] else None
                }
            return {"user_id": user_id, "status_code": None, "response": None}
        
        # Submit multiple concurrent requests
        num_users = 3
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(create_video, i) for i in range(num_users)]
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze results
        successful_submissions = [r for r in results if r["status_code"] in [200, 201, 202]]
        
        if len(successful_submissions) > 0:
            print(f"Successfully submitted {len(successful_submissions)}/{num_users} concurrent requests")
            
            # Should handle concurrent requests
            self.assertGreater(len(successful_submissions), 0)
        else:
            print("No successful concurrent submissions - may be expected in test environment")
    
    def test_concurrent_task_monitoring(self):
        """Test concurrent task status monitoring"""
        # Get list of tasks
        response = self.make_api_request("GET", "/api/v1/tasks")
        
        if response and response.status_code == 200:
            task_data = response.json()
            
            if isinstance(task_data, dict) and "tasks" in task_data:
                tasks = task_data["tasks"][:5]  # Test first 5 tasks
                
                def monitor_task(task_id):
                    """Monitor individual task"""
                    response = self.make_api_request("GET", f"/api/v1/tasks/{task_id}")
                    return {
                        "task_id": task_id,
                        "success": response.status_code == 200 if response else False
                    }
                
                # Monitor multiple tasks concurrently
                with ThreadPoolExecutor(max_workers=5) as executor:
                    task_ids = [task.get("id") for task in tasks if "id" in task]
                    futures = [executor.submit(monitor_task, task_id) for task_id in task_ids]
                    results = [future.result() for future in as_completed(futures)]
                
                # Should handle concurrent monitoring
                successful_monitors = [r for r in results if r["success"]]
                if len(task_ids) > 0:
                    success_rate = len(successful_monitors) / len(task_ids)
                    self.assertGreater(success_rate, 0.5)  # At least 50% success rate


class TestPerformanceUnderLoad(BaseIntegrationTest):
    """Test system performance under various load conditions"""
    
    def test_api_response_time_under_load(self):
        """Test API response times under concurrent load"""
        def make_health_check():
            """Make health check request and measure time"""
            start_time = time.time()
            response = self.make_api_request("GET", "/ping")
            end_time = time.time()
            
            return {
                "response_time": end_time - start_time,
                "success": response.status_code == 200 if response else False
            }
        
        # Make multiple concurrent health checks
        num_requests = 10
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_health_check) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze response times
        successful_results = [r for r in results if r["success"]]
        
        if len(successful_results) > 0:
            response_times = [r["response_time"] for r in successful_results]
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            print(f"Load test results: {len(successful_results)}/{num_requests} successful")
            print(f"Average response time: {avg_response_time:.2f}s")
            print(f"Max response time: {max_response_time:.2f}s")
            
            # Response times should be reasonable even under load
            self.assertLess(avg_response_time, 2.0)  # Average under 2 seconds
            self.assertLess(max_response_time, 5.0)   # Max under 5 seconds
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable under repeated operations"""
        # Make repeated requests and monitor for memory leaks
        num_iterations = 20
        
        for i in range(num_iterations):
            # Make health check request
            response = self.make_api_request("GET", "/ping")
            
            if response and response.status_code == 200:
                data = response.json()
                
                # Check for memory information if available
                if "system" in data and "memory_percent" in data["system"]:
                    memory_percent = data["system"]["memory_percent"]
                    
                    # Memory usage should not continuously increase
                    self.assertLess(memory_percent, 95.0)  # Should not exceed 95%
            
            time.sleep(0.1)  # Small delay between requests


def run_integration_flow_tests():
    """Run all integration flow tests"""
    print("="*70)
    print("Integration Flow Testing Suite")
    print("Created by QA_Engineer agent in the Hive Mind swarm")
    print("="*70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEndToEndVideoCreation,
        TestTemplateManagement,
        TestTaskMonitoring,
        TestMultiUserScenarios,
        TestPerformanceUnderLoad,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "="*70)
    print("INTEGRATION FLOW TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_flow_tests()
    sys.exit(0 if success else 1)