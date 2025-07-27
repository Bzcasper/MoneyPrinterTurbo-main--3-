"""
Enhanced debug logging validation and implementation for video processing pipeline.

This module provides comprehensive debug logging enhancements throughout the video
processing pipeline to help diagnose issues and monitor performance.
"""

import unittest
import os
import sys
import tempfile
import logging
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class DebugLogger:
    """Enhanced debug logger for video processing pipeline"""
    
    def __init__(self, name="VideoProcessor", level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create console handler with detailed formatting
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)8s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_clip_processing(self, clip_index, action, **kwargs):
        """Log clip processing actions with context"""
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.debug(f"CLIP[{clip_index:03d}] {action} | {context}")
    
    def log_performance(self, operation, duration, **metrics):
        """Log performance metrics"""
        metrics_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
        self.logger.info(f"PERF: {operation} completed in {duration:.3f}s | {metrics_str}")
    
    def log_memory(self, operation, memory_mb, change=None):
        """Log memory usage"""
        change_str = f" | Change: {change:+.1f}MB" if change else ""
        self.logger.debug(f"MEM: {operation} | Usage: {memory_mb:.1f}MB{change_str}")
    
    def log_codec(self, encoder_type, codec, settings_summary):
        """Log codec selection and settings"""
        self.logger.info(f"CODEC: Using {encoder_type} encoder | Codec: {codec} | Settings: {settings_summary}")
    
    def log_error_recovery(self, operation, error, recovery_action):
        """Log error recovery actions"""
        self.logger.warning(f"ERROR_RECOVERY: {operation} failed with '{error}' | Recovery: {recovery_action}")
    
    def log_batch_progress(self, batch_num, total_batches, items_in_batch, progress):
        """Log batch processing progress"""
        self.logger.info(f"BATCH[{batch_num}/{total_batches}] Processing {items_in_batch} items | Progress: {progress:.1f}%")
    
    def log_hardware_detection(self, hardware_type, available, details=None):
        """Log hardware capability detection"""
        status = "AVAILABLE" if available else "NOT_AVAILABLE"
        details_str = f" | Details: {details}" if details else ""
        self.logger.info(f"HW_DETECT: {hardware_type} - {status}{details_str}")


class TestDebugLoggingEnhancements(unittest.TestCase):
    """Test enhanced debug logging functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_output = StringIO()
        
        # Capture log output
        self.handler = logging.StreamHandler(self.log_output)
        self.handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s:%(message)s')
        self.handler.setFormatter(formatter)
        
    def tearDown(self):
        if hasattr(self, 'handler'):
            self.handler.close()
    
    def test_debug_logger_initialization(self):
        """Test DebugLogger initialization and configuration"""
        print("\n=== Testing Debug Logger Initialization ===")
        
        logger = DebugLogger("TestProcessor")
        
        self.assertIsNotNone(logger.logger)
        self.assertEqual(logger.logger.level, logging.DEBUG)
        self.assertTrue(len(logger.logger.handlers) > 0)
        
        print("✅ Debug logger initialized successfully")
        print(f"✅ Logger level: {logging.getLevelName(logger.logger.level)}")
        print(f"✅ Number of handlers: {len(logger.logger.handlers)}")
    
    def test_clip_processing_logging(self):
        """Test clip processing action logging"""
        print("\n=== Testing Clip Processing Logging ===")
        
        logger = DebugLogger("ClipProcessor")
        
        # Add our test handler
        logger.logger.addHandler(self.handler)
        
        # Test various clip processing logs
        logger.log_clip_processing(1, "STARTED", file="video1.mp4", duration=5.2)
        logger.log_clip_processing(1, "RESIZING", from_size="1920x1080", to_size="720x1280")
        logger.log_clip_processing(1, "TRANSITION", type="fade_in", duration=1.0)
        logger.log_clip_processing(1, "ENCODING", codec="h264_qsv", quality=23)
        logger.log_clip_processing(1, "COMPLETED", output_file="temp-clip-1.mp4", size_mb=12.3)
        
        log_content = self.log_output.getvalue()
        
        # Verify log entries
        self.assertIn("CLIP[001] STARTED", log_content)
        self.assertIn("video1.mp4", log_content)
        self.assertIn("CLIP[001] RESIZING", log_content)
        self.assertIn("1920x1080", log_content)
        self.assertIn("CLIP[001] COMPLETED", log_content)
        
        print("✅ Clip processing logs generated successfully")
        print(f"✅ Log entries captured: {len(log_content.split('CLIP['))}")
    
    def test_performance_logging(self):
        """Test performance metrics logging"""
        print("\n=== Testing Performance Logging ===")
        
        logger = DebugLogger("PerfMonitor")
        logger.logger.addHandler(self.handler)
        
        # Test performance logging
        logger.log_performance("clip_processing", 2.456, clips=5, threads=4, memory_peak=156.7)
        logger.log_performance("ffmpeg_concat", 1.234, files=8, method="stream_copy", speedup="3.2x")
        logger.log_performance("parallel_encode", 15.678, clips=20, workers=8, success_rate=95.0)
        
        log_content = self.log_output.getvalue()
        
        # Verify performance logs
        self.assertIn("PERF: clip_processing completed", log_content)
        self.assertIn("2.456s", log_content)
        self.assertIn("clips=5", log_content)
        self.assertIn("PERF: ffmpeg_concat", log_content)
        self.assertIn("speedup=3.2x", log_content)
        
        print("✅ Performance metrics logged successfully")
        print(f"✅ Performance entries: {log_content.count('PERF:')}")
    
    def test_memory_logging(self):
        """Test memory usage logging"""
        print("\n=== Testing Memory Usage Logging ===")
        
        logger = DebugLogger("MemoryMonitor")
        logger.logger.addHandler(self.handler)
        
        # Test memory logging
        logger.log_memory("initialization", 245.6)
        logger.log_memory("clip_loading", 312.4, change=66.8)
        logger.log_memory("after_processing", 198.2, change=-114.2)
        logger.log_memory("garbage_collection", 187.5, change=-10.7)
        
        log_content = self.log_output.getvalue()
        
        # Verify memory logs
        self.assertIn("MEM: initialization", log_content)
        self.assertIn("245.6MB", log_content)
        self.assertIn("MEM: clip_loading", log_content)
        self.assertIn("Change: +66.8MB", log_content)
        self.assertIn("Change: -114.2MB", log_content)
        
        print("✅ Memory usage logged successfully")
        print(f"✅ Memory entries: {log_content.count('MEM:')}")
    
    def test_codec_logging(self):
        """Test codec selection and settings logging"""
        print("\n=== Testing Codec Logging ===")
        
        logger = DebugLogger("CodecOptimizer")
        logger.logger.addHandler(self.handler)
        
        # Test codec logging
        logger.log_codec("hardware", "h264_qsv", "preset=fast, quality=23, lookahead=1")
        logger.log_codec("software", "libx264", "preset=superfast, crf=23, tune=film")
        logger.log_codec("hardware", "h264_nvenc", "preset=p4, cq=23, rc=vbr")
        
        log_content = self.log_output.getvalue()
        
        # Verify codec logs
        self.assertIn("CODEC: Using hardware encoder", log_content)
        self.assertIn("h264_qsv", log_content)
        self.assertIn("CODEC: Using software encoder", log_content)
        self.assertIn("libx264", log_content)
        self.assertIn("preset=fast", log_content)
        
        print("✅ Codec information logged successfully")
        print(f"✅ Codec entries: {log_content.count('CODEC:')}")
    
    def test_error_recovery_logging(self):
        """Test error recovery action logging"""
        print("\n=== Testing Error Recovery Logging ===")
        
        logger = DebugLogger("ErrorHandler")
        logger.logger.addHandler(self.handler)
        
        # Test error recovery logging
        logger.log_error_recovery("hardware_encode", "NVENC initialization failed", "falling back to software")
        logger.log_error_recovery("stream_copy", "incompatible streams", "re-encoding with libx264")
        logger.log_error_recovery("parallel_process", "timeout after 300s", "reducing thread count to 2")
        
        log_content = self.log_output.getvalue()
        
        # Verify error recovery logs
        self.assertIn("ERROR_RECOVERY: hardware_encode failed", log_content)
        self.assertIn("NVENC initialization failed", log_content)
        self.assertIn("falling back to software", log_content)
        self.assertIn("ERROR_RECOVERY: stream_copy failed", log_content)
        self.assertIn("re-encoding with libx264", log_content)
        
        print("✅ Error recovery logged successfully")
        print(f"✅ Error recovery entries: {log_content.count('ERROR_RECOVERY:')}")
    
    def test_batch_progress_logging(self):
        """Test batch processing progress logging"""
        print("\n=== Testing Batch Progress Logging ===")
        
        logger = DebugLogger("BatchProcessor")
        logger.logger.addHandler(self.handler)
        
        # Test batch progress logging
        logger.log_batch_progress(1, 4, 8, 25.0)
        logger.log_batch_progress(2, 4, 8, 50.0)
        logger.log_batch_progress(3, 4, 6, 75.0)
        logger.log_batch_progress(4, 4, 5, 100.0)
        
        log_content = self.log_output.getvalue()
        
        # Verify batch progress logs
        self.assertIn("BATCH[1/4] Processing 8 items", log_content)
        self.assertIn("Progress: 25.0%", log_content)
        self.assertIn("BATCH[4/4] Processing 5 items", log_content)
        self.assertIn("Progress: 100.0%", log_content)
        
        print("✅ Batch progress logged successfully")
        print(f"✅ Batch entries: {log_content.count('BATCH[')}")
    
    def test_hardware_detection_logging(self):
        """Test hardware capability detection logging"""
        print("\n=== Testing Hardware Detection Logging ===")
        
        logger = DebugLogger("HardwareDetector")
        logger.logger.addHandler(self.handler)
        
        # Test hardware detection logging
        logger.log_hardware_detection("QSV", True, "Intel i7-12700K detected")
        logger.log_hardware_detection("NVENC", False, "No NVIDIA GPU found")
        logger.log_hardware_detection("VAAPI", True, "Intel UHD Graphics 770")
        logger.log_hardware_detection("CPU_Threads", True, "12 cores, 20 threads")
        
        log_content = self.log_output.getvalue()
        
        # Verify hardware detection logs
        self.assertIn("HW_DETECT: QSV - AVAILABLE", log_content)
        self.assertIn("Intel i7-12700K detected", log_content)
        self.assertIn("HW_DETECT: NVENC - NOT_AVAILABLE", log_content)
        self.assertIn("No NVIDIA GPU found", log_content)
        self.assertIn("HW_DETECT: VAAPI - AVAILABLE", log_content)
        
        print("✅ Hardware detection logged successfully")
        print(f"✅ Hardware entries: {log_content.count('HW_DETECT:')}")


class TestIntegratedDebugLogging(unittest.TestCase):
    """Test integrated debug logging in video processing scenarios"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = DebugLogger("IntegratedTest")
        
    def test_complete_video_processing_log_flow(self):
        """Test complete video processing pipeline logging"""
        print("\n=== Testing Complete Pipeline Logging ===")
        
        # Simulate complete video processing flow with logging
        start_time = time.time()
        
        # 1. Hardware detection phase
        self.logger.log_hardware_detection("QSV", True, "Available")
        self.logger.log_hardware_detection("NVENC", False, "Not found")
        
        # 2. Initialization phase
        initial_memory = 245.6
        self.logger.log_memory("initialization", initial_memory)
        self.logger.log_codec("software", "libx264", "preset=superfast, crf=23")
        
        # 3. Clip processing phase
        for i in range(3):
            clip_start = time.time()
            
            self.logger.log_clip_processing(i, "STARTED", file=f"video_{i}.mp4", duration=5.0)
            self.logger.log_clip_processing(i, "RESIZING", from_size="1920x1080", to_size="720x1280")
            self.logger.log_clip_processing(i, "ENCODING", codec="libx264", preset="superfast")
            
            clip_end = time.time()
            self.logger.log_clip_processing(i, "COMPLETED", duration=clip_end - clip_start)
            
            # Memory tracking
            memory_change = 15.2 + i * 5.1
            self.logger.log_memory(f"after_clip_{i}", initial_memory + memory_change, change=memory_change)
        
        # 4. Batch concatenation phase
        self.logger.log_batch_progress(1, 1, 3, 100.0)
        
        concat_start = time.time()
        # Simulate concatenation error and recovery
        self.logger.log_error_recovery("stream_copy", "incompatible streams", "re-encoding")
        concat_end = time.time()
        
        self.logger.log_performance("ffmpeg_concat", concat_end - concat_start, 
                                   files=3, method="re-encode", success=True)
        
        # 5. Cleanup phase
        self.logger.log_memory("after_cleanup", initial_memory + 5.2, change=-55.6)
        
        total_time = time.time() - start_time
        self.logger.log_performance("complete_pipeline", total_time, 
                                   clips=3, success_rate=100.0, memory_efficient=True)
        
        print("✅ Complete pipeline logging simulation completed")
        print(f"✅ Total simulation time: {total_time:.3f}s")
    
    def test_concurrent_logging(self):
        """Test logging in concurrent processing scenarios"""
        print("\n=== Testing Concurrent Logging ===")
        
        import threading
        import concurrent.futures
        
        def simulate_worker(worker_id):
            """Simulate worker thread with logging"""
            worker_logger = DebugLogger(f"Worker-{worker_id}")
            
            # Simulate work with logging
            worker_logger.log_clip_processing(worker_id, "STARTED", thread_id=threading.current_thread().ident)
            
            # Simulate processing time
            time.sleep(0.01)
            
            worker_logger.log_clip_processing(worker_id, "COMPLETED", 
                                            thread_id=threading.current_thread().ident,
                                            success=True)
            return worker_id
        
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(simulate_worker, i) for i in range(6)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        self.assertEqual(len(results), 6)
        print(f"✅ Concurrent logging completed for {len(results)} workers")
        
    def test_error_scenario_logging(self):
        """Test logging during error scenarios"""
        print("\n=== Testing Error Scenario Logging ===")
        
        # Simulate various error scenarios
        error_scenarios = [
            ("file_not_found", "video.mp4 does not exist", "skipping clip"),
            ("memory_exhaustion", "insufficient memory", "reducing batch size"),
            ("codec_failure", "h264_qsv init failed", "fallback to software"),
            ("timeout", "processing exceeded 300s", "terminating and cleanup"),
        ]
        
        for operation, error, recovery in error_scenarios:
            self.logger.log_error_recovery(operation, error, recovery)
            
            # Log memory and performance impact
            self.logger.log_memory(f"after_{operation}_error", 320.5, change=25.3)
            self.logger.log_performance(f"error_recovery_{operation}", 0.5, 
                                      success=False, recovery_attempted=True)
        
        print(f"✅ Error scenario logging completed for {len(error_scenarios)} scenarios")


class TestLogAnalysis(unittest.TestCase):
    """Test log analysis and debugging utilities"""
    
    def test_log_pattern_detection(self):
        """Test detection of common log patterns for debugging"""
        print("\n=== Testing Log Pattern Detection ===")
        
        # Sample log entries that might indicate issues
        sample_logs = [
            "MEM: after_clip_5 | Usage: 1950.5MB | Change: +156.7MB",  # High memory usage
            "PERF: clip_processing completed in 45.678s | clips=1",      # Slow processing
            "ERROR_RECOVERY: hardware_encode failed with 'timeout'",     # Hardware issues
            "BATCH[1/8] Processing 100 items | Progress: 12.5%",        # Large batch
            "CODEC: Using software encoder | Codec: libx264",           # No hardware accel
        ]
        
        # Pattern detection logic (simplified)
        patterns = {
            'high_memory': lambda log: 'Usage:' in log and float(log.split('Usage: ')[1].split('MB')[0]) > 1500,
            'slow_processing': lambda log: 'completed in' in log and float(log.split('completed in ')[1].split('s')[0]) > 30,
            'hardware_failure': lambda log: 'ERROR_RECOVERY' in log and 'hardware' in log,
            'large_batch': lambda log: 'BATCH[' in log and 'items' in log and int(log.split('Processing ')[1].split(' items')[0]) > 50,
            'no_hardware_accel': lambda log: 'software encoder' in log,
        }
        
        detected_patterns = {}
        for pattern_name, pattern_func in patterns.items():
            detected_patterns[pattern_name] = []
            for log in sample_logs:
                try:
                    if pattern_func(log):
                        detected_patterns[pattern_name].append(log)
                except:
                    pass  # Skip malformed logs
        
        # Verify pattern detection
        self.assertTrue(len(detected_patterns['high_memory']) > 0)
        self.assertTrue(len(detected_patterns['slow_processing']) > 0)
        self.assertTrue(len(detected_patterns['hardware_failure']) > 0)
        
        for pattern, logs in detected_patterns.items():
            if logs:
                print(f"✅ Detected {pattern}: {len(logs)} instances")
            else:
                print(f"ℹ️  No {pattern} detected")
                
    def test_performance_trend_analysis(self):
        """Test analysis of performance trends from logs"""
        print("\n=== Testing Performance Trend Analysis ===")
        
        # Simulate performance data over time
        performance_data = [
            ("clip_processing", 2.456, {"clips": 5}),
            ("clip_processing", 2.123, {"clips": 5}),
            ("clip_processing", 1.987, {"clips": 5}),  # Improving trend
            ("ffmpeg_concat", 5.234, {"files": 8}),
            ("ffmpeg_concat", 3.456, {"files": 8}),
            ("ffmpeg_concat", 2.123, {"files": 8}),    # Significant improvement
        ]
        
        # Analyze trends
        trends = {}
        for operation, duration, metadata in performance_data:
            if operation not in trends:
                trends[operation] = []
            trends[operation].append(duration)
        
        # Calculate trend direction
        for operation, durations in trends.items():
            if len(durations) >= 3:
                recent_avg = sum(durations[-3:]) / 3
                early_avg = sum(durations[:3]) / 3
                improvement = (early_avg - recent_avg) / early_avg * 100
                
                trend = "improving" if improvement > 5 else "stable" if abs(improvement) <= 5 else "degrading"
                print(f"✅ {operation}: {trend} (improvement: {improvement:+.1f}%)")
                
                self.assertIsInstance(improvement, float)
                self.assertIn(trend, ["improving", "stable", "degrading"])


if __name__ == '__main__':
    print("=" * 70)
    print("DEBUG LOGGING VALIDATION SUITE")
    print("=" * 70)
    print("""
    This test suite validates enhanced debug logging functionality:
    
    ✅ Debug logger initialization and configuration
    ✅ Clip processing action logging with context
    ✅ Performance metrics logging and analysis
    ✅ Memory usage tracking and monitoring
    ✅ Codec selection and settings logging
    ✅ Error recovery action documentation
    ✅ Batch processing progress tracking
    ✅ Hardware capability detection logging
    ✅ Integrated pipeline logging scenarios
    ✅ Concurrent processing log coordination
    ✅ Error scenario logging and analysis
    ✅ Log pattern detection and trend analysis
    """)
    print("=" * 70)
    
    # Run the test suite
    unittest.main(verbosity=2, exit=False)
    
    print("=" * 70)
    print("DEBUG LOGGING VALIDATION COMPLETED")
    print("=" * 70)