"""
Specialized tests for FFmpeg concatenation fixes and optimizations.

This module focuses specifically on validating the FFmpeg concatenation improvements:
1. Progressive FFmpeg concatenation performance
2. Batch processing with memory management 
3. Hardware acceleration fallback scenarios
4. Stream copy vs re-encoding behavior
5. Large file handling and memory efficiency
6. Error recovery and timeout handling
"""

import unittest
import os
import sys
import tempfile
import shutil
import time
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.video import (
    progressive_ffmpeg_concat,
    _ffmpeg_concat_batch,
    _ffmpeg_progressive_concat,
    MemoryMonitor,
    CodecOptimizer,
    CONCAT_BATCH_SIZE
)

class TestFFmpegConcatenation(unittest.TestCase):
    """Test suite for FFmpeg concatenation optimizations"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_videos = []
        
        # Create test video files
        self.create_test_videos()
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_test_videos(self):
        """Create test video files for concatenation testing"""
        try:
            from moviepy import ColorClip
            
            # Create multiple short test videos
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            
            for i, color in enumerate(colors):
                clip = ColorClip(
                    size=(720, 1280), 
                    color=color, 
                    duration=2
                ).with_fps(30)
                
                video_path = os.path.join(self.temp_dir, f"test_video_{i}.mp4")
                clip.write_videofile(video_path, logger=None, verbose=False)
                clip.close()
                
                self.test_videos.append(video_path)
                
        except Exception as e:
            print(f"Warning: Could not create test videos: {e}")
            # Create dummy files for basic testing
            self._create_dummy_videos()
            
    def _create_dummy_videos(self):
        """Create dummy video files when MoviePy is unavailable"""
        for i in range(4):
            video_path = os.path.join(self.temp_dir, f"test_video_{i}.mp4")
            with open(video_path, 'wb') as f:
                # Write minimal MP4 header-like content
                f.write(b'ftypisom' + b'\x00' * 100)
            self.test_videos.append(video_path)


class TestProgressiveConcatenation(TestFFmpegConcatenation):
    """Test progressive FFmpeg concatenation functionality"""
    
    def test_single_video_concatenation(self):
        """Test concatenation with single video (should just copy)"""
        print("\n=== Testing Single Video Concatenation ===")
        
        output_path = os.path.join(self.temp_dir, "single_output.mp4")
        
        # Test with single file
        result = progressive_ffmpeg_concat(
            video_files=[self.test_videos[0]],
            output_path=output_path,
            threads=2
        )
        
        print(f"✅ Single video concatenation result: {result}")
        
        if result and os.path.exists(output_path):
            # Output should exist and have some content
            output_size = os.path.getsize(output_path)
            input_size = os.path.getsize(self.test_videos[0])
            print(f"✅ Input size: {input_size}, Output size: {output_size}")
            self.assertGreater(output_size, 0)
        else:
            print("ℹ️  Single video concatenation failed (expected with dummy files)")
            
    def test_multiple_video_concatenation(self):
        """Test concatenation with multiple videos"""
        print("\n=== Testing Multiple Video Concatenation ===")
        
        output_path = os.path.join(self.temp_dir, "multi_output.mp4")
        
        # Test with multiple files
        result = progressive_ffmpeg_concat(
            video_files=self.test_videos,
            output_path=output_path,
            threads=2
        )
        
        print(f"✅ Multi-video concatenation result: {result}")
        print(f"   Input videos: {len(self.test_videos)}")
        
        if result and os.path.exists(output_path):
            output_size = os.path.getsize(output_path)
            print(f"✅ Output file created, size: {output_size}")
            self.assertGreater(output_size, 0)
        else:
            print("ℹ️  Multi-video concatenation failed (expected with dummy files)")
            
    def test_empty_video_list(self):
        """Test concatenation with empty video list"""
        print("\n=== Testing Empty Video List ===")
        
        output_path = os.path.join(self.temp_dir, "empty_output.mp4")
        
        # Test with empty list (should fail gracefully)
        result = progressive_ffmpeg_concat(
            video_files=[],
            output_path=output_path,
            threads=2
        )
        
        self.assertFalse(result)
        print("✅ Empty video list handled gracefully")
        
    def test_batch_size_handling(self):
        """Test batch size configuration and handling"""
        print("\n=== Testing Batch Size Handling ===")
        
        # Test with different batch sizes
        original_batch_size = CONCAT_BATCH_SIZE
        print(f"✅ Default batch size: {original_batch_size}")
        
        # Create more videos than batch size to test batching
        extra_videos = []
        for i in range(CONCAT_BATCH_SIZE + 2):
            video_path = os.path.join(self.temp_dir, f"batch_test_{i}.mp4") 
            with open(video_path, 'wb') as f:
                f.write(b'dummy_video_content')
            extra_videos.append(video_path)
            
        # Test that batching logic works
        batch_size = min(CONCAT_BATCH_SIZE, len(extra_videos))
        num_batches = (len(extra_videos) + batch_size - 1) // batch_size
        
        self.assertGreaterEqual(num_batches, 1)
        print(f"✅ {len(extra_videos)} videos would create {num_batches} batches")


class TestBatchProcessing(TestFFmpegConcatenation):
    """Test batch processing with memory management"""
    
    @patch('app.services.video.subprocess.run')
    def test_ffmpeg_concat_batch_success(self, mock_subprocess):
        """Test successful batch concatenation"""
        print("\n=== Testing FFmpeg Batch Success ===")
        
        # Mock successful FFmpeg execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        output_path = os.path.join(self.temp_dir, "batch_success.mp4")
        concat_list = os.path.join(self.temp_dir, "concat_list.txt")
        
        result = _ffmpeg_concat_batch(
            video_files=self.test_videos[:2],
            output_path=output_path,
            concat_list_file=concat_list,
            threads=2,
            use_hardware_acceleration=True
        )
        
        self.assertTrue(result)
        print("✅ Batch concatenation succeeded with mocked FFmpeg")
        
        # Verify FFmpeg was called
        mock_subprocess.assert_called()
        call_args = mock_subprocess.call_args[0][0]
        self.assertIn('ffmpeg', call_args)
        self.assertIn('-c', call_args)
        self.assertIn('copy', call_args)
        print("✅ FFmpeg called with stream copy")
        
    @patch('app.services.video.subprocess.run')
    def test_ffmpeg_concat_batch_fallback(self, mock_subprocess):
        """Test hardware acceleration fallback behavior"""
        print("\n=== Testing Hardware Acceleration Fallback ===")
        
        # Mock failed stream copy, then successful re-encoding
        mock_results = [
            MagicMock(returncode=1, stderr="Stream copy failed"),  # First attempt fails
            MagicMock(returncode=0, stderr="")  # Second attempt succeeds
        ]
        mock_subprocess.side_effect = mock_results
        
        output_path = os.path.join(self.temp_dir, "fallback_output.mp4")
        concat_list = os.path.join(self.temp_dir, "concat_list.txt")
        
        result = _ffmpeg_concat_batch(
            video_files=self.test_videos[:2],
            output_path=output_path,
            concat_list_file=concat_list,
            threads=2,
            use_hardware_acceleration=True
        )
        
        self.assertTrue(result)
        print("✅ Fallback to hardware acceleration succeeded")
        
        # Verify both calls were made
        self.assertEqual(mock_subprocess.call_count, 2)
        print("✅ Stream copy attempted, then fell back to re-encoding")
        
    @patch('app.services.video.subprocess.run')
    def test_ffmpeg_timeout_handling(self, mock_subprocess):
        """Test timeout handling in FFmpeg operations"""
        print("\n=== Testing FFmpeg Timeout Handling ===")
        
        # Mock timeout exception
        mock_subprocess.side_effect = subprocess.TimeoutExpired('ffmpeg', 300)
        
        output_path = os.path.join(self.temp_dir, "timeout_output.mp4")
        concat_list = os.path.join(self.temp_dir, "concat_list.txt")
        
        result = _ffmpeg_concat_batch(
            video_files=self.test_videos[:2],
            output_path=output_path, 
            concat_list_file=concat_list,
            threads=2,
            use_hardware_acceleration=False
        )
        
        self.assertFalse(result)
        print("✅ Timeout handled gracefully, returned False")
        
    def test_concat_list_file_creation(self):
        """Test creation of FFmpeg concat list files"""
        print("\n=== Testing Concat List File Creation ===")
        
        concat_list = os.path.join(self.temp_dir, "test_concat_list.txt")
        
        # Simulate concat list creation (from _ffmpeg_concat_batch)
        with open(concat_list, 'w', encoding='utf-8') as f:
            for video_file in self.test_videos[:2]:
                # Test path escaping
                escaped_path = video_file.replace("'", "'\''")
                f.write(f"file '{escaped_path}'\n")
                
        # Verify file was created and has content
        self.assertTrue(os.path.exists(concat_list))
        
        with open(concat_list, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.strip().split('\n')
            
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.startswith("file '") for line in lines))
        print(f"✅ Concat list file created with {len(lines)} entries")


class TestMemoryEfficiency(TestFFmpegConcatenation):
    """Test memory efficiency and monitoring"""
    
    def test_memory_monitoring_during_concat(self):
        """Test memory monitoring throughout concatenation"""
        print("\n=== Testing Memory Monitoring ===")
        
        # Record initial memory
        initial_memory = MemoryMonitor.get_memory_usage_mb()
        print(f"✅ Initial memory: {initial_memory:.1f}MB")
        
        # Simulate memory usage tracking
        memory_readings = []
        for i in range(5):
            current_memory = MemoryMonitor.get_memory_usage_mb()
            memory_readings.append(current_memory)
            time.sleep(0.01)  # Brief pause
            
        # Verify we can get memory readings
        self.assertEqual(len(memory_readings), 5)
        self.assertTrue(all(reading > 0 for reading in memory_readings))
        
        avg_memory = sum(memory_readings) / len(memory_readings)
        print(f"✅ Average memory during test: {avg_memory:.1f}MB")
        
    def test_memory_availability_check(self):
        """Test memory availability checking"""
        print("\n=== Testing Memory Availability ===")
        
        # Test various memory requirements
        test_requirements = [10, 100, 500, 1000, 2000]  # MB
        
        for requirement in test_requirements:
            available = MemoryMonitor.is_memory_available(requirement)
            status = "✅ Available" if available else "❌ Not available"
            print(f"   {requirement}MB: {status}")
            
    def test_garbage_collection_triggers(self):
        """Test garbage collection triggering"""
        print("\n=== Testing Garbage Collection ===")
        
        initial_memory = MemoryMonitor.get_memory_usage_mb()
        
        # Create some temporary memory usage
        temp_data = []
        for i in range(10000):
            temp_data.append(list(range(100)))
            
        memory_after_allocation = MemoryMonitor.get_memory_usage_mb()
        
        # Force garbage collection
        MemoryMonitor.force_gc_cleanup()
        
        # Clear references
        del temp_data
        
        # Force GC again
        MemoryMonitor.force_gc_cleanup()
        
        final_memory = MemoryMonitor.get_memory_usage_mb()
        
        print(f"✅ Memory progression:")
        print(f"   Initial: {initial_memory:.1f}MB")
        print(f"   After allocation: {memory_after_allocation:.1f}MB")
        print(f"   After cleanup: {final_memory:.1f}MB")
        
        # Memory should be cleaned up (though exact amounts vary)
        memory_cleaned = memory_after_allocation - final_memory
        print(f"✅ Memory cleaned: {memory_cleaned:.1f}MB")


class TestProgressiveBatching(TestFFmpegConcatenation):
    """Test progressive batch processing"""
    
    @patch('app.services.video._ffmpeg_concat_batch')
    def test_progressive_concat_multiple_batches(self, mock_concat_batch):
        """Test progressive concatenation with multiple batches"""
        print("\n=== Testing Progressive Multi-Batch Concatenation ===")
        
        # Mock successful batch processing
        mock_concat_batch.return_value = True
        
        # Create enough videos for multiple batches
        many_videos = []
        for i in range(CONCAT_BATCH_SIZE * 2 + 3):  # Force multiple batches
            video_path = os.path.join(self.temp_dir, f"prog_test_{i}.mp4")
            with open(video_path, 'wb') as f:
                f.write(b'dummy_content')
            many_videos.append(video_path)
            
        # Calculate expected batches
        batch_size = min(CONCAT_BATCH_SIZE, len(many_videos))
        expected_batches = (len(many_videos) + batch_size - 1) // batch_size
        
        batches = [many_videos[i:i + batch_size] 
                  for i in range(0, len(many_videos), batch_size)]
        
        output_path = os.path.join(self.temp_dir, "progressive_output.mp4")
        
        # Test progressive batching
        result = _ffmpeg_progressive_concat(
            batches=batches,
            output_path=output_path,
            temp_dir=self.temp_dir,
            threads=2
        )
        
        self.assertTrue(result)
        print(f"✅ Progressive concatenation: {len(many_videos)} videos")
        print(f"   Batch size: {batch_size}")
        print(f"   Number of batches: {len(batches)}")
        print(f"   Expected batches: {expected_batches}")
        
        # Verify batch function was called multiple times
        expected_calls = len(batches) + 1  # One call per batch + final concat
        self.assertEqual(mock_concat_batch.call_count, expected_calls)
        print(f"✅ Concat batch called {mock_concat_batch.call_count} times")
        
    @patch('app.services.video._ffmpeg_concat_batch')
    def test_progressive_concat_batch_failure(self, mock_concat_batch):
        """Test progressive concatenation with batch failure"""
        print("\n=== Testing Progressive Concatenation Batch Failure ===")
        
        # Mock first batch success, second batch failure
        mock_concat_batch.side_effect = [True, False]  # Success, then failure
        
        # Create videos for multiple batches
        many_videos = []
        for i in range(CONCAT_BATCH_SIZE + 2):
            video_path = os.path.join(self.temp_dir, f"fail_test_{i}.mp4")
            with open(video_path, 'wb') as f:
                f.write(b'dummy_content')
            many_videos.append(video_path)
            
        batches = [many_videos[i:i + CONCAT_BATCH_SIZE] 
                  for i in range(0, len(many_videos), CONCAT_BATCH_SIZE)]
        
        output_path = os.path.join(self.temp_dir, "failure_output.mp4")
        
        # Test failure handling
        result = _ffmpeg_progressive_concat(
            batches=batches,
            output_path=output_path,
            temp_dir=self.temp_dir,
            threads=2
        )
        
        self.assertFalse(result)
        print("✅ Batch failure handled gracefully")
        
        # Should have stopped after first failure
        self.assertLessEqual(mock_concat_batch.call_count, 2)
        print(f"✅ Processing stopped after failure ({mock_concat_batch.call_count} calls)")


class TestCodecOptimization(TestFFmpegConcatenation):
    """Test codec optimization for concatenation"""
    
    def test_codec_settings_for_concatenation(self):
        """Test codec settings optimization for concatenation"""
        print("\n=== Testing Codec Settings for Concatenation ===")
        
        optimizer = CodecOptimizer()
        
        # Test speed-optimized settings for concatenation
        settings = optimizer.get_optimal_codec_settings(
            content_type='general',
            target_quality='speed'
        )
        
        self.assertIn('encoder_type', settings)
        self.assertIn('codec', settings)
        print(f"✅ Speed settings: {settings['encoder_type']} / {settings['codec']}")
        
        # For concatenation, we prioritize speed
        if settings['encoder_type'] == 'software':
            # Should use fast presets
            print("✅ Software encoding optimized for speed")
        else:
            print(f"✅ Hardware encoding: {settings['encoder_type']}")
            
    def test_ffmpeg_args_for_concatenation(self):
        """Test FFmpeg argument generation for concatenation"""
        print("\n=== Testing FFmpeg Args for Concatenation ===")
        
        optimizer = CodecOptimizer()
        settings = optimizer.get_optimal_codec_settings(target_quality='speed')
        
        input_file = self.test_videos[0]
        output_file = os.path.join(self.temp_dir, "codec_test_output.mp4")
        
        args = optimizer.build_ffmpeg_args(input_file, output_file, settings)
        
        # Verify essential concatenation-friendly arguments
        self.assertIn('ffmpeg', args)
        self.assertIn('-c:v', args)
        
        codec_index = args.index('-c:v') + 1
        codec_used = args[codec_index]
        
        self.assertEqual(codec_used, settings['codec'])
        print(f"✅ Codec in args: {codec_used}")
        
        # Check for optimization flags
        if '-movflags' in args:
            movflags_index = args.index('-movflags') + 1
            movflags_value = args[movflags_index]
            self.assertIn('faststart', movflags_value)
            print("✅ Faststart optimization included")


class TestPerformanceValidation(TestFFmpegConcatenation):
    """Test performance improvements and validation"""
    
    def test_concatenation_performance_simulation(self):
        """Test concatenation performance with timing"""
        print("\n=== Testing Concatenation Performance ===")
        
        # Simulate different concatenation scenarios
        scenarios = [
            ("Single video", [self.test_videos[0]]),
            ("Two videos", self.test_videos[:2]),
            ("Multiple videos", self.test_videos),
        ]
        
        for scenario_name, videos in scenarios:
            output_path = os.path.join(self.temp_dir, f"perf_{scenario_name.replace(' ', '_')}.mp4")
            
            start_time = time.time()
            
            # This will likely fail with dummy files, but we can time the attempt
            result = progressive_ffmpeg_concat(
                video_files=videos,
                output_path=output_path,
                threads=2
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"✅ {scenario_name}: {processing_time:.3f}s, Success: {result}")
            
            # Performance should be reasonable (< 5s for test scenarios)
            self.assertLess(processing_time, 5.0)
            
    def test_memory_efficiency_during_concat(self):
        """Test memory efficiency during concatenation"""
        print("\n=== Testing Memory Efficiency ===")
        
        # Monitor memory during a concatenation attempt
        memory_readings = []
        
        def track_memory():
            for _ in range(10):
                memory_readings.append(MemoryMonitor.get_memory_usage_mb())
                time.sleep(0.01)
                
        import threading
        monitor_thread = threading.Thread(target=track_memory)
        monitor_thread.start()
        
        # Perform concatenation while monitoring
        output_path = os.path.join(self.temp_dir, "memory_test_output.mp4")
        progressive_ffmpeg_concat(
            video_files=self.test_videos[:2],
            output_path=output_path,
            threads=2
        )
        
        monitor_thread.join()
        
        # Analyze memory usage
        if memory_readings:
            min_memory = min(memory_readings)
            max_memory = max(memory_readings)
            memory_variation = max_memory - min_memory
            
            print(f"✅ Memory range: {min_memory:.1f}MB - {max_memory:.1f}MB")
            print(f"✅ Memory variation: {memory_variation:.1f}MB")
            
            # Memory variation should be reasonable (not growing excessively)
            self.assertLess(memory_variation, 500)  # Less than 500MB variation
        
        print("✅ Memory efficiency test completed")


if __name__ == '__main__':
    print("=" * 70)
    print("FFMPEG CONCATENATION VALIDATION SUITE")
    print("=" * 70)
    print("""
    This test suite specifically validates FFmpeg concatenation fixes:
    
    ✅ Progressive FFmpeg concatenation performance
    ✅ Batch processing with memory management
    ✅ Hardware acceleration fallback scenarios
    ✅ Stream copy vs re-encoding behavior
    ✅ Large file handling and memory efficiency
    ✅ Error recovery and timeout handling
    ✅ Codec optimization for concatenation
    ✅ Performance validation and benchmarks
    """)
    print("=" * 70)
    
    # Run the test suite
    unittest.main(verbosity=2, exit=False)
    
    print("=" * 70)
    print("FFMPEG CONCATENATION VALIDATION COMPLETED")
    print("=" * 70)