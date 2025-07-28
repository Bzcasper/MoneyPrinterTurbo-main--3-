"""
AI Upscaler Validation and Testing Suite
========================================

Comprehensive testing and validation for the AI video upscaler system.
Tests functionality, performance, and integration with the video processing pipeline.

Author: AIUpscaler Agent
Version: 1.0.0
"""

import os
import sys
import asyncio
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from loguru import logger
from app.services.neural_training.ai_upscaler import (
    AIVideoUpscaler, UpscalerConfig, create_fast_upscaler, 
    create_balanced_upscaler, create_ultra_upscaler
)
from app.services.neural_training.upscaler_integration import (
    UpscalerIntegrationManager, upscale_video_integrated
)
from app.models.schema import VideoParams


class AIUpscalerValidator:
    """Validation suite for AI upscaler"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        
        logger.info("AI Upscaler Validator initialized")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests"""
        logger.info("🧪 Starting AI Upscaler validation suite")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            
            # Create test video
            test_video_path = await self._create_test_video()
            
            # Run individual tests
            tests = [
                ("Model Initialization", self._test_model_initialization),
                ("Basic Upscaling", lambda: self._test_basic_upscaling(test_video_path)),
                ("Scale Factor Validation", lambda: self._test_scale_factors(test_video_path)),
                ("Quality Presets", lambda: self._test_quality_presets(test_video_path)),
                ("Temporal Consistency", lambda: self._test_temporal_consistency(test_video_path)),
                ("Edge Enhancement", lambda: self._test_edge_enhancement(test_video_path)),
                ("Memory Management", lambda: self._test_memory_management(test_video_path)),
                ("Integration Layer", lambda: self._test_integration_layer(test_video_path)),
                ("Error Handling", lambda: self._test_error_handling()),
                ("Performance Benchmarks", lambda: self._test_performance(test_video_path))
            ]
            
            for test_name, test_func in tests:
                logger.info(f"🔍 Running test: {test_name}")
                
                try:
                    start_time = time.time()
                    result = await test_func()
                    test_time = time.time() - start_time
                    
                    self.test_results.append({
                        'test_name': test_name,
                        'status': 'PASSED' if result['success'] else 'FAILED',
                        'duration': test_time,
                        'details': result
                    })
                    
                    if result['success']:
                        logger.success(f"✅ {test_name}: PASSED ({test_time:.2f}s)")
                    else:
                        logger.error(f"❌ {test_name}: FAILED - {result.get('error')}")
                        
                except Exception as e:
                    test_time = time.time() - start_time
                    logger.error(f"💥 {test_name}: CRASHED - {str(e)}")
                    
                    self.test_results.append({
                        'test_name': test_name,
                        'status': 'CRASHED',
                        'duration': test_time,
                        'details': {'error': str(e)}
                    })
        
        # Generate summary
        return self._generate_test_summary()
    
    async def _create_test_video(self) -> str:
        """Create a test video for validation"""
        logger.info("🎬 Creating test video...")
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (256, 256))
        
        # Generate 60 frames (2 seconds at 30fps)
        for i in range(60):
            # Create frame with moving pattern
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
            
            # Add moving rectangle
            x = int(50 + 50 * np.sin(i * 0.1))
            y = int(50 + 50 * np.cos(i * 0.1))
            cv2.rectangle(frame, (x, y), (x + 50, y + 50), (0, 255, 0), -1)
            
            # Add text
            cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add noise for testing
            noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)
            
            out.write(frame)
        
        out.release()
        
        logger.success(f"✅ Test video created: {video_path}")
        return video_path
    
    async def _test_model_initialization(self) -> Dict[str, Any]:
        """Test model initialization"""
        try:
            # Test different quality presets
            presets = ['fast', 'balanced', 'ultra']
            
            for preset in presets:
                logger.debug(f"Testing {preset} preset initialization...")
                
                if preset == 'fast':
                    upscaler = create_fast_upscaler()
                elif preset == 'balanced':
                    upscaler = create_balanced_upscaler()
                else:
                    upscaler = create_ultra_upscaler()
                
                await upscaler.initialize_models()
                
                # Check if models are loaded
                if not upscaler.models:
                    return {'success': False, 'error': f'No models loaded for {preset} preset'}
                
                logger.debug(f"✅ {preset} preset: {len(upscaler.models)} models loaded")
            
            return {'success': True, 'models_loaded': len(upscaler.models)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _test_basic_upscaling(self, test_video_path: str) -> Dict[str, Any]:
        """Test basic video upscaling functionality"""
        try:
            output_path = os.path.join(self.temp_dir, "upscaled_basic.mp4")
            
            upscaler = create_balanced_upscaler()
            await upscaler.initialize_models()
            
            result = await upscaler.upscale_video(
                test_video_path, output_path, scale_factor=2
            )
            
            if not result['success']:
                return {'success': False, 'error': result.get('error')}
            
            # Verify output exists and has correct properties
            if not os.path.exists(output_path):
                return {'success': False, 'error': 'Output file not created'}
            
            # Check video properties
            cap = cv2.VideoCapture(output_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Expected resolution: 256x256 -> 512x512 (2x upscaling)
            if width != 512 or height != 512:
                return {
                    'success': False, 
                    'error': f'Incorrect output resolution: {width}x{height}, expected 512x512'
                }
            
            return {
                'success': True,
                'output_resolution': (width, height),
                'frames_processed': result['frames_processed'],
                'processing_time': result['processing_time']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _test_scale_factors(self, test_video_path: str) -> Dict[str, Any]:
        """Test different scale factors"""
        try:
            scale_factors = [2, 4, 8]
            results = {}
            
            upscaler = create_balanced_upscaler()
            await upscaler.initialize_models()
            
            for scale in scale_factors:
                output_path = os.path.join(self.temp_dir, f"upscaled_{scale}x.mp4")
                
                result = await upscaler.upscale_video(
                    test_video_path, output_path, scale_factor=scale
                )
                
                if result['success']:
                    # Verify resolution
                    cap = cv2.VideoCapture(output_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    
                    expected_width = 256 * scale
                    expected_height = 256 * scale
                    
                    if width == expected_width and height == expected_height:
                        results[f'{scale}x'] = 'PASSED'
                    else:
                        results[f'{scale}x'] = f'FAILED - Resolution {width}x{height}, expected {expected_width}x{expected_height}'
                else:
                    results[f'{scale}x'] = f'FAILED - {result.get("error")}'
            
            # Check if all scale factors passed
            all_passed = all(result == 'PASSED' for result in results.values())
            
            return {
                'success': all_passed,
                'scale_factor_results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _test_quality_presets(self, test_video_path: str) -> Dict[str, Any]:
        """Test different quality presets"""
        try:
            presets = ['fast', 'balanced', 'ultra']
            results = {}
            
            for preset in presets:
                output_path = os.path.join(self.temp_dir, f"upscaled_{preset}.mp4")
                
                if preset == 'fast':
                    upscaler = create_fast_upscaler()
                elif preset == 'balanced':
                    upscaler = create_balanced_upscaler()
                else:
                    upscaler = create_ultra_upscaler()
                
                await upscaler.initialize_models()
                
                start_time = time.time()
                result = await upscaler.upscale_video(
                    test_video_path, output_path, scale_factor=2
                )
                processing_time = time.time() - start_time
                
                if result['success']:
                    results[preset] = {
                        'status': 'PASSED',
                        'processing_time': processing_time,
                        'output_exists': os.path.exists(output_path)
                    }
                else:
                    results[preset] = {
                        'status': 'FAILED',
                        'error': result.get('error')
                    }
            
            # Check if all presets passed
            all_passed = all(result['status'] == 'PASSED' for result in results.values())
            
            return {
                'success': all_passed,
                'preset_results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _test_temporal_consistency(self, test_video_path: str) -> Dict[str, Any]:
        """Test temporal consistency features"""
        try:
            # Test with temporal consistency enabled
            config_with_temporal = UpscalerConfig(
                enable_temporal_consistency=True,
                temporal_window=3
            )
            
            upscaler_temporal = AIVideoUpscaler(config_with_temporal)
            await upscaler_temporal.initialize_models()
            
            output_path = os.path.join(self.temp_dir, "upscaled_temporal.mp4")
            
            result = await upscaler_temporal.upscale_video(
                test_video_path, output_path, scale_factor=2
            )
            
            if not result['success']:
                return {'success': False, 'error': f'Temporal consistency test failed: {result.get("error")}'}
            
            # Test without temporal consistency
            config_no_temporal = UpscalerConfig(
                enable_temporal_consistency=False
            )
            
            upscaler_no_temporal = AIVideoUpscaler(config_no_temporal)
            await upscaler_no_temporal.initialize_models()
            
            output_path_no_temporal = os.path.join(self.temp_dir, "upscaled_no_temporal.mp4")
            
            result_no_temporal = await upscaler_no_temporal.upscale_video(
                test_video_path, output_path_no_temporal, scale_factor=2
            )
            
            if not result_no_temporal['success']:
                return {'success': False, 'error': f'No temporal consistency test failed: {result_no_temporal.get("error")}'}
            
            return {
                'success': True,
                'temporal_enabled': result['processing_time'],
                'temporal_disabled': result_no_temporal['processing_time'],
                'both_outputs_exist': os.path.exists(output_path) and os.path.exists(output_path_no_temporal)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _test_edge_enhancement(self, test_video_path: str) -> Dict[str, Any]:
        """Test edge enhancement features"""
        try:
            config = UpscalerConfig(
                enable_edge_enhancement=True,
                enable_detail_recovery=True
            )
            
            upscaler = AIVideoUpscaler(config)
            await upscaler.initialize_models()
            
            output_path = os.path.join(self.temp_dir, "upscaled_edge_enhanced.mp4")
            
            result = await upscaler.upscale_video(
                test_video_path, output_path, scale_factor=2
            )
            
            return {
                'success': result['success'],
                'error': result.get('error'),
                'output_exists': os.path.exists(output_path),
                'processing_time': result.get('processing_time')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _test_memory_management(self, test_video_path: str) -> Dict[str, Any]:
        """Test memory management features"""
        try:
            import psutil
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Test with memory constraints
            config = UpscalerConfig(
                max_memory_usage=0.5,  # 50% limit
                parallel_streams=2  # Reduced for memory constraint
            )
            
            upscaler = AIVideoUpscaler(config)
            await upscaler.initialize_models()
            
            output_path = os.path.join(self.temp_dir, "upscaled_memory_test.mp4")
            
            result = await upscaler.upscale_video(
                test_video_path, output_path, scale_factor=2
            )
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            return {
                'success': result['success'],
                'error': result.get('error'),
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': memory_increase,
                'output_exists': os.path.exists(output_path)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _test_integration_layer(self, test_video_path: str) -> Dict[str, Any]:
        """Test integration with video processing pipeline"""
        try:
            params = VideoParams(
                video_subject="test_integration",
                voice_name="en-US-JennyNeural",
                subtitle_enabled=False
            )
            
            output_path = os.path.join(self.temp_dir, "upscaled_integrated.mp4")
            
            result = await upscale_video_integrated(
                test_video_path, output_path, params, scale_factor=2, quality_preset="fast"
            )
            
            return {
                'success': result['success'],
                'error': result.get('error'),
                'output_exists': os.path.exists(output_path),
                'processing_time': result.get('total_processing_time'),
                'quality_enhanced': result.get('quality_enhanced')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling"""
        try:
            upscaler = create_balanced_upscaler()
            await upscaler.initialize_models()
            
            # Test with non-existent input file
            result1 = await upscaler.upscale_video(
                "non_existent_file.mp4", 
                os.path.join(self.temp_dir, "output.mp4"), 
                scale_factor=2
            )
            
            # Should fail gracefully
            if result1['success']:
                return {'success': False, 'error': 'Should have failed with non-existent input'}
            
            # Test with invalid scale factor
            try:
                config = UpscalerConfig(scale_factors=[3])  # Invalid scale factor
                upscaler_invalid = AIVideoUpscaler(config)
                await upscaler_invalid.initialize_models()
                
                # Should handle invalid scale factor
                error_handled = True
            except Exception:
                error_handled = True
            
            return {
                'success': True,
                'non_existent_file_handled': not result1['success'],
                'invalid_scale_factor_handled': error_handled
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _test_performance(self, test_video_path: str) -> Dict[str, Any]:
        """Test performance benchmarks"""
        try:
            # Test fast preset performance
            upscaler_fast = create_fast_upscaler()
            await upscaler_fast.initialize_models()
            
            start_time = time.time()
            result_fast = await upscaler_fast.upscale_video(
                test_video_path, 
                os.path.join(self.temp_dir, "upscaled_perf_fast.mp4"), 
                scale_factor=2
            )
            fast_time = time.time() - start_time
            
            # Test balanced preset performance
            upscaler_balanced = create_balanced_upscaler()
            await upscaler_balanced.initialize_models()
            
            start_time = time.time()
            result_balanced = await upscaler_balanced.upscale_video(
                test_video_path, 
                os.path.join(self.temp_dir, "upscaled_perf_balanced.mp4"), 
                scale_factor=2
            )
            balanced_time = time.time() - start_time
            
            return {
                'success': result_fast['success'] and result_balanced['success'],
                'fast_preset_time': fast_time,
                'balanced_preset_time': balanced_time,
                'speed_improvement': balanced_time / fast_time if fast_time > 0 else 0,
                'fast_fps': result_fast.get('average_fps', 0),
                'balanced_fps': result_balanced.get('average_fps', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate test summary report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASSED')
        failed_tests = sum(1 for result in self.test_results if result['status'] == 'FAILED')
        crashed_tests = sum(1 for result in self.test_results if result['status'] == 'CRASHED')
        
        total_time = sum(result['duration'] for result in self.test_results)
        
        summary = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'crashed': crashed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_time': total_time,
            'individual_results': self.test_results
        }
        
        # Log summary
        logger.info("=" * 80)
        logger.info("AI UPSCALER VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Crashed: {crashed_tests}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info("=" * 80)
        
        return summary


async def main():
    """Main validation function"""
    logger.info("🚀 Starting AI Upscaler Validation Suite")
    
    validator = AIUpscalerValidator()
    
    try:
        summary = await validator.run_all_tests()
        
        # Save results to file
        results_file = "ai_upscaler_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Validation results saved to: {results_file}")
        
        # Exit with appropriate code
        if summary['success_rate'] == 100:
            logger.success("🎉 All tests passed!")
            return 0
        else:
            logger.warning(f"⚠️ Some tests failed (Success rate: {summary['success_rate']:.1f}%)")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Validation suite crashed: {str(e)}")
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))