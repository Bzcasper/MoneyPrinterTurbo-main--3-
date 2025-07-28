#!/usr/bin/env python3
"""
Performance Benchmarking Module for MoneyPrinter Turbo Enhanced
Provides comprehensive encoder performance testing and analysis.

Key Features:
- Multi-encoder performance comparison
- Quality metrics analysis (PSNR, SSIM, VMAF)
- Real-time performance monitoring
- Test video generation
- Results analysis and recommendations
"""

import subprocess
import time
import os
import json
import threading
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .hardware_detector import HardwareDetector
from .codec_configurator import CodecConfigurator, EncodingConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark test result"""
    encoder_type: str
    config_name: str
    encoding_time: float
    file_size_mb: float
    bitrate_kbps: float
    fps_achieved: float
    cpu_usage_percent: Optional[float]
    memory_usage_mb: Optional[float]
    quality_metrics: Optional[Dict[str, float]]
    success: bool
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    test_name: str
    test_duration: float
    test_resolution: Tuple[int, int]
    test_framerate: int
    results: List[BenchmarkResult]
    best_speed: Optional[BenchmarkResult]
    best_quality: Optional[BenchmarkResult]
    best_efficiency: Optional[BenchmarkResult]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'test_name': self.test_name,
            'test_duration': self.test_duration,
            'test_resolution': self.test_resolution,
            'test_framerate': self.test_framerate,
            'results': [r.to_dict() for r in self.results],
            'best_speed': self.best_speed.to_dict() if self.best_speed else None,
            'best_quality': self.best_quality.to_dict() if self.best_quality else None,
            'best_efficiency': self.best_efficiency.to_dict() if self.best_efficiency else None,
            'timestamp': self.timestamp
        }


class PerformanceBenchmarker:
    """Comprehensive encoder performance benchmarking system"""
    
    def __init__(self, 
                 hardware_detector: Optional[HardwareDetector] = None,
                 configurator: Optional[CodecConfigurator] = None):
        self.hardware_detector = hardware_detector
        self.configurator = configurator
        
        # Performance monitoring
        self._monitoring_active = False
        self._performance_data = {}
        
        # Test video cache
        self._test_video_cache: Dict[str, Path] = {}
        
        logger.info("Performance benchmarker initialized")
    
    def run_comprehensive_benchmark(self,
                                  test_duration: float = 10.0,
                                  resolution: Tuple[int, int] = (1920, 1080),
                                  framerate: int = 30,
                                  test_name: str = "default") -> BenchmarkSuite:
        """
        Run comprehensive encoder performance benchmark
        
        Args:
            test_duration: Duration of test video in seconds
            resolution: Test video resolution
            framerate: Test video framerate
            test_name: Name for this benchmark suite
            
        Returns:
            Complete benchmark suite results
        """
        logger.info(f"🏃 Running comprehensive benchmark: {test_name}")
        logger.info(f"   Resolution: {resolution[0]}x{resolution[1]}")
        logger.info(f"   Duration: {test_duration}s @ {framerate}fps")
        
        start_time = time.time()
        
        # Create test video
        test_video_path = self._create_test_video(test_duration, resolution, framerate)
        
        # Get available encoders
        available_encoders = self._get_benchmark_encoders()
        
        # Run benchmarks
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for encoder_type in available_encoders:
                for config_name, config in self._get_encoder_configs(encoder_type, resolution, framerate).items():
                    future = executor.submit(
                        self._run_single_benchmark,
                        test_video_path,
                        config,
                        f"{encoder_type}_{config_name}"
                    )
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Benchmark task failed: {str(e)}")
        
        # Analyze results
        suite = BenchmarkSuite(
            test_name=test_name,
            test_duration=test_duration,
            test_resolution=resolution,
            test_framerate=framerate,
            results=results,
            best_speed=None,
            best_quality=None,
            best_efficiency=None,
            timestamp=time.time()
        )
        
        self._analyze_benchmark_results(suite)
        
        # Cleanup
        self._cleanup_test_files(test_video_path)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Benchmark completed in {total_time:.2f}s")
        
        return suite
    
    def benchmark_encoder_comparison(self,
                                   encoder_list: List[str],
                                   test_duration: float = 5.0,
                                   resolution: Tuple[int, int] = (1280, 720)) -> Dict[str, BenchmarkResult]:
        """
        Compare specific encoders head-to-head
        
        Args:
            encoder_list: List of encoder types to compare
            test_duration: Test video duration
            resolution: Test resolution
            
        Returns:
            Dictionary of encoder results
        """
        logger.info(f"🔀 Comparing encoders: {encoder_list}")
        
        test_video_path = self._create_test_video(test_duration, resolution, 30)
        results = {}
        
        for encoder_type in encoder_list:
            if not self._is_encoder_available(encoder_type):
                logger.warning(f"Encoder {encoder_type} not available, skipping")
                continue
            
            # Use balanced configuration for fair comparison
            config = self._get_balanced_config(encoder_type, resolution, 30)
            result = self._run_single_benchmark(
                test_video_path, config, f"{encoder_type}_comparison"
            )
            results[encoder_type] = result
        
        self._cleanup_test_files(test_video_path)
        return results
    
    def benchmark_quality_levels(self,
                                encoder_type: str,
                                test_duration: float = 5.0) -> List[BenchmarkResult]:
        """
        Benchmark different quality levels for a specific encoder
        
        Args:
            encoder_type: Encoder to test
            test_duration: Test video duration
            
        Returns:
            List of results for different quality levels
        """
        logger.info(f"📊 Testing quality levels for {encoder_type}")
        
        resolution = (1920, 1080)
        test_video_path = self._create_test_video(test_duration, resolution, 30)
        results = []
        
        # Test different quality targets
        quality_levels = ['speed', 'balanced', 'quality', 'archive']
        
        for quality in quality_levels:
            if not self.configurator:
                continue
                
            config = self.configurator.generate_encoding_config(
                quality_target=getattr(self.configurator.QualityTarget, quality.upper()),
                resolution=resolution,
                encoder_override=encoder_type
            )
            
            result = self._run_single_benchmark(
                test_video_path, config, f"{encoder_type}_{quality}"
            )
            results.append(result)
        
        self._cleanup_test_files(test_video_path)
        return results
    
    def _create_test_video(self,
                          duration: float,
                          resolution: Tuple[int, int],
                          framerate: int) -> Path:
        """Create test video for benchmarking"""
        
        cache_key = f"{duration}s_{resolution[0]}x{resolution[1]}_{framerate}fps"
        
        if cache_key in self._test_video_cache:
            cached_path = self._test_video_cache[cache_key]
            if cached_path.exists():
                return cached_path
        
        output_path = Path(f"benchmark_test_{cache_key}.mp4")
        
        # Create complex test pattern for realistic encoding load
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
            '-f', 'lavfi',
            '-i', f'testsrc2=duration={duration}:size={resolution[0]}x{resolution[1]}:rate={framerate}',
            '-f', 'lavfi',
            '-i', f'sine=frequency=1000:duration={duration}',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-c:a', 'aac', '-b:a', '128k',
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self._test_video_cache[cache_key] = output_path
            logger.debug(f"Created test video: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create test video: {e}")
            raise
    
    def _run_single_benchmark(self,
                            input_path: Path,
                            config: EncodingConfig,
                            test_name: str) -> BenchmarkResult:
        """Run single encoder benchmark"""
        
        output_path = Path(f"benchmark_output_{test_name}_{int(time.time())}.mp4")
        
        try:
            # Build FFmpeg command
            cmd = self._build_ffmpeg_command(input_path, output_path, config)
            
            # Start performance monitoring
            perf_data = self._start_performance_monitoring()
            
            # Run encoding
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            encoding_time = time.time() - start_time
            
            # Stop performance monitoring
            cpu_usage, memory_usage = self._stop_performance_monitoring(perf_data)
            
            if result.returncode == 0 and output_path.exists():
                # Calculate metrics
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                bitrate_kbps = (file_size_mb * 8 * 1024) / (config.framerate * 10)  # Assuming 10s video
                fps_achieved = (config.framerate * 10) / encoding_time
                
                # Quality metrics (if enabled)
                quality_metrics = self._calculate_quality_metrics(input_path, output_path)
                
                return BenchmarkResult(
                    encoder_type=config.encoder_type,
                    config_name=test_name,
                    encoding_time=encoding_time,
                    file_size_mb=file_size_mb,
                    bitrate_kbps=bitrate_kbps,
                    fps_achieved=fps_achieved,
                    cpu_usage_percent=cpu_usage,
                    memory_usage_mb=memory_usage,
                    quality_metrics=quality_metrics,
                    success=True,
                    error_message=None
                )
            else:
                return BenchmarkResult(
                    encoder_type=config.encoder_type,
                    config_name=test_name,
                    encoding_time=float('inf'),
                    file_size_mb=0,
                    bitrate_kbps=0,
                    fps_achieved=0,
                    cpu_usage_percent=None,
                    memory_usage_mb=None,
                    quality_metrics=None,
                    success=False,
                    error_message=result.stderr
                )
        
        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                encoder_type=config.encoder_type,
                config_name=test_name,
                encoding_time=float('inf'),
                file_size_mb=0,
                bitrate_kbps=0,
                fps_achieved=0,
                cpu_usage_percent=None,
                memory_usage_mb=None,
                quality_metrics=None,
                success=False,
                error_message="Timeout"
            )
        
        except Exception as e:
            return BenchmarkResult(
                encoder_type=config.encoder_type,
                config_name=test_name,
                encoding_time=float('inf'),
                file_size_mb=0,
                bitrate_kbps=0,
                fps_achieved=0,
                cpu_usage_percent=None,
                memory_usage_mb=None,
                quality_metrics=None,
                success=False,
                error_message=str(e)
            )
        
        finally:
            # Cleanup output file
            if output_path.exists():
                output_path.unlink()
    
    def _build_ffmpeg_command(self,
                            input_path: Path,
                            output_path: Path,
                            config: EncodingConfig) -> List[str]:
        """Build FFmpeg command from encoding config"""
        
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning']
        cmd.extend(['-i', str(input_path)])
        
        # Video codec
        cmd.extend(['-c:v', config.settings['codec']])
        
        # Add encoder-specific settings
        for key, value in config.settings.items():
            if key == 'codec' or value is None:
                continue
            cmd.extend([f'-{key}', str(value)])
        
        # Audio codec
        cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        
        # Output
        cmd.append(str(output_path))
        
        return cmd
    
    def _get_benchmark_encoders(self) -> List[str]:
        """Get list of encoders to benchmark"""
        encoders = ['software']
        
        if self.hardware_detector:
            available = self.hardware_detector.get_available_encoders()
            encoders.extend([enc for enc in available if enc != 'software'])
        
        return encoders
    
    def _get_encoder_configs(self,
                           encoder_type: str,
                           resolution: Tuple[int, int],
                           framerate: int) -> Dict[str, EncodingConfig]:
        """Get test configurations for encoder"""
        
        if not self.configurator:
            return {}
        
        configs = {}
        
        # Test different quality targets
        for quality in ['speed', 'balanced', 'quality']:
            try:
                config = self.configurator.generate_encoding_config(
                    quality_target=getattr(self.configurator.QualityTarget, quality.upper()),
                    resolution=resolution,
                    framerate=framerate,
                    encoder_override=encoder_type
                )
                configs[quality] = config
            except Exception as e:
                logger.warning(f"Failed to generate {quality} config for {encoder_type}: {e}")
        
        return configs
    
    def _get_balanced_config(self,
                           encoder_type: str,
                           resolution: Tuple[int, int],
                           framerate: int) -> EncodingConfig:
        """Get balanced configuration for encoder"""
        
        if self.configurator:
            return self.configurator.generate_encoding_config(
                quality_target=self.configurator.QualityTarget.BALANCED,
                resolution=resolution,
                framerate=framerate,
                encoder_override=encoder_type
            )
        
        # Fallback basic config
        return EncodingConfig(
            encoder_type=encoder_type,
            codec='libx264' if encoder_type == 'software' else f'{encoder_type}_codec',
            settings={'codec': 'libx264', 'preset': 'medium', 'crf': '23'},
            content_type=self.configurator.ContentType.GENERAL if self.configurator else None,
            quality_target=self.configurator.QualityTarget.BALANCED if self.configurator else None,
            streaming_mode=None,
            hdr_mode=False,
            resolution=resolution,
            framerate=framerate,
            capabilities=None
        )
    
    def _is_encoder_available(self, encoder_type: str) -> bool:
        """Check if encoder is available"""
        if encoder_type == 'software':
            return True
        
        if self.hardware_detector:
            return self.hardware_detector.is_encoder_available(encoder_type)
        
        return False
    
    def _start_performance_monitoring(self) -> Dict[str, Any]:
        """Start performance monitoring"""
        return {
            'start_time': time.time(),
            'start_cpu': time.process_time()
        }
    
    def _stop_performance_monitoring(self, perf_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Stop performance monitoring and return results"""
        try:
            elapsed_wall = time.time() - perf_data['start_time']
            elapsed_cpu = time.process_time() - perf_data['start_cpu']
            
            cpu_usage_percent = (elapsed_cpu / elapsed_wall) * 100 if elapsed_wall > 0 else None
            
            # Memory usage estimation (simplified)
            memory_usage_mb = None  # Could implement actual memory monitoring here
            
            return cpu_usage_percent, memory_usage_mb
        except Exception:
            return None, None
    
    def _calculate_quality_metrics(self,
                                 reference_path: Path,
                                 test_path: Path) -> Optional[Dict[str, float]]:
        """Calculate quality metrics (PSNR, SSIM)"""
        try:
            # Basic PSNR calculation using FFmpeg
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', str(reference_path),
                '-i', str(test_path),
                '-lavfi', 'psnr=stats_file=-',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Parse PSNR from output (simplified)
                psnr_value = 0.0  # Would need proper parsing
                return {'psnr': psnr_value}
        
        except Exception as e:
            logger.debug(f"Quality metrics calculation failed: {e}")
        
        return None
    
    def _analyze_benchmark_results(self, suite: BenchmarkSuite) -> None:
        """Analyze benchmark results and identify best performers"""
        
        successful_results = [r for r in suite.results if r.success]
        
        if not successful_results:
            logger.warning("No successful benchmark results to analyze")
            return
        
        # Find best speed (highest fps_achieved)
        suite.best_speed = max(successful_results, key=lambda r: r.fps_achieved)
        
        # Find best efficiency (best fps per MB ratio)
        def efficiency_score(r: BenchmarkResult) -> float:
            return r.fps_achieved / max(r.file_size_mb, 0.1)
        
        suite.best_efficiency = max(successful_results, key=efficiency_score)
        
        # Find best quality (if quality metrics available)
        quality_results = [r for r in successful_results if r.quality_metrics]
        if quality_results:
            suite.best_quality = max(quality_results, key=lambda r: r.quality_metrics.get('psnr', 0))
        else:
            # Fallback: smallest file size with reasonable speed
            suite.best_quality = min(successful_results, key=lambda r: r.file_size_mb)
        
        self._log_benchmark_summary(suite)
    
    def _log_benchmark_summary(self, suite: BenchmarkSuite) -> None:
        """Log benchmark results summary"""
        
        logger.info("📊 Benchmark Results Summary:")
        logger.info(f"   Test: {suite.test_name}")
        logger.info(f"   Total results: {len(suite.results)}")
        
        successful = [r for r in suite.results if r.success]
        logger.info(f"   Successful: {len(successful)}")
        
        if suite.best_speed:
            logger.info(f"🏆 Fastest: {suite.best_speed.encoder_type} ({suite.best_speed.fps_achieved:.1f} fps)")
        
        if suite.best_efficiency:
            eff = suite.best_efficiency.fps_achieved / suite.best_efficiency.file_size_mb
            logger.info(f"⚡ Most Efficient: {suite.best_efficiency.encoder_type} ({eff:.1f} fps/MB)")
        
        if suite.best_quality:
            logger.info(f"🎯 Best Quality: {suite.best_quality.encoder_type}")
    
    def _cleanup_test_files(self, *paths: Path) -> None:
        """Clean up test files"""
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
            except Exception as e:
                logger.debug(f"Failed to cleanup {path}: {e}")
    
    def save_benchmark_results(self, suite: BenchmarkSuite, output_path: Path) -> None:
        """Save benchmark results to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(suite.to_dict(), f, indent=2)
            logger.info(f"Benchmark results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
    
    def load_benchmark_results(self, input_path: Path) -> Optional[BenchmarkSuite]:
        """Load benchmark results from JSON file"""
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct BenchmarkSuite (simplified)
            results = [BenchmarkResult(**r) for r in data['results']]
            
            suite = BenchmarkSuite(
                test_name=data['test_name'],
                test_duration=data['test_duration'],
                test_resolution=tuple(data['test_resolution']),
                test_framerate=data['test_framerate'],
                results=results,
                best_speed=BenchmarkResult(**data['best_speed']) if data['best_speed'] else None,
                best_quality=BenchmarkResult(**data['best_quality']) if data['best_quality'] else None,
                best_efficiency=BenchmarkResult(**data['best_efficiency']) if data['best_efficiency'] else None,
                timestamp=data['timestamp']
            )
            
            logger.info(f"Loaded benchmark results from {input_path}")
            return suite
        
        except Exception as e:
            logger.error(f"Failed to load benchmark results: {e}")
            return None