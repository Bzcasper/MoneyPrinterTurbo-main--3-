#!/usr/bin/env python3
"""
Enhanced Codec Optimizer for MoneyPrinter Turbo Enhanced
Main orchestrator module that integrates hardware detection, codec configuration, 
and performance benchmarking for optimal video encoding.

Key Features:
- Unified API for codec optimization
- Intelligent encoder selection
- Performance-based recommendations
- Comprehensive error handling
- Modular architecture with dependency injection
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path

from .hardware_detector import HardwareDetector, hardware_detector
from .codec_configurator import (
    CodecConfigurator, 
    EncodingConfig, 
    ContentType, 
    QualityTarget, 
    StreamingMode
)
from .performance_benchmarker import PerformanceBenchmarker, BenchmarkSuite, BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRequest:
    """Request parameters for codec optimization"""
    content_type: ContentType = ContentType.GENERAL
    quality_target: QualityTarget = QualityTarget.BALANCED
    streaming_mode: Optional[StreamingMode] = None
    hdr_mode: bool = False
    resolution: Tuple[int, int] = (1920, 1080)
    framerate: int = 30
    encoder_override: Optional[str] = None
    platform_target: Optional[str] = None
    enable_benchmarking: bool = False
    multi_pass: bool = False


@dataclass
class OptimizationResult:
    """Result of codec optimization"""
    config: EncodingConfig
    benchmark_suite: Optional[BenchmarkSuite] = None
    recommendations: List[str] = None
    performance_score: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    optimization_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'config': self.config.to_dict(),
            'benchmark_suite': self.benchmark_suite.to_dict() if self.benchmark_suite else None,
            'recommendations': self.recommendations or [],
            'performance_score': self.performance_score,
            'success': self.success,
            'error_message': self.error_message,
            'optimization_time': self.optimization_time
        }


class EnhancedCodecOptimizer:
    """Enhanced codec optimizer with comprehensive optimization capabilities"""
    
    def __init__(self,
                 hardware_detector: Optional[HardwareDetector] = None,
                 configurator: Optional[CodecConfigurator] = None,
                 benchmarker: Optional[PerformanceBenchmarker] = None):
        """
        Initialize enhanced codec optimizer
        
        Args:
            hardware_detector: Hardware detection module
            configurator: Codec configuration module  
            benchmarker: Performance benchmarking module
        """
        # Use dependency injection with fallbacks
        self.hardware_detector = hardware_detector or globals().get('hardware_detector')
        self.configurator = configurator or CodecConfigurator(self.hardware_detector)
        self.benchmarker = benchmarker or PerformanceBenchmarker(
            self.hardware_detector, self.configurator
        )
        
        # Optimization cache
        self._optimization_cache: Dict[str, OptimizationResult] = {}
        self._cache_max_size = 100
        
        logger.info("Enhanced codec optimizer initialized successfully")
    
    def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        """
        Perform comprehensive codec optimization
        
        Args:
            request: Optimization request parameters
            
        Returns:
            Complete optimization result
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self._optimization_cache:
                cached_result = self._optimization_cache[cache_key]
                logger.info(f"🔄 Using cached optimization result for {cache_key}")
                return cached_result
            
            logger.info(f"🚀 Starting codec optimization...")
            logger.info(f"   Content: {request.content_type.value}")
            logger.info(f"   Quality: {request.quality_target.value}")
            logger.info(f"   Resolution: {request.resolution[0]}x{request.resolution[1]}")
            
            # Generate base configuration
            config = self._generate_base_config(request)
            
            # Apply platform optimizations if specified
            if request.platform_target:
                config = self.configurator.optimize_for_platform(config, request.platform_target)
            
            # Setup multi-pass if requested
            if request.multi_pass:
                pass_configs = self.configurator.setup_multi_pass_encoding(config)
                logger.info(f"📈 Multi-pass encoding configured: {len(pass_configs)} passes")
            
            # Run benchmarking if enabled
            benchmark_suite = None
            if request.enable_benchmarking:
                benchmark_suite = self._run_performance_benchmark(request)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(config, benchmark_suite)
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(config, benchmark_suite)
            
            optimization_time = time.time() - start_time
            
            result = OptimizationResult(
                config=config,
                benchmark_suite=benchmark_suite,
                recommendations=recommendations,
                performance_score=performance_score,
                success=True,
                optimization_time=optimization_time
            )
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            logger.info(f"✅ Optimization completed in {optimization_time:.2f}s")
            logger.info(f"   Encoder: {config.encoder_type}")
            logger.info(f"   Performance Score: {performance_score:.1f}/100")
            
            return result
            
        except Exception as e:
            optimization_time = time.time() - start_time
            error_message = f"Optimization failed: {str(e)}"
            logger.error(error_message)
            
            return OptimizationResult(
                config=self._generate_fallback_config(request),
                success=False,
                error_message=error_message,
                optimization_time=optimization_time
            )
    
    def get_encoder_recommendations(self, request: OptimizationRequest) -> List[Dict[str, Any]]:
        """
        Get encoder recommendations without full optimization
        
        Args:
            request: Optimization request parameters
            
        Returns:
            List of encoder recommendations with scores
        """
        recommendations = []
        
        if not self.hardware_detector:
            return [{'encoder': 'software', 'score': 50, 'reason': 'Hardware detection unavailable'}]
        
        available_encoders = self.hardware_detector.get_available_encoders()
        
        for encoder in available_encoders:
            score = self._score_encoder_for_request(encoder, request)
            reason = self._get_encoder_recommendation_reason(encoder, request)
            
            recommendations.append({
                'encoder': encoder,
                'score': score,
                'reason': reason,
                'available': True
            })
        
        # Sort by score descending
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    def quick_optimize(self, 
                      content_type: ContentType = ContentType.GENERAL,
                      quality_target: QualityTarget = QualityTarget.BALANCED,
                      resolution: Tuple[int, int] = (1920, 1080)) -> EncodingConfig:
        """
        Quick optimization without benchmarking
        
        Args:
            content_type: Type of content
            quality_target: Quality preference
            resolution: Video resolution
            
        Returns:
            Optimized encoding configuration
        """
        request = OptimizationRequest(
            content_type=content_type,
            quality_target=quality_target,
            resolution=resolution,
            enable_benchmarking=False
        )
        
        result = self.optimize(request)
        return result.config
    
    def benchmark_current_system(self, 
                                test_duration: float = 10.0,
                                resolution: Tuple[int, int] = (1920, 1080)) -> BenchmarkSuite:
        """
        Run comprehensive system benchmark
        
        Args:
            test_duration: Duration of benchmark test
            resolution: Test resolution
            
        Returns:
            Complete benchmark suite results
        """
        logger.info("🧪 Running comprehensive system benchmark...")
        
        return self.benchmarker.run_comprehensive_benchmark(
            test_duration=test_duration,
            resolution=resolution,
            test_name="system_benchmark"
        )
    
    def _generate_base_config(self, request: OptimizationRequest) -> EncodingConfig:
        """Generate base encoding configuration"""
        return self.configurator.generate_encoding_config(
            content_type=request.content_type,
            quality_target=request.quality_target,
            streaming_mode=request.streaming_mode,
            hdr_mode=request.hdr_mode,
            resolution=request.resolution,
            framerate=request.framerate,
            encoder_override=request.encoder_override
        )
    
    def _run_performance_benchmark(self, request: OptimizationRequest) -> Optional[BenchmarkSuite]:
        """Run performance benchmark for the request"""
        try:
            return self.benchmarker.run_comprehensive_benchmark(
                test_duration=5.0,  # Short benchmark for optimization
                resolution=request.resolution,
                framerate=request.framerate,
                test_name=f"optimization_{request.content_type.value}"
            )
        except Exception as e:
            logger.warning(f"Benchmarking failed: {e}")
            return None
    
    def _generate_recommendations(self, 
                                config: EncodingConfig, 
                                benchmark_suite: Optional[BenchmarkSuite]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Hardware-based recommendations
        if config.encoder_type == 'software' and self.hardware_detector:
            available = self.hardware_detector.get_available_encoders()
            if any(enc in available for enc in ['nvenc', 'qsv', 'vaapi']):
                recommendations.append(
                    "Consider using hardware acceleration for better performance"
                )
        
        # Quality recommendations
        if config.quality_target == QualityTarget.SPEED:
            recommendations.append(
                "For better quality, consider using 'balanced' or 'quality' targets"
            )
        
        # Resolution recommendations
        width, height = config.resolution
        if width * height > 2073600:  # > 1080p
            recommendations.append(
                "High resolution detected - hardware acceleration strongly recommended"
            )
        
        # Benchmark-based recommendations
        if benchmark_suite and benchmark_suite.best_speed:
            best_encoder = benchmark_suite.best_speed.encoder_type
            if best_encoder != config.encoder_type:
                recommendations.append(
                    f"Performance benchmark suggests {best_encoder} for better speed"
                )
        
        return recommendations
    
    def _calculate_performance_score(self, 
                                   config: EncodingConfig,
                                   benchmark_suite: Optional[BenchmarkSuite]) -> float:
        """Calculate overall performance score"""
        score = 50.0  # Base score
        
        # Hardware acceleration bonus
        if config.encoder_type in ['nvenc', 'qsv', 'vaapi']:
            score += 20.0
        
        # Quality target adjustments
        quality_scores = {
            QualityTarget.SPEED: 30.0,
            QualityTarget.BALANCED: 20.0,
            QualityTarget.QUALITY: 10.0,
            QualityTarget.ARCHIVE: 5.0
        }
        score += quality_scores.get(config.quality_target, 0)
        
        # Benchmark-based scoring
        if benchmark_suite:
            successful_results = [r for r in benchmark_suite.results if r.success]
            if successful_results:
                score += 10.0  # Bonus for successful benchmarking
        
        return min(score, 100.0)
    
    def _score_encoder_for_request(self, encoder: str, request: OptimizationRequest) -> float:
        """Score an encoder for a specific request"""
        score = 50.0
        
        # Base hardware acceleration score
        if encoder in ['nvenc', 'qsv', 'vaapi']:
            score += 30.0
        
        # Content type specific scoring
        if request.content_type == ContentType.HIGH_MOTION:
            if encoder in ['nvenc', 'qsv']:
                score += 20.0
        elif request.content_type == ContentType.TEXT_HEAVY:
            if encoder == 'software':
                score += 15.0
        
        # Quality target scoring
        if request.quality_target == QualityTarget.SPEED:
            if encoder in ['nvenc', 'qsv']:
                score += 15.0
        elif request.quality_target in [QualityTarget.QUALITY, QualityTarget.ARCHIVE]:
            if encoder == 'software':
                score += 10.0
        
        return min(score, 100.0)
    
    def _get_encoder_recommendation_reason(self, encoder: str, request: OptimizationRequest) -> str:
        """Get reason for encoder recommendation"""
        if encoder == 'nvenc':
            return "NVIDIA hardware acceleration - excellent speed"
        elif encoder == 'qsv':
            return "Intel Quick Sync - good balance of speed and quality"
        elif encoder == 'vaapi':
            return "Linux hardware acceleration - good performance"
        elif encoder == 'software':
            return "Software encoding - maximum quality and compatibility"
        else:
            return "Alternative encoder option"
    
    def _generate_fallback_config(self, request: OptimizationRequest) -> EncodingConfig:
        """Generate fallback configuration when optimization fails"""
        return EncodingConfig(
            encoder_type='software',
            codec='libx264',
            settings={'codec': 'libx264', 'preset': 'fast', 'crf': '23'},
            content_type=request.content_type,
            quality_target=request.quality_target,
            streaming_mode=request.streaming_mode,
            hdr_mode=request.hdr_mode,
            resolution=request.resolution,
            framerate=request.framerate,
            capabilities=None
        )
    
    def _generate_cache_key(self, request: OptimizationRequest) -> str:
        """Generate cache key for optimization request"""
        return (
            f"{request.content_type.value}_{request.quality_target.value}_"
            f"{request.resolution[0]}x{request.resolution[1]}_{request.framerate}fps_"
            f"hdr:{request.hdr_mode}_encoder:{request.encoder_override}_"
            f"platform:{request.platform_target}"
        )
    
    def _cache_result(self, key: str, result: OptimizationResult) -> None:
        """Cache optimization result"""
        if len(self._optimization_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._optimization_cache))
            del self._optimization_cache[oldest_key]
        
        self._optimization_cache[key] = result
    
    def clear_cache(self) -> None:
        """Clear optimization cache"""
        self._optimization_cache.clear()
        logger.info("Optimization cache cleared")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'hardware_detection': None,
            'available_encoders': [],
            'optimization_cache_size': len(self._optimization_cache)
        }
        
        if self.hardware_detector:
            info['hardware_detection'] = self.hardware_detector.get_detection_summary()
            info['available_encoders'] = self.hardware_detector.get_available_encoders()
        
        return info


# Global instance for singleton access
enhanced_codec_optimizer = EnhancedCodecOptimizer()