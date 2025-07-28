#!/usr/bin/env python3
"""
Codec Module for MoneyPrinter Turbo Enhanced
Provides comprehensive video encoding optimization capabilities.

Main Components:
- HardwareDetector: Cross-platform encoder detection and validation
- CodecConfigurator: Intelligent encoder configuration and optimization
- PerformanceBenchmarker: Comprehensive encoder performance testing
- EnhancedCodecOptimizer: Main orchestrator for codec optimization

Usage:
    from app.services.codec import enhanced_codec_optimizer, OptimizationRequest, ContentType, QualityTarget
    
    # Quick optimization
    config = enhanced_codec_optimizer.quick_optimize(
        content_type=ContentType.HIGH_MOTION,
        quality_target=QualityTarget.BALANCED
    )
    
    # Full optimization with benchmarking
    request = OptimizationRequest(
        content_type=ContentType.GENERAL,
        quality_target=QualityTarget.QUALITY,
        enable_benchmarking=True
    )
    result = enhanced_codec_optimizer.optimize(request)
"""

# Import main components
from .hardware_detector import (
    HardwareDetector,
    EncoderType,
    EncoderCapabilities,
    hardware_detector
)

from .codec_configurator import (
    CodecConfigurator,
    EncodingConfig,
    ContentType,
    QualityTarget,
    StreamingMode
)

from .performance_benchmarker import (
    PerformanceBenchmarker,
    BenchmarkResult,
    BenchmarkSuite
)

from .enhanced_codec_optimizer import (
    EnhancedCodecOptimizer,
    OptimizationRequest,
    OptimizationResult,
    enhanced_codec_optimizer
)

# Version information
__version__ = "1.0.0"
__author__ = "MoneyPrinter Turbo Enhanced Team"

# Public API
__all__ = [
    # Main optimizer
    'enhanced_codec_optimizer',
    'EnhancedCodecOptimizer',
    'OptimizationRequest',
    'OptimizationResult',
    
    # Hardware detection
    'hardware_detector',
    'HardwareDetector',
    'EncoderType',
    'EncoderCapabilities',
    
    # Configuration
    'CodecConfigurator',
    'EncodingConfig',
    'ContentType',
    'QualityTarget',
    'StreamingMode',
    
    # Benchmarking
    'PerformanceBenchmarker',
    'BenchmarkResult',
    'BenchmarkSuite',
]

# Module-level convenience functions
def quick_optimize(content_type=ContentType.GENERAL, 
                  quality_target=QualityTarget.BALANCED,
                  resolution=(1920, 1080)):
    """
    Quick codec optimization without benchmarking
    
    Args:
        content_type: Type of content to optimize for
        quality_target: Quality vs speed preference
        resolution: Target video resolution
        
    Returns:
        EncodingConfig: Optimized encoding configuration
    """
    return enhanced_codec_optimizer.quick_optimize(
        content_type=content_type,
        quality_target=quality_target,
        resolution=resolution
    )

def get_available_encoders():
    """
    Get list of available hardware encoders
    
    Returns:
        List[str]: Available encoder names
    """
    return hardware_detector.get_available_encoders()

def get_system_info():
    """
    Get comprehensive system information
    
    Returns:
        Dict: System and hardware information
    """
    return enhanced_codec_optimizer.get_system_info()

def benchmark_system(test_duration=10.0, resolution=(1920, 1080)):
    """
    Run comprehensive system benchmark
    
    Args:
        test_duration: Duration of benchmark test in seconds
        resolution: Test resolution tuple
        
    Returns:
        BenchmarkSuite: Complete benchmark results
    """
    return enhanced_codec_optimizer.benchmark_current_system(
        test_duration=test_duration,
        resolution=resolution
    )