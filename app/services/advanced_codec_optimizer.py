#!/usr/bin/env python3
"""
Advanced Codec Optimizer for MoneyPrinter Turbo Enhanced
Implements hardware-accelerated encoding with HDR, multi-pass, and streaming optimization

Key Features:
- Hardware encoder detection (NVENC, QSV, VAAPI)
- HDR and wide color gamut processing
- Multi-pass encoding strategies
- Real-time streaming optimization
- Quality assessment coordination
- Performance benchmarking
"""

import subprocess
import multiprocessing
import time
import os
import json
import threading
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from concurrent.futures import ThreadPoolExecutor


class EnhancedCodecOptimizer:
    """Advanced hardware-accelerated codec optimization system"""
    
    def __init__(self):
        self._hw_encoders = {}
        self._codec_capabilities = {}
        self._performance_cache = {}
        self._quality_metrics = {}
        self._streaming_presets = {}
        self._hdr_support = {}
        
        # Initialize hardware detection and capabilities
        self._initialize_hardware_detection()
        self._initialize_codec_capabilities()
        self._initialize_streaming_presets()
        self._initialize_hdr_support()
        
        logger.info(f"Enhanced Codec Optimizer initialized with hardware: {list(self._hw_encoders.keys())}")
    
    def _initialize_hardware_detection(self):
        """Comprehensive hardware encoder detection"""
        logger.info("🔍 Detecting hardware acceleration capabilities...")
        
        # Test NVIDIA NVENC
        self._hw_encoders['nvenc'] = self._test_encoder_capability(
            'h264_nvenc', 'NVIDIA NVENC'
        )
        
        # Test Intel Quick Sync Video (QSV)
        self._hw_encoders['qsv'] = self._test_encoder_capability(
            'h264_qsv', 'Intel Quick Sync Video'
        )
        
        # Test VAAPI (Linux)
        self._hw_encoders['vaapi'] = self._test_encoder_capability(
            'h264_vaapi', 'VAAPI (Linux HW)'
        )
        
        # Test AMD AMF (Windows)
        self._hw_encoders['amf'] = self._test_encoder_capability(
            'h264_amf', 'AMD AMF'
        )
        
        # Test Apple VideoToolbox (macOS)
        self._hw_encoders['videotoolbox'] = self._test_encoder_capability(
            'h264_videotoolbox', 'Apple VideoToolbox'
        )
        
        # Check for HEVC/H.265 support
        if self._hw_encoders.get('nvenc'):
            self._hw_encoders['nvenc_hevc'] = self._test_encoder_capability(
                'hevc_nvenc', 'NVIDIA NVENC HEVC'
            )
        
        if self._hw_encoders.get('qsv'):
            self._hw_encoders['qsv_hevc'] = self._test_encoder_capability(
                'hevc_qsv', 'Intel QSV HEVC'
            )
        
        available_encoders = [k for k, v in self._hw_encoders.items() if v]
        logger.info(f"✅ Hardware encoders detected: {available_encoders}")
    
    def _test_encoder_capability(self, codec: str, name: str) -> bool:
        """Test if a specific encoder is available"""
        try:
            result = subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'quiet',
                '-f', 'lavfi', '-i', 'testsrc=duration=0.1:size=320x240:rate=1',
                '-c:v', codec, '-f', 'null', '-'
            ], capture_output=True, timeout=15)
            
            available = result.returncode == 0
            status = "✅" if available else "❌"
            logger.debug(f"   {status} {name}: {'Available' if available else 'Not available'}")
            return available
            
        except Exception as e:
            logger.debug(f"   ❌ {name}: Error - {str(e)}")
            return False
    
    def _initialize_codec_capabilities(self):
        """Initialize codec-specific capabilities and optimal settings"""
        
        # NVENC capabilities
        if self._hw_encoders.get('nvenc'):
            self._codec_capabilities['nvenc'] = {
                'max_resolution': '4096x4096',
                'max_framerate': 120,
                'supports_b_frames': True,
                'supports_multi_pass': True,
                'supports_vbr': True,
                'supports_cbr': True,
                'presets': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
                'quality_modes': ['cq', 'vbr', 'cbr'],
                'hdr_support': self._hw_encoders.get('nvenc_hevc', False)
            }
        
        # QSV capabilities
        if self._hw_encoders.get('qsv'):
            self._codec_capabilities['qsv'] = {
                'max_resolution': '4096x4096',
                'max_framerate': 60,
                'supports_b_frames': True,
                'supports_multi_pass': True,
                'supports_vbr': True,
                'supports_cbr': True,
                'presets': ['veryfast', 'faster', 'fast', 'medium', 'slow', 'slower'],
                'quality_modes': ['global_quality', 'bitrate'],
                'hdr_support': self._hw_encoders.get('qsv_hevc', False)
            }
        
        # VAAPI capabilities
        if self._hw_encoders.get('vaapi'):
            self._codec_capabilities['vaapi'] = {
                'max_resolution': '3840x2160',
                'max_framerate': 60,
                'supports_b_frames': False,
                'supports_multi_pass': False,
                'supports_vbr': True,
                'supports_cbr': True,
                'presets': [],
                'quality_modes': ['quality', 'bitrate'],
                'hdr_support': False
            }
        
        # Software encoder capabilities (fallback)
        self._codec_capabilities['software'] = {
            'max_resolution': '8192x8192',
            'max_framerate': 60,
            'supports_b_frames': True,
            'supports_multi_pass': True,
            'supports_vbr': True,
            'supports_cbr': True,
            'presets': ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
            'quality_modes': ['crf', 'bitrate'],
            'hdr_support': True  # Software can handle HDR if configured
        }
    
    def _initialize_streaming_presets(self):
        """Initialize real-time streaming optimization presets"""
        
        # Ultra-low latency streaming
        self._streaming_presets['ultra_low_latency'] = {
            'target_latency_ms': 100,
            'buffer_size': 'verysmall',
            'keyframe_interval': 1,
            'b_frames': 0,
            'force_idr': True,
            'tune': 'zerolatency'
        }
        
        # Low latency streaming
        self._streaming_presets['low_latency'] = {
            'target_latency_ms': 500,
            'buffer_size': 'small',
            'keyframe_interval': 2,
            'b_frames': 1,
            'force_idr': False,
            'tune': 'zerolatency'
        }
        
        # Balanced streaming
        self._streaming_presets['balanced_streaming'] = {
            'target_latency_ms': 2000,
            'buffer_size': 'medium',
            'keyframe_interval': 4,
            'b_frames': 2,
            'force_idr': False,
            'tune': 'film'
        }
        
        # Quality streaming
        self._streaming_presets['quality_streaming'] = {
            'target_latency_ms': 5000,
            'buffer_size': 'large',
            'keyframe_interval': 8,
            'b_frames': 3,
            'force_idr': False,
            'tune': 'film'
        }
    
    def _initialize_hdr_support(self):
        """Initialize HDR and wide color gamut support"""
        
        self._hdr_support = {
            'color_spaces': ['bt2020', 'rec2020', 'bt709'],
            'transfer_functions': ['smpte2084', 'arib-std-b67', 'bt709'],
            'color_primaries': ['bt2020', 'bt709', 'dci-p3'],
            'supported_formats': ['hevc', 'av1', 'vp9'],
            'tone_mapping': {
                'methods': ['reinhard', 'hable', 'mobius', 'linear'],
                'peak_luminance': [1000, 4000, 10000],  # nits
                'target_luminance': 100  # nits for SDR displays
            }
        }
    
    def get_optimal_codec_settings(self, 
                                   content_type: str = 'general',
                                   target_quality: str = 'balanced',
                                   streaming_mode: Optional[str] = None,
                                   hdr_mode: bool = False,
                                   resolution: Tuple[int, int] = (1920, 1080),
                                   framerate: int = 30) -> Dict[str, Any]:
        """
        Get optimal codec settings based on comprehensive parameters
        
        Args:
            content_type: 'general', 'high_motion', 'text_heavy', 'animation'
            target_quality: 'speed', 'balanced', 'quality', 'archive'
            streaming_mode: None, 'ultra_low_latency', 'low_latency', 'balanced_streaming', 'quality_streaming'
            hdr_mode: Enable HDR/wide color gamut processing
            resolution: Target resolution (width, height)
            framerate: Target framerate
        
        Returns:
            Dictionary with optimal codec settings
        """
        
        # Select best encoder based on capabilities and requirements
        encoder_type = self._select_optimal_encoder(
            content_type, target_quality, streaming_mode, hdr_mode, resolution, framerate
        )
        
        # Get base settings for selected encoder
        settings = self._get_base_encoder_settings(encoder_type)
        
        # Apply content-specific optimizations
        settings = self._apply_content_optimizations(settings, content_type, encoder_type)
        
        # Apply quality target optimizations
        settings = self._apply_quality_optimizations(settings, target_quality, encoder_type)
        
        # Apply streaming optimizations if needed
        if streaming_mode:
            settings = self._apply_streaming_optimizations(settings, streaming_mode, encoder_type)
        
        # Apply HDR optimizations if needed
        if hdr_mode:
            settings = self._apply_hdr_optimizations(settings, encoder_type)
        
        # Add metadata
        settings.update({
            'encoder_type': encoder_type,
            'content_type': content_type,
            'target_quality': target_quality,
            'streaming_mode': streaming_mode,
            'hdr_mode': hdr_mode,
            'resolution': resolution,
            'framerate': framerate,
            'capabilities': self._codec_capabilities.get(encoder_type, {})
        })
        
        logger.debug(f"Optimal codec settings: {encoder_type} for {content_type}/{target_quality}")
        return settings
    
    def _select_optimal_encoder(self, content_type: str, target_quality: str, 
                               streaming_mode: Optional[str], hdr_mode: bool,
                               resolution: Tuple[int, int], framerate: int) -> str:
        """Select the best available encoder for the given requirements"""
        
        width, height = resolution
        pixel_count = width * height
        
        # HDR requirements
        if hdr_mode:
            if self._hw_encoders.get('nvenc_hevc'):
                return 'nvenc_hevc'
            elif self._hw_encoders.get('qsv_hevc'):
                return 'qsv_hevc'
            else:
                return 'software_hevc'
        
        # Ultra-low latency streaming requirements  
        if streaming_mode == 'ultra_low_latency':
            if self._hw_encoders.get('nvenc'):
                return 'nvenc'  # Best for ultra-low latency
            elif self._hw_encoders.get('qsv'):
                return 'qsv'
            else:
                return 'software'
        
        # High-resolution or high-motion content
        if pixel_count > 2073600 or content_type == 'high_motion' or framerate > 60:  # > 1080p
            if self._hw_encoders.get('nvenc'):
                return 'nvenc'  # Best performance for high-res/high-motion
            elif self._hw_encoders.get('qsv'):
                return 'qsv'
            elif self._hw_encoders.get('vaapi'):
                return 'vaapi'
            else:
                return 'software'
        
        # Quality-focused encoding
        if target_quality == 'archive' or target_quality == 'quality':
            if content_type == 'text_heavy':
                return 'software'  # Better text quality
            elif self._hw_encoders.get('qsv'):
                return 'qsv'  # Good quality/speed balance
            elif self._hw_encoders.get('nvenc'):
                return 'nvenc'
            else:
                return 'software'
        
        # Speed-focused encoding
        if target_quality == 'speed':
            if self._hw_encoders.get('nvenc'):
                return 'nvenc'
            elif self._hw_encoders.get('qsv'):
                return 'qsv'  
            elif self._hw_encoders.get('vaapi'):
                return 'vaapi'
            else:
                return 'software'
        
        # Default balanced selection
        if self._hw_encoders.get('qsv'):
            return 'qsv'  # Generally good balance
        elif self._hw_encoders.get('nvenc'):
            return 'nvenc'
        elif self._hw_encoders.get('vaapi'):
            return 'vaapi'
        else:
            return 'software'
    
    def _get_base_encoder_settings(self, encoder_type: str) -> Dict[str, Any]:
        """Get base settings for a specific encoder type"""
        
        cpu_count = multiprocessing.cpu_count()
        
        if encoder_type == 'nvenc':
            return {
                'codec': 'h264_nvenc',
                'preset': 'p4',
                'cq': '23',
                'b_frames': '2',
                'bf': '2',
                'spatial_aq': '1',
                'temporal_aq': '1',
                'rc': 'vbr',
                'threads': '1'  # NVENC uses minimal CPU threads
            }
        
        elif encoder_type == 'nvenc_hevc':
            return {
                'codec': 'hevc_nvenc',
                'preset': 'p4',
                'cq': '25',
                'b_frames': '2',
                'bf': '2',
                'spatial_aq': '1',
                'temporal_aq': '1',
                'rc': 'vbr',
                'threads': '1'
            }
        
        elif encoder_type == 'qsv':
            return {
                'codec': 'h264_qsv',
                'preset': 'balanced',
                'global_quality': '23',
                'b_frames': '2',
                'bf': '2',
                'look_ahead': '1',
                'threads': '1'
            }
        
        elif encoder_type == 'qsv_hevc':
            return {
                'codec': 'hevc_qsv',
                'preset': 'balanced',
                'global_quality': '25',
                'b_frames': '2',
                'bf': '2',
                'look_ahead': '1',
                'threads': '1'
            }
        
        elif encoder_type == 'vaapi':
            return {
                'codec': 'h264_vaapi',
                'quality': '23',
                'threads': '1'
            }
        
        elif encoder_type == 'software_hevc':
            return {
                'codec': 'libx265',
                'preset': 'fast',
                'crf': '25',
                'threads': str(min(cpu_count, 8))
            }
        
        else:  # software
            return {
                'codec': 'libx264',
                'preset': 'fast',
                'crf': '23',
                'threads': str(min(cpu_count, 8))
            }
    
    def _apply_content_optimizations(self, settings: Dict[str, Any], 
                                   content_type: str, encoder_type: str) -> Dict[str, Any]:
        """Apply content-specific optimizations"""
        
        if content_type == 'high_motion':
            # High motion content (games, sports, action)
            if encoder_type == 'nvenc':
                settings.update({
                    'preset': 'p1',  # Fastest
                    'temporal_aq': '1',
                    'spatial_aq': '1'
                })
            elif encoder_type == 'qsv':
                settings.update({
                    'preset': 'veryfast',
                    'look_ahead': '0'  # Disable for speed
                })
            elif encoder_type == 'software':
                settings.update({
                    'preset': 'veryfast',
                    'tune': 'grain'
                })
        
        elif content_type == 'text_heavy':
            # Text and graphics content
            if encoder_type == 'nvenc':
                settings.update({
                    'preset': 'p6',  # Higher quality
                    'cq': '20'
                })
            elif encoder_type == 'qsv':
                settings.update({
                    'preset': 'slower',
                    'global_quality': '20'
                })
            elif encoder_type == 'software':
                settings.update({
                    'preset': 'slow',
                    'tune': 'stillimage',
                    'crf': '20'
                })
        
        elif content_type == 'animation':
            # Animated content
            if encoder_type == 'software':
                settings.update({
                    'tune': 'animation'
                })
        
        return settings
    
    def _apply_quality_optimizations(self, settings: Dict[str, Any], 
                                   target_quality: str, encoder_type: str) -> Dict[str, Any]:
        """Apply quality target optimizations"""
        
        if target_quality == 'speed':
            if encoder_type == 'nvenc':
                settings.update({
                    'preset': 'p1',
                    'cq': '25'
                })
            elif encoder_type == 'qsv':
                settings.update({
                    'preset': 'veryfast',
                    'global_quality': '25'
                })
            elif encoder_type == 'software':
                settings.update({
                    'preset': 'ultrafast',
                    'crf': '25'
                })
        
        elif target_quality == 'quality':
            if encoder_type == 'nvenc':
                settings.update({
                    'preset': 'p6',
                    'cq': '20'
                })
            elif encoder_type == 'qsv':
                settings.update({
                    'preset': 'slower',
                    'global_quality': '20'
                })
            elif encoder_type == 'software':
                settings.update({
                    'preset': 'slow',
                    'crf': '20'
                })
        
        elif target_quality == 'archive':
            if encoder_type == 'nvenc':
                settings.update({
                    'preset': 'p7',
                    'cq': '18'
                })
            elif encoder_type == 'qsv':
                settings.update({
                    'preset': 'veryslow',
                    'global_quality': '18'
                })
            elif encoder_type == 'software':
                settings.update({
                    'preset': 'veryslow',
                    'crf': '18'
                })
        
        return settings
    
    def _apply_streaming_optimizations(self, settings: Dict[str, Any], 
                                     streaming_mode: str, encoder_type: str) -> Dict[str, Any]:
        """Apply streaming-specific optimizations"""
        
        preset = self._streaming_presets.get(streaming_mode, {})
        
        if streaming_mode == 'ultra_low_latency':
            if encoder_type == 'nvenc':
                settings.update({
                    'preset': 'p1',
                    'tune': 'ull',  # Ultra-low latency
                    'delay': '0',
                    'rc': 'cbr',
                    'b_frames': '0',
                    'bf': '0'
                })
            elif encoder_type == 'qsv':
                settings.update({
                    'preset': 'veryfast',
                    'b_frames': '0',
                    'bf': '0'
                })
            elif encoder_type == 'software':
                settings.update({
                    'preset': 'ultrafast',
                    'tune': 'zerolatency',
                    'b_frames': '0'
                })
        
        elif streaming_mode == 'low_latency':
            if encoder_type == 'nvenc':
                settings.update({
                    'preset': 'p2',
                    'tune': 'll',  # Low latency
                    'delay': '0',
                    'b_frames': '1',
                    'bf': '1'
                })
            elif encoder_type == 'software':
                settings.update({
                    'tune': 'zerolatency',
                    'b_frames': '1'
                })
        
        # Add keyframe interval settings
        keyframe_interval = preset.get('keyframe_interval', 4)
        settings['keyint_min'] = str(keyframe_interval)
        settings['g'] = str(keyframe_interval * 30)  # GOP size
        
        return settings
    
    def _apply_hdr_optimizations(self, settings: Dict[str, Any], encoder_type: str) -> Dict[str, Any]:
        """Apply HDR and wide color gamut optimizations"""
        
        # HDR color space and transfer function settings
        hdr_settings = {
            'colorspace': 'bt2020nc',
            'color_primaries': 'bt2020',
            'color_trc': 'smpte2084',  # PQ (Perceptual Quantizer)
            'color_range': 'tv'
        }
        
        if encoder_type in ['nvenc_hevc', 'qsv_hevc', 'software_hevc']:
            # HEVC HDR settings
            hdr_settings.update({
                'x265-params': 'hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc'
            })
        
        settings.update(hdr_settings)
        return settings
    
    def setup_multi_pass_encoding(self, settings: Dict[str, Any], 
                                 passes: int = 2) -> List[Dict[str, Any]]:
        """Setup multi-pass encoding configuration"""
        
        if passes < 2:
            return [settings]
        
        encoder_type = settings.get('encoder_type', 'software')
        
        # Check if encoder supports multi-pass
        capabilities = self._codec_capabilities.get(encoder_type, {})
        if not capabilities.get('supports_multi_pass', False):
            logger.warning(f"Multi-pass encoding not supported by {encoder_type}, using single pass")
            return [settings]
        
        pass_configs = []
        
        for pass_num in range(1, passes + 1):
            pass_settings = settings.copy()
            
            if encoder_type == 'software':
                if pass_num == 1:
                    # First pass: analysis only
                    pass_settings.update({
                        'pass': '1',
                        'passlogfile': 'video_pass',
                        'an': None,  # No audio in first pass
                        'f': 'null'  # No output file
                    })
                else:
                    # Second pass: final encoding
                    pass_settings.update({
                        'pass': '2',
                        'passlogfile': 'video_pass'
                    })
            
            elif encoder_type == 'nvenc':
                # NVENC multi-pass
                pass_settings.update({
                    'multipass': 'fullres' if passes == 2 else 'qres'
                })
            
            elif encoder_type == 'qsv':
                # QSV multi-pass
                pass_settings.update({
                    'extbrc': '1',
                    'look_ahead': '1'
                })
            
            pass_configs.append(pass_settings)
        
        return pass_configs
    
    def optimize_for_streaming_platform(self, settings: Dict[str, Any], 
                                       platform: str) -> Dict[str, Any]:
        """Optimize settings for specific streaming platforms"""
        
        platform_configs = {
            'youtube': {
                'max_bitrate': '50000k',
                'bufsize': '75000k',
                'keyint_min': '4',
                'g': '120',  # 4 second GOP at 30fps
                'profile': 'high',
                'level': '4.2'
            },
            'twitch': {
                'max_bitrate': '6000k',
                'bufsize': '12000k',
                'keyint_min': '2',
                'g': '60',  # 2 second GOP
                'profile': 'main',
                'level': '4.1'
            },
            'facebook': {
                'max_bitrate': '4000k',
                'bufsize': '8000k',
                'keyint_min': '2',
                'g': '60',
                'profile': 'main'
            },
            'twitter': {
                'max_bitrate': '5000k',
                'bufsize': '10000k',
                'keyint_min': '4',
                'g': '120',
                'profile': 'main'
            }
        }
        
        platform_settings = platform_configs.get(platform.lower(), {})
        if platform_settings:
            settings.update(platform_settings)
            logger.info(f"Applied {platform} streaming optimizations")
        
        return settings
    
    def benchmark_encoder_performance(self, test_duration: int = 10) -> Dict[str, Any]:
        """Benchmark available encoders for performance comparison"""
        
        logger.info("🏃 Running encoder performance benchmark...")
        
        # Create test video
        test_input = Path("test_benchmark_input.mp4")
        self._create_test_video(test_input, test_duration)
        
        results = {}
        
        # Test each available encoder
        for encoder_type in ['software', 'nvenc', 'qsv', 'vaapi']:
            if encoder_type == 'software' or self._hw_encoders.get(encoder_type):
                
                settings = self.get_optimal_codec_settings(
                    target_quality='balanced',
                    content_type='general'
                )
                
                # Override encoder type for testing
                settings['encoder_type'] = encoder_type
                settings = self._get_base_encoder_settings(encoder_type)
                
                result = self._benchmark_single_encoder(test_input, settings, encoder_type)
                results[encoder_type] = result
        
        # Cleanup
        if test_input.exists():
            test_input.unlink()
        
        # Analyze results
        self._analyze_benchmark_results(results)
        
        return results
    
    def _create_test_video(self, output_path: Path, duration: int = 10):
        """Create a test video for benchmarking"""
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'quiet',
            '-f', 'lavfi',
            '-i', f'testsrc=duration={duration}:size=1280x720:rate=30',
            '-f', 'lavfi',
            '-i', f'sine=frequency=1000:duration={duration}',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac',
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True)
    
    def _benchmark_single_encoder(self, input_path: Path, settings: Dict[str, Any], 
                                 encoder_type: str) -> Dict[str, Any]:
        """Benchmark a single encoder configuration"""
        
        output_path = Path(f"test_benchmark_{encoder_type}.mp4")
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y', '-hide_banner', '-i', str(input_path)]
        
        # Add codec settings
        cmd.extend(['-c:v', settings['codec']])
        
        # Add encoder-specific settings
        if encoder_type == 'software':
            cmd.extend(['-preset', settings.get('preset', 'fast')])
            cmd.extend(['-crf', settings.get('crf', '23')])
        elif encoder_type == 'nvenc':
            cmd.extend(['-preset', settings.get('preset', 'p4')])
            cmd.extend(['-cq', settings.get('cq', '23')])
        elif encoder_type == 'qsv':
            cmd.extend(['-preset', settings.get('preset', 'balanced')])
            cmd.extend(['-global_quality', settings.get('global_quality', '23')])
        elif encoder_type == 'vaapi':
            cmd.extend(['-quality', settings.get('quality', '23')])
        
        cmd.extend(['-c:a', 'copy', str(output_path)])
        
        # Measure encoding time
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            encoding_time = time.time() - start_time
            
            if result.returncode == 0 and output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                
                # Cleanup
                output_path.unlink()
                
                return {
                    'success': True,
                    'encoding_time': encoding_time,
                    'file_size_mb': file_size,
                    'fps': 30 * 10 / encoding_time,  # 30fps * 10s duration
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'encoding_time': float('inf'),
                    'file_size_mb': 0,
                    'fps': 0,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'encoding_time': float('inf'),
                'file_size_mb': 0,
                'fps': 0,
                'error': 'Timeout'
            }
    
    def _analyze_benchmark_results(self, results: Dict[str, Any]):
        """Analyze and log benchmark results"""
        
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if not successful_results:
            logger.warning("No successful benchmark results")
            return
        
        # Find best performers
        fastest = min(successful_results.items(), key=lambda x: x[1]['encoding_time'])
        most_efficient = max(successful_results.items(), key=lambda x: x[1]['fps'])
        
        logger.info("📊 Benchmark Results:")
        for encoder, result in successful_results.items():
            speedup = fastest[1]['encoding_time'] / result['encoding_time'] if result['encoding_time'] > 0 else 0
            logger.info(f"  {encoder}: {result['encoding_time']:.2f}s, {result['fps']:.1f} fps, {result['file_size_mb']:.1f}MB")
        
        logger.info(f"🏆 Fastest: {fastest[0]} ({fastest[1]['encoding_time']:.2f}s)")
        logger.info(f"🎯 Most Efficient: {most_efficient[0]} ({most_efficient[1]['fps']:.1f} fps)")
    
    def get_quality_assessment_hooks(self) -> Dict[str, callable]:
        """Return hooks for quality assessment coordination"""
        
        def pre_encode_hook(settings: Dict[str, Any]) -> Dict[str, Any]:
            """Hook called before encoding starts"""
            logger.debug(f"Pre-encode: Using {settings.get('encoder_type')} encoder")
            return settings
        
        def post_encode_hook(settings: Dict[str, Any], output_path: str, 
                           encoding_time: float) -> Dict[str, Any]:
            """Hook called after encoding completes"""
            
            # Store performance metrics
            metrics = {
                'encoder_type': settings.get('encoder_type'),
                'encoding_time': encoding_time,
                'output_size': os.path.getsize(output_path) if os.path.exists(output_path) else 0,
                'timestamp': time.time()
            }
            
            # Cache performance data
            cache_key = f"{settings.get('encoder_type')}_{settings.get('target_quality')}"
            self._performance_cache[cache_key] = metrics
            
            logger.debug(f"Post-encode: {settings.get('encoder_type')} completed in {encoding_time:.2f}s")
            
            return metrics
        
        return {
            'pre_encode': pre_encode_hook,
            'post_encode': post_encode_hook
        }


# Global instance
enhanced_codec_optimizer = EnhancedCodecOptimizer()