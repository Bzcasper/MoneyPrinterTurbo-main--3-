#!/usr/bin/env python3
"""
Hardware Encoder Detection Module for MoneyPrinter Turbo Enhanced
Provides comprehensive hardware acceleration capability detection and validation.

Key Features:
- Cross-platform encoder detection (NVENC, QSV, VAAPI, AMF, VideoToolbox)
- Performance validation and capability testing
- Detailed encoder feature support mapping
- Error handling and graceful fallbacks
"""

import subprocess
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EncoderType(Enum):
    """Supported hardware encoder types"""
    NVENC = "nvenc"
    QSV = "qsv"
    VAAPI = "vaapi"
    AMF = "amf"
    VIDEOTOOLBOX = "videotoolbox"
    SOFTWARE = "software"


@dataclass
class EncoderCapabilities:
    """Encoder capability configuration"""
    max_resolution: str
    max_framerate: int
    supports_b_frames: bool
    supports_multi_pass: bool
    supports_vbr: bool
    supports_cbr: bool
    presets: List[str]
    quality_modes: List[str]
    hdr_support: bool


class HardwareDetector:
    """Hardware encoder detection and capability management"""
    
    def __init__(self):
        self._available_encoders: Dict[str, bool] = {}
        self._encoder_capabilities: Dict[str, EncoderCapabilities] = {}
        self._detection_cache: Dict[str, bool] = {}
        
        # Initialize detection
        self._detect_all_encoders()
        self._initialize_capabilities()
        
        logger.info(f"Hardware detection complete. Available encoders: {self.get_available_encoders()}")
    
    def _detect_all_encoders(self) -> None:
        """Detect all available hardware encoders"""
        
        # Test NVIDIA NVENC (H.264)
        self._available_encoders['nvenc'] = self._test_encoder_capability(
            'h264_nvenc', 'NVIDIA NVENC H.264'
        )
        
        # Test NVIDIA NVENC (HEVC)
        self._available_encoders['nvenc_hevc'] = self._test_encoder_capability(
            'hevc_nvenc', 'NVIDIA NVENC HEVC'
        )
        
        # Test Intel Quick Sync Video (H.264)
        self._available_encoders['qsv'] = self._test_encoder_capability(
            'h264_qsv', 'Intel Quick Sync Video H.264'
        )
        
        # Test Intel Quick Sync Video (HEVC)
        self._available_encoders['qsv_hevc'] = self._test_encoder_capability(
            'hevc_qsv', 'Intel QSV HEVC'
        )
        
        # Test VAAPI (Linux hardware acceleration)
        self._available_encoders['vaapi'] = self._test_encoder_capability(
            'h264_vaapi', 'VAAPI Hardware Acceleration'
        )
        
        # Test AMD AMF (Windows)
        self._available_encoders['amf'] = self._test_encoder_capability(
            'h264_amf', 'AMD AMF'
        )
        
        # Test Apple VideoToolbox (macOS)
        self._available_encoders['videotoolbox'] = self._test_encoder_capability(
            'h264_videotoolbox', 'Apple VideoToolbox'
        )
        
        # Software encoders are always available
        self._available_encoders['software'] = True
        self._available_encoders['software_hevc'] = True
    
    def _test_encoder_capability(self, codec: str, name: str) -> bool:
        """
        Test if a specific encoder is available and functional
        
        Args:
            codec: FFmpeg codec name
            name: Human-readable encoder name
            
        Returns:
            True if encoder is available and working
        """
        # Check cache first
        if codec in self._detection_cache:
            return self._detection_cache[codec]
        
        try:
            # Create a minimal test encoding
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'quiet',
                '-f', 'lavfi', '-i', 'testsrc=duration=0.1:size=320x240:rate=1',
                '-c:v', codec, '-f', 'null', '-'
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                timeout=15,
                check=False
            )
            
            available = result.returncode == 0
            self._detection_cache[codec] = available
            
            status = "✅" if available else "❌"
            logger.debug(f"   {status} {name}: {'Available' if available else 'Not available'}")
            
            return available
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
            logger.debug(f"   ❌ {name}: Error during detection - {str(e)}")
            self._detection_cache[codec] = False
            return False
        except Exception as e:
            logger.warning(f"Unexpected error testing {name}: {str(e)}")
            self._detection_cache[codec] = False
            return False
    
    def _initialize_capabilities(self) -> None:
        """Initialize encoder-specific capabilities"""
        
        # NVIDIA NVENC capabilities
        if self._available_encoders.get('nvenc'):
            self._encoder_capabilities['nvenc'] = EncoderCapabilities(
                max_resolution='4096x4096',
                max_framerate=120,
                supports_b_frames=True,
                supports_multi_pass=True,
                supports_vbr=True,
                supports_cbr=True,
                presets=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
                quality_modes=['cq', 'vbr', 'cbr'],
                hdr_support=self._available_encoders.get('nvenc_hevc', False)
            )
        
        # Intel QSV capabilities
        if self._available_encoders.get('qsv'):
            self._encoder_capabilities['qsv'] = EncoderCapabilities(
                max_resolution='4096x4096',
                max_framerate=60,
                supports_b_frames=True,
                supports_multi_pass=True,
                supports_vbr=True,
                supports_cbr=True,
                presets=['veryfast', 'faster', 'fast', 'medium', 'slow', 'slower'],
                quality_modes=['global_quality', 'bitrate'],
                hdr_support=self._available_encoders.get('qsv_hevc', False)
            )
        
        # VAAPI capabilities
        if self._available_encoders.get('vaapi'):
            self._encoder_capabilities['vaapi'] = EncoderCapabilities(
                max_resolution='3840x2160',
                max_framerate=60,
                supports_b_frames=False,
                supports_multi_pass=False,
                supports_vbr=True,
                supports_cbr=True,
                presets=[],
                quality_modes=['quality', 'bitrate'],
                hdr_support=False
            )
        
        # AMD AMF capabilities
        if self._available_encoders.get('amf'):
            self._encoder_capabilities['amf'] = EncoderCapabilities(
                max_resolution='3840x2160',
                max_framerate=60,
                supports_b_frames=True,
                supports_multi_pass=False,
                supports_vbr=True,
                supports_cbr=True,
                presets=['speed', 'balanced', 'quality'],
                quality_modes=['cqp', 'cbr', 'vbr'],
                hdr_support=False
            )
        
        # Software encoder capabilities (always available)
        self._encoder_capabilities['software'] = EncoderCapabilities(
            max_resolution='8192x8192',
            max_framerate=60,
            supports_b_frames=True,
            supports_multi_pass=True,
            supports_vbr=True,
            supports_cbr=True,
            presets=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
            quality_modes=['crf', 'bitrate'],
            hdr_support=True
        )
    
    def get_available_encoders(self) -> List[str]:
        """
        Get list of available encoder types
        
        Returns:
            List of available encoder names
        """
        return [name for name, available in self._available_encoders.items() if available]
    
    def is_encoder_available(self, encoder_type: str) -> bool:
        """
        Check if a specific encoder type is available
        
        Args:
            encoder_type: Encoder type name
            
        Returns:
            True if encoder is available
        """
        return self._available_encoders.get(encoder_type, False)
    
    def get_encoder_capabilities(self, encoder_type: str) -> Optional[EncoderCapabilities]:
        """
        Get capabilities for a specific encoder
        
        Args:
            encoder_type: Encoder type name
            
        Returns:
            EncoderCapabilities object or None if encoder not available
        """
        return self._encoder_capabilities.get(encoder_type)
    
    def get_best_encoder_for_use_case(self, 
                                     hdr_required: bool = False,
                                     high_performance: bool = False,
                                     quality_priority: bool = False) -> str:
        """
        Get the best available encoder for specific use case
        
        Args:
            hdr_required: Whether HDR support is required
            high_performance: Whether to prioritize encoding speed
            quality_priority: Whether to prioritize encoding quality
            
        Returns:
            Best encoder type name
        """
        available = self.get_available_encoders()
        
        # HDR requirement filters
        if hdr_required:
            hdr_encoders = [enc for enc in available 
                           if self._encoder_capabilities.get(enc, EncoderCapabilities(
                               '', 0, False, False, False, False, [], [], False
                           )).hdr_support]
            if hdr_encoders:
                available = hdr_encoders
        
        # Performance priority
        if high_performance:
            priority_order = ['nvenc', 'qsv', 'amf', 'vaapi', 'software']
        # Quality priority
        elif quality_priority:
            priority_order = ['software', 'qsv', 'nvenc', 'vaapi', 'amf']
        # Balanced approach
        else:
            priority_order = ['qsv', 'nvenc', 'vaapi', 'amf', 'software']
        
        # Find first available encoder in priority order
        for encoder in priority_order:
            if encoder in available:
                return encoder
        
        # Fallback to software
        return 'software'
    
    def validate_encoder_for_settings(self, 
                                    encoder_type: str,
                                    resolution: tuple,
                                    framerate: int,
                                    use_b_frames: bool = False) -> bool:
        """
        Validate if encoder can handle specific settings
        
        Args:
            encoder_type: Encoder type to validate
            resolution: Target resolution (width, height)
            framerate: Target framerate
            use_b_frames: Whether B-frames are required
            
        Returns:
            True if encoder can handle the settings
        """
        if not self.is_encoder_available(encoder_type):
            return False
        
        capabilities = self.get_encoder_capabilities(encoder_type)
        if not capabilities:
            return False
        
        # Check resolution limits
        width, height = resolution
        max_width, max_height = map(int, capabilities.max_resolution.split('x'))
        if width > max_width or height > max_height:
            return False
        
        # Check framerate limits
        if framerate > capabilities.max_framerate:
            return False
        
        # Check B-frame support
        if use_b_frames and not capabilities.supports_b_frames:
            return False
        
        return True
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive hardware detection summary
        
        Returns:
            Dictionary with detection results and capabilities
        """
        return {
            'available_encoders': self.get_available_encoders(),
            'encoder_details': {
                name: {
                    'available': available,
                    'capabilities': self._encoder_capabilities.get(name).__dict__ 
                    if self._encoder_capabilities.get(name) else None
                }
                for name, available in self._available_encoders.items()
            },
            'recommended_encoder': self.get_best_encoder_for_use_case(),
            'hdr_capable_encoders': [
                name for name, caps in self._encoder_capabilities.items()
                if caps.hdr_support
            ]
        }


# Global instance for singleton access
hardware_detector = HardwareDetector()