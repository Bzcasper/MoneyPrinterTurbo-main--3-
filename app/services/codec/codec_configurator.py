#!/usr/bin/env python3
"""
Codec Configuration Module for MoneyPrinter Turbo Enhanced
Manages encoder settings, optimizations, and configuration generation.

Key Features:
- Content-specific optimizations (high motion, text, animation)
- Quality target configurations (speed, balanced, quality, archive)
- Multi-pass encoding setup
- Platform-specific optimizations
- HDR and streaming configurations
"""

import multiprocessing
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .hardware_detector import HardwareDetector, EncoderCapabilities

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type classifications for optimization"""
    GENERAL = "general"
    HIGH_MOTION = "high_motion"
    TEXT_HEAVY = "text_heavy"
    ANIMATION = "animation"


class QualityTarget(Enum):
    """Quality vs speed target configurations"""
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"
    ARCHIVE = "archive"


class StreamingMode(Enum):
    """Streaming optimization modes"""
    ULTRA_LOW_LATENCY = "ultra_low_latency"
    LOW_LATENCY = "low_latency"
    BALANCED_STREAMING = "balanced_streaming"
    QUALITY_STREAMING = "quality_streaming"


@dataclass
class EncodingConfig:
    """Complete encoding configuration"""
    encoder_type: str
    codec: str
    settings: Dict[str, str]
    content_type: ContentType
    quality_target: QualityTarget
    streaming_mode: Optional[StreamingMode]
    hdr_mode: bool
    resolution: Tuple[int, int]
    framerate: int
    capabilities: Optional[EncoderCapabilities]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'encoder_type': self.encoder_type,
            'codec': self.codec,
            'settings': self.settings,
            'content_type': self.content_type.value,
            'quality_target': self.quality_target.value,
            'streaming_mode': self.streaming_mode.value if self.streaming_mode else None,
            'hdr_mode': self.hdr_mode,
            'resolution': self.resolution,
            'framerate': self.framerate,
            'capabilities': self.capabilities.__dict__ if self.capabilities else None
        }


class CodecConfigurator:
    """Codec configuration and optimization manager"""
    
    def __init__(self, hardware_detector: Optional[HardwareDetector] = None):
        self.hardware_detector = hardware_detector
        self._streaming_presets = self._initialize_streaming_presets()
        self._hdr_settings = self._initialize_hdr_settings()
        
        # CPU count for threading optimization
        self._cpu_count = multiprocessing.cpu_count()
        
        logger.info("Codec configurator initialized")
    
    def _initialize_streaming_presets(self) -> Dict[str, Dict[str, Any]]:
        """Initialize streaming optimization presets"""
        return {
            StreamingMode.ULTRA_LOW_LATENCY.value: {
                'target_latency_ms': 100,
                'buffer_size': 'verysmall',
                'keyframe_interval': 1,
                'b_frames': 0,
                'force_idr': True,
                'tune': 'zerolatency'
            },
            StreamingMode.LOW_LATENCY.value: {
                'target_latency_ms': 500,
                'buffer_size': 'small',
                'keyframe_interval': 2,
                'b_frames': 1,
                'force_idr': False,
                'tune': 'zerolatency'
            },
            StreamingMode.BALANCED_STREAMING.value: {
                'target_latency_ms': 2000,
                'buffer_size': 'medium',
                'keyframe_interval': 4,
                'b_frames': 2,
                'force_idr': False,
                'tune': 'film'
            },
            StreamingMode.QUALITY_STREAMING.value: {
                'target_latency_ms': 5000,
                'buffer_size': 'large',
                'keyframe_interval': 8,
                'b_frames': 3,
                'force_idr': False,
                'tune': 'film'
            }
        }
    
    def _initialize_hdr_settings(self) -> Dict[str, Any]:
        """Initialize HDR and wide color gamut settings"""
        return {
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
    
    def generate_encoding_config(self,
                                content_type: ContentType = ContentType.GENERAL,
                                quality_target: QualityTarget = QualityTarget.BALANCED,
                                streaming_mode: Optional[StreamingMode] = None,
                                hdr_mode: bool = False,
                                resolution: Tuple[int, int] = (1920, 1080),
                                framerate: int = 30,
                                encoder_override: Optional[str] = None) -> EncodingConfig:
        """
        Generate comprehensive encoding configuration
        
        Args:
            content_type: Type of content being encoded
            quality_target: Quality vs speed preference
            streaming_mode: Streaming optimization mode
            hdr_mode: Enable HDR processing
            resolution: Target resolution
            framerate: Target framerate
            encoder_override: Force specific encoder type
            
        Returns:
            Complete encoding configuration
        """
        # Select optimal encoder
        if encoder_override:
            encoder_type = encoder_override
        else:
            encoder_type = self._select_optimal_encoder(
                content_type, quality_target, streaming_mode, hdr_mode, resolution, framerate
            )
        
        # Get base encoder settings
        base_settings = self._get_base_encoder_settings(encoder_type)
        
        # Apply optimizations
        settings = self._apply_content_optimizations(
            base_settings.copy(), content_type, encoder_type
        )
        settings = self._apply_quality_optimizations(
            settings, quality_target, encoder_type
        )
        
        if streaming_mode:
            settings = self._apply_streaming_optimizations(
                settings, streaming_mode, encoder_type
            )
        
        if hdr_mode:
            settings = self._apply_hdr_optimizations(settings, encoder_type)
        
        # Get capabilities
        capabilities = None
        if self.hardware_detector:
            capabilities = self.hardware_detector.get_encoder_capabilities(encoder_type)
        
        return EncodingConfig(
            encoder_type=encoder_type,
            codec=settings.get('codec', 'libx264'),
            settings=settings,
            content_type=content_type,
            quality_target=quality_target,
            streaming_mode=streaming_mode,
            hdr_mode=hdr_mode,
            resolution=resolution,
            framerate=framerate,
            capabilities=capabilities
        )
    
    def _select_optimal_encoder(self,
                               content_type: ContentType,
                               quality_target: QualityTarget,
                               streaming_mode: Optional[StreamingMode],
                               hdr_mode: bool,
                               resolution: Tuple[int, int],
                               framerate: int) -> str:
        """Select optimal encoder based on requirements"""
        
        if not self.hardware_detector:
            return 'software'
        
        width, height = resolution
        pixel_count = width * height
        
        # HDR requirements
        if hdr_mode:
            if self.hardware_detector.is_encoder_available('nvenc_hevc'):
                return 'nvenc_hevc'
            elif self.hardware_detector.is_encoder_available('qsv_hevc'):
                return 'qsv_hevc'
            else:
                return 'software_hevc'
        
        # Ultra-low latency streaming
        if streaming_mode == StreamingMode.ULTRA_LOW_LATENCY:
            if self.hardware_detector.is_encoder_available('nvenc'):
                return 'nvenc'
            elif self.hardware_detector.is_encoder_available('qsv'):
                return 'qsv'
            else:
                return 'software'
        
        # High-resolution or high-motion content
        if pixel_count > 2073600 or content_type == ContentType.HIGH_MOTION or framerate > 60:
            if self.hardware_detector.is_encoder_available('nvenc'):
                return 'nvenc'
            elif self.hardware_detector.is_encoder_available('qsv'):
                return 'qsv'
            elif self.hardware_detector.is_encoder_available('vaapi'):
                return 'vaapi'
            else:
                return 'software'
        
        # Quality-focused encoding
        if quality_target in [QualityTarget.ARCHIVE, QualityTarget.QUALITY]:
            if content_type == ContentType.TEXT_HEAVY:
                return 'software'  # Better text quality
            elif self.hardware_detector.is_encoder_available('qsv'):
                return 'qsv'
            elif self.hardware_detector.is_encoder_available('nvenc'):
                return 'nvenc'
            else:
                return 'software'
        
        # Speed-focused encoding
        if quality_target == QualityTarget.SPEED:
            if self.hardware_detector.is_encoder_available('nvenc'):
                return 'nvenc'
            elif self.hardware_detector.is_encoder_available('qsv'):
                return 'qsv'
            elif self.hardware_detector.is_encoder_available('vaapi'):
                return 'vaapi'
            else:
                return 'software'
        
        # Default balanced selection
        return self.hardware_detector.get_best_encoder_for_use_case()
    
    def _get_base_encoder_settings(self, encoder_type: str) -> Dict[str, str]:
        """Get base settings for encoder type"""
        
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
                'threads': '1'
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
                'threads': str(min(self._cpu_count, 8))
            }
        
        else:  # software
            return {
                'codec': 'libx264',
                'preset': 'fast',
                'crf': '23',
                'threads': str(min(self._cpu_count, 8))
            }
    
    def _apply_content_optimizations(self,
                                   settings: Dict[str, str],
                                   content_type: ContentType,
                                   encoder_type: str) -> Dict[str, str]:
        """Apply content-specific optimizations"""
        
        if content_type == ContentType.HIGH_MOTION:
            if encoder_type == 'nvenc':
                settings.update({
                    'preset': 'p1',
                    'temporal_aq': '1',
                    'spatial_aq': '1'
                })
            elif encoder_type == 'qsv':
                settings.update({
                    'preset': 'veryfast',
                    'look_ahead': '0'
                })
            elif encoder_type == 'software':
                settings.update({
                    'preset': 'veryfast',
                    'tune': 'grain'
                })
        
        elif content_type == ContentType.TEXT_HEAVY:
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
                    'tune': 'stillimage',
                    'crf': '20'
                })
        
        elif content_type == ContentType.ANIMATION:
            if encoder_type == 'software':
                settings['tune'] = 'animation'
        
        return settings
    
    def _apply_quality_optimizations(self,
                                   settings: Dict[str, str],
                                   quality_target: QualityTarget,
                                   encoder_type: str) -> Dict[str, str]:
        """Apply quality target optimizations"""
        
        if quality_target == QualityTarget.SPEED:
            if encoder_type == 'nvenc':
                settings.update({'preset': 'p1', 'cq': '25'})
            elif encoder_type == 'qsv':
                settings.update({'preset': 'veryfast', 'global_quality': '25'})
            elif encoder_type == 'software':
                settings.update({'preset': 'ultrafast', 'crf': '25'})
        
        elif quality_target == QualityTarget.QUALITY:
            if encoder_type == 'nvenc':
                settings.update({'preset': 'p6', 'cq': '20'})
            elif encoder_type == 'qsv':
                settings.update({'preset': 'slower', 'global_quality': '20'})
            elif encoder_type == 'software':
                settings.update({'preset': 'slow', 'crf': '20'})
        
        elif quality_target == QualityTarget.ARCHIVE:
            if encoder_type == 'nvenc':
                settings.update({'preset': 'p7', 'cq': '18'})
            elif encoder_type == 'qsv':
                settings.update({'preset': 'veryslow', 'global_quality': '18'})
            elif encoder_type == 'software':
                settings.update({'preset': 'veryslow', 'crf': '18'})
        
        return settings
    
    def _apply_streaming_optimizations(self,
                                     settings: Dict[str, str],
                                     streaming_mode: StreamingMode,
                                     encoder_type: str) -> Dict[str, str]:
        """Apply streaming-specific optimizations"""
        
        preset = self._streaming_presets.get(streaming_mode.value, {})
        
        if streaming_mode == StreamingMode.ULTRA_LOW_LATENCY:
            if encoder_type == 'nvenc':
                settings.update({
                    'preset': 'p1',
                    'tune': 'ull',
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
        
        elif streaming_mode == StreamingMode.LOW_LATENCY:
            if encoder_type == 'nvenc':
                settings.update({
                    'preset': 'p2',
                    'tune': 'll',
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
    
    def _apply_hdr_optimizations(self,
                               settings: Dict[str, str],
                               encoder_type: str) -> Dict[str, str]:
        """Apply HDR and wide color gamut optimizations"""
        
        hdr_settings = {
            'colorspace': 'bt2020nc',
            'color_primaries': 'bt2020',
            'color_trc': 'smpte2084',
            'color_range': 'tv'
        }
        
        if encoder_type in ['nvenc_hevc', 'qsv_hevc', 'software_hevc']:
            hdr_settings['x265-params'] = 'hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc'
        
        settings.update(hdr_settings)
        return settings
    
    def setup_multi_pass_encoding(self,
                                 config: EncodingConfig,
                                 passes: int = 2) -> List[EncodingConfig]:
        """Setup multi-pass encoding configuration"""
        
        if passes < 2:
            return [config]
        
        # Check if encoder supports multi-pass
        if config.capabilities and not config.capabilities.supports_multi_pass:
            logger.warning(f"Multi-pass encoding not supported by {config.encoder_type}")
            return [config]
        
        pass_configs = []
        
        for pass_num in range(1, passes + 1):
            pass_config = EncodingConfig(
                encoder_type=config.encoder_type,
                codec=config.codec,
                settings=config.settings.copy(),
                content_type=config.content_type,
                quality_target=config.quality_target,
                streaming_mode=config.streaming_mode,
                hdr_mode=config.hdr_mode,
                resolution=config.resolution,
                framerate=config.framerate,
                capabilities=config.capabilities
            )
            
            if config.encoder_type == 'software':
                if pass_num == 1:
                    pass_config.settings.update({
                        'pass': '1',
                        'passlogfile': 'video_pass',
                        'an': None,
                        'f': 'null'
                    })
                else:
                    pass_config.settings.update({
                        'pass': '2',
                        'passlogfile': 'video_pass'
                    })
            
            elif config.encoder_type == 'nvenc':
                pass_config.settings['multipass'] = 'fullres' if passes == 2 else 'qres'
            
            elif config.encoder_type == 'qsv':
                pass_config.settings.update({
                    'extbrc': '1',
                    'look_ahead': '1'
                })
            
            pass_configs.append(pass_config)
        
        return pass_configs
    
    def optimize_for_platform(self,
                             config: EncodingConfig,
                             platform: str) -> EncodingConfig:
        """Optimize configuration for specific streaming platform"""
        
        platform_configs = {
            'youtube': {
                'max_bitrate': '50000k',
                'bufsize': '75000k',
                'keyint_min': '4',
                'g': '120',
                'profile': 'high',
                'level': '4.2'
            },
            'twitch': {
                'max_bitrate': '6000k',
                'bufsize': '12000k',
                'keyint_min': '2',
                'g': '60',
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
            config.settings.update(platform_settings)
            logger.info(f"Applied {platform} platform optimizations")
        
        return config
    
    def validate_configuration(self, config: EncodingConfig) -> bool:
        """Validate encoding configuration"""
        
        if not self.hardware_detector:
            return True
        
        return self.hardware_detector.validate_encoder_for_settings(
            config.encoder_type,
            config.resolution,
            config.framerate,
            config.settings.get('b_frames', '0') != '0'
        )