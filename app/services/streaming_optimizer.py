#!/usr/bin/env python3
"""
Real-Time Streaming Optimization for MoneyPrinter Turbo Enhanced

Implements:
- Ultra-low latency streaming optimization
- Adaptive bitrate streaming (ABR)
- Platform-specific optimizations (YouTube, Twitch, etc.)
- Network-aware encoding adjustments
- Real-time quality adaptation
- Buffer management and congestion control
"""

import subprocess
import time
import json
import threading
import psutil
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import queue


@dataclass
class StreamingProfile:
    """Streaming profile configuration"""
    name: str
    platform: str
    max_bitrate: int
    target_fps: int
    resolution: Tuple[int, int]
    keyframe_interval: int
    latency_mode: str
    buffer_size: str
    profile: str
    level: str


@dataclass 
class NetworkConditions:
    """Network conditions for adaptive streaming"""
    bandwidth_kbps: int
    rtt_ms: int
    packet_loss: float
    jitter_ms: int
    congestion_level: str


class StreamingOptimizer:
    """Advanced real-time streaming optimization system"""
    
    def __init__(self):
        self.streaming_profiles = self._initialize_streaming_profiles()
        self.network_monitor = NetworkMonitor()
        self.quality_monitor = StreamingQualityMonitor()
        self.adaptive_controller = AdaptiveBitrateController()
        
        # Streaming state
        self.current_profile = None
        self.is_streaming = False
        self.quality_metrics = {}
        
        logger.info("Streaming Optimizer initialized with platform profiles")
    
    def _initialize_streaming_profiles(self) -> Dict[str, StreamingProfile]:
        """Initialize streaming profiles for different platforms"""
        
        profiles = {}
        
        # YouTube streaming profiles
        profiles['youtube_1080p'] = StreamingProfile(
            name='YouTube 1080p',
            platform='youtube',
            max_bitrate=6000,
            target_fps=60,
            resolution=(1920, 1080),
            keyframe_interval=2,
            latency_mode='normal',
            buffer_size='12000k',
            profile='high',
            level='4.2'
        )
        
        profiles['youtube_720p'] = StreamingProfile(
            name='YouTube 720p',
            platform='youtube',
            max_bitrate=3000,
            target_fps=30,
            resolution=(1280, 720),
            keyframe_interval=2,
            latency_mode='low',
            buffer_size='6000k',
            profile='main',
            level='3.1'
        )
        
        # Twitch streaming profiles
        profiles['twitch_1080p'] = StreamingProfile(
            name='Twitch 1080p',
            platform='twitch',
            max_bitrate=6000,
            target_fps=60,
            resolution=(1920, 1080),
            keyframe_interval=2,
            latency_mode='low',
            buffer_size='9000k',
            profile='main',
            level='4.1'
        )
        
        profiles['twitch_720p'] = StreamingProfile(
            name='Twitch 720p',
            platform='twitch',
            max_bitrate=3000,
            target_fps=30,
            resolution=(1280, 720),
            keyframe_interval=2,
            latency_mode='ultra_low',
            buffer_size='4500k',
            profile='main',
            level='3.1'
        )
        
        # Facebook Live profiles
        profiles['facebook_1080p'] = StreamingProfile(
            name='Facebook 1080p',
            platform='facebook',
            max_bitrate=4000,
            target_fps=30,
            resolution=(1920, 1080),
            keyframe_interval=2,
            latency_mode='normal',
            buffer_size='8000k',
            profile='main',
            level='4.0'
        )
        
        # Ultra-low latency profiles
        profiles['ultra_low_latency'] = StreamingProfile(
            name='Ultra Low Latency',
            platform='custom',
            max_bitrate=2000,
            target_fps=30,
            resolution=(1280, 720),
            keyframe_interval=1,
            latency_mode='ultra_low',
            buffer_size='2000k',
            profile='baseline',
            level='3.0'
        )
        
        return profiles
    
    def optimize_for_streaming(self, codec_settings: Dict[str, Any],
                              profile_name: str,
                              network_conditions: Optional[NetworkConditions] = None) -> Dict[str, Any]:
        """Optimize codec settings for streaming with specific profile"""
        
        if profile_name not in self.streaming_profiles:
            logger.warning(f"Unknown streaming profile: {profile_name}")
            return codec_settings
        
        profile = self.streaming_profiles[profile_name]
        optimized_settings = codec_settings.copy()
        
        # Update basic streaming parameters
        optimized_settings.update({
            'bitrate': f"{profile.max_bitrate}k",
            'maxrate': f"{profile.max_bitrate}k", 
            'bufsize': profile.buffer_size,
            'profile': profile.profile,
            'level': profile.level,
            'fps': str(profile.target_fps)
        })
        
        # Keyframe interval settings
        optimized_settings.update({
            'keyint_min': str(profile.keyframe_interval),
            'g': str(profile.keyframe_interval * profile.target_fps),  # GOP size
            'force_key_frames': f"expr:gte(t,n_forced*{profile.keyframe_interval})"
        })
        
        # Latency optimizations
        if profile.latency_mode == 'ultra_low':
            optimized_settings.update(self._get_ultra_low_latency_settings(optimized_settings))
        elif profile.latency_mode == 'low':
            optimized_settings.update(self._get_low_latency_settings(optimized_settings))
        
        # Network-adaptive adjustments
        if network_conditions:
            optimized_settings = self._adapt_to_network_conditions(
                optimized_settings, network_conditions, profile
            )
        
        # Encoder-specific streaming optimizations
        encoder_type = codec_settings.get('encoder_type', 'software')
        optimized_settings = self._apply_encoder_streaming_optimizations(
            optimized_settings, encoder_type, profile
        )
        
        logger.info(f"Streaming optimization applied: {profile_name}")
        return optimized_settings
    
    def _get_ultra_low_latency_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Get ultra-low latency streaming settings"""
        
        latency_settings = {
            'tune': 'zerolatency',
            'preset': 'ultrafast',
            'b_frames': '0',
            'bf': '0',
            'refs': '1',
            'sc_threshold': '0',
            'rc_lookahead': '0',
            'slices': '4',  # Multiple slices for parallel processing
            'slice_type': 'i',
            'intra_refresh': '1'
        }
        
        # Encoder-specific ultra-low latency
        encoder_type = settings.get('encoder_type', 'software')
        
        if encoder_type == 'nvenc':
            latency_settings.update({
                'preset': 'p1',  # Fastest NVENC preset
                'tune': 'ull',   # Ultra-low latency
                'delay': '0',
                'rc': 'cbr',     # Constant bitrate for predictable latency
                'zerolatency': '1'
            })
        
        elif encoder_type == 'qsv':
            latency_settings.update({
                'preset': 'veryfast',
                'async_depth': '1',
                'look_ahead': '0'
            })
        
        return latency_settings
    
    def _get_low_latency_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Get low latency streaming settings"""
        
        latency_settings = {
            'tune': 'zerolatency',
            'preset': 'veryfast',
            'b_frames': '1',
            'bf': '1',
            'refs': '2',
            'rc_lookahead': '10'
        }
        
        encoder_type = settings.get('encoder_type', 'software')
        
        if encoder_type == 'nvenc':
            latency_settings.update({
                'preset': 'p2',
                'tune': 'll',  # Low latency
                'delay': '0',
                'rc': 'vbr'
            })
        
        elif encoder_type == 'qsv':
            latency_settings.update({
                'preset': 'faster',
                'async_depth': '2'
            })
        
        return latency_settings
    
    def _adapt_to_network_conditions(self, settings: Dict[str, Any],
                                   network: NetworkConditions,
                                   profile: StreamingProfile) -> Dict[str, Any]:
        """Adapt encoding settings based on network conditions"""
        
        adapted_settings = settings.copy()
        
        # Adjust bitrate based on available bandwidth
        available_bandwidth = network.bandwidth_kbps * 0.8  # Use 80% of available bandwidth
        target_bitrate = min(profile.max_bitrate, int(available_bandwidth))
        
        adapted_settings['bitrate'] = f"{target_bitrate}k"
        adapted_settings['maxrate'] = f"{target_bitrate}k"
        
        # Adjust buffer size based on RTT
        if network.rtt_ms > 100:
            # High latency - increase buffer
            buffer_multiplier = 1.5
        elif network.rtt_ms > 50:
            buffer_multiplier = 1.2
        else:
            buffer_multiplier = 1.0
        
        base_buffer = int(profile.buffer_size.replace('k', ''))
        new_buffer = int(base_buffer * buffer_multiplier)
        adapted_settings['bufsize'] = f"{new_buffer}k"
        
        # Adjust based on packet loss
        if network.packet_loss > 0.01:  # > 1% packet loss
            # More conservative settings for lossy networks
            adapted_settings.update({
                'refs': '1',
                'b_frames': '0',
                'intra_refresh': '1'
            })
        
        # Congestion control adjustments
        if network.congestion_level == 'high':
            # Reduce quality to maintain smooth streaming
            if 'crf' in adapted_settings:
                current_crf = int(adapted_settings['crf'])
                adapted_settings['crf'] = str(min(28, current_crf + 2))
        
        logger.info(f"Network adaptation: {target_bitrate}kbps, buffer: {new_buffer}k")
        return adapted_settings
    
    def _apply_encoder_streaming_optimizations(self, settings: Dict[str, Any],
                                             encoder_type: str,
                                             profile: StreamingProfile) -> Dict[str, Any]:
        """Apply encoder-specific streaming optimizations"""
        
        streaming_settings = settings.copy()
        
        if encoder_type == 'nvenc':
            # NVENC streaming optimizations
            streaming_settings.update({
                'spatial_aq': '1',
                'temporal_aq': '1',
                'aq_strength': '8',
                'rc': 'vbr' if profile.latency_mode != 'ultra_low' else 'cbr',
                'multipass': 'qres' if profile.latency_mode == 'normal' else 'disabled'
            })
            
        elif encoder_type == 'qsv':
            # QSV streaming optimizations
            streaming_settings.update({
                'look_ahead': '1' if profile.latency_mode == 'normal' else '0',
                'extbrc': '1',
                'mbbrc': '1',  # Macroblock-level rate control
                'adaptive_i': '1',
                'adaptive_b': '1'
            })
            
        elif encoder_type == 'software':
            # Software encoder streaming optimizations
            streaming_settings.update({
                'aq_mode': '2',  # Variance-based adaptive quantization
                'aq_strength': '1.0',
                'psy_rd': '1.0:0.0',
                'rc_lookahead': '30' if profile.latency_mode == 'normal' else '0',
                'mbtree': '1' if profile.latency_mode == 'normal' else '0'
            })
        
        return streaming_settings
    
    def setup_adaptive_bitrate_streaming(self, base_settings: Dict[str, Any],
                                       quality_levels: List[str] = None) -> List[Dict[str, Any]]:
        """Setup adaptive bitrate streaming with multiple quality levels"""
        
        if quality_levels is None:
            quality_levels = ['1080p', '720p', '480p', '360p']
        
        abr_settings = []
        
        for quality in quality_levels:
            level_settings = base_settings.copy()
            
            if quality == '1080p':
                level_settings.update({
                    'scale': '1920x1080',
                    'bitrate': '5000k',
                    'maxrate': '5000k',
                    'bufsize': '10000k'
                })
            elif quality == '720p':
                level_settings.update({
                    'scale': '1280x720',
                    'bitrate': '3000k',
                    'maxrate': '3000k',
                    'bufsize': '6000k'
                })
            elif quality == '480p':
                level_settings.update({
                    'scale': '854x480',
                    'bitrate': '1500k',
                    'maxrate': '1500k',
                    'bufsize': '3000k'
                })
            elif quality == '360p':
                level_settings.update({
                    'scale': '640x360',
                    'bitrate': '800k',
                    'maxrate': '800k',
                    'bufsize': '1600k'
                })
            
            level_settings['quality_level'] = quality
            abr_settings.append(level_settings)
        
        logger.info(f"ABR streaming configured with {len(abr_settings)} quality levels")
        return abr_settings
    
    def monitor_streaming_quality(self, stream_url: str,
                                callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Monitor streaming quality metrics in real-time"""
        
        self.quality_monitor.start_monitoring(stream_url, callback)
        return self.quality_monitor.get_current_metrics()
    
    def get_platform_recommendations(self, platform: str,
                                   content_type: str = 'general') -> Dict[str, Any]:
        """Get platform-specific encoding recommendations"""
        
        recommendations = {
            'youtube': {
                'general': {
                    'resolution': '1920x1080',
                    'fps': 60,
                    'bitrate_range': '4500-6000k',
                    'keyframe_interval': 2,
                    'profile': 'high',
                    'audio_bitrate': '128k'
                },
                'gaming': {
                    'resolution': '1920x1080',
                    'fps': 60,
                    'bitrate_range': '6000-8000k',
                    'keyframe_interval': 2,
                    'profile': 'high',
                    'tune': 'grain'
                }
            },
            'twitch': {
                'general': {
                    'resolution': '1920x1080',
                    'fps': 60,
                    'bitrate_range': '4000-6000k',
                    'keyframe_interval': 2,
                    'profile': 'main',
                    'audio_bitrate': '160k'
                },
                'gaming': {
                    'resolution': '1920x1080',
                    'fps': 60,
                    'bitrate_range': '5000-6000k',
                    'keyframe_interval': 1,
                    'profile': 'main',
                    'tune': 'grain'
                }
            },
            'facebook': {
                'general': {
                    'resolution': '1280x720',
                    'fps': 30,
                    'bitrate_range': '2000-4000k',
                    'keyframe_interval': 2,
                    'profile': 'main',
                    'audio_bitrate': '128k'
                }
            }
        }
        
        platform_recs = recommendations.get(platform.lower(), {})
        content_recs = platform_recs.get(content_type, platform_recs.get('general', {}))
        
        logger.info(f"Platform recommendations for {platform}/{content_type}: {content_recs}")
        return content_recs


class NetworkMonitor:
    """Network conditions monitoring"""
    
    def __init__(self):
        self.is_monitoring = False
        self.current_conditions = None
        self.history = []
    
    def start_monitoring(self, interval: int = 5):
        """Start network monitoring"""
        
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                conditions = self._measure_network_conditions()
                self.current_conditions = conditions
                self.history.append((time.time(), conditions))
                
                # Keep only last 100 measurements
                if len(self.history) > 100:
                    self.history.pop(0)
                
                time.sleep(interval)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        
        logger.info("Network monitoring started")
    
    def stop_monitoring(self):
        """Stop network monitoring"""
        self.is_monitoring = False
        logger.info("Network monitoring stopped")
    
    def _measure_network_conditions(self) -> NetworkConditions:
        """Measure current network conditions"""
        
        try:
            # Get network stats
            net_io = psutil.net_io_counters()
            
            # Simple bandwidth estimation (would be more sophisticated in production)
            if hasattr(self, '_last_net_io'):
                time_delta = time.time() - self._last_time
                bytes_delta = net_io.bytes_sent + net_io.bytes_recv - (
                    self._last_net_io.bytes_sent + self._last_net_io.bytes_recv
                )
                bandwidth_bps = bytes_delta / time_delta if time_delta > 0 else 0
                bandwidth_kbps = int(bandwidth_bps / 1024 * 8)
            else:
                bandwidth_kbps = 10000  # Default assumption
            
            self._last_net_io = net_io
            self._last_time = time.time()
            
            # Mock RTT and packet loss (would use actual network tools in production)
            rtt_ms = 50  # Default RTT
            packet_loss = 0.0  # Default packet loss
            jitter_ms = 5  # Default jitter
            
            # Determine congestion level
            if bandwidth_kbps < 1000:
                congestion_level = 'high'
            elif bandwidth_kbps < 5000:
                congestion_level = 'medium'
            else:
                congestion_level = 'low'
            
            return NetworkConditions(
                bandwidth_kbps=bandwidth_kbps,
                rtt_ms=rtt_ms,
                packet_loss=packet_loss,
                jitter_ms=jitter_ms,
                congestion_level=congestion_level
            )
            
        except Exception as e:
            logger.error(f"Error measuring network conditions: {e}")
            return NetworkConditions(
                bandwidth_kbps=5000,
                rtt_ms=50,
                packet_loss=0.0,
                jitter_ms=5,
                congestion_level='medium'
            )
    
    def get_current_conditions(self) -> Optional[NetworkConditions]:
        """Get current network conditions"""
        return self.current_conditions


class StreamingQualityMonitor:
    """Real-time streaming quality monitoring"""
    
    def __init__(self):
        self.is_monitoring = False
        self.metrics = {}
        self.callbacks = []
    
    def start_monitoring(self, stream_url: str, callback: Optional[Callable] = None):
        """Start monitoring streaming quality"""
        
        if callback:
            self.callbacks.append(callback)
        
        self.is_monitoring = True
        
        def quality_monitor_loop():
            while self.is_monitoring:
                metrics = self._collect_quality_metrics(stream_url)
                self.metrics.update(metrics)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Quality monitor callback error: {e}")
                
                time.sleep(10)  # Check every 10 seconds
        
        thread = threading.Thread(target=quality_monitor_loop, daemon=True)
        thread.start()
        
        logger.info(f"Quality monitoring started for: {stream_url}")
    
    def stop_monitoring(self):
        """Stop quality monitoring"""
        self.is_monitoring = False
        logger.info("Quality monitoring stopped")
    
    def _collect_quality_metrics(self, stream_url: str) -> Dict[str, Any]:
        """Collect streaming quality metrics"""
        
        # This would integrate with actual streaming analytics in production
        # For now, return mock metrics
        
        metrics = {
            'timestamp': time.time(),
            'bitrate_actual': 4500,  # kbps
            'fps_actual': 59.2,
            'dropped_frames': 12,
            'buffer_health': 85,  # percentage
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'network_utilization': 75  # percentage
        }
        
        return metrics
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current quality metrics"""
        return self.metrics.copy()


class AdaptiveBitrateController:
    """Adaptive bitrate control for streaming"""
    
    def __init__(self):
        self.current_bitrate = 5000  # kbps
        self.target_bitrate = 5000
        self.adjustment_history = []
        self.controller_active = False
    
    def start_adaptive_control(self, quality_monitor: StreamingQualityMonitor,
                             network_monitor: NetworkMonitor):
        """Start adaptive bitrate control"""
        
        self.controller_active = True
        
        def control_loop():
            while self.controller_active:
                # Get current conditions
                quality_metrics = quality_monitor.get_current_metrics()
                network_conditions = network_monitor.get_current_conditions()
                
                if quality_metrics and network_conditions:
                    # Determine if bitrate adjustment is needed
                    adjustment = self._calculate_bitrate_adjustment(
                        quality_metrics, network_conditions
                    )
                    
                    if abs(adjustment) > 0.1:  # Significant adjustment needed
                        self._apply_bitrate_adjustment(adjustment)
                
                time.sleep(5)  # Adjust every 5 seconds
        
        thread = threading.Thread(target=control_loop, daemon=True)
        thread.start()
        
        logger.info("Adaptive bitrate control started")
    
    def stop_adaptive_control(self):
        """Stop adaptive bitrate control"""
        self.controller_active = False
        logger.info("Adaptive bitrate control stopped")
    
    def _calculate_bitrate_adjustment(self, quality_metrics: Dict[str, Any],
                                    network_conditions: NetworkConditions) -> float:
        """Calculate bitrate adjustment factor"""
        
        adjustment = 0.0
        
        # Buffer health based adjustment
        buffer_health = quality_metrics.get('buffer_health', 100)
        if buffer_health < 30:
            adjustment -= 0.2  # Reduce bitrate by 20%
        elif buffer_health > 90:
            adjustment += 0.1  # Increase bitrate by 10%
        
        # Dropped frames based adjustment
        dropped_frames = quality_metrics.get('dropped_frames', 0)
        if dropped_frames > 50:
            adjustment -= 0.15
        elif dropped_frames < 5:
            adjustment += 0.05
        
        # Network conditions based adjustment
        available_bandwidth = network_conditions.bandwidth_kbps * 0.8
        utilization = self.current_bitrate / available_bandwidth
        
        if utilization > 0.9:
            adjustment -= 0.25  # Aggressive reduction
        elif utilization < 0.6:
            adjustment += 0.15  # Conservative increase
        
        return max(-0.5, min(0.3, adjustment))  # Limit adjustment range
    
    def _apply_bitrate_adjustment(self, adjustment_factor: float):
        """Apply bitrate adjustment"""
        
        old_bitrate = self.current_bitrate
        self.current_bitrate = int(self.current_bitrate * (1 + adjustment_factor))
        
        # Clamp to reasonable range
        self.current_bitrate = max(500, min(8000, self.current_bitrate))
        
        self.adjustment_history.append({
            'timestamp': time.time(),
            'old_bitrate': old_bitrate,
            'new_bitrate': self.current_bitrate,
            'adjustment_factor': adjustment_factor
        })
        
        logger.info(f"Bitrate adjusted: {old_bitrate} -> {self.current_bitrate} kbps ({adjustment_factor:+.2f})")


# Global instance
streaming_optimizer = StreamingOptimizer()