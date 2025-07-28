#!/usr/bin/env python3
"""
Multi-Pass Encoding Strategy System for MoneyPrinter Turbo Enhanced

Implements advanced multi-pass encoding strategies:
- Two-pass VBR encoding for optimal quality/size balance
- Multi-pass analysis for complex content
- Adaptive bitrate allocation
- Quality-driven encoding decisions
- Performance vs quality optimization
"""

import subprocess
import os
import time
import json
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class PassConfig:
    """Configuration for a single encoding pass"""
    pass_number: int
    pass_type: str  # 'analysis', 'encode', 'final'
    output_file: Optional[str] = None
    stats_file: Optional[str] = None
    analysis_only: bool = False
    extra_params: Dict[str, str] = None


@dataclass
class EncodingStrategy:
    """Multi-pass encoding strategy configuration"""
    name: str
    description: str
    passes: List[PassConfig]
    target_quality: str
    expected_improvement: float
    use_cases: List[str]


class MultiPassEncoder:
    """Advanced multi-pass encoding system"""
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
        self.temp_dir = None
        self.current_stats = {}
        
        logger.info("Multi-Pass Encoder initialized with strategies")
    
    def _initialize_strategies(self) -> Dict[str, EncodingStrategy]:
        """Initialize encoding strategies"""
        
        strategies = {}
        
        # Two-pass VBR strategy
        strategies['two_pass_vbr'] = EncodingStrategy(
            name='Two-Pass VBR',
            description='Classic two-pass VBR encoding for optimal quality/size balance',
            passes=[
                PassConfig(
                    pass_number=1,
                    pass_type='analysis',
                    analysis_only=True,
                    extra_params={'pass': '1', 'an': None, 'f': 'null'}
                ),
                PassConfig(
                    pass_number=2,
                    pass_type='encode',
                    analysis_only=False,
                    extra_params={'pass': '2'}
                )
            ],
            target_quality='balanced',
            expected_improvement=0.15,  # 15% better quality/size ratio
            use_cases=['archive', 'high_quality', 'streaming']
        )
        
        # Three-pass quality strategy
        strategies['three_pass_quality'] = EncodingStrategy(
            name='Three-Pass Quality',
            description='Three-pass encoding with detailed analysis for maximum quality',
            passes=[
                PassConfig(
                    pass_number=1,
                    pass_type='analysis',
                    analysis_only=True,
                    extra_params={'pass': '1', 'an': None, 'f': 'null'}
                ),
                PassConfig(
                    pass_number=2,
                    pass_type='analysis',
                    analysis_only=True,
                    extra_params={'pass': '1', 'an': None, 'f': 'null'}  # Re-analyze with refined settings
                ),
                PassConfig(
                    pass_number=3,
                    pass_type='encode',
                    analysis_only=False,
                    extra_params={'pass': '2'}
                )
            ],
            target_quality='quality',
            expected_improvement=0.25,  # 25% better quality
            use_cases=['archive', 'professional', 'high_motion']
        )
        
        # CRF with pre-analysis strategy
        strategies['crf_preanalysis'] = EncodingStrategy(
            name='CRF with Pre-Analysis',
            description='Single-pass CRF with pre-analysis for optimal CRF selection',
            passes=[
                PassConfig(
                    pass_number=1,
                    pass_type='analysis',
                    analysis_only=True,
                    extra_params={'ss': '60', 't': '60', 'an': None, 'f': 'null'}  # Analyze 1 minute sample
                ),
                PassConfig(
                    pass_number=2,
                    pass_type='encode',
                    analysis_only=False,
                    extra_params={}  # CRF will be determined from analysis
                )
            ],
            target_quality='balanced',
            expected_improvement=0.20,  # 20% better quality selection
            use_cases=['general', 'balanced', 'content_adaptive']
        )
        
        # Adaptive bitrate strategy
        strategies['adaptive_bitrate'] = EncodingStrategy(
            name='Adaptive Bitrate',
            description='Multi-pass encoding with scene-aware bitrate allocation',
            passes=[
                PassConfig(
                    pass_number=1,
                    pass_type='analysis',
                    analysis_only=True,
                    extra_params={'pass': '1', 'an': None, 'f': 'null'}
                ),
                PassConfig(
                    pass_number=2,
                    pass_type='encode',
                    analysis_only=False,
                    extra_params={'pass': '2', 'mbtree': '1', 'rc_lookahead': '60'}
                )
            ],
            target_quality='balanced',
            expected_improvement=0.18,  # 18% better bitrate efficiency
            use_cases=['streaming', 'variable_content', 'bandwidth_limited']
        )
        
        return strategies
    
    def select_strategy(self, content_type: str, target_quality: str, 
                       encoder_type: str, time_constraint: Optional[float] = None) -> str:
        """Select optimal multi-pass strategy based on requirements"""
        
        # Check encoder support for multi-pass
        if encoder_type in ['vaapi', 'videotoolbox']:
            logger.info(f"Multi-pass not supported by {encoder_type}, using single-pass")
            return None
        
        # Time constraint check
        if time_constraint and time_constraint < 5.0:  # Less than 5x encoding time
            logger.info("Time constraint too tight for multi-pass encoding")
            return None
        
        # Select strategy based on requirements
        if target_quality == 'archive' or target_quality == 'quality':
            if content_type == 'high_motion':
                return 'three_pass_quality'
            else:
                return 'two_pass_vbr'
        
        elif target_quality == 'balanced':
            if content_type == 'variable_content':
                return 'adaptive_bitrate'
            else:
                return 'crf_preanalysis'
        
        elif target_quality == 'speed':
            # No multi-pass for speed priority
            return None
        
        # Default to two-pass VBR
        return 'two_pass_vbr'
    
    def analyze_content_complexity(self, input_path: str, 
                                  sample_duration: int = 60) -> Dict[str, Any]:
        """Analyze content complexity for optimal encoding settings"""
        
        try:
            # Create temporary directory for analysis
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # Extract sample for analysis
                sample_path = os.path.join(temp_dir, 'sample.mp4')
                
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'quiet',
                    '-i', input_path,
                    '-t', str(sample_duration),
                    '-c:v', 'copy', '-c:a', 'copy',
                    sample_path
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    logger.warning("Failed to extract sample, using full file analysis")
                    sample_path = input_path
                
                # Analyze complexity using FFmpeg's stats
                stats_path = os.path.join(temp_dir, 'stats.log')
                
                cmd = [
                    'ffmpeg', '-y', '-hide_banner',
                    '-i', sample_path,
                    '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-pass', '1', '-passlogfile', stats_path,
                    '-an', '-f', 'null', '-'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(f"{stats_path}-0.log"):
                    # Parse x264 stats for complexity analysis
                    complexity_stats = self._parse_x264_stats(f"{stats_path}-0.log")
                else:
                    # Fallback to basic analysis
                    complexity_stats = self._basic_complexity_analysis(sample_path)
                
                logger.info(f"Content complexity analysis: {complexity_stats}")
                return complexity_stats
                
        except Exception as e:
            logger.error(f"Error analyzing content complexity: {e}")
            return {'complexity': 'medium', 'motion_level': 'medium', 'detail_level': 'medium'}
    
    def _parse_x264_stats(self, stats_file: str) -> Dict[str, Any]:
        """Parse x264 statistics for complexity metrics"""
        
        try:
            with open(stats_file, 'r') as f:
                stats_data = f.read()
            
            # Parse key metrics from x264 stats
            lines = stats_data.strip().split('\n')
            
            total_bits = 0
            frame_count = 0
            motion_vectors = []
            complexity_scores = []
            
            for line in lines:
                if line.startswith('in:'):
                    parts = line.split()
                    
                    # Extract frame type and size
                    frame_type = None
                    frame_size = 0
                    
                    for part in parts:
                        if part.startswith('type:'):
                            frame_type = part.split(':')[1]
                        elif part.startswith('q:'):
                            try:
                                qp = float(part.split(':')[1])
                                complexity_scores.append(qp)
                            except:
                                pass
                        elif part.startswith('size:'):
                            try:
                                frame_size = int(part.split(':')[1])
                                total_bits += frame_size * 8
                            except:
                                pass
                    
                    frame_count += 1
            
            # Calculate complexity metrics
            avg_bits_per_frame = total_bits / frame_count if frame_count > 0 else 0
            avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 25
            
            # Determine complexity level
            if avg_complexity < 20:
                complexity_level = 'low'
            elif avg_complexity < 30:
                complexity_level = 'medium'
            else:
                complexity_level = 'high'
            
            # Motion level based on bits per frame
            if avg_bits_per_frame < 50000:
                motion_level = 'low'
            elif avg_bits_per_frame < 150000:
                motion_level = 'medium'
            else:
                motion_level = 'high'
            
            return {
                'complexity': complexity_level,
                'motion_level': motion_level,
                'detail_level': complexity_level,
                'avg_qp': avg_complexity,
                'avg_bits_per_frame': avg_bits_per_frame,
                'frame_count': frame_count
            }
            
        except Exception as e:
            logger.error(f"Error parsing x264 stats: {e}")
            return {'complexity': 'medium', 'motion_level': 'medium', 'detail_level': 'medium'}
    
    def _basic_complexity_analysis(self, input_path: str) -> Dict[str, Any]:
        """Basic complexity analysis using FFprobe"""
        
        try:
            # Get basic video info
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-show_format',
                input_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Find video stream
                video_stream = None
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        video_stream = stream
                        break
                
                if video_stream:
                    width = int(video_stream.get('width', 1920))
                    height = int(video_stream.get('height', 1080))
                    bitrate = int(video_stream.get('bit_rate', 0)) if video_stream.get('bit_rate') else 0
                    
                    # Estimate complexity based on resolution and bitrate
                    pixel_count = width * height
                    
                    if pixel_count > 2073600:  # > 1080p
                        complexity = 'high'
                    elif pixel_count > 921600:  # > 720p
                        complexity = 'medium'
                    else:
                        complexity = 'low'
                    
                    # Motion estimation based on bitrate density
                    if bitrate > 0:
                        bitrate_density = bitrate / pixel_count
                        if bitrate_density > 0.1:
                            motion_level = 'high'
                        elif bitrate_density > 0.05:
                            motion_level = 'medium'
                        else:
                            motion_level = 'low'
                    else:
                        motion_level = 'medium'
                    
                    return {
                        'complexity': complexity,
                        'motion_level': motion_level,
                        'detail_level': complexity,
                        'resolution': f"{width}x{height}",
                        'bitrate': bitrate
                    }
            
            return {'complexity': 'medium', 'motion_level': 'medium', 'detail_level': 'medium'}
            
        except Exception as e:
            logger.error(f"Error in basic complexity analysis: {e}")
            return {'complexity': 'medium', 'motion_level': 'medium', 'detail_level': 'medium'}
    
    def optimize_crf_from_analysis(self, complexity_stats: Dict[str, Any], 
                                  target_quality: str) -> int:
        """Determine optimal CRF value based on content analysis"""
        
        base_crf = {
            'speed': 25,
            'balanced': 23,
            'quality': 20,
            'archive': 18
        }.get(target_quality, 23)
        
        # Adjust based on complexity
        complexity_level = complexity_stats.get('complexity', 'medium')
        motion_level = complexity_stats.get('motion_level', 'medium')
        
        crf_adjustment = 0
        
        # Complex content needs lower CRF (higher quality)
        if complexity_level == 'high':
            crf_adjustment -= 2
        elif complexity_level == 'low':
            crf_adjustment += 1
        
        # High motion content needs lower CRF
        if motion_level == 'high':
            crf_adjustment -= 1
        elif motion_level == 'low':
            crf_adjustment += 1
        
        optimal_crf = max(16, min(28, base_crf + crf_adjustment))
        
        logger.info(f"Optimal CRF determined: {optimal_crf} (base: {base_crf}, adjustment: {crf_adjustment})")
        return optimal_crf
    
    def execute_multipass_encoding(self, input_path: str, output_path: str,
                                  strategy_name: str, codec_settings: Dict[str, Any],
                                  progress_callback: Optional[callable] = None) -> bool:
        """Execute multi-pass encoding with specified strategy"""
        
        if strategy_name not in self.strategies:
            logger.error(f"Unknown strategy: {strategy_name}")
            return False
        
        strategy = self.strategies[strategy_name]
        
        try:
            # Create temporary directory for pass files
            with tempfile.TemporaryDirectory() as temp_dir:
                self.temp_dir = temp_dir
                stats_file = os.path.join(temp_dir, 'passlogfile')
                
                # Content analysis for adaptive strategies
                if strategy_name == 'crf_preanalysis':
                    complexity_stats = self.analyze_content_complexity(input_path)
                    optimal_crf = self.optimize_crf_from_analysis(
                        complexity_stats, codec_settings.get('target_quality', 'balanced')
                    )
                    codec_settings['crf'] = str(optimal_crf)
                
                total_passes = len(strategy.passes)
                
                for i, pass_config in enumerate(strategy.passes):
                    logger.info(f"Executing pass {pass_config.pass_number}/{total_passes}: {pass_config.pass_type}")
                    
                    # Build command for this pass
                    cmd = self._build_pass_command(
                        input_path, output_path, pass_config, codec_settings, stats_file
                    )
                    
                    # Execute pass with timeout and better error handling
                    start_time = time.time()
                    try:
                        # Set timeout based on pass type (analysis passes should be faster)
                        timeout = 3600 if pass_config.pass_type == 'encode' else 1800  # 1hr for encode, 30min for analysis
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                        pass_time = time.time() - start_time
                        
                        if result.returncode != 0:
                            logger.error(f"Pass {pass_config.pass_number} failed: {result.stderr}")
                            return False
                            
                    except subprocess.TimeoutExpired:
                        logger.error(f"Pass {pass_config.pass_number} timed out after {timeout}s")
                        return False
                    except Exception as e:
                        logger.error(f"Pass {pass_config.pass_number} failed with exception: {e}")
                        return False
                    
                    logger.success(f"Pass {pass_config.pass_number} completed in {pass_time:.2f}s")
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(pass_config.pass_number, total_passes, pass_time)
                
                logger.success(f"Multi-pass encoding completed: {strategy_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error in multi-pass encoding: {e}")
            return False
    
    def _build_pass_command(self, input_path: str, output_path: str,
                           pass_config: PassConfig, codec_settings: Dict[str, Any],
                           stats_file: str) -> List[str]:
        """Build FFmpeg command for a specific pass"""
        
        cmd = ['ffmpeg', '-y', '-hide_banner', '-i', input_path]
        
        # Video codec
        cmd.extend(['-c:v', codec_settings.get('codec', 'libx264')])
        
        # Base codec settings
        encoder_type = codec_settings.get('encoder_type', 'software')
        
        if encoder_type == 'software':
            # x264/x265 multi-pass settings
            if 'preset' in codec_settings:
                cmd.extend(['-preset', codec_settings['preset']])
            
            if pass_config.pass_type == 'analysis':
                # First pass settings
                cmd.extend(['-pass', '1', '-passlogfile', stats_file])
                cmd.extend(['-an', '-f', 'null', '-'])  # No audio, null output
            else:
                # Second pass settings
                cmd.extend(['-pass', '2', '-passlogfile', stats_file])
                
                # Quality setting
                if 'crf' in codec_settings:
                    # For CRF mode in second pass
                    cmd.extend(['-crf', codec_settings['crf']])
                elif 'bitrate' in codec_settings:
                    # For VBR mode
                    cmd.extend(['-b:v', codec_settings['bitrate']])
                    if 'maxrate' in codec_settings:
                        cmd.extend(['-maxrate', codec_settings['maxrate']])
                        cmd.extend(['-bufsize', codec_settings.get('bufsize', codec_settings['maxrate'])])
        
        elif encoder_type == 'nvenc':
            # NVENC multi-pass
            cmd.extend(['-preset', codec_settings.get('preset', 'p4')])
            cmd.extend(['-multipass', 'fullres'])
            cmd.extend(['-rc', 'vbr'])
            
            if 'cq' in codec_settings:
                cmd.extend(['-cq', codec_settings['cq']])
        
        elif encoder_type == 'qsv':
            # QSV multi-pass
            cmd.extend(['-preset', codec_settings.get('preset', 'balanced')])
            cmd.extend(['-extbrc', '1'])
            cmd.extend(['-look_ahead', '1'])
            
            if 'global_quality' in codec_settings:
                cmd.extend(['-global_quality', codec_settings['global_quality']])
        
        # Add extra parameters from pass config
        if pass_config.extra_params:
            for key, value in pass_config.extra_params.items():
                if value is not None:
                    cmd.extend([f'-{key}', str(value)])
                else:
                    cmd.append(f'-{key}')
        
        # Audio handling
        if not pass_config.analysis_only:
            cmd.extend(['-c:a', 'copy'])
        
        # Output
        if not pass_config.analysis_only:
            cmd.append(output_path)
        
        logger.debug(f"Pass command: {' '.join(cmd)}")
        return cmd
    
    def estimate_encoding_time(self, strategy_name: str, base_time: float) -> float:
        """Estimate total encoding time for multi-pass strategy"""
        
        if strategy_name not in self.strategies:
            return base_time
        
        strategy = self.strategies[strategy_name]
        
        # Time multipliers for different pass types
        pass_multipliers = {
            'analysis': 0.6,  # Analysis passes are faster
            'encode': 1.0,    # Full encoding passes
            'final': 1.0
        }
        
        total_multiplier = sum(
            pass_multipliers.get(pass_config.pass_type, 1.0) 
            for pass_config in strategy.passes
        )
        
        estimated_time = base_time * total_multiplier
        
        logger.info(f"Estimated encoding time for {strategy_name}: {estimated_time:.1f}s (base: {base_time:.1f}s)")
        return estimated_time
    
    def get_strategy_recommendation(self, content_analysis: Dict[str, Any],
                                   target_quality: str, time_budget: Optional[float] = None) -> str:
        """Get recommended strategy based on content analysis and constraints"""
        
        complexity = content_analysis.get('complexity', 'medium')
        motion_level = content_analysis.get('motion_level', 'medium')
        
        # High complexity/motion content benefits more from multi-pass
        if complexity == 'high' or motion_level == 'high':
            if target_quality in ['quality', 'archive']:
                if time_budget is None or time_budget > 10.0:
                    return 'three_pass_quality'
                else:
                    return 'two_pass_vbr'
            else:
                return 'adaptive_bitrate'
        
        # Medium complexity content
        elif complexity == 'medium':
            if target_quality == 'balanced':
                return 'crf_preanalysis'
            elif target_quality in ['quality', 'archive']:
                return 'two_pass_vbr'
            else:
                return None  # Single pass for speed
        
        # Low complexity content
        else:
            if target_quality in ['quality', 'archive']:
                return 'two_pass_vbr'
            else:
                return 'crf_preanalysis'  # Still benefits from CRF optimization


# Global instance
multipass_encoder = MultiPassEncoder()