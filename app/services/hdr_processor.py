#!/usr/bin/env python3
"""
HDR Processing and Wide Color Gamut Handler for MoneyPrinter Turbo Enhanced

Handles:
- HDR10 and HDR10+ processing
- Wide color gamut conversion (Rec.2020, DCI-P3)
- Tone mapping for SDR displays
- Color space conversions
- Dynamic range optimization
"""

import subprocess
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from loguru import logger
from dataclasses import dataclass


@dataclass
class HDRMetadata:
    """HDR metadata container"""
    color_primaries: str = 'bt709'
    transfer_characteristics: str = 'bt709'
    matrix_coefficients: str = 'bt709'
    max_content_light_level: Optional[int] = None
    max_frame_average_light_level: Optional[int] = None
    master_display_primaries: Optional[str] = None
    master_display_white_point: Optional[str] = None
    master_display_luminance: Optional[str] = None


class HDRProcessor:
    """Advanced HDR and wide color gamut processing"""
    
    def __init__(self):
        self.color_spaces = {
            'bt709': {
                'primaries': 'bt709',
                'transfer': 'bt709',
                'matrix': 'bt709',
                'range': 'tv'
            },
            'bt2020': {
                'primaries': 'bt2020',
                'transfer': 'smpte2084',  # PQ
                'matrix': 'bt2020nc',
                'range': 'tv'
            },
            'dci_p3': {
                'primaries': 'smpte432',
                'transfer': 'smpte2084',
                'matrix': 'bt2020nc',
                'range': 'tv'
            },
            'rec2020_hlg': {
                'primaries': 'bt2020',
                'transfer': 'arib-std-b67',  # HLG
                'matrix': 'bt2020nc',
                'range': 'tv'
            }
        }
        
        self.tone_mapping_algorithms = {
            'reinhard': {
                'description': 'Simple Reinhard tone mapping',
                'params': {'peak': 10000, 'contrast': 0.5, 'brightness': 0.0}
            },
            'hable': {
                'description': 'Uncharted 2 Hable tone mapping',
                'params': {'exposure_bias': 2.0, 'shoulder_strength': 0.15, 'linear_strength': 0.5}
            },
            'mobius': {
                'description': 'Mobius tone mapping (preserves detail)',
                'params': {'linear_knee': 0.3, 'linear_start': 0.2}
            },
            'bt2390': {
                'description': 'ITU-R BT.2390 EETF',
                'params': {'target_luma': 100, 'knee_point': 0.75}
            }
        }
        
        logger.info("HDR Processor initialized with color space support")
    
    def detect_hdr_content(self, input_path: str) -> HDRMetadata:
        """Detect HDR metadata from input video"""
        
        try:
            # Use ffprobe to analyze video metadata
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-show_format', '-show_frames', '-read_intervals', '%+#1',
                input_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"Failed to probe HDR metadata: {result.stderr}")
                return HDRMetadata()
            
            data = json.loads(result.stdout)
            
            # Extract HDR metadata from video stream
            hdr_metadata = HDRMetadata()
            
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    
                    # Color space information
                    hdr_metadata.color_primaries = stream.get('color_primaries', 'bt709')
                    hdr_metadata.transfer_characteristics = stream.get('color_trc', 'bt709')
                    hdr_metadata.matrix_coefficients = stream.get('color_space', 'bt709')
                    
                    # HDR10 metadata
                    side_data = stream.get('side_data_list', [])
                    for side_data_item in side_data:
                        side_data_type = side_data_item.get('side_data_type')
                        
                        if side_data_type == 'Content light level metadata':
                            hdr_metadata.max_content_light_level = side_data_item.get('max_content')
                            hdr_metadata.max_frame_average_light_level = side_data_item.get('max_average')
                        
                        elif side_data_type == 'Mastering display metadata':
                            hdr_metadata.master_display_primaries = side_data_item.get('red_x')
                            hdr_metadata.master_display_white_point = side_data_item.get('white_point_x')
                            hdr_metadata.master_display_luminance = side_data_item.get('max_luminance')
            
            # Check if content is actually HDR
            is_hdr = (
                hdr_metadata.color_primaries in ['bt2020', 'smpte432'] or
                hdr_metadata.transfer_characteristics in ['smpte2084', 'arib-std-b67'] or
                hdr_metadata.max_content_light_level is not None
            )
            
            if is_hdr:
                logger.info(f"HDR content detected: {hdr_metadata.color_primaries}/{hdr_metadata.transfer_characteristics}")
            else:
                logger.debug("SDR content detected")
            
            return hdr_metadata
            
        except Exception as e:
            logger.error(f"Error detecting HDR metadata: {e}")
            return HDRMetadata()
    
    def setup_hdr_encoding_params(self, target_color_space: str = 'bt2020',
                                 peak_luminance: int = 4000,
                                 content_light_level: int = 1000) -> Dict[str, str]:
        """Setup HDR encoding parameters"""
        
        color_config = self.color_spaces.get(target_color_space, self.color_spaces['bt709'])
        
        hdr_params = {
            'colorspace': color_config['matrix'],
            'color_primaries': color_config['primaries'],
            'color_trc': color_config['transfer'],
            'color_range': color_config['range']
        }
        
        # Add HDR10 metadata
        if target_color_space in ['bt2020', 'dci_p3']:
            
            # Master display metadata (standard Rec.2020 primaries)
            master_display = (
                f"G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)"
                f"L({peak_luminance * 10000},50)"  # Max/min luminance in 0.0001 nits
            )
            
            hdr_params.update({
                'master_display': master_display,
                'max_cll': f"{content_light_level},{content_light_level // 4}",  # MaxCLL,MaxFALL
                'hdr10_opt': '1' if target_color_space == 'bt2020' else '0'
            })
        
        return hdr_params
    
    def setup_tone_mapping(self, algorithm: str = 'bt2390',
                          target_peak: int = 100,
                          source_peak: int = 4000) -> Dict[str, str]:
        """Setup tone mapping parameters for HDR to SDR conversion"""
        
        if algorithm not in self.tone_mapping_algorithms:
            logger.warning(f"Unknown tone mapping algorithm: {algorithm}, using bt2390")
            algorithm = 'bt2390'
        
        algo_config = self.tone_mapping_algorithms[algorithm]
        
        tone_map_params = {
            'tonemap': algorithm,
            'tonemap_param': str(target_peak / source_peak)
        }
        
        # Algorithm-specific parameters
        if algorithm == 'reinhard':
            tone_map_params.update({
                'peak': str(source_peak),
                'contrast': str(algo_config['params']['contrast'])
            })
        
        elif algorithm == 'hable':
            tone_map_params.update({
                'exposure': str(algo_config['params']['exposure_bias']),
                'a': str(algo_config['params']['shoulder_strength']),
                'b': str(algo_config['params']['linear_strength'])
            })
        
        elif algorithm == 'mobius':
            tone_map_params.update({
                'transition': str(algo_config['params']['linear_knee']),
                'peak': str(source_peak)
            })
        
        elif algorithm == 'bt2390':
            tone_map_params.update({
                'target_nits': str(target_peak),
                'knee': str(algo_config['params']['knee_point'])
            })
        
        logger.info(f"Tone mapping configured: {algorithm} ({source_peak} -> {target_peak} nits)")
        return tone_map_params
    
    def build_hdr_filter_chain(self, input_metadata: HDRMetadata,
                              target_color_space: str = 'bt709',
                              enable_tone_mapping: bool = True,
                              tone_map_algorithm: str = 'bt2390') -> str:
        """Build FFmpeg filter chain for HDR processing"""
        
        filters = []
        
        # Determine if we need color space conversion
        input_is_hdr = (
            input_metadata.color_primaries in ['bt2020', 'smpte432'] or
            input_metadata.transfer_characteristics in ['smpte2084', 'arib-std-b67']
        )
        
        target_is_hdr = target_color_space in ['bt2020', 'dci_p3', 'rec2020_hlg']
        
        if input_is_hdr and not target_is_hdr and enable_tone_mapping:
            # HDR to SDR conversion with tone mapping
            
            # First, ensure proper color space
            filters.append(
                f"zscale=matrix=bt2020nc:primaries=bt2020:transfer=smpte2084"
            )
            
            # Apply tone mapping
            tone_params = self.setup_tone_mapping(tone_map_algorithm)
            tonemap_filter = f"tonemap={tone_map_algorithm}"
            
            if tone_params.get('target_nits'):
                tonemap_filter += f":target_nits={tone_params['target_nits']}"
            if tone_params.get('peak'):
                tonemap_filter += f":peak={tone_params['peak']}"
            
            filters.append(tonemap_filter)
            
            # Final color space conversion to target
            target_config = self.color_spaces[target_color_space]
            filters.append(
                f"zscale=matrix={target_config['matrix']}:"
                f"primaries={target_config['primaries']}:"
                f"transfer={target_config['transfer']}:"
                f"range={target_config['range']}"
            )
            
        elif input_is_hdr and target_is_hdr:
            # HDR to HDR conversion (color space change only)
            target_config = self.color_spaces[target_color_space]
            filters.append(
                f"zscale=matrix={target_config['matrix']}:"
                f"primaries={target_config['primaries']}:"
                f"transfer={target_config['transfer']}:"
                f"range={target_config['range']}"
            )
            
        elif not input_is_hdr and target_is_hdr:
            # SDR to HDR conversion (upconversion)
            target_config = self.color_spaces[target_color_space]
            
            # First convert to linear RGB
            filters.append("zscale=transfer=linear")
            
            # Scale to HDR range and apply EOTF
            filters.append(
                f"zscale=matrix={target_config['matrix']}:"
                f"primaries={target_config['primaries']}:"
                f"transfer={target_config['transfer']}:"
                f"range={target_config['range']}"
            )
        
        filter_chain = ','.join(filters) if filters else None
        
        if filter_chain:
            logger.info(f"HDR filter chain: {filter_chain}")
        
        return filter_chain
    
    def build_hdr_encoding_args(self, encoder_type: str, hdr_params: Dict[str, str],
                               codec_settings: Dict[str, Any]) -> List[str]:
        """Build encoding arguments with HDR support"""
        
        args = []
        
        # Add color space parameters
        if 'colorspace' in hdr_params:
            args.extend(['-colorspace', hdr_params['colorspace']])
        if 'color_primaries' in hdr_params:
            args.extend(['-color_primaries', hdr_params['color_primaries']])
        if 'color_trc' in hdr_params:
            args.extend(['-color_trc', hdr_params['color_trc']])
        if 'color_range' in hdr_params:
            args.extend(['-color_range', hdr_params['color_range']])
        
        # Encoder-specific HDR parameters
        if encoder_type == 'nvenc_hevc':
            # NVIDIA NVENC HEVC HDR
            args.extend(['-c:v', 'hevc_nvenc'])
            
            if 'master_display' in hdr_params:
                args.extend(['-master_display', hdr_params['master_display']])
            if 'max_cll' in hdr_params:
                args.extend(['-max_cll', hdr_params['max_cll']])
            
            # NVENC-specific HDR settings
            args.extend([
                '-profile:v', 'main10',  # 10-bit profile
                '-pix_fmt', 'p010le',    # 10-bit pixel format
                '-rc', 'vbr',
                '-spatial_aq', '1',
                '-temporal_aq', '1'
            ])
            
        elif encoder_type == 'qsv_hevc':
            # Intel QSV HEVC HDR
            args.extend(['-c:v', 'hevc_qsv'])
            
            if 'master_display' in hdr_params:
                args.extend(['-master_display', hdr_params['master_display']])
            if 'max_cll' in hdr_params:
                args.extend(['-max_cll', hdr_params['max_cll']])
            
            args.extend([
                '-profile:v', 'main10',
                '-pix_fmt', 'p010le'
            ])
            
        elif encoder_type == 'software_hevc':
            # Software x265 HEVC HDR
            args.extend(['-c:v', 'libx265'])
            
            x265_params = ['hdr-opt=1', 'repeat-headers=1']
            
            if 'master_display' in hdr_params:
                x265_params.append(f"master-display={hdr_params['master_display']}")
            if 'max_cll' in hdr_params:
                max_cll_parts = hdr_params['max_cll'].split(',')
                if len(max_cll_parts) == 2:
                    x265_params.append(f"max-cll={max_cll_parts[0]},{max_cll_parts[1]}")
            
            # Color space parameters for x265
            x265_params.extend([
                f"colorprim={hdr_params.get('color_primaries', 'bt2020')}",
                f"transfer={hdr_params.get('color_trc', 'smpte2084')}",
                f"colormatrix={hdr_params.get('colorspace', 'bt2020nc')}"
            ])
            
            args.extend([
                '-x265-params', ':'.join(x265_params),
                '-profile:v', 'main10',
                '-pix_fmt', 'yuv420p10le'
            ])
        
        return args
    
    def process_hdr_video(self, input_path: str, output_path: str,
                         target_color_space: str = 'bt709',
                         enable_tone_mapping: bool = True,
                         tone_map_algorithm: str = 'bt2390',
                         encoder_settings: Optional[Dict[str, Any]] = None) -> bool:
        """Process HDR video with color space conversion and tone mapping"""
        
        try:
            # Detect input HDR metadata
            input_metadata = self.detect_hdr_content(input_path)
            
            # Build filter chain
            filter_chain = self.build_hdr_filter_chain(
                input_metadata, target_color_space, enable_tone_mapping, tone_map_algorithm
            )
            
            # Setup encoding parameters
            if target_color_space in ['bt2020', 'dci_p3']:
                hdr_params = self.setup_hdr_encoding_params(target_color_space)
                encoder_type = encoder_settings.get('encoder_type', 'software_hevc') if encoder_settings else 'software_hevc'
            else:
                hdr_params = self.color_spaces['bt709']
                encoder_type = encoder_settings.get('encoder_type', 'software') if encoder_settings else 'software'
            
            # Build FFmpeg command
            cmd = ['ffmpeg', '-y', '-hide_banner', '-i', input_path]
            
            # Add filter chain if needed
            if filter_chain:
                cmd.extend(['-vf', filter_chain])
            
            # Add HDR encoding arguments
            if target_color_space in ['bt2020', 'dci_p3']:
                hdr_args = self.build_hdr_encoding_args(encoder_type, hdr_params, encoder_settings or {})
                cmd.extend(hdr_args)
            else:
                # Standard SDR encoding
                cmd.extend(['-c:v', encoder_settings.get('codec', 'libx264') if encoder_settings else 'libx264'])
                if encoder_settings:
                    if 'preset' in encoder_settings:
                        cmd.extend(['-preset', encoder_settings['preset']])
                    if 'crf' in encoder_settings:
                        cmd.extend(['-crf', encoder_settings['crf']])
            
            # Audio copy
            cmd.extend(['-c:a', 'copy'])
            
            # Output
            cmd.append(output_path)
            
            logger.info(f"Processing HDR video: {input_path} -> {output_path}")
            logger.debug(f"HDR command: {' '.join(cmd)}")
            
            # Execute processing
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.success(f"HDR processing completed: {output_path}")
                return True
            else:
                logger.error(f"HDR processing failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing HDR video: {e}")
            return False
    
    def create_hdr_test_pattern(self, output_path: str, duration: int = 10) -> bool:
        """Create HDR test pattern for validation"""
        
        try:
            # Create HDR test pattern with various brightness levels
            cmd = [
                'ffmpeg', '-y', '-hide_banner',
                '-f', 'lavfi',
                '-i', f'testsrc2=duration={duration}:size=1920x1080:rate=25',
                '-vf', (
                    'split=4[a][b][c][d];'
                    '[a]crop=960:540:0:0,lut=y=val*4[tl];'      # Top-left: 4x brightness
                    '[b]crop=960:540:960:0,lut=y=val*2[tr];'    # Top-right: 2x brightness  
                    '[c]crop=960:540:0:540,lut=y=val*1[bl];'    # Bottom-left: normal
                    '[d]crop=960:540:960:540,lut=y=val*0.5[br];' # Bottom-right: 0.5x brightness
                    '[tl][tr]hstack[top];'
                    '[bl][br]hstack[bottom];'
                    '[top][bottom]vstack'
                ),
                '-c:v', 'libx265',
                '-profile:v', 'main10',
                '-pix_fmt', 'yuv420p10le',
                '-x265-params', (
                    'hdr-opt=1:repeat-headers=1:'
                    'colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:'
                    'master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(40000000,50):'
                    'max-cll=4000,1000'
                ),
                '-colorspace', 'bt2020nc',
                '-color_primaries', 'bt2020', 
                '-color_trc', 'smpte2084',
                '-t', str(duration),
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.success(f"HDR test pattern created: {output_path}")
                return True
            else:
                logger.error(f"Failed to create HDR test pattern: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating HDR test pattern: {e}")
            return False


# Global instance
hdr_processor = HDRProcessor()