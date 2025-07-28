#!/usr/bin/env python3
"""
Codec Quality Assessment Coordinator for MoneyPrinter Turbo Enhanced

Coordinates with QualityAssessor for:
- Automated quality validation
- Performance benchmarking
- Codec selection optimization
- Quality metrics analysis
- Encoding parameter tuning
"""

import subprocess
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
import numpy as np


@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    psnr: float
    ssim: float
    vmaf: Optional[float] = None
    bitrate_kbps: float = 0.0
    file_size_mb: float = 0.0
    encoding_time_s: float = 0.0
    quality_score: float = 0.0  # Composite quality score


@dataclass
class EncodingTest:
    """Encoding test configuration"""
    test_id: str
    input_path: str
    codec_settings: Dict[str, Any]
    reference_path: Optional[str] = None
    quality_target: str = 'balanced'


class CodecQualityCoordinator:
    """Coordinates codec optimization with quality assessment"""
    
    def __init__(self):
        self.quality_cache = {}
        self.benchmark_results = {}
        self.optimization_history = []
        
        # Quality assessment tools
        self.available_tools = self._detect_quality_tools()
        
        logger.info(f"Quality Coordinator initialized with tools: {list(self.available_tools.keys())}")
    
    def _detect_quality_tools(self) -> Dict[str, bool]:
        """Detect available quality assessment tools"""
        
        tools = {}
        
        # Check for FFmpeg with SSIM/PSNR support
        try:
            result = subprocess.run(['ffmpeg', '-hide_banner', '-filters'], 
                                  capture_output=True, text=True)
            tools['ffmpeg_ssim'] = 'ssim' in result.stdout
            tools['ffmpeg_psnr'] = 'psnr' in result.stdout
        except:
            tools['ffmpeg_ssim'] = False
            tools['ffmpeg_psnr'] = False
        
        # Check for VMAF
        try:
            result = subprocess.run(['ffmpeg', '-hide_banner', '-filters'], 
                                  capture_output=True, text=True)
            tools['vmaf'] = 'libvmaf' in result.stdout
        except:
            tools['vmaf'] = False
        
        # Check for standalone quality assessment tools
        for tool in ['vmafossexec', 'ssim', 'psnr']:
            try:
                result = subprocess.run([tool, '--help'], 
                                      capture_output=True, timeout=5)
                tools[tool] = result.returncode == 0
            except:
                tools[tool] = False
        
        return tools
    
    def assess_encoding_quality(self, encoded_path: str, reference_path: str,
                              codec_settings: Dict[str, Any]) -> QualityMetrics:
        """Assess encoding quality against reference"""
        
        try:
            metrics = QualityMetrics(psnr=0.0, ssim=0.0)
            
            # Get file information
            if os.path.exists(encoded_path):
                file_size = os.path.getsize(encoded_path) / (1024 * 1024)  # MB
                metrics.file_size_mb = file_size
                
                # Get bitrate from file
                bitrate = self._get_video_bitrate(encoded_path)
                metrics.bitrate_kbps = bitrate
            
            # Calculate PSNR
            if self.available_tools.get('ffmpeg_psnr'):
                metrics.psnr = self._calculate_psnr(encoded_path, reference_path)
            
            # Calculate SSIM
            if self.available_tools.get('ffmpeg_ssim'):
                metrics.ssim = self._calculate_ssim(encoded_path, reference_path)
            
            # Calculate VMAF if available
            if self.available_tools.get('vmaf'):
                metrics.vmaf = self._calculate_vmaf(encoded_path, reference_path)
            
            # Calculate composite quality score
            metrics.quality_score = self._calculate_quality_score(metrics)
            
            # Cache results
            cache_key = f"{encoded_path}_{reference_path}"
            self.quality_cache[cache_key] = metrics
            
            logger.info(f"Quality assessment: PSNR={metrics.psnr:.2f}, SSIM={metrics.ssim:.4f}, Score={metrics.quality_score:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error assessing quality: {e}")
            return QualityMetrics(psnr=0.0, ssim=0.0)
    
    def _calculate_psnr(self, encoded_path: str, reference_path: str) -> float:
        """Calculate PSNR between encoded and reference video"""
        
        try:
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', encoded_path, '-i', reference_path,
                '-lavfi', 'psnr=stats_file=/dev/stdout',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse PSNR from output
                output_lines = result.stderr.split('\n')
                for line in output_lines:
                    if 'average:' in line and 'PSNR' in line:
                        # Extract PSNR value
                        psnr_part = line.split('average:')[1].strip()
                        psnr_value = float(psnr_part.split()[0])
                        return psnr_value
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating PSNR: {e}")
            return 0.0
    
    def _calculate_ssim(self, encoded_path: str, reference_path: str) -> float:
        """Calculate SSIM between encoded and reference video"""
        
        try:
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', encoded_path, '-i', reference_path,
                '-lavfi', 'ssim=stats_file=/dev/stdout',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse SSIM from output
                output_lines = result.stderr.split('\n')
                for line in output_lines:
                    if 'All:' in line and 'SSIM' in line:
                        # Extract SSIM value
                        ssim_part = line.split('All:')[1].strip()
                        ssim_value = float(ssim_part.split()[0])
                        return ssim_value
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating SSIM: {e}")
            return 0.0
    
    def _calculate_vmaf(self, encoded_path: str, reference_path: str) -> Optional[float]:
        """Calculate VMAF score between encoded and reference video"""
        
        try:
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', encoded_path, '-i', reference_path,
                '-lavfi', 'libvmaf=log_path=/dev/stdout:log_fmt=json',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                try:
                    # Parse VMAF JSON output
                    vmaf_data = json.loads(result.stdout)
                    if 'pooled_metrics' in vmaf_data:
                        vmaf_score = vmaf_data['pooled_metrics']['vmaf']['mean']
                        return float(vmaf_score)
                except json.JSONDecodeError:
                    pass
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating VMAF: {e}")
            return None
    
    def _get_video_bitrate(self, video_path: str) -> float:
        """Get video bitrate in kbps"""
        
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        bit_rate = stream.get('bit_rate')
                        if bit_rate:
                            return float(bit_rate) / 1000  # Convert to kbps
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting bitrate: {e}")
            return 0.0
    
    def _calculate_quality_score(self, metrics: QualityMetrics) -> float:
        """Calculate composite quality score"""
        
        score = 0.0
        weight_sum = 0.0
        
        # PSNR contribution (weight: 0.3)
        if metrics.psnr > 0:
            psnr_normalized = min(1.0, metrics.psnr / 50.0)  # Normalize to 0-1
            score += psnr_normalized * 0.3
            weight_sum += 0.3
        
        # SSIM contribution (weight: 0.4)
        if metrics.ssim > 0:
            score += metrics.ssim * 0.4
            weight_sum += 0.4
        
        # VMAF contribution (weight: 0.3)
        if metrics.vmaf is not None:
            vmaf_normalized = metrics.vmaf / 100.0  # VMAF is 0-100
            score += vmaf_normalized * 0.3
            weight_sum += 0.3
        
        # Normalize by actual weights used
        if weight_sum > 0:
            score = score / weight_sum * 100  # Scale to 0-100
        
        return min(100.0, max(0.0, score))
    
    def benchmark_codec_configurations(self, test_configs: List[EncodingTest],
                                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Benchmark multiple codec configurations"""
        
        logger.info(f"Starting codec benchmark with {len(test_configs)} configurations")
        
        results = {}
        total_tests = len(test_configs)
        
        for i, test in enumerate(test_configs):
            logger.info(f"Running test {i+1}/{total_tests}: {test.test_id}")
            
            # Create test output path
            output_path = f"benchmark_{test.test_id}.mp4"
            
            try:
                # Encode with test settings
                encoding_start = time.time()
                success = self._encode_with_settings(
                    test.input_path, output_path, test.codec_settings
                )
                encoding_time = time.time() - encoding_start
                
                if success and os.path.exists(output_path):
                    # Assess quality
                    reference_path = test.reference_path or test.input_path
                    metrics = self.assess_encoding_quality(
                        output_path, reference_path, test.codec_settings
                    )
                    metrics.encoding_time_s = encoding_time
                    
                    # Calculate efficiency score (quality per bitrate)
                    efficiency = metrics.quality_score / metrics.bitrate_kbps if metrics.bitrate_kbps > 0 else 0
                    
                    results[test.test_id] = {
                        'metrics': metrics,
                        'efficiency': efficiency,
                        'settings': test.codec_settings,
                        'success': True
                    }
                    
                    logger.success(f"Test {test.test_id}: Quality={metrics.quality_score:.1f}, "
                                 f"Efficiency={efficiency:.4f}, Time={encoding_time:.1f}s")
                    
                    # Cleanup
                    os.remove(output_path)
                    
                else:
                    results[test.test_id] = {
                        'success': False,
                        'error': 'Encoding failed',
                        'settings': test.codec_settings
                    }
                    logger.error(f"Test {test.test_id} failed")
                
            except Exception as e:
                results[test.test_id] = {
                    'success': False,
                    'error': str(e),
                    'settings': test.codec_settings
                }
                logger.error(f"Test {test.test_id} error: {e}")
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total_tests, results[test.test_id])
        
        # Analyze results
        analysis = self._analyze_benchmark_results(results)
        
        # Store results
        self.benchmark_results[f"benchmark_{int(time.time())}"] = {
            'results': results,
            'analysis': analysis,
            'timestamp': time.time()
        }
        
        return {
            'results': results,
            'analysis': analysis
        }
    
    def _encode_with_settings(self, input_path: str, output_path: str,
                            settings: Dict[str, Any]) -> bool:
        """Encode video with specified settings"""
        
        try:
            cmd = ['ffmpeg', '-y', '-hide_banner', '-i', input_path]
            
            # Add codec settings
            encoder_type = settings.get('encoder_type', 'software')
            codec = settings.get('codec', 'libx264')
            
            cmd.extend(['-c:v', codec])
            
            # Add encoder-specific parameters
            if encoder_type == 'software':
                if 'preset' in settings:
                    cmd.extend(['-preset', settings['preset']])
                if 'crf' in settings:
                    cmd.extend(['-crf', settings['crf']])
                if 'bitrate' in settings:
                    cmd.extend(['-b:v', settings['bitrate']])
            
            elif encoder_type == 'nvenc':
                if 'preset' in settings:
                    cmd.extend(['-preset', settings['preset']])
                if 'cq' in settings:
                    cmd.extend(['-cq', settings['cq']])
            
            elif encoder_type == 'qsv':
                if 'preset' in settings:
                    cmd.extend(['-preset', settings['preset']])
                if 'global_quality' in settings:
                    cmd.extend(['-global_quality', settings['global_quality']])
            
            # Add common settings
            cmd.extend(['-c:a', 'copy', output_path])
            
            # Execute encoding
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            return False
    
    def _analyze_benchmark_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze benchmark results"""
        
        successful_results = {k: v for k, v in results.items() if v.get('success')}
        
        if not successful_results:
            return {'error': 'No successful encodings'}
        
        # Find best performers
        best_quality = max(successful_results.items(), 
                          key=lambda x: x[1]['metrics'].quality_score)
        best_efficiency = max(successful_results.items(),
                            key=lambda x: x[1]['efficiency'])
        fastest = min(successful_results.items(),
                     key=lambda x: x[1]['metrics'].encoding_time_s)
        
        # Calculate averages
        avg_quality = sum(r['metrics'].quality_score for r in successful_results.values()) / len(successful_results)
        avg_efficiency = sum(r['efficiency'] for r in successful_results.values()) / len(successful_results)
        avg_time = sum(r['metrics'].encoding_time_s for r in successful_results.values()) / len(successful_results)
        
        analysis = {
            'total_tests': len(results),
            'successful_tests': len(successful_results),
            'best_quality': {
                'test_id': best_quality[0],
                'score': best_quality[1]['metrics'].quality_score,
                'settings': best_quality[1]['settings']
            },
            'best_efficiency': {
                'test_id': best_efficiency[0],
                'efficiency': best_efficiency[1]['efficiency'],
                'settings': best_efficiency[1]['settings']
            },
            'fastest': {
                'test_id': fastest[0],
                'time': fastest[1]['metrics'].encoding_time_s,
                'settings': fastest[1]['settings']
            },
            'averages': {
                'quality': avg_quality,
                'efficiency': avg_efficiency,
                'encoding_time': avg_time
            }
        }
        
        logger.info(f"Benchmark analysis: Best quality={avg_quality:.1f}, "
                   f"Best efficiency={best_efficiency[1]['efficiency']:.4f}")
        
        return analysis
    
    def optimize_codec_settings(self, base_settings: Dict[str, Any],
                              input_path: str, target_metric: str = 'efficiency',
                              iterations: int = 5) -> Dict[str, Any]:
        """Optimize codec settings using iterative quality assessment"""
        
        logger.info(f"Starting codec optimization for {target_metric}")
        
        current_settings = base_settings.copy()
        best_settings = current_settings.copy()
        best_score = 0.0
        
        # Parameter ranges for optimization
        param_ranges = self._get_optimization_ranges(base_settings)
        
        for iteration in range(iterations):
            logger.info(f"Optimization iteration {iteration + 1}/{iterations}")
            
            # Generate test configurations
            test_configs = self._generate_optimization_tests(
                current_settings, param_ranges, iteration
            )
            
            # Benchmark test configurations
            benchmark_results = self.benchmark_codec_configurations(test_configs)
            
            # Select best configuration
            best_result = self._select_best_result(
                benchmark_results['results'], target_metric
            )
            
            if best_result:
                score = self._get_result_score(best_result, target_metric)
                
                if score > best_score:
                    best_score = score
                    best_settings = best_result['settings'].copy()
                    current_settings = best_settings.copy()
                    
                    logger.info(f"New best settings found: score={score:.4f}")
                else:
                    logger.info(f"No improvement: score={score:.4f} (best: {best_score:.4f})")
            
            # Refine parameter ranges for next iteration
            param_ranges = self._refine_parameter_ranges(param_ranges, best_settings)
        
        # Store optimization history
        self.optimization_history.append({
            'timestamp': time.time(),
            'base_settings': base_settings,
            'optimized_settings': best_settings,
            'target_metric': target_metric,
            'iterations': iterations,
            'final_score': best_score
        })
        
        logger.success(f"Codec optimization completed: final score={best_score:.4f}")
        return best_settings
    
    def _get_optimization_ranges(self, base_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameter ranges for optimization"""
        
        encoder_type = base_settings.get('encoder_type', 'software')
        ranges = {}
        
        if encoder_type == 'software':
            ranges = {
                'crf': {'min': 18, 'max': 28, 'step': 2},
                'preset': {'values': ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium']},
                'tune': {'values': ['film', 'animation', 'grain', 'stillimage', 'fastdecode']}
            }
        elif encoder_type == 'nvenc':
            ranges = {
                'cq': {'min': 18, 'max': 28, 'step': 2},
                'preset': {'values': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']},
                'rc': {'values': ['vbr', 'cbr', 'constqp']}
            }
        elif encoder_type == 'qsv':
            ranges = {
                'global_quality': {'min': 18, 'max': 28, 'step': 2},
                'preset': {'values': ['veryfast', 'faster', 'fast', 'medium', 'slow']}
            }
        
        return ranges
    
    def _generate_optimization_tests(self, base_settings: Dict[str, Any],
                                   param_ranges: Dict[str, Any],
                                   iteration: int) -> List[EncodingTest]:
        """Generate test configurations for optimization"""
        
        tests = []
        test_counter = 0
        
        # Generate parameter combinations
        for param_name, param_range in param_ranges.items():
            if 'values' in param_range:
                # Discrete values
                for value in param_range['values']:
                    test_settings = base_settings.copy()
                    test_settings[param_name] = value
                    
                    test = EncodingTest(
                        test_id=f"opt_{iteration}_{test_counter}",
                        input_path="test_input.mp4",  # Would be provided
                        codec_settings=test_settings
                    )
                    tests.append(test)
                    test_counter += 1
            
            elif 'min' in param_range and 'max' in param_range:
                # Numeric range
                current_value = base_settings.get(param_name, param_range['min'])
                step = param_range.get('step', 1)
                
                # Test values around current
                for offset in [-step, 0, step]:
                    new_value = current_value + offset
                    if param_range['min'] <= new_value <= param_range['max']:
                        test_settings = base_settings.copy()
                        test_settings[param_name] = str(new_value)
                        
                        test = EncodingTest(
                            test_id=f"opt_{iteration}_{test_counter}",
                            input_path="test_input.mp4",
                            codec_settings=test_settings
                        )
                        tests.append(test)
                        test_counter += 1
        
        return tests
    
    def _select_best_result(self, results: Dict[str, Any], target_metric: str) -> Optional[Dict[str, Any]]:
        """Select best result based on target metric"""
        
        successful_results = {k: v for k, v in results.items() if v.get('success')}
        
        if not successful_results:
            return None
        
        if target_metric == 'quality':
            return max(successful_results.values(), 
                      key=lambda x: x['metrics'].quality_score)
        elif target_metric == 'efficiency':
            return max(successful_results.values(),
                      key=lambda x: x['efficiency'])
        elif target_metric == 'speed':
            return min(successful_results.values(),
                      key=lambda x: x['metrics'].encoding_time_s)
        else:
            return max(successful_results.values(),
                      key=lambda x: x['efficiency'])  # Default to efficiency
    
    def _get_result_score(self, result: Dict[str, Any], target_metric: str) -> float:
        """Get score for result based on target metric"""
        
        if target_metric == 'quality':
            return result['metrics'].quality_score
        elif target_metric == 'efficiency':
            return result['efficiency']
        elif target_metric == 'speed':
            return 1.0 / result['metrics'].encoding_time_s  # Inverse for "higher is better"
        else:
            return result['efficiency']
    
    def _refine_parameter_ranges(self, param_ranges: Dict[str, Any],
                               best_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Refine parameter ranges around best settings"""
        
        refined_ranges = {}
        
        for param_name, param_range in param_ranges.items():
            if param_name in best_settings:
                best_value = best_settings[param_name]
                
                if 'values' in param_range:
                    # For discrete values, keep same range
                    refined_ranges[param_name] = param_range
                elif 'min' in param_range and 'max' in param_range:
                    # For numeric ranges, narrow around best value
                    try:
                        best_numeric = float(best_value)
                        step = param_range.get('step', 1)
                        
                        refined_ranges[param_name] = {
                            'min': max(param_range['min'], best_numeric - step),
                            'max': min(param_range['max'], best_numeric + step),
                            'step': step / 2  # Finer granularity
                        }
                    except ValueError:
                        refined_ranges[param_name] = param_range
                else:
                    refined_ranges[param_name] = param_range
            else:
                refined_ranges[param_name] = param_range
        
        return refined_ranges
    
    def get_quality_assessment_report(self, test_id: str) -> Dict[str, Any]:
        """Generate quality assessment report"""
        
        if test_id in self.benchmark_results:
            benchmark_data = self.benchmark_results[test_id]
            
            report = {
                'test_id': test_id,
                'timestamp': benchmark_data['timestamp'],
                'summary': benchmark_data['analysis'],
                'detailed_results': benchmark_data['results'],
                'recommendations': self._generate_recommendations(benchmark_data)
            }
            
            return report
        
        return {'error': f'Test {test_id} not found'}
    
    def _generate_recommendations(self, benchmark_data: Dict[str, Any]) -> List[str]:
        """Generate codec recommendations based on benchmark results"""
        
        recommendations = []
        analysis = benchmark_data['analysis']
        
        # Quality recommendations
        if analysis['averages']['quality'] < 70:
            recommendations.append("Consider using slower presets or lower CRF values for better quality")
        
        # Efficiency recommendations
        best_efficiency = analysis['best_efficiency']['efficiency']
        if best_efficiency < 0.01:
            recommendations.append("Consider hardware acceleration for better encoding efficiency")
        
        # Speed recommendations
        if analysis['averages']['encoding_time'] > 300:  # 5 minutes
            recommendations.append("Consider faster presets or hardware acceleration for better speed")
        
        return recommendations


# Global instance
codec_quality_coordinator = CodecQualityCoordinator()