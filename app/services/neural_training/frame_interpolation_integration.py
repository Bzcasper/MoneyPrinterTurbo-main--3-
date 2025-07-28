"""
Frame Interpolation Integration with Video Pipeline
==================================================

Seamless integration of neural frame interpolation with the existing video pipeline:
- Pipeline integration points
- AIUpscaler coordination
- Batch processing optimization
- Memory management
- Performance monitoring

Author: FrameInterpolator Agent  
Version: 1.0.0
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
from pathlib import Path
import cv2

from loguru import logger

# Local imports
from .frame_interpolation_models import (
    RIFEModel, DAINModel, AdaCOFModel, SepConvModel, 
    FrameInterpolationPipeline, create_frame_interpolator
)
from .adaptive_interpolation import (
    AdaptiveInterpolationController, DynamicFPSController,
    SceneAnalyzer, create_adaptive_interpolator
)
from .motion_compensation import MotionCompensationPipeline
from .temporal_consistency import (
    TemporalConsistencyLoss, MultiFrameTemporalModel,
    compute_temporal_consistency_metrics
)


class FrameInterpolationManager:
    """
    Central manager for neural frame interpolation in video pipeline
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.models = {}
        self.adaptive_controller = None
        self.motion_compensator = None
        self.fps_controller = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_frames_processed': 0,
            'total_processing_time': 0.0,
            'average_fps': 0.0,
            'quality_scores': [],
            'memory_usage': []
        }
        
        self._initialize_components()
        logger.info(f"FrameInterpolationManager initialized on {self.device}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'models': {
                'rife': {'num_levels': 4, 'scale_factor': 1.0},
                'dain': {'filter_size': 51},
                'adacof': {'num_flows': 4, 'kernel_size': 5},
                'sepconv': {'kernel_size': 51}
            },
            'adaptive_interpolation': True,
            'motion_compensation': True,
            'temporal_consistency': True,
            'target_fps_range': (30, 120),
            'batch_size': 4,
            'memory_limit_mb': 4096,
            'quality_threshold': 0.8
        }
    
    def _initialize_components(self):
        """Initialize all interpolation components"""
        # Initialize individual models
        model_configs = self.config.get('models', {})
        
        for model_name, model_config in model_configs.items():
            try:
                self.models[model_name] = create_frame_interpolator(
                    model_type=model_name, 
                    device=self.device
                )
                logger.info(f"Initialized {model_name} interpolation model")
            except Exception as e:
                logger.warning(f"Failed to initialize {model_name}: {e}")
        
        # Initialize adaptive controller if enabled
        if self.config.get('adaptive_interpolation', False) and self.models:
            try:
                # Create neural models for adaptive controller
                neural_models = {}
                for model_name in self.models.keys():
                    if model_name == 'rife':
                        neural_models[model_name] = RIFEModel(**model_configs.get(model_name, {}))
                    elif model_name == 'dain':
                        neural_models[model_name] = DAINModel(**model_configs.get(model_name, {}))
                    elif model_name == 'adacof':
                        neural_models[model_name] = AdaCOFModel(**model_configs.get(model_name, {}))
                    elif model_name == 'sepconv':
                        neural_models[model_name] = SepConvModel(**model_configs.get(model_name, {}))
                
                self.adaptive_controller = AdaptiveInterpolationController(neural_models)
                logger.info("Initialized adaptive interpolation controller")
            except Exception as e:
                logger.warning(f"Failed to initialize adaptive controller: {e}")
        
        # Initialize motion compensator if enabled
        if self.config.get('motion_compensation', False):
            try:
                self.motion_compensator = MotionCompensationPipeline(device=self.device)
                logger.info("Initialized motion compensation pipeline")
            except Exception as e:
                logger.warning(f"Failed to initialize motion compensator: {e}")
        
        # Initialize FPS controller
        target_fps_range = self.config.get('target_fps_range', (30, 120))
        self.fps_controller = DynamicFPSController(target_fps_range)
        logger.info(f"Initialized FPS controller with range {target_fps_range}")
    
    async def enhance_video_framerate(self, 
                                    input_video_path: str,
                                    output_video_path: str,
                                    target_fps: int = None,
                                    model_preference: str = None) -> Dict[str, Any]:
        """
        Main entry point for video frame rate enhancement
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to output enhanced video
            target_fps: Target frame rate (auto-determined if None)
            model_preference: Preferred interpolation model
        
        Returns:
            Dictionary with enhancement results and metrics
        """
        start_time = time.time()
        
        try:
            # Load and analyze input video
            video_info = await self._analyze_input_video(input_video_path)
            source_fps = video_info['fps']
            
            # Determine optimal target FPS
            if target_fps is None:
                target_fps = self.fps_controller.adapt_target_fps(
                    scene_complexity=video_info['complexity'],
                    available_compute=self._get_available_compute()
                )
            
            logger.info(f"Enhancing video: {source_fps}fps → {target_fps}fps")
            
            # Process video in batches
            enhancement_result = await self._process_video_batches(
                input_video_path, output_video_path, 
                source_fps, target_fps, model_preference
            )
            
            # Calculate final metrics
            total_time = time.time() - start_time
            self.performance_metrics['total_processing_time'] += total_time
            
            enhancement_result.update({
                'source_fps': source_fps,
                'target_fps': target_fps,
                'processing_time': total_time,
                'speedup_factor': target_fps / source_fps,
                'performance_metrics': self.performance_metrics.copy()
            })
            
            logger.success(f"Video enhancement completed in {total_time:.2f}s")
            return enhancement_result
            
        except Exception as e:
            logger.error(f"Video enhancement failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def _analyze_input_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze input video properties"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get basic properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Sample frames for complexity analysis
        sample_frames = []
        frame_indices = np.linspace(0, frame_count - 1, min(10, frame_count), dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sample_frames.append(frame_rgb)
        
        cap.release()
        
        # Analyze scene complexity using first two frames
        if len(sample_frames) >= 2 and self.adaptive_controller:
            frame1 = torch.from_numpy(sample_frames[0]).permute(2, 0, 1).float() / 255.0
            frame2 = torch.from_numpy(sample_frames[1]).permute(2, 0, 1).float() / 255.0
            
            frame1 = frame1.unsqueeze(0).to(self.device)
            frame2 = frame2.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                analysis = self.adaptive_controller.scene_analyzer(frame1, frame2)
                complexity = analysis['scene_complexity'][0]
        else:
            from .adaptive_interpolation import SceneComplexity
            complexity = SceneComplexity.MODERATE  # Default
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': frame_count / fps,
            'complexity': complexity,
            'sample_frames': sample_frames
        }
    
    async def _process_video_batches(self,
                                   input_path: str,
                                   output_path: str,
                                   source_fps: int,
                                   target_fps: int,
                                   model_preference: str = None) -> Dict[str, Any]:
        """Process video in optimized batches"""
        
        # Calculate interpolation requirements
        interpolation_factor = target_fps / source_fps
        frames_to_generate = int((interpolation_factor - 1) * 100)  # Per 100 source frames
        
        logger.info(f"Interpolation factor: {interpolation_factor:.2f}x")
        logger.info(f"Will generate ~{frames_to_generate} frames per 100 source frames")
        
        # Open video files
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, 
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        batch_size = self.config.get('batch_size', 4)
        frame_buffer = []
        processed_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Store results for analysis
        quality_scores = []
        processing_times = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buffer.append(frame_rgb)
                
                # Process batch when full or at end
                if len(frame_buffer) >= batch_size + 1 or not ret:
                    if len(frame_buffer) >= 2:
                        batch_start = time.time()
                        
                        # Process frame pairs in batch
                        batch_results = await self._process_frame_batch(
                            frame_buffer, interpolation_factor, model_preference
                        )
                        
                        # Write interpolated frames
                        for result_frame in batch_results['frames']:
                            frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                            out.write(frame_bgr)
                        
                        # Update metrics
                        batch_time = time.time() - batch_start
                        processing_times.append(batch_time)
                        
                        if 'quality_score' in batch_results:
                            quality_scores.append(batch_results['quality_score'])
                        
                        processed_count += len(frame_buffer) - 1
                        
                        # Keep last frame for next batch
                        frame_buffer = [frame_buffer[-1]] if frame_buffer else []
                        
                        # Progress update
                        progress = processed_count / total_frames * 100
                        logger.info(f"Progress: {progress:.1f}% "
                                  f"({processed_count}/{total_frames} frames)")
                        
                        # Memory management
                        await self._manage_memory()
            
        finally:
            cap.release()
            out.release()
        
        # Calculate final metrics
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        
        return {
            'success': True,
            'processed_frames': processed_count,
            'quality_score': avg_quality,
            'avg_processing_time': avg_processing_time,
            'interpolation_factor': interpolation_factor
        }
    
    async def _process_frame_batch(self,
                                 frame_buffer: List[np.ndarray],
                                 interpolation_factor: float,
                                 model_preference: str = None) -> Dict[str, Any]:
        """Process a batch of frames with interpolation"""
        
        output_frames = []
        quality_scores = []
        
        # Process consecutive frame pairs
        for i in range(len(frame_buffer) - 1):
            frame1 = frame_buffer[i]
            frame2 = frame_buffer[i + 1]
            
            # Add original frame
            output_frames.append(frame1)
            
            # Generate intermediate frames
            num_intermediate = int(interpolation_factor) - 1
            
            if num_intermediate > 0:
                # Convert to tensors
                tensor1 = torch.from_numpy(frame1).permute(2, 0, 1).float() / 255.0
                tensor2 = torch.from_numpy(frame2).permute(2, 0, 1).float() / 255.0
                tensor1 = tensor1.unsqueeze(0).to(self.device)
                tensor2 = tensor2.unsqueeze(0).to(self.device)
                
                # Generate interpolated frames
                for j in range(1, num_intermediate + 1):
                    timestep = j / (num_intermediate + 1)
                    
                    # Choose interpolation method
                    if self.adaptive_controller and not model_preference:
                        # Adaptive selection
                        with torch.no_grad():
                            result = self.adaptive_controller(tensor1, tensor2, timestep)
                            interpolated = result['interpolated_frame']
                            
                            # Track quality if available
                            if 'quality_predictions' in result:
                                quality_pred = result['quality_predictions']
                                avg_quality = np.mean([
                                    pred['predicted_ssim'].item() 
                                    for pred in quality_pred.values()
                                ])
                                quality_scores.append(avg_quality)
                                
                    else:
                        # Use specific model or default
                        model_name = model_preference or list(self.models.keys())[0]
                        pipeline = self.models[model_name]
                        
                        with torch.no_grad():
                            interpolated = pipeline.model(tensor1, tensor2, timestep)['interpolated_frame']
                    
                    # Convert back to numpy
                    interpolated_np = interpolated.squeeze(0).cpu().numpy()
                    interpolated_np = np.transpose(interpolated_np, (1, 2, 0))
                    interpolated_np = (interpolated_np * 255).astype(np.uint8)
                    
                    output_frames.append(interpolated_np)
        
        # Add the last frame
        if frame_buffer:
            output_frames.append(frame_buffer[-1])
        
        return {
            'frames': output_frames,
            'quality_score': np.mean(quality_scores) if quality_scores else 0.0
        }
    
    async def _manage_memory(self):
        """Manage GPU memory usage"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            # Force garbage collection if memory usage is high
            memory_limit = self.config.get('memory_limit_mb', 4096)
            if current_memory > memory_limit * 0.8:
                torch.cuda.empty_cache()
                logger.debug(f"Cleared CUDA cache. Memory: {current_memory:.1f}MB")
    
    def _get_available_compute(self) -> float:
        """Estimate available computational resources"""
        if torch.cuda.is_available():
            # GPU compute estimation
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            current_memory = torch.cuda.memory_allocated()
            memory_utilization = current_memory / gpu_memory
            
            # Simple heuristic: higher available memory = more compute
            return max(0.1, 1.0 - memory_utilization)
        else:
            # CPU-based estimation
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            return max(0.1, 1.0 - cpu_percent / 100.0)
    
    async def integrate_with_video_pipeline(self,
                                          video_clips: List[str],
                                          target_fps: int,
                                          output_dir: str) -> Dict[str, Any]:
        """
        Integration point with the main video pipeline
        
        Args:
            video_clips: List of video clip paths to enhance
            target_fps: Target frame rate for enhancement
            output_dir: Output directory for enhanced clips
        
        Returns:
            Dictionary with enhancement results
        """
        logger.info(f"Integrating frame interpolation with video pipeline")
        logger.info(f"Processing {len(video_clips)} clips at {target_fps}fps")
        
        enhanced_clips = []
        total_processing_time = 0.0
        
        for i, clip_path in enumerate(video_clips):
            try:
                # Generate output path
                clip_name = Path(clip_path).stem
                output_path = Path(output_dir) / f"{clip_name}_enhanced_{target_fps}fps.mp4"
                
                # Enhance frame rate
                result = await self.enhance_video_framerate(
                    input_video_path=clip_path,
                    output_video_path=str(output_path),
                    target_fps=target_fps
                )
                
                if result.get('success', False):
                    enhanced_clips.append(str(output_path))
                    total_processing_time += result.get('processing_time', 0.0)
                    
                    logger.info(f"Enhanced clip {i+1}/{len(video_clips)}: {output_path}")
                else:
                    logger.warning(f"Failed to enhance clip: {clip_path}")
                    enhanced_clips.append(clip_path)  # Use original
                    
            except Exception as e:
                logger.error(f"Error processing clip {clip_path}: {e}")
                enhanced_clips.append(clip_path)  # Use original
        
        # Coordinate with memory system
        await self._store_coordination_results({
            'enhanced_clips': enhanced_clips,
            'total_processing_time': total_processing_time,
            'target_fps': target_fps,
            'performance_metrics': self.performance_metrics
        })
        
        return {
            'enhanced_clips': enhanced_clips,
            'original_count': len(video_clips),
            'enhanced_count': len([c for c in enhanced_clips if 'enhanced' in c]),
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(video_clips),
            'success_rate': len([c for c in enhanced_clips if 'enhanced' in c]) / len(video_clips)
        }
    
    async def _store_coordination_results(self, results: Dict[str, Any]):
        """Store results in coordination memory system"""
        try:
            import subprocess
            import json
            
            # Store in Claude Flow memory
            memory_data = {
                'timestamp': time.time(),
                'component': 'frame_interpolator',
                'results': results,
                'performance': self.performance_metrics
            }
            
            subprocess.run([
                'npx', 'claude-flow@alpha', 'memory', 'store',
                '--key', f'interpolation/results/{int(time.time())}',
                '--value', json.dumps(memory_data),
                '--namespace', 'video-enhancement'
            ], capture_output=True, timeout=10)
            
            logger.debug("Stored frame interpolation results in coordination memory")
            
        except Exception as e:
            logger.debug(f"Failed to store coordination results: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        metrics = self.performance_metrics.copy()
        
        # Calculate derived metrics
        if metrics['total_frames_processed'] > 0:
            metrics['frames_per_second'] = (
                metrics['total_frames_processed'] / 
                max(metrics['total_processing_time'], 0.001)
            )
        
        if metrics['quality_scores']:
            metrics['average_quality'] = np.mean(metrics['quality_scores'])
            metrics['quality_std'] = np.std(metrics['quality_scores'])
        
        return {
            'performance_metrics': metrics,
            'model_count': len(self.models),
            'device': self.device,
            'config': self.config,
            'adaptive_enabled': self.adaptive_controller is not None,
            'motion_compensation_enabled': self.motion_compensator is not None
        }


# Integration utility functions
async def enhance_video_clips_for_pipeline(
    clip_paths: List[str],
    target_fps: int = 60,
    output_dir: str = None,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Utility function for pipeline integration
    
    Args:
        clip_paths: List of video clip paths
        target_fps: Target frame rate
        output_dir: Output directory (temp if None)
        config: Optional configuration
    
    Returns:
        Enhancement results
    """
    if output_dir is None:
        import tempfile
        output_dir = tempfile.mkdtemp(prefix='frame_interpolation_')
    
    # Initialize manager
    manager = FrameInterpolationManager(config)
    
    # Process clips
    result = await manager.integrate_with_video_pipeline(
        video_clips=clip_paths,
        target_fps=target_fps,
        output_dir=output_dir
    )
    
    return result


def create_interpolation_config(
    models: List[str] = None,
    target_fps_range: Tuple[int, int] = (30, 120),
    adaptive: bool = True,
    device: str = None
) -> Dict[str, Any]:
    """
    Create configuration for frame interpolation
    
    Args:
        models: List of models to use
        target_fps_range: FPS range tuple
        adaptive: Enable adaptive interpolation
        device: Device to use
    
    Returns:
        Configuration dictionary
    """
    if models is None:
        models = ['rife', 'dain']  # Default models
    
    config = {
        'device': device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        'models': {},
        'adaptive_interpolation': adaptive,
        'motion_compensation': True,
        'temporal_consistency': True,
        'target_fps_range': target_fps_range,
        'batch_size': 4,
        'memory_limit_mb': 4096,
        'quality_threshold': 0.8
    }
    
    # Add model configurations
    default_configs = {
        'rife': {'num_levels': 4, 'scale_factor': 1.0},
        'dain': {'filter_size': 51},
        'adacof': {'num_flows': 4, 'kernel_size': 5},
        'sepconv': {'kernel_size': 51}
    }
    
    for model in models:
        if model in default_configs:
            config['models'][model] = default_configs[model]
    
    return config


# Export main classes and functions
__all__ = [
    'FrameInterpolationManager',
    'enhance_video_clips_for_pipeline',
    'create_interpolation_config'
]