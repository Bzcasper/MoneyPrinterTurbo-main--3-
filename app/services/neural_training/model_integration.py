"""
Neural Model Integration for MoneyPrinterTurbo
==============================================

Integration of trained neural models with the existing video processing pipeline
including model serving, inference optimization, A/B testing, and monitoring.

Features:
- Model serving and inference optimization
- A/B testing framework for model performance
- Model monitoring and drift detection
- Integration with existing video processing workflow
- Performance benchmarking and comparison

Author: ML Model Developer
Version: 1.0.0
"""

import os
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import statistics
import warnings

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

from loguru import logger
from app.services.gpu_manager import get_gpu_manager, GPUVendor
from app.services.quality_enhancement import QualityEnhancementConfig
from app.services.neural_training.video_enhancement_models import ModelFactory, load_pretrained_model
from app.services.neural_training.training_infrastructure import ModelType, get_model_versioning


class InferenceMode(Enum):
    """Inference execution modes"""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class ModelStatus(Enum):
    """Model deployment status"""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UPDATING = "updating"


@dataclass
class InferenceConfig:
    """Configuration for model inference"""
    device: str = "auto"  # auto, cpu, cuda:0, etc.
    batch_size: int = 1
    max_batch_size: int = 8
    timeout_seconds: float = 30.0
    use_mixed_precision: bool = True
    enable_optimization: bool = True
    cache_results: bool = True
    
    # TensorRT optimization (if available)
    use_tensorrt: bool = False
    tensorrt_precision: str = "fp16"  # fp32, fp16, int8
    
    # ONNX optimization (if available)
    use_onnx: bool = False
    onnx_providers: List[str] = field(default_factory=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'])


@dataclass
class ModelMetrics:
    """Metrics for model performance monitoring"""
    model_id: str
    inference_time: float
    memory_usage: float
    accuracy_score: Optional[float] = None
    quality_score: Optional[float] = None
    user_rating: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "inference_time": self.inference_time,
            "memory_usage": self.memory_usage,
            "accuracy_score": self.accuracy_score,
            "quality_score": self.quality_score,
            "user_rating": self.user_rating,
            "timestamp": self.timestamp.isoformat()
        }


class ModelServer:
    """Neural model serving system"""
    
    def __init__(self, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.gpu_manager = get_gpu_manager()
        self.model_versioning = get_model_versioning()
        
        # Model registry
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.model_status: Dict[str, ModelStatus] = {}
        
        # Performance monitoring
        self.metrics_history: List[ModelMetrics] = []
        self.performance_cache: Dict[str, Any] = {}
        
        # Device selection
        self._setup_device()
        
        logger.info(f"Neural model server initialized on device: {self.device}")
    
    def _setup_device(self):
        """Setup inference device"""
        if self.config.device == "auto":
            # Auto-select best available device
            best_gpu = self.gpu_manager.get_best_gpu_for_task(
                required_memory_mb=1024,
                preferred_vendor=GPUVendor.NVIDIA
            )
            
            if best_gpu and torch.cuda.is_available():
                self.device = f"cuda:{best_gpu.id}"
                logger.info(f"Auto-selected GPU: {best_gpu.name}")
            else:
                self.device = "cpu"
                logger.info("Using CPU for inference")
        else:
            self.device = self.config.device
            
        # Validate device
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
    
    def load_model(
        self,
        model_type: ModelType,
        model_id: str = None,
        model_path: str = None,
        version: str = "latest",
        **model_kwargs
    ) -> str:
        """Load a neural model for inference"""
        
        if model_id is None:
            model_id = f"{model_type.value}_{version}_{int(time.time())}"
        
        logger.info(f"Loading model: {model_id} ({model_type.value})")
        
        try:
            self.model_status[model_id] = ModelStatus.LOADING
            
            # Get model path from registry if not provided
            if model_path is None:
                model_path = self.model_versioning.get_model_path(model_type, version)
                if model_path is None:
                    raise FileNotFoundError(f"No model found for {model_type.value} version {version}")
            
            # Load model
            model = load_pretrained_model(model_path, model_type.value, **model_kwargs)
            model.to(self.device)
            model.eval()
            
            # Optimize model if enabled
            if self.config.enable_optimization:
                model = self._optimize_model(model, model_type)
            
            # Store model info
            self.loaded_models[model_id] = {
                'model': model,
                'model_type': model_type,
                'model_path': model_path,
                'version': version,
                'load_time': datetime.now(),
                'inference_count': 0,
                'total_time': 0.0
            }
            
            self.model_status[model_id] = ModelStatus.READY
            logger.success(f"Model loaded successfully: {model_id}")
            
            return model_id
            
        except Exception as e:
            self.model_status[model_id] = ModelStatus.ERROR
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise
    
    def _optimize_model(self, model: nn.Module, model_type: ModelType) -> nn.Module:
        """Optimize model for inference"""
        try:
            # Compile model (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.device.startswith('cuda'):
                logger.info("Compiling model with torch.compile")
                model = torch.compile(model, mode='reduce-overhead')
            
            # Enable mixed precision if supported
            if self.config.use_mixed_precision and self.device.startswith('cuda'):
                model = model.half()
                logger.info("Enabled mixed precision inference")
            
            # TensorRT optimization (if available)
            if self.config.use_tensorrt:
                try:
                    import torch_tensorrt
                    logger.info("Optimizing with TensorRT")
                    # Note: This would require proper TensorRT setup and example inputs
                    # model = torch_tensorrt.compile(model, inputs=example_inputs)
                except ImportError:
                    logger.warning("TensorRT not available, skipping optimization")
            
            return model
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {str(e)}")
            return model
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            self.model_status.pop(model_id, None)
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Model unloaded: {model_id}")
            return True
        
        return False
    
    def get_model_status(self, model_id: str) -> Optional[ModelStatus]:
        """Get status of a loaded model"""
        return self.model_status.get(model_id)
    
    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """List all loaded models"""
        models = []
        
        for model_id, model_info in self.loaded_models.items():
            models.append({
                'model_id': model_id,
                'model_type': model_info['model_type'].value,
                'version': model_info['version'],
                'status': self.model_status.get(model_id, ModelStatus.ERROR).value,
                'load_time': model_info['load_time'].isoformat(),
                'inference_count': model_info['inference_count'],
                'avg_inference_time': (model_info['total_time'] / model_info['inference_count']) 
                                    if model_info['inference_count'] > 0 else 0.0
            })
        
        return models
    
    async def inference(
        self,
        model_id: str,
        input_data: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        **kwargs
    ) -> Dict[str, Any]:
        """Run inference on loaded model"""
        
        if model_id not in self.loaded_models:
            raise ValueError(f"Model not loaded: {model_id}")
        
        if self.model_status[model_id] != ModelStatus.READY:
            raise RuntimeError(f"Model not ready: {model_id}")
        
        model_info = self.loaded_models[model_id]
        model = model_info['model']
        model_type = model_info['model_type']
        
        start_time = time.time()
        
        try:
            # Preprocess input
            processed_input = self._preprocess_input(input_data, model_type)
            
            # Run inference
            with torch.no_grad():
                if self.config.use_mixed_precision and self.device.startswith('cuda'):
                    with torch.cuda.amp.autocast():
                        output = model(processed_input)
                else:
                    output = model(processed_input)
            
            # Postprocess output
            processed_output = self._postprocess_output(output, model_type)
            
            # Update metrics
            inference_time = time.time() - start_time
            model_info['inference_count'] += 1
            model_info['total_time'] += inference_time
            
            # Record metrics
            memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            metrics = ModelMetrics(
                model_id=model_id,
                inference_time=inference_time,
                memory_usage=memory_usage
            )
            self.metrics_history.append(metrics)
            
            logger.debug(f"Inference completed: {model_id} ({inference_time:.3f}s)")
            
            return {
                'success': True,
                'output': processed_output,
                'inference_time': inference_time,
                'memory_usage': memory_usage,
                'model_id': model_id
            }
            
        except Exception as e:
            logger.error(f"Inference failed for {model_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_id': model_id
            }
    
    def _preprocess_input(
        self,
        input_data: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        model_type: ModelType
    ) -> torch.Tensor:
        """Preprocess input data for model inference"""
        
        if isinstance(input_data, torch.Tensor):
            return input_data.to(self.device)
        
        elif isinstance(input_data, np.ndarray):
            # Convert numpy array to tensor
            if len(input_data.shape) == 3:  # Single image (H, W, C)
                input_data = input_data.transpose(2, 0, 1)  # (C, H, W)
                input_data = np.expand_dims(input_data, 0)  # (1, C, H, W)
            elif len(input_data.shape) == 4:  # Batch of images (B, H, W, C)
                input_data = input_data.transpose(0, 3, 1, 2)  # (B, C, H, W)
            
            tensor = torch.from_numpy(input_data).float()
            
            # Normalize to [0, 1] if needed
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            
            return tensor.to(self.device)
        
        elif isinstance(input_data, list):
            # List of numpy arrays (video sequence)
            processed_frames = []
            
            for frame in input_data:
                if len(frame.shape) == 3:  # (H, W, C)
                    frame = frame.transpose(2, 0, 1)  # (C, H, W)
                
                processed_frames.append(frame)
            
            tensor = torch.from_numpy(np.stack(processed_frames)).float()
            
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            
            return tensor.to(self.device)
        
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _postprocess_output(self, output: torch.Tensor, model_type: ModelType) -> Any:
        """Postprocess model output"""
        
        if isinstance(output, dict):
            # Handle dictionary outputs (e.g., from scene detection model)
            processed_output = {}
            
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    processed_output[key] = value.cpu().numpy()
                else:
                    processed_output[key] = value
            
            return processed_output
        
        elif isinstance(output, torch.Tensor):
            # Convert tensor to numpy
            output_np = output.cpu().numpy()
            
            # Handle different model types
            if model_type in [ModelType.VIDEO_UPSCALER, ModelType.QUALITY_ENHANCER, ModelType.STYLE_TRANSFER]:
                # Image/video output - ensure proper range and format
                if output_np.max() <= 1.0:
                    output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
                
                # Convert from (B, C, H, W) to (B, H, W, C)
                if len(output_np.shape) == 4:
                    output_np = output_np.transpose(0, 2, 3, 1)
                elif len(output_np.shape) == 3:
                    output_np = output_np.transpose(1, 2, 0)
            
            return output_np
        
        else:
            return output


class ABTestingFramework:
    """A/B testing framework for model performance comparison"""
    
    def __init__(self, model_server: ModelServer):
        self.model_server = model_server
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.experiment_results: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("A/B Testing framework initialized")
    
    def create_experiment(
        self,
        experiment_name: str,
        model_a_id: str,
        model_b_id: str,
        traffic_split: float = 0.5,
        duration_hours: int = 24,
        metrics: List[str] = None
    ) -> str:
        """Create A/B testing experiment"""
        
        experiment_id = str(uuid.uuid4())
        
        if metrics is None:
            metrics = ['inference_time', 'memory_usage', 'quality_score']
        
        experiment = {
            'experiment_id': experiment_id,
            'name': experiment_name,
            'model_a_id': model_a_id,
            'model_b_id': model_b_id,
            'traffic_split': traffic_split,
            'duration_hours': duration_hours,
            'metrics': metrics,
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(hours=duration_hours),
            'status': 'active',
            'results': {'model_a': [], 'model_b': []}
        }
        
        self.experiments[experiment_id] = experiment
        self.experiment_results[experiment_id] = []
        
        logger.info(f"Created A/B experiment: {experiment_name} ({experiment_id})")
        return experiment_id
    
    async def run_experiment_inference(
        self,
        experiment_id: str,
        input_data: Any,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Run inference through A/B experiment"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment = self.experiments[experiment_id]
        
        # Check if experiment is still active
        if datetime.now() > experiment['end_time']:
            experiment['status'] = 'completed'
            return {'error': 'Experiment has ended'}
        
        # Determine which model to use
        if np.random.random() < experiment['traffic_split']:
            model_id = experiment['model_a_id']
            variant = 'model_a'
        else:
            model_id = experiment['model_b_id']
            variant = 'model_b'
        
        # Run inference
        result = await self.model_server.inference(model_id, input_data)
        
        # Record result
        experiment_result = {
            'timestamp': datetime.now(),
            'variant': variant,
            'model_id': model_id,
            'user_id': user_id,
            'inference_time': result.get('inference_time', 0),
            'memory_usage': result.get('memory_usage', 0),
            'success': result.get('success', False)
        }
        
        experiment['results'][variant].append(experiment_result)
        self.experiment_results[experiment_id].append(experiment_result)
        
        return {
            'result': result,
            'variant': variant,
            'experiment_id': experiment_id
        }
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get results for an experiment"""
        
        if experiment_id not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_id]
        results = experiment['results']
        
        # Calculate statistics
        stats = {}
        
        for variant in ['model_a', 'model_b']:
            variant_results = results[variant]
            
            if variant_results:
                inference_times = [r['inference_time'] for r in variant_results if r['success']]
                memory_usage = [r['memory_usage'] for r in variant_results if r['success']]
                success_rate = sum(1 for r in variant_results if r['success']) / len(variant_results)
                
                stats[variant] = {
                    'total_requests': len(variant_results),
                    'success_rate': success_rate,
                    'avg_inference_time': statistics.mean(inference_times) if inference_times else 0,
                    'median_inference_time': statistics.median(inference_times) if inference_times else 0,
                    'avg_memory_usage': statistics.mean(memory_usage) if memory_usage else 0,
                    'p95_inference_time': np.percentile(inference_times, 95) if inference_times else 0
                }
            else:
                stats[variant] = {
                    'total_requests': 0,
                    'success_rate': 0.0,
                    'avg_inference_time': 0.0,
                    'median_inference_time': 0.0,
                    'avg_memory_usage': 0.0,
                    'p95_inference_time': 0.0
                }
        
        return {
            'experiment': experiment,
            'statistics': stats,
            'winner': self._determine_winner(stats)
        }
    
    def _determine_winner(self, stats: Dict[str, Dict[str, float]]) -> str:
        """Determine experiment winner based on statistics"""
        model_a_stats = stats['model_a']
        model_b_stats = stats['model_b']
        
        # Simple scoring based on inference time and success rate
        model_a_score = (
            model_a_stats['success_rate'] * 100 +
            (1 / (model_a_stats['avg_inference_time'] + 0.001)) * 10
        )
        
        model_b_score = (
            model_b_stats['success_rate'] * 100 +
            (1 / (model_b_stats['avg_inference_time'] + 0.001)) * 10
        )
        
        if model_a_score > model_b_score:
            return 'model_a'
        elif model_b_score > model_a_score:
            return 'model_b'
        else:
            return 'tie'


class ModelMonitor:
    """Model performance monitoring and drift detection"""
    
    def __init__(self, model_server: ModelServer):
        self.model_server = model_server
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.alert_thresholds = {
            'inference_time_increase': 2.0,  # 2x slower
            'memory_usage_increase': 1.5,    # 50% more memory
            'error_rate_threshold': 0.05     # 5% error rate
        }
        
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        logger.info("Model monitor initialized")
    
    def set_baseline_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Set baseline metrics for a model"""
        self.baseline_metrics[model_id] = metrics
        logger.info(f"Set baseline metrics for model: {model_id}")
    
    def start_monitoring(self, check_interval: int = 300):  # 5 minutes
        """Start continuous monitoring"""
        
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(check_interval,),
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Model monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring_active = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10.0)
        
        logger.info("Model monitoring stopped")
    
    def _monitor_loop(self, check_interval: int):
        """Main monitoring loop"""
        
        while self._monitoring_active:
            try:
                self._check_model_performance()
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _check_model_performance(self):
        """Check performance of all loaded models"""
        
        current_time = datetime.now()
        window_start = current_time - timedelta(hours=1)  # Check last hour
        
        for model_id in self.model_server.loaded_models.keys():
            try:
                # Get recent metrics
                recent_metrics = [
                    m for m in self.model_server.metrics_history
                    if m.model_id == model_id and m.timestamp >= window_start
                ]
                
                if not recent_metrics:
                    continue
                
                # Calculate current performance
                current_performance = {
                    'avg_inference_time': statistics.mean([m.inference_time for m in recent_metrics]),
                    'avg_memory_usage': statistics.mean([m.memory_usage for m in recent_metrics]),
                    'error_rate': 0.0  # Would need to track errors separately
                }
                
                # Check for performance degradation
                self._check_performance_alerts(model_id, current_performance)
                
            except Exception as e:
                logger.error(f"Error checking performance for {model_id}: {str(e)}")
    
    def _check_performance_alerts(self, model_id: str, current_performance: Dict[str, float]):
        """Check for performance alerts"""
        
        if model_id not in self.baseline_metrics:
            return
        
        baseline = self.baseline_metrics[model_id]
        
        # Check inference time
        if 'avg_inference_time' in baseline:
            baseline_time = baseline['avg_inference_time']
            current_time = current_performance['avg_inference_time']
            
            if current_time > baseline_time * self.alert_thresholds['inference_time_increase']:
                logger.warning(
                    f"Performance alert: {model_id} inference time increased "
                    f"{current_time:.3f}s vs baseline {baseline_time:.3f}s"
                )
        
        # Check memory usage
        if 'avg_memory_usage' in baseline:
            baseline_memory = baseline['avg_memory_usage']
            current_memory = current_performance['avg_memory_usage']
            
            if current_memory > baseline_memory * self.alert_thresholds['memory_usage_increase']:
                logger.warning(
                    f"Performance alert: {model_id} memory usage increased "
                    f"{current_memory:.1f}MB vs baseline {baseline_memory:.1f}MB"
                )


class NeuralVideoProcessor:
    """Integration wrapper for neural models in video processing pipeline"""
    
    def __init__(self):
        self.model_server = ModelServer()
        self.ab_testing = ABTestingFramework(self.model_server)
        self.monitor = ModelMonitor(self.model_server)
        
        # Load default models
        self._load_default_models()
        
        logger.info("Neural video processor initialized")
    
    def _load_default_models(self):
        """Load commonly used models"""
        try:
            # Load video upscaler
            upscaler_id = self.model_server.load_model(
                ModelType.VIDEO_UPSCALER,
                model_id="default_upscaler",
                scale_factor=2,
                num_channels=3
            )
            
            # Load quality enhancer
            quality_id = self.model_server.load_model(
                ModelType.QUALITY_ENHANCER,
                model_id="default_quality",
                num_channels=3
            )
            
            logger.info(f"Loaded default models: {upscaler_id}, {quality_id}")
            
        except Exception as e:
            logger.warning(f"Failed to load default models: {str(e)}")
    
    async def enhance_video_quality(
        self,
        input_frames: Union[np.ndarray, List[np.ndarray]],
        model_id: str = "default_quality"
    ) -> np.ndarray:
        """Enhance video quality using neural model"""
        
        result = await self.model_server.inference(model_id, input_frames)
        
        if result['success']:
            return result['output']
        else:
            raise RuntimeError(f"Quality enhancement failed: {result['error']}")
    
    async def upscale_video(
        self,
        input_frames: Union[np.ndarray, List[np.ndarray]],
        model_id: str = "default_upscaler"
    ) -> np.ndarray:
        """Upscale video using neural model"""
        
        result = await self.model_server.inference(model_id, input_frames)
        
        if result['success']:
            return result['output']
        else:
            raise RuntimeError(f"Video upscaling failed: {result['error']}")
    
    async def detect_scenes(
        self,
        input_frames: Union[np.ndarray, List[np.ndarray]],
        model_id: str = "scene_detector"
    ) -> Dict[str, Any]:
        """Detect scenes using neural model"""
        
        result = await self.model_server.inference(model_id, input_frames)
        
        if result['success']:
            return result['output']
        else:
            raise RuntimeError(f"Scene detection failed: {result['error']}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        
        models = self.model_server.list_loaded_models()
        metrics_summary = {}
        
        for model in models:
            model_id = model['model_id']
            
            # Get recent metrics
            recent_metrics = [
                m for m in self.model_server.metrics_history
                if m.model_id == model_id and 
                m.timestamp >= datetime.now() - timedelta(hours=1)
            ]
            
            if recent_metrics:
                metrics_summary[model_id] = {
                    'model_info': model,
                    'recent_requests': len(recent_metrics),
                    'avg_inference_time': statistics.mean([m.inference_time for m in recent_metrics]),
                    'avg_memory_usage': statistics.mean([m.memory_usage for m in recent_metrics])
                }
        
        return metrics_summary


# Global neural video processor instance
neural_processor = NeuralVideoProcessor()


def get_neural_processor() -> NeuralVideoProcessor:
    """Get global neural video processor instance"""
    return neural_processor


# Export main classes and functions
__all__ = [
    'InferenceConfig',
    'ModelMetrics',
    'ModelServer',
    'ABTestingFramework',
    'ModelMonitor',
    'NeuralVideoProcessor',
    'get_neural_processor'
]