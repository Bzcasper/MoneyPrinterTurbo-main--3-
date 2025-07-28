"""
Adaptive Interpolation Algorithms for Frame Rate Enhancement
===========================================================

Intelligent algorithms that adapt interpolation strategy based on:
- Scene content analysis
- Motion complexity assessment  
- Temporal pattern recognition
- Quality metric optimization
- Dynamic algorithm selection

Author: FrameInterpolator Agent
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import math

from loguru import logger


class InterpolationStrategy(Enum):
    """Available interpolation strategies"""
    LINEAR = "linear"
    RIFE = "rife"
    DAIN = "dain"
    ADACOF = "adacof"
    SEPCONV = "sepconv"
    HYBRID = "hybrid"


class SceneComplexity(Enum):
    """Scene complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXTREME = "extreme"


class SceneAnalyzer(nn.Module):
    """
    Analyzes scene content to determine optimal interpolation strategy
    """
    
    def __init__(self, feature_channels: int = 128):
        super().__init__()
        
        self.feature_channels = feature_channels
        
        # Feature extraction backbone
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(6, 64, 7, stride=2, padding=3),  # 2 frames concatenated
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Motion complexity analyzer
        self.motion_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),  # 4 complexity levels
            nn.Softmax(dim=1)
        )
        
        # Texture complexity analyzer
        self.texture_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),  # 4 texture complexity levels
            nn.Softmax(dim=1)
        )
        
        # Edge density analyzer
        self.edge_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True), 
            nn.Linear(64, 3),  # Low, medium, high edge density
            nn.Softmax(dim=1)
        )
        
        # Strategy recommendation network
        self.strategy_recommender = nn.Sequential(
            nn.Linear(256 + 4 + 4 + 3, 256),  # features + analyses
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(InterpolationStrategy)),
            nn.Softmax(dim=1)
        )
    
    def compute_optical_flow_magnitude(self, frame1: torch.Tensor, 
                                     frame2: torch.Tensor) -> torch.Tensor:
        """Compute simple optical flow magnitude for motion analysis"""
        # Simple frame difference as motion proxy
        diff = torch.abs(frame2 - frame1)
        motion_magnitude = torch.mean(diff, dim=1, keepdim=True)
        return torch.mean(motion_magnitude, dim=[2, 3])  # (B, 1)
    
    def compute_texture_complexity(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute texture complexity using local variance"""
        # Convert to grayscale
        gray = torch.mean(frames, dim=1, keepdim=True)
        
        # Compute local variance using convolution
        kernel_size = 5
        padding = kernel_size // 2
        
        # Mean filter
        mean_kernel = torch.ones(1, 1, kernel_size, kernel_size, 
                               device=frames.device) / (kernel_size * kernel_size)
        local_mean = F.conv2d(gray, mean_kernel, padding=padding)
        
        # Variance computation
        squared_diff = (gray - local_mean) ** 2
        local_variance = F.conv2d(squared_diff, mean_kernel, padding=padding)
        
        # Average texture complexity
        texture_complexity = torch.mean(local_variance, dim=[2, 3])
        return texture_complexity
    
    def compute_edge_density(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute edge density using Sobel filters"""
        # Convert to grayscale
        gray = torch.mean(frames, dim=1, keepdim=True)
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=frames.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=frames.device).view(1, 1, 3, 3)
        
        # Compute gradients
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        
        # Edge magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        edge_density = torch.mean(edge_magnitude, dim=[2, 3])
        
        return edge_density
    
    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze scene content and recommend interpolation strategy
        
        Args:
            frame1: First frame (B, 3, H, W)
            frame2: Second frame (B, 3, H, W) 
        
        Returns:
            Dictionary with scene analysis and strategy recommendation
        """
        # Concatenate frames for feature extraction
        frame_pair = torch.cat([frame1, frame2], dim=1)
        
        # Extract features
        features = self.feature_extractor(frame_pair)
        features_flat = features.view(features.size(0), -1)
        
        # Analyze different aspects
        motion_complexity = self.motion_analyzer(features_flat)
        texture_complexity = self.texture_analyzer(features_flat)
        edge_density = self.edge_analyzer(features_flat)
        
        # Additional handcrafted features  
        optical_flow_mag = self.compute_optical_flow_magnitude(frame1, frame2)
        texture_var = self.compute_texture_complexity(frame_pair)
        edge_var = self.compute_edge_density(frame_pair)
        
        # Combine all analyses
        combined_analysis = torch.cat([
            features_flat,
            motion_complexity,
            texture_complexity, 
            edge_density
        ], dim=1)
        
        # Recommend strategy
        strategy_weights = self.strategy_recommender(combined_analysis)
        recommended_strategy = torch.argmax(strategy_weights, dim=1)
        
        return {
            'strategy_weights': strategy_weights,
            'recommended_strategy': recommended_strategy,
            'motion_complexity': motion_complexity,
            'texture_complexity': texture_complexity,
            'edge_density': edge_density,
            'optical_flow_magnitude': optical_flow_mag,
            'texture_variance': texture_var,
            'edge_variance': edge_var,
            'scene_complexity': self._determine_scene_complexity(
                motion_complexity, texture_complexity, edge_density
            )
        }
    
    def _determine_scene_complexity(self, motion_comp: torch.Tensor, 
                                   texture_comp: torch.Tensor,
                                   edge_dens: torch.Tensor) -> List[SceneComplexity]:
        """Determine overall scene complexity"""
        batch_size = motion_comp.size(0)
        complexities = []
        
        for i in range(batch_size):
            # Get dominant complexity levels
            motion_level = torch.argmax(motion_comp[i]).item()
            texture_level = torch.argmax(texture_comp[i]).item()
            edge_level = torch.argmax(edge_dens[i]).item()
            
            # Combine scores
            avg_complexity = (motion_level + texture_level + edge_level) / 3.0
            
            if avg_complexity < 1.0:
                complexities.append(SceneComplexity.SIMPLE)
            elif avg_complexity < 2.0:
                complexities.append(SceneComplexity.MODERATE)
            elif avg_complexity < 3.0:
                complexities.append(SceneComplexity.COMPLEX)
            else:
                complexities.append(SceneComplexity.EXTREME)
        
        return complexities


class QualityMetrics(nn.Module):
    """
    Neural network for predicting interpolation quality metrics
    """
    
    def __init__(self):
        super().__init__()
        
        # Feature extraction for quality assessment
        self.quality_extractor = nn.Sequential(
            nn.Conv2d(9, 64, 3, padding=1),  # frame1, frame2, interpolated
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Quality predictors
        self.psnr_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.ReLU(inplace=True)  # PSNR is positive
        )
        
        self.ssim_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # SSIM is in [0, 1]
        )
        
        self.lpips_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # LPIPS is in [0, 1]
        )
    
    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor, 
                interpolated: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict quality metrics for interpolated frame
        
        Args:
            frame1: First frame (B, 3, H, W)
            frame2: Second frame (B, 3, H, W)
            interpolated: Interpolated frame (B, 3, H, W)
        
        Returns:
            Dictionary with predicted quality metrics
        """
        # Combine all frames
        combined = torch.cat([frame1, frame2, interpolated], dim=1)
        
        # Extract features
        features = self.quality_extractor(combined)
        features_flat = features.view(features.size(0), -1)
        
        # Predict metrics
        psnr = self.psnr_predictor(features_flat)
        ssim = self.ssim_predictor(features_flat)
        lpips = self.lpips_predictor(features_flat)
        
        return {
            'predicted_psnr': psnr,
            'predicted_ssim': ssim,
            'predicted_lpips': lpips
        }


class AdaptiveInterpolationController(nn.Module):
    """
    Main controller that adaptively selects and blends interpolation methods
    """
    
    def __init__(self, available_models: Dict[str, nn.Module]):
        super().__init__()
        
        self.available_models = available_models
        self.model_names = list(available_models.keys())
        
        # Scene analyzer
        self.scene_analyzer = SceneAnalyzer()
        
        # Quality predictor
        self.quality_predictor = QualityMetrics()
        
        # Adaptive blending network
        self.blending_net = nn.Sequential(
            nn.Conv2d(len(self.model_names) * 3, 128, 3, padding=1),  # All interpolated frames
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, len(self.model_names), 3, padding=1),
            nn.Softmax(dim=1)
        )
        
        # Strategy performance tracking
        self.strategy_performance = {}
        for model_name in self.model_names:
            self.strategy_performance[model_name] = {
                'count': 0,
                'avg_quality': 0.0,
                'avg_speed': 0.0
            }
    
    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor, 
                timestep: float = 0.5, adaptive_blend: bool = True) -> Dict[str, Any]:
        """
        Adaptively interpolate frames using optimal strategy
        
        Args:
            frame1: First frame (B, 3, H, W)
            frame2: Second frame (B, 3, H, W)
            timestep: Temporal position (0.0 to 1.0)
            adaptive_blend: Whether to use adaptive blending
        
        Returns:
            Dictionary with interpolation results and analysis
        """
        # Analyze scene content
        scene_analysis = self.scene_analyzer(frame1, frame2)
        
        # Generate interpolations with all available models
        interpolations = {}
        quality_predictions = {}
        
        for model_name, model in self.available_models.items():
            try:
                with torch.no_grad():
                    result = model(frame1, frame2, timestep)
                    interpolated = result['interpolated_frame']
                    interpolations[model_name] = interpolated
                    
                    # Predict quality
                    quality_pred = self.quality_predictor(frame1, frame2, interpolated)
                    quality_predictions[model_name] = quality_pred
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {str(e)}")
                continue
        
        if not interpolations:
            raise RuntimeError("No interpolation models succeeded")
        
        # Select best strategy based on analysis
        if adaptive_blend and len(interpolations) > 1:
            # Adaptive blending of multiple interpolations
            final_result = self._adaptive_blend(
                interpolations, quality_predictions, scene_analysis
            )
        else:
            # Select single best model
            best_model = self._select_best_model(
                scene_analysis, quality_predictions
            )
            final_result = {
                'interpolated_frame': interpolations[best_model],
                'selected_model': best_model,
                'blend_weights': {best_model: 1.0}
            }
        
        # Combine all results
        result = {
            **final_result,
            'scene_analysis': scene_analysis,
            'individual_interpolations': interpolations,
            'quality_predictions': quality_predictions,
            'strategy_performance': self.strategy_performance
        }
        
        return result
    
    def _adaptive_blend(self, interpolations: Dict[str, torch.Tensor],
                       quality_predictions: Dict[str, Dict[str, torch.Tensor]],
                       scene_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptively blend multiple interpolations"""
        
        # Stack all interpolations
        interpolation_list = list(interpolations.values())
        stacked = torch.cat(interpolation_list, dim=1)  # (B, num_models*3, H, W)
        
        # Compute adaptive blend weights
        blend_weights = self.blending_net(stacked)  # (B, num_models, H, W)
        
        # Apply blending
        blended_frame = torch.zeros_like(interpolation_list[0])
        weight_dict = {}
        
        for i, (model_name, interpolation) in enumerate(interpolations.items()):
            weight = blend_weights[:, i:i+1, :, :]
            blended_frame += weight * interpolation
            weight_dict[model_name] = torch.mean(weight).item()
        
        return {
            'interpolated_frame': blended_frame,
            'blend_weights': weight_dict,
            'spatial_blend_weights': blend_weights
        }
    
    def _select_best_model(self, scene_analysis: Dict[str, Any],
                          quality_predictions: Dict[str, Dict[str, torch.Tensor]]) -> str:
        """Select single best model based on analysis"""
        
        # Get recommended strategy from scene analysis
        strategy_weights = scene_analysis['strategy_weights']
        strategy_idx = torch.argmax(strategy_weights, dim=1)[0].item()
        
        # Map strategy index to model name (simplified)
        strategy_names = [s.value for s in InterpolationStrategy]
        if strategy_idx < len(strategy_names):
            preferred_strategy = strategy_names[strategy_idx]
            
            # Find matching model
            for model_name in self.model_names:
                if preferred_strategy.lower() in model_name.lower():
                    return model_name
        
        # Fallback: select based on predicted quality
        best_model = None
        best_score = -float('inf')
        
        for model_name, quality_pred in quality_predictions.items():
            # Composite quality score
            psnr = quality_pred['predicted_psnr'].item()
            ssim = quality_pred['predicted_ssim'].item()
            lpips = 1.0 - quality_pred['predicted_lpips'].item()  # Lower is better
            
            composite_score = 0.4 * psnr + 0.4 * ssim + 0.2 * lpips
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = model_name
        
        return best_model or list(self.model_names)[0]
    
    def update_performance_stats(self, model_name: str, quality_score: float, 
                               processing_time: float):
        """Update performance statistics for strategy selection"""
        if model_name in self.strategy_performance:
            stats = self.strategy_performance[model_name]
            count = stats['count']
            
            # Running average update
            stats['avg_quality'] = (stats['avg_quality'] * count + quality_score) / (count + 1)
            stats['avg_speed'] = (stats['avg_speed'] * count + 1.0/processing_time) / (count + 1)
            stats['count'] = count + 1


class DynamicFPSController:
    """
    Controls dynamic frame rate enhancement based on content and resources
    """
    
    def __init__(self, target_fps_range: Tuple[int, int] = (30, 120)):
        self.min_fps, self.max_fps = target_fps_range
        self.current_fps = 30  # Default
        
        # Performance tracking
        self.frame_processing_times = []
        self.quality_scores = []
        
    def adapt_target_fps(self, scene_complexity: SceneComplexity, 
                        available_compute: float, quality_threshold: float = 0.8) -> int:
        """
        Dynamically adapt target FPS based on scene and resources
        
        Args:
            scene_complexity: Analyzed scene complexity
            available_compute: Available computational resources (0-1)
            quality_threshold: Minimum quality threshold
        
        Returns:
            Optimal target FPS
        """
        # Base FPS based on scene complexity
        if scene_complexity == SceneComplexity.SIMPLE:
            base_fps = self.max_fps
        elif scene_complexity == SceneComplexity.MODERATE:
            base_fps = int(self.max_fps * 0.8)
        elif scene_complexity == SceneComplexity.COMPLEX:
            base_fps = int(self.max_fps * 0.6)
        else:  # EXTREME
            base_fps = int(self.max_fps * 0.4)
        
        # Adjust based on available compute
        compute_adjusted_fps = int(base_fps * available_compute)
        
        # Ensure within bounds
        target_fps = max(self.min_fps, min(self.max_fps, compute_adjusted_fps))
        
        # Update current FPS with smoothing
        alpha = 0.3  # Smoothing factor
        self.current_fps = int(alpha * target_fps + (1 - alpha) * self.current_fps)
        
        logger.info(f"Adapted target FPS: {self.current_fps} "
                   f"(complexity: {scene_complexity.value}, compute: {available_compute:.2f})")
        
        return self.current_fps
    
    def get_interpolation_schedule(self, source_fps: int, target_fps: int) -> List[float]:
        """
        Generate interpolation schedule for FPS conversion
        
        Args:
            source_fps: Source frame rate
            target_fps: Target frame rate
        
        Returns:
            List of timesteps for interpolation
        """
        if target_fps <= source_fps:
            return []  # No interpolation needed
        
        ratio = target_fps / source_fps
        timesteps = []
        
        for i in range(int(ratio) - 1):
            timestep = (i + 1) / ratio
            timesteps.append(timestep)
        
        return timesteps


# Factory function for creating adaptive interpolation system
def create_adaptive_interpolator(model_configs: Dict[str, Dict[str, Any]]) -> AdaptiveInterpolationController:
    """
    Factory function to create adaptive interpolation system
    
    Args:
        model_configs: Dictionary with model configurations
    
    Returns:
        Configured adaptive interpolation controller
    """
    # Import interpolation models
    from .frame_interpolation_models import (
        RIFEModel, DAINModel, AdaCOFModel, SepConvModel
    )
    
    # Create model instances
    available_models = {}
    
    for model_name, config in model_configs.items():
        if model_name.lower() == 'rife':
            available_models['rife'] = RIFEModel(**config)
        elif model_name.lower() == 'dain':
            available_models['dain'] = DAINModel(**config)
        elif model_name.lower() == 'adacof':
            available_models['adacof'] = AdaCOFModel(**config)
        elif model_name.lower() == 'sepconv':
            available_models['sepconv'] = SepConvModel(**config)
    
    return AdaptiveInterpolationController(available_models)


# Export main classes and functions
__all__ = [
    'InterpolationStrategy',
    'SceneComplexity',
    'SceneAnalyzer',
    'QualityMetrics',
    'AdaptiveInterpolationController',
    'DynamicFPSController',
    'create_adaptive_interpolator'
]