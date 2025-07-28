"""
Advanced Motion Compensation for Frame Interpolation
====================================================

Sophisticated motion compensation algorithms including:
- Hierarchical motion estimation
- Sub-pixel motion refinement  
- Motion field regularization
- Occlusion detection and handling
- Adaptive motion models

Author: FrameInterpolator Agent
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import cv2
import math

from loguru import logger


class OpticalFlowPyramid(nn.Module):
    """
    Hierarchical optical flow estimation using image pyramids
    """
    
    def __init__(self, num_levels: int = 4, feature_channels: int = 64):
        super().__init__()
        
        self.num_levels = num_levels
        self.feature_channels = feature_channels
        
        # Feature extraction networks for each pyramid level
        self.feature_extractors = nn.ModuleList()
        for level in range(num_levels):
            # Different receptive fields for different levels
            kernel_size = 3 + 2 * level
            padding = kernel_size // 2
            
            extractor = nn.Sequential(
                nn.Conv2d(3, feature_channels, kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_channels, feature_channels // 2, 3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.feature_extractors.append(extractor)
        
        # Flow estimation networks for each level
        self.flow_estimators = nn.ModuleList()
        for level in range(num_levels):
            input_channels = feature_channels + 2  # features + upsampled flow from coarser level
            if level == num_levels - 1:  # Coarsest level
                input_channels = feature_channels
                
            estimator = nn.Sequential(
                nn.Conv2d(input_channels, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), 
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2, 3, padding=1)  # 2 channels for x,y flow
            )
            self.flow_estimators.append(estimator)
        
        # Flow upsampling layers
        self.upsample_layers = nn.ModuleList()
        for level in range(num_levels - 1):
            upsample = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1, bias=False)
            self.upsample_layers.append(upsample)
    
    def build_pyramid(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Build image pyramid"""
        pyramid = [image]
        
        for level in range(1, self.num_levels):
            # Downsample by factor of 2
            downsampled = F.avg_pool2d(pyramid[-1], kernel_size=2, stride=2)
            pyramid.append(downsampled)
        
        return pyramid[::-1]  # Coarsest to finest
    
    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> Dict[str, Any]:
        """
        Estimate hierarchical optical flow
        
        Args:
            frame1: First frame (B, 3, H, W)
            frame2: Second frame (B, 3, H, W)
        
        Returns:
            Dictionary with flow estimates at multiple scales
        """
        # Build pyramids
        pyramid1 = self.build_pyramid(frame1)
        pyramid2 = self.build_pyramid(frame2)
        
        flows = []
        flow_upsampled = None
        
        # Process from coarsest to finest
        for level in range(self.num_levels):
            # Extract features
            feat1 = self.feature_extractors[level](pyramid1[level])
            feat2 = self.feature_extractors[level](pyramid2[level])
            
            # Concatenate features
            correlation = torch.cat([feat1, feat2], dim=1)
            
            # Add upsampled flow from coarser level
            if flow_upsampled is not None:
                correlation = torch.cat([correlation, flow_upsampled], dim=1)
            
            # Estimate flow at this level
            flow = self.flow_estimators[level](correlation)
            flows.append(flow)
            
            # Upsample flow for next level
            if level < self.num_levels - 1:
                flow_upsampled = self.upsample_layers[level](flow)
                # Scale flow values by upsampling factor
                flow_upsampled = flow_upsampled * 2.0
        
        return {
            'flows': flows[::-1],  # Return finest to coarsest
            'pyramid1': pyramid1[::-1],
            'pyramid2': pyramid2[::-1]
        }


class SubPixelMotionRefinement(nn.Module):
    """
    Sub-pixel motion estimation refinement
    """
    
    def __init__(self, search_radius: int = 4, feature_channels: int = 64):
        super().__init__()
        
        self.search_radius = search_radius
        self.feature_channels = feature_channels
        
        # Feature extraction for sub-pixel refinement
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Sub-pixel refinement network
        cost_volume_channels = (2 * search_radius + 1) ** 2
        self.refinement_net = nn.Sequential(
            nn.Conv2d(cost_volume_channels + 2, 128, 3, padding=1),  # +2 for initial flow
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1)  # Flow refinement
        )
        
        # Generate search grid
        self.register_buffer('search_grid', self._generate_search_grid())
    
    def _generate_search_grid(self) -> torch.Tensor:
        """Generate search grid for sub-pixel refinement"""
        coords = torch.arange(-self.search_radius, self.search_radius + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        search_grid = torch.stack([grid_x, grid_y], dim=0)  # (2, 2*R+1, 2*R+1)
        return search_grid
    
    def compute_cost_volume(self, feat1: torch.Tensor, feat2: torch.Tensor, 
                           initial_flow: torch.Tensor) -> torch.Tensor:
        """
        Compute cost volume for sub-pixel refinement
        
        Args:
            feat1: Features from frame 1 (B, C, H, W)
            feat2: Features from frame 2 (B, C, H, W)
            initial_flow: Initial flow estimate (B, 2, H, W)
        
        Returns:
            Cost volume (B, (2*R+1)^2, H, W)
        """
        B, C, H, W = feat1.shape
        R = self.search_radius
        
        # Warp feat2 using initial flow
        grid = self._flow_to_grid(initial_flow)
        feat2_warped = F.grid_sample(feat2, grid, mode='bilinear', 
                                   padding_mode='border', align_corners=True)
        
        cost_volume = torch.zeros(B, (2*R+1)**2, H, W, device=feat1.device)
        
        # Compute costs for each search position
        for i, (dx, dy) in enumerate(self.search_grid.view(2, -1).t()):
            # Shift warped features
            shifted_feat2 = self._shift_features(feat2_warped, dx.item(), dy.item())
            
            # Compute matching cost (normalized cross-correlation)
            cost = torch.sum(feat1 * shifted_feat2, dim=1, keepdim=True)
            cost_volume[:, i:i+1] = cost
        
        return cost_volume
    
    def _flow_to_grid(self, flow: torch.Tensor) -> torch.Tensor:
        """Convert optical flow to sampling grid"""
        B, _, H, W = flow.shape
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=flow.device),
            torch.arange(0, W, dtype=torch.float32, device=flow.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Add flow
        new_grid = grid + flow
        
        # Normalize to [-1, 1]
        new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0
        new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0
        
        return new_grid.permute(0, 2, 3, 1)
    
    def _shift_features(self, features: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        """Shift features by sub-pixel amounts"""
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return features
        
        B, C, H, W = features.shape
        
        # Create shift grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=features.device),
            torch.arange(0, W, dtype=torch.float32, device=features.device),
            indexing='ij'
        )
        
        # Apply shift
        grid_x = grid_x + dx
        grid_y = grid_y + dy
        
        # Normalize
        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        return F.grid_sample(features, grid, mode='bilinear', 
                           padding_mode='border', align_corners=True)
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor, 
                initial_flow: torch.Tensor) -> torch.Tensor:
        """
        Refine optical flow to sub-pixel accuracy
        
        Args:
            feat1: Features from frame 1 (B, 3, H, W) 
            feat2: Features from frame 2 (B, 3, H, W)
            initial_flow: Initial flow estimate (B, 2, H, W)
        
        Returns:
            Refined optical flow (B, 2, H, W)
        """
        # Extract features
        f1 = self.feature_extractor(feat1)
        f2 = self.feature_extractor(feat2)
        
        # Compute cost volume
        cost_volume = self.compute_cost_volume(f1, f2, initial_flow)
        
        # Predict flow refinement
        refinement_input = torch.cat([cost_volume, initial_flow], dim=1)
        flow_refinement = self.refinement_net(refinement_input)
        
        # Add refinement to initial flow
        refined_flow = initial_flow + flow_refinement
        
        return refined_flow


class MotionFieldRegularizer(nn.Module):
    """
    Motion field regularization for smooth and consistent motion
    """
    
    def __init__(self, regularization_strength: float = 0.1):
        super().__init__()
        self.regularization_strength = regularization_strength
        
        # Spatial smoothness kernels
        self.register_buffer('laplacian_kernel', torch.tensor([
            [[0., -1., 0.],
             [-1., 4., -1.],
             [0., -1., 0.]]
        ]).view(1, 1, 3, 3))
        
        # Edge-preserving regularization network
        self.edge_detector = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def compute_motion_smoothness(self, flow: torch.Tensor) -> torch.Tensor:
        """Compute spatial smoothness of motion field"""
        # Apply Laplacian to both flow components
        flow_x = flow[:, 0:1]
        flow_y = flow[:, 1:2]
        
        smoothness_x = F.conv2d(flow_x, self.laplacian_kernel, padding=1)
        smoothness_y = F.conv2d(flow_y, self.laplacian_kernel, padding=1)
        
        total_smoothness = torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))
        return total_smoothness
    
    def compute_edge_aware_smoothness(self, flow: torch.Tensor, 
                                    reference_frame: torch.Tensor) -> torch.Tensor:
        """Compute edge-aware motion smoothness"""
        # Detect edges in reference frame
        edge_map = self.edge_detector(reference_frame)
        
        # Compute flow gradients
        flow_grad_x = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        flow_grad_y = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        
        # Apply edge-aware weighting
        edge_weight_x = 1.0 - edge_map[:, :, :, 1:]
        edge_weight_y = 1.0 - edge_map[:, :, 1:, :]
        
        weighted_smoothness_x = torch.mean(flow_grad_x * edge_weight_x)
        weighted_smoothness_y = torch.mean(flow_grad_y * edge_weight_y)
        
        return weighted_smoothness_x + weighted_smoothness_y
    
    def forward(self, flow: torch.Tensor, reference_frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply motion field regularization
        
        Args:
            flow: Optical flow field (B, 2, H, W)
            reference_frame: Reference frame for edge detection (B, 3, H, W)
        
        Returns:
            Dictionary with regularization losses
        """
        losses = {}
        
        # Spatial smoothness
        smoothness_loss = self.compute_motion_smoothness(flow)
        losses['spatial_smoothness'] = smoothness_loss * self.regularization_strength
        
        # Edge-aware smoothness
        edge_aware_loss = self.compute_edge_aware_smoothness(flow, reference_frame)
        losses['edge_aware_smoothness'] = edge_aware_loss * self.regularization_strength * 0.5
        
        # Motion magnitude regularization
        flow_magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
        magnitude_loss = torch.mean(flow_magnitude)
        losses['magnitude_regularization'] = magnitude_loss * 0.01
        
        # Total regularization loss
        losses['total_regularization'] = sum(losses.values())
        
        return losses


class OcclusionDetector(nn.Module):
    """
    Occlusion detection for motion compensation
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        
        # Occlusion detection network 
        self.occlusion_net = nn.Sequential(
            nn.Conv2d(8, 64, 3, padding=1),  # 3+3+2 for frame1, frame2, flow
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Warping utility
        self.warping_layer = None  # Will be initialized when needed
    
    def _warp_frame(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp frame using optical flow"""
        B, C, H, W = frame.shape
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=frame.device),
            torch.arange(0, W, dtype=torch.float32, device=frame.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Add flow
        new_grid = grid + flow
        
        # Normalize to [-1, 1]
        new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0
        new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0
        
        # Sample
        warped = F.grid_sample(frame, new_grid.permute(0, 2, 3, 1), 
                              mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped
    
    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor, 
                flow_forward: torch.Tensor, flow_backward: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect occlusions between frames
        
        Args:
            frame1: First frame (B, 3, H, W)
            frame2: Second frame (B, 3, H, W)
            flow_forward: Forward optical flow (B, 2, H, W)
            flow_backward: Backward optical flow (B, 2, H, W)
        
        Returns:
            Dictionary with occlusion maps and metrics
        """
        # Forward-backward consistency check
        warped_flow_backward = self._warp_frame(flow_backward, flow_forward)
        flow_consistency = torch.norm(flow_forward + warped_flow_backward, dim=1, keepdim=True)
        
        # Normalize consistency
        flow_consistency = flow_consistency / (torch.norm(flow_forward, dim=1, keepdim=True) + 1e-8)
        
        # Warping-based occlusion detection
        warped_frame2 = self._warp_frame(frame2, flow_forward)
        warping_error = torch.mean(torch.abs(frame1 - warped_frame2), dim=1, keepdim=True)
        
        # Neural network based occlusion detection
        occlusion_input = torch.cat([frame1, frame2, flow_forward], dim=1)
        neural_occlusion = self.occlusion_net(occlusion_input)
        
        # Combine different occlusion cues
        consistency_occlusion = (flow_consistency > self.threshold).float()
        warping_occlusion = (warping_error > 0.1).float()  # Threshold for warping error
        
        # Final occlusion map (weighted combination)
        final_occlusion = (
            0.4 * neural_occlusion + 
            0.3 * consistency_occlusion + 
            0.3 * warping_occlusion
        )
        final_occlusion = (final_occlusion > 0.5).float()
        
        return {
            'occlusion_map': final_occlusion,
            'flow_consistency': flow_consistency,
            'warping_error': warping_error,
            'neural_occlusion': neural_occlusion,
            'occlusion_ratio': torch.mean(final_occlusion).item()
        }


class AdaptiveMotionModel(nn.Module):
    """
    Adaptive motion model that adjusts based on scene content
    """
    
    def __init__(self, num_motion_types: int = 4):
        super().__init__()
        
        self.num_motion_types = num_motion_types
        
        # Motion type classifier
        self.motion_classifier = nn.Sequential(
            nn.Conv2d(6, 64, 7, stride=2, padding=3),  # 2 frames
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_motion_types),
            nn.Softmax(dim=1)
        )
        
        # Specialized motion estimators for different motion types
        self.motion_estimators = nn.ModuleList([
            OpticalFlowPyramid(num_levels=3) for _ in range(num_motion_types)
        ])
        
        # Motion fusion network
        self.motion_fusion = nn.Sequential(
            nn.Conv2d(2 * num_motion_types + num_motion_types, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1)
        )
    
    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> Dict[str, Any]:
        """
        Adaptively estimate motion based on scene content
        
        Args:
            frame1: First frame (B, 3, H, W)
            frame2: Second frame (B, 3, H, W)
        
        Returns:
            Dictionary with adaptive motion estimation results
        """
        B, C, H, W = frame1.shape
        
        # Classify motion type
        motion_input = torch.cat([frame1, frame2], dim=1)
        motion_weights = self.motion_classifier(motion_input)  # (B, num_motion_types)
        
        # Estimate motion using specialized estimators
        motion_estimates = []
        for estimator in self.motion_estimators:
            result = estimator(frame1, frame2)
            motion_estimates.append(result['flows'][0])  # Use finest flow
        
        # Stack motion estimates
        stacked_motions = torch.stack(motion_estimates, dim=1)  # (B, num_types, 2, H, W)
        
        # Reshape motion weights for broadcasting
        motion_weights_spatial = motion_weights.view(B, self.num_motion_types, 1, 1, 1)
        motion_weights_spatial = motion_weights_spatial.expand(-1, -1, 2, H, W)
        
        # Weighted combination of motion estimates
        adaptive_motion = torch.sum(stacked_motions * motion_weights_spatial, dim=1)
        
        # Prepare fusion input
        fusion_input_list = [adaptive_motion]
        
        # Add individual motion estimates
        for motion_est in motion_estimates:
            fusion_input_list.append(motion_est)
        
        # Add motion weights as spatial maps
        motion_weights_map = motion_weights.view(B, self.num_motion_types, 1, 1)
        motion_weights_map = motion_weights_map.expand(-1, -1, H, W)
        fusion_input_list.append(motion_weights_map)
        
        fusion_input = torch.cat(fusion_input_list, dim=1)
        
        # Final motion fusion
        refined_motion = self.motion_fusion(fusion_input)
        
        return {
            'adaptive_motion': refined_motion,
            'motion_weights': motion_weights,
            'individual_motions': motion_estimates,
            'motion_types': torch.argmax(motion_weights, dim=1)
        }


class MotionCompensationPipeline:
    """
    Complete motion compensation pipeline
    """
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize motion compensation components
        self.optical_flow_pyramid = OpticalFlowPyramid(num_levels=4).to(self.device)
        self.subpixel_refinement = SubPixelMotionRefinement(search_radius=4).to(self.device)
        self.motion_regularizer = MotionFieldRegularizer(regularization_strength=0.1).to(self.device)
        self.occlusion_detector = OcclusionDetector(threshold=0.5).to(self.device)
        self.adaptive_motion_model = AdaptiveMotionModel(num_motion_types=4).to(self.device)
        
        logger.info(f"Initialized motion compensation pipeline on {self.device}")
    
    def estimate_motion(self, frame1: torch.Tensor, frame2: torch.Tensor, 
                       refinement: bool = True, adaptive: bool = True) -> Dict[str, Any]:
        """
        Complete motion estimation with all components
        
        Args:
            frame1: First frame (B, 3, H, W)
            frame2: Second frame (B, 3, H, W)
            refinement: Whether to apply sub-pixel refinement
            adaptive: Whether to use adaptive motion model
        
        Returns:
            Dictionary with comprehensive motion estimation results
        """
        results = {}
        
        with torch.no_grad():
            # Initial hierarchical flow estimation
            if adaptive:
                adaptive_result = self.adaptive_motion_model(frame1, frame2)
                initial_flow = adaptive_result['adaptive_motion']
                results.update(adaptive_result)
            else:
                pyramid_result = self.optical_flow_pyramid(frame1, frame2)
                initial_flow = pyramid_result['flows'][0]
                results.update(pyramid_result)
            
            # Sub-pixel refinement
            if refinement:
                refined_flow = self.subpixel_refinement(frame1, frame2, initial_flow)
                results['refined_flow'] = refined_flow
                final_flow = refined_flow
            else:
                final_flow = initial_flow
            
            # Motion field regularization
            regularization_losses = self.motion_regularizer(final_flow, frame1)
            results['regularization_losses'] = regularization_losses
            
            # Occlusion detection (requires backward flow)
            backward_result = self.optical_flow_pyramid(frame2, frame1)
            backward_flow = backward_result['flows'][0]
            
            occlusion_result = self.occlusion_detector(
                frame1, frame2, final_flow, backward_flow
            )
            results.update(occlusion_result)
            
            # Final motion field
            results['final_motion'] = final_flow
            results['backward_motion'] = backward_flow
        
        return results
    
    def compensate_motion(self, frame: torch.Tensor, motion_field: torch.Tensor,
                         occlusion_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply motion compensation to frame
        
        Args:
            frame: Input frame (B, 3, H, W)
            motion_field: Motion field (B, 2, H, W)
            occlusion_map: Optional occlusion map (B, 1, H, W)
        
        Returns:
            Motion compensated frame
        """
        B, C, H, W = frame.shape
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=frame.device),
            torch.arange(0, W, dtype=torch.float32, device=frame.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Apply motion
        compensated_grid = grid + motion_field
        
        # Normalize for grid_sample
        compensated_grid[:, 0] = 2.0 * compensated_grid[:, 0] / (W - 1) - 1.0
        compensated_grid[:, 1] = 2.0 * compensated_grid[:, 1] / (H - 1) - 1.0
        
        # Sample compensated frame
        compensated_frame = F.grid_sample(
            frame, compensated_grid.permute(0, 2, 3, 1),
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        # Handle occlusions if provided
        if occlusion_map is not None:
            # In occluded regions, use original frame
            compensated_frame = compensated_frame * (1 - occlusion_map) + frame * occlusion_map
        
        return compensated_frame


# Utility functions
def visualize_motion_field(motion_field: torch.Tensor, scale: float = 10.0) -> np.ndarray:
    """
    Visualize optical flow as color-coded image
    
    Args:
        motion_field: Motion field tensor (1, 2, H, W) 
        scale: Scaling factor for visualization
    
    Returns:
        Color-coded flow visualization
    """
    flow = motion_field.squeeze(0).cpu().numpy()  # (2, H, W)
    flow = np.transpose(flow, (1, 2, 0))  # (H, W, 2)
    
    # Convert to polar coordinates
    magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
    
    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi) * 179  # Hue from angle
    hsv[:, :, 1] = 255  # Full saturation
    hsv[:, :, 2] = np.clip(magnitude * scale, 0, 255)  # Value from magnitude
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


# Export main classes and functions
__all__ = [
    'OpticalFlowPyramid',
    'SubPixelMotionRefinement',
    'MotionFieldRegularizer', 
    'OcclusionDetector',
    'AdaptiveMotionModel',
    'MotionCompensationPipeline',
    'visualize_motion_field'
]