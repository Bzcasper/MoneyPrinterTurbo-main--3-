"""
Temporal Consistency Preservation for Frame Interpolation
========================================================

Advanced algorithms for maintaining temporal consistency in interpolated video sequences:
- Multi-frame temporal modeling
- Consistency loss functions
- Temporal flow regularization
- Cross-frame feature alignment
- Perceptual temporal smoothness

Author: FrameInterpolator Agent
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import math

from loguru import logger


class TemporalConsistencyLoss(nn.Module):
    """
    Multi-component temporal consistency loss function
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        
        # Default loss weights
        self.weights = weights or {
            'warp_loss': 1.0,
            'smooth_loss': 0.5,
            'perceptual_loss': 0.3,
            'edge_loss': 0.2,
            'flow_consistency': 0.4
        }
        
        # Perceptual loss network (VGG-like)
        self.perceptual_net = self._build_perceptual_network()
        
        # Edge detection kernel
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2], 
            [-1, 0, 1]
        ]).float().view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]).float().view(1, 1, 3, 3))
    
    def _build_perceptual_network(self):
        """Build perceptual feature extraction network"""
        from torchvision.models import vgg16
        
        vgg = vgg16(pretrained=True).features[:16]  # Up to conv3_3
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        return vgg
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive temporal consistency loss
        
        Args:
            predictions: Dictionary with predicted tensors
            targets: Dictionary with target tensors
        
        Returns:
            Dictionary with individual loss components
        """
        losses = {}
        
        # Basic reconstruction loss
        if 'interpolated_frames' in predictions and 'target_frames' in targets:
            losses['reconstruction'] = F.mse_loss(
                predictions['interpolated_frames'], 
                targets['target_frames']
            )
        
        # Warping consistency loss
        if 'warped_frames' in predictions and 'target_frames' in targets:
            losses['warp_loss'] = self._compute_warp_loss(
                predictions['warped_frames'], 
                targets['target_frames']
            ) * self.weights['warp_loss']
        
        # Temporal smoothness loss
        if 'interpolated_frames' in predictions:
            losses['smooth_loss'] = self._compute_smoothness_loss(
                predictions['interpolated_frames']
            ) * self.weights['smooth_loss']
        
        # Perceptual consistency loss
        if 'interpolated_frames' in predictions and 'target_frames' in targets:
            losses['perceptual_loss'] = self._compute_perceptual_loss(
                predictions['interpolated_frames'],
                targets['target_frames']
            ) * self.weights['perceptual_loss']
        
        # Edge consistency loss
        if 'interpolated_frames' in predictions and 'target_frames' in targets:
            losses['edge_loss'] = self._compute_edge_loss(
                predictions['interpolated_frames'],
                targets['target_frames']
            ) * self.weights['edge_loss']
        
        # Flow consistency loss
        if 'flows' in predictions:
            losses['flow_consistency'] = self._compute_flow_consistency_loss(
                predictions['flows']
            ) * self.weights['flow_consistency']
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def _compute_warp_loss(self, warped_frames: torch.Tensor, 
                          target_frames: torch.Tensor) -> torch.Tensor:
        """Compute warping consistency loss"""
        return F.l1_loss(warped_frames, target_frames)
    
    def _compute_smoothness_loss(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute temporal smoothness loss"""
        if frames.dim() == 5:  # (B, T, C, H, W)
            # Temporal gradient
            temporal_diff = frames[:, 1:] - frames[:, :-1]
            smoothness = torch.mean(torch.abs(temporal_diff))
        else:  # Single frame case
            # Spatial smoothness as proxy
            dx = F.conv2d(frames, self.sobel_x.expand(frames.size(1), 1, 3, 3), 
                         padding=1, groups=frames.size(1))
            dy = F.conv2d(frames, self.sobel_y.expand(frames.size(1), 1, 3, 3), 
                         padding=1, groups=frames.size(1))
            smoothness = torch.mean(torch.sqrt(dx**2 + dy**2))
        
        return smoothness
    
    def _compute_perceptual_loss(self, pred_frames: torch.Tensor, 
                                target_frames: torch.Tensor) -> torch.Tensor:
        """Compute perceptual consistency loss using VGG features"""
        # Extract features
        pred_features = self.perceptual_net(pred_frames)
        target_features = self.perceptual_net(target_frames)
        
        # Compute perceptual loss
        perceptual_loss = F.mse_loss(pred_features, target_features)
        
        return perceptual_loss
    
    def _compute_edge_loss(self, pred_frames: torch.Tensor, 
                          target_frames: torch.Tensor) -> torch.Tensor:
        """Compute edge consistency loss"""
        # Compute edges for predicted frames
        pred_gray = torch.mean(pred_frames, dim=1, keepdim=True)
        pred_edges_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)
        
        # Compute edges for target frames
        target_gray = torch.mean(target_frames, dim=1, keepdim=True)
        target_edges_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)
        
        # Edge consistency loss
        edge_loss = F.l1_loss(pred_edges, target_edges)
        
        return edge_loss
    
    def _compute_flow_consistency_loss(self, flows: List[torch.Tensor]) -> torch.Tensor:
        """Compute optical flow consistency loss"""
        if len(flows) < 2:
            return torch.tensor(0.0, device=flows[0].device)
        
        consistency_loss = 0.0
        count = 0
        
        # Forward-backward consistency
        for i in range(len(flows) - 1):
            flow_forward = flows[i]
            flow_backward = flows[i + 1]
            
            # Flow should be approximately opposite
            consistency = F.l1_loss(flow_forward, -flow_backward)
            consistency_loss += consistency
            count += 1
        
        return consistency_loss / count if count > 0 else torch.tensor(0.0)


class TemporalFeatureAlignment(nn.Module):
    """
    Module for aligning features across temporal frames
    """
    
    def __init__(self, feature_channels: int, alignment_type: str = 'attention'):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.alignment_type = alignment_type
        
        if alignment_type == 'attention':
            self.alignment_net = self._build_attention_alignment()
        elif alignment_type == 'correlation':
            self.alignment_net = self._build_correlation_alignment()
        else:
            raise ValueError(f"Unknown alignment type: {alignment_type}")
    
    def _build_attention_alignment(self):
        """Build attention-based feature alignment"""
        return nn.MultiheadAttention(
            embed_dim=self.feature_channels,
            num_heads=8,
            batch_first=True
        )
    
    def _build_correlation_alignment(self):
        """Build correlation-based feature alignment"""
        return nn.Sequential(
            nn.Conv2d(self.feature_channels * 2, self.feature_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels, self.feature_channels, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features_t0: torch.Tensor, features_t1: torch.Tensor) -> torch.Tensor:
        """
        Align features between two temporal frames
        
        Args:
            features_t0: Features from frame at time t0 (B, C, H, W)
            features_t1: Features from frame at time t1 (B, C, H, W)
        
        Returns:
            Aligned features
        """
        if self.alignment_type == 'attention':
            return self._attention_alignment(features_t0, features_t1)
        else:
            return self._correlation_alignment(features_t0, features_t1)
    
    def _attention_alignment(self, feat_t0: torch.Tensor, feat_t1: torch.Tensor) -> torch.Tensor:
        """Attention-based alignment"""
        B, C, H, W = feat_t0.shape
        
        # Reshape for attention: (B, H*W, C)
        feat_t0_flat = feat_t0.view(B, C, H*W).transpose(1, 2)
        feat_t1_flat = feat_t1.view(B, C, H*W).transpose(1, 2)
        
        # Apply multi-head attention
        aligned_features, _ = self.alignment_net(feat_t0_flat, feat_t1_flat, feat_t1_flat)
        
        # Reshape back: (B, C, H, W)
        aligned_features = aligned_features.transpose(1, 2).view(B, C, H, W)
        
        return aligned_features
    
    def _correlation_alignment(self, feat_t0: torch.Tensor, feat_t1: torch.Tensor) -> torch.Tensor:
        """Correlation-based alignment"""
        # Concatenate features
        combined = torch.cat([feat_t0, feat_t1], dim=1)
        
        # Compute alignment weights
        alignment_weights = self.alignment_net(combined)
        
        # Apply alignment
        aligned = feat_t0 * alignment_weights + feat_t1 * (1 - alignment_weights)
        
        return aligned


class TemporalFlowRegularizer(nn.Module):
    """
    Regularization module for optical flow temporal consistency
    """
    
    def __init__(self, regularization_strength: float = 0.1):
        super().__init__()
        self.regularization_strength = regularization_strength
    
    def forward(self, flows: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply temporal flow regularization
        
        Args:
            flows: List of optical flow tensors
        
        Returns:
            Dictionary with regularization losses
        """
        losses = {}
        
        if len(flows) < 2:
            losses['flow_regularization'] = torch.tensor(0.0, device=flows[0].device)
            return losses
        
        # Temporal flow smoothness
        temporal_smoothness = 0.0
        for i in range(len(flows) - 1):
            flow_diff = flows[i+1] - flows[i]
            temporal_smoothness += torch.mean(torch.abs(flow_diff))
        
        losses['temporal_smoothness'] = temporal_smoothness * self.regularization_strength
        
        # Flow magnitude regularization
        magnitude_reg = 0.0
        for flow in flows:
            flow_magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
            magnitude_reg += torch.mean(flow_magnitude)
        
        losses['magnitude_regularization'] = magnitude_reg * 0.01
        
        # Spatial flow smoothness
        spatial_smoothness = 0.0
        for flow in flows:
            # Spatial gradients
            flow_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
            flow_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
            
            spatial_smoothness += torch.mean(torch.abs(flow_dx)) + torch.mean(torch.abs(flow_dy))
        
        losses['spatial_smoothness'] = spatial_smoothness * 0.05
        
        # Total regularization
        losses['total_regularization'] = sum(losses.values())
        
        return losses


class MultiFrameTemporalModel(nn.Module):
    """
    Multi-frame temporal modeling for enhanced consistency
    """
    
    def __init__(self, num_frames: int = 5, feature_channels: int = 128):
        super().__init__()
        
        self.num_frames = num_frames
        self.feature_channels = feature_channels
        
        # Temporal convolution for multi-frame processing
        self.temporal_conv = nn.Conv3d(
            in_channels=3,  # RGB
            out_channels=feature_channels,
            kernel_size=(num_frames, 3, 3),
            padding=(0, 1, 1)
        )
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_channels,
            num_heads=8,
            batch_first=True
        )
        
        # Feature fusion network
        self.fusion_net = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 2, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        
        # Temporal consistency enforcer
        self.consistency_enforcer = nn.LSTM(
            input_size=feature_channels,
            hidden_size=feature_channels,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, frame_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process multi-frame sequence for temporal consistency
        
        Args:
            frame_sequence: Sequence of frames (B, T, C, H, W)
        
        Returns:
            Dictionary with processed results
        """
        B, T, C, H, W = frame_sequence.shape
        
        # Temporal convolution
        # Reshape for 3D conv: (B, C, T, H, W)
        sequence_3d = frame_sequence.permute(0, 2, 1, 3, 4)
        temporal_features = self.temporal_conv(sequence_3d)  # (B, feat_channels, 1, H, W)
        temporal_features = temporal_features.squeeze(2)  # (B, feat_channels, H, W)
        
        # Temporal attention
        # Reshape for attention: (B, H*W, feat_channels)
        feat_flat = temporal_features.view(B, self.feature_channels, H*W).transpose(1, 2)
        attended_features, attention_weights = self.temporal_attention(
            feat_flat, feat_flat, feat_flat
        )
        
        # Reshape back: (B, feat_channels, H, W)
        attended_features = attended_features.transpose(1, 2).view(B, self.feature_channels, H, W)
        
        # LSTM for temporal consistency
        # Use spatial averages as sequence features
        spatial_features = F.adaptive_avg_pool2d(attended_features, (1, 1))  # (B, feat_channels, 1, 1)
        spatial_features = spatial_features.view(B, 1, self.feature_channels)  # (B, 1, feat_channels)
        
        # Expand for sequence processing
        sequence_features = spatial_features.expand(B, T, self.feature_channels)
        consistent_features, _ = self.consistency_enforcer(sequence_features)
        
        # Take the middle frame's consistent features
        middle_idx = T // 2
        middle_consistent = consistent_features[:, middle_idx, :]  # (B, feat_channels*2)
        
        # Project back to spatial dimensions
        middle_consistent = middle_consistent.view(B, self.feature_channels * 2, 1, 1)
        middle_consistent = F.interpolate(middle_consistent, size=(H, W), mode='bilinear', align_corners=False)
        
        # Reduce channels and combine with attended features
        middle_consistent = middle_consistent[:, :self.feature_channels, :, :]
        combined_features = attended_features + middle_consistent
        
        # Generate final output
        output_frame = self.fusion_net(combined_features)
        
        return {
            'output_frame': output_frame,
            'temporal_features': temporal_features,
            'attention_weights': attention_weights,
            'consistent_features': combined_features
        }


class AdaptiveTemporalFiltering(nn.Module):
    """
    Adaptive temporal filtering for dynamic consistency adjustment
    """
    
    def __init__(self, num_channels: int = 3, filter_size: int = 5):
        super().__init__()
        
        self.num_channels = num_channels
        self.filter_size = filter_size
        
        # Motion estimation network
        self.motion_estimator = nn.Sequential(
            nn.Conv2d(num_channels * 2, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
        # Adaptive filter generator
        self.filter_generator = nn.Sequential(
            nn.Conv2d(num_channels * 2 + 1, 128, 3, 1, 1),  # +1 for motion
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, filter_size * filter_size, 3, 1, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, frame_pair: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive temporal filtering
        
        Args:
            frame_pair: Pair of consecutive frames (B, 2*C, H, W)
        
        Returns:
            Temporally filtered intermediate frame
        """
        # Estimate motion
        motion_map = self.motion_estimator(frame_pair)
        
        # Generate adaptive filters
        filter_input = torch.cat([frame_pair, motion_map], dim=1)
        adaptive_filters = self.filter_generator(filter_input)
        
        # Apply adaptive filtering
        frame1 = frame_pair[:, :self.num_channels, :, :]
        frame2 = frame_pair[:, self.num_channels:, :, :]
        
        # Simple linear blending with adaptive weights
        alpha = motion_map
        filtered_frame = alpha * frame1 + (1 - alpha) * frame2
        
        return filtered_frame


# Utility functions for temporal consistency evaluation
def compute_temporal_consistency_metrics(interpolated_sequence: torch.Tensor) -> Dict[str, float]:
    """
    Compute temporal consistency metrics for interpolated sequence
    
    Args:
        interpolated_sequence: Sequence of interpolated frames (B, T, C, H, W)
    
    Returns:
        Dictionary with consistency metrics
    """
    metrics = {}
    
    if interpolated_sequence.dim() != 5:
        logger.warning("Expected 5D tensor for temporal consistency evaluation")
        return metrics
    
    B, T, C, H, W = interpolated_sequence.shape
    
    # Temporal smoothness metric
    temporal_diffs = interpolated_sequence[:, 1:] - interpolated_sequence[:, :-1]
    temporal_smoothness = torch.mean(torch.abs(temporal_diffs)).item()
    metrics['temporal_smoothness'] = temporal_smoothness
    
    # Optical flow consistency (simplified)
    total_flow_variance = 0.0
    for t in range(T - 1):
        frame1 = interpolated_sequence[:, t]
        frame2 = interpolated_sequence[:, t + 1]
        
        # Simple frame difference as flow proxy
        flow_proxy = torch.mean((frame2 - frame1)**2, dim=1)
        flow_variance = torch.var(flow_proxy).item()
        total_flow_variance += flow_variance
    
    metrics['flow_consistency'] = total_flow_variance / (T - 1) if T > 1 else 0.0
    
    # Edge consistency
    edge_consistency = 0.0
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)
    
    if interpolated_sequence.is_cuda:
        sobel_x = sobel_x.cuda()
        sobel_y = sobel_y.cuda()
    
    for t in range(T):
        frame = torch.mean(interpolated_sequence[:, t], dim=1, keepdim=True)
        edges_x = F.conv2d(frame, sobel_x, padding=1)
        edges_y = F.conv2d(frame, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        edge_consistency += torch.mean(edges).item()
    
    metrics['edge_consistency'] = edge_consistency / T
    
    return metrics


# Export main classes and functions
__all__ = [
    'TemporalConsistencyLoss',
    'TemporalFeatureAlignment', 
    'TemporalFlowRegularizer',
    'MultiFrameTemporalModel',
    'AdaptiveTemporalFiltering',
    'compute_temporal_consistency_metrics'
]