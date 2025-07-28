"""
Neural Frame Interpolation Models for MoneyPrinterTurbo
======================================================

Advanced neural network models for frame rate enhancement including:
- RIFE (Real-Time Intermediate Flow Estimation)
- DAIN (Depth-Aware Video Frame Interpolation)
- AdaCoF (Adaptive Collaboration of Flows)
- SepConv (Separable Convolution-based Interpolation)

Features:
- Motion-compensated frame generation
- Temporal consistency preservation
- Multi-target FPS (30, 60, 120)
- Adaptive interpolation algorithms
- Neural pattern learning

Author: FrameInterpolator Agent
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
from pathlib import Path
import math

from loguru import logger


class WarpingLayer(nn.Module):
    """Grid sampling layer for optical flow warping"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
    
    def forward(self, x, flow):
        """
        Warp input tensor using optical flow
        
        Args:
            x: Input tensor (B, C, H, W)
            flow: Optical flow tensor (B, 2, H, W)
        """
        B, C, H, W = x.size()
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32),
            torch.arange(0, W, dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1).to(self.device)
        
        # Add flow to grid
        new_grid = grid + flow
        
        # Normalize to [-1, 1] for grid_sample
        new_grid[:, 0, :, :] = 2.0 * new_grid[:, 0, :, :] / (W - 1) - 1.0
        new_grid[:, 1, :, :] = 2.0 * new_grid[:, 1, :, :] / (H - 1) - 1.0
        
        # Transpose for grid_sample (B, H, W, 2)
        new_grid = new_grid.permute(0, 2, 3, 1)
        
        # Warp using bilinear interpolation
        warped = F.grid_sample(x, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped


class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, activation: str = 'relu', 
                 use_bn: bool = False):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'prelu':
            self.activation = nn.PReLU(out_channels)
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class FlowEstimationBlock(nn.Module):
    """Block for optical flow estimation between frames"""
    
    def __init__(self, in_channels: int, num_levels: int = 4):
        super().__init__()
        
        # Feature pyramids for multi-scale flow estimation
        self.feature_pyramids = nn.ModuleList()
        channels = in_channels
        
        for i in range(num_levels):
            pyramid_block = nn.Sequential(
                ConvBlock(channels, 64, kernel_size=3, padding=1),
                ConvBlock(64, 64, kernel_size=3, padding=1),
                nn.MaxPool2d(2) if i < num_levels - 1 else nn.Identity()
            )
            self.feature_pyramids.append(pyramid_block)
            channels = 64
        
        # Flow prediction heads
        self.flow_heads = nn.ModuleList()
        for i in range(num_levels):
            flow_head = nn.Sequential(
                ConvBlock(128, 64, kernel_size=3, padding=1),  # 128 from concatenated features
                ConvBlock(64, 32, kernel_size=3, padding=1),
                nn.Conv2d(32, 2, kernel_size=3, padding=1)  # 2 channels for optical flow
            )
            self.flow_heads.append(flow_head)
    
    def forward(self, frame1, frame2):
        """
        Estimate optical flow between two frames
        
        Args:
            frame1: First frame (B, C, H, W)
            frame2: Second frame (B, C, H, W)
        
        Returns:
            Multi-scale optical flow maps
        """
        # Extract feature pyramids
        feat1_pyramid = []
        feat2_pyramid = []
        
        f1, f2 = frame1, frame2
        for pyramid in self.feature_pyramids:
            f1 = pyramid(f1)
            f2 = pyramid(f2)
            feat1_pyramid.append(f1)
            feat2_pyramid.append(f2)
        
        # Estimate flow at multiple scales
        flows = []
        for i, (feat1, feat2, flow_head) in enumerate(zip(feat1_pyramid, feat2_pyramid, self.flow_heads)):
            # Concatenate features for flow estimation
            combined_feat = torch.cat([feat1, feat2], dim=1)
            flow = flow_head(combined_feat)
            flows.append(flow)
        
        return flows


class RIFEModel(nn.Module):
    """
    RIFE (Real-Time Intermediate Flow Estimation) Model
    
    High-performance frame interpolation using privileged information
    and learnable intermediate flow estimation.
    """
    
    def __init__(self, num_levels: int = 4, scale_factor: float = 1.0):
        super().__init__()
        
        self.num_levels = num_levels
        self.scale_factor = scale_factor
        
        # IFNet for flow estimation
        self.flow_estimator = FlowEstimationBlock(6, num_levels)  # 6 channels for 2 RGB frames
        
        # Context encoder for intermediate frame synthesis
        self.context_encoder = nn.Sequential(
            ConvBlock(6, 32, kernel_size=3, padding=1),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
        )
        
        # Intermediate frame decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 + 6, 128, 4, 2, 1),  # +6 for warped frames
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            ConvBlock(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Warping layer
        self.warping = WarpingLayer()
        
        # Refinement network
        self.refine_net = nn.Sequential(
            ConvBlock(9, 64, kernel_size=3, padding=1),  # 9 = 3*3 (input, prediction, gt)
            ConvBlock(64, 64, kernel_size=3, padding=1),
            ConvBlock(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, frame1, frame2, timestep=0.5):
        """
        Generate intermediate frame at given timestep
        
        Args:
            frame1: First frame (B, 3, H, W)
            frame2: Second frame (B, 3, H, W)  
            timestep: Temporal position (0.0 to 1.0)
        
        Returns:
            Interpolated frame at timestep
        """
        B, C, H, W = frame1.shape
        
        # Concatenate input frames
        input_frames = torch.cat([frame1, frame2], dim=1)
        
        # Estimate bidirectional flows
        flows = self.flow_estimator(input_frames)
        
        # Use highest resolution flow
        flow_0_1 = flows[0]  # Flow from frame1 to frame2
        flow_1_0 = -flow_0_1  # Reverse flow (approximation)
        
        # Scale flows by timestep
        flow1_t = flow_0_1 * timestep
        flow2_t = flow_1_0 * (1 - timestep)
        
        # Warp frames to timestep
        warped1 = self.warping(frame1, flow1_t)
        warped2 = self.warping(frame2, flow2_t)
        
        # Encode context
        context_input = torch.cat([frame1, frame2], dim=1)
        context_features = self.context_encoder(context_input)
        
        # Concatenate warped frames with context
        decoder_input = torch.cat([context_features, warped1, warped2], dim=1)
        
        # Decode intermediate frame
        intermediate = self.decoder(decoder_input)
        
        # Refinement
        refine_input = torch.cat([input_frames, intermediate], dim=1)
        residual = self.refine_net(refine_input)
        final_frame = intermediate + residual * 0.1  # Small residual connection
        
        # Ensure output is in valid range
        final_frame = torch.clamp(final_frame, 0, 1)
        
        return {
            'interpolated_frame': final_frame,
            'warped_frame1': warped1,
            'warped_frame2': warped2,
            'flow_0_1': flow_0_1,
            'intermediate_raw': intermediate
        }


class DAINModel(nn.Module):
    """
    DAIN (Depth-Aware Video Frame Interpolation) Model
    
    Uses depth information and adaptive convolution for better interpolation
    """
    
    def __init__(self, filter_size: int = 51):
        super().__init__()
        
        self.filter_size = filter_size
        
        # Depth estimation network
        self.depth_estimator = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            ConvBlock(64, 128, kernel_size=5, stride=2, padding=2),
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ConvBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ConvBlock(512, 256, kernel_size=3, padding=1),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )
        
        # Flow estimation
        self.flow_estimator = FlowEstimationBlock(6, num_levels=3)
        
        # Adaptive convolution kernel generation
        self.kernel_estimator = nn.Sequential(
            ConvBlock(8, 128, kernel_size=3, padding=1),  # 6 (frames) + 2 (flows)
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, filter_size * filter_size, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        
        # Warping layer
        self.warping = WarpingLayer()
    
    def adaptive_convolution(self, input_tensor, kernels):
        """
        Apply adaptive convolution using predicted kernels
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            kernels: Predicted kernels (B, K*K, H, W)
        """
        B, C, H, W = input_tensor.shape
        K = int(math.sqrt(kernels.shape[1]))
        
        # Unfold input tensor
        unfolded = F.unfold(input_tensor, kernel_size=K, padding=K//2)  # (B, C*K*K, H*W)
        unfolded = unfolded.view(B, C, K*K, H, W)
        
        # Apply kernels
        kernels = kernels.unsqueeze(1)  # (B, 1, K*K, H, W)
        output = torch.sum(unfolded * kernels, dim=2)  # (B, C, H, W)
        
        return output
    
    def forward(self, frame1, frame2, timestep=0.5):
        """
        Generate depth-aware interpolated frame
        
        Args:
            frame1: First frame (B, 3, H, W)
            frame2: Second frame (B, 3, H, W)
            timestep: Temporal position (0.0 to 1.0)
        
        Returns:
            Interpolated frame with depth awareness
        """
        # Estimate depth for both frames
        depth1 = self.depth_estimator(frame1)
        depth2 = self.depth_estimator(frame2)
        
        # Estimate optical flow
        flows = self.flow_estimator(torch.cat([frame1, frame2], dim=1))
        flow_0_1 = flows[0]
        
        # Scale flow by timestep
        flow1_t = flow_0_1 * timestep
        flow2_t = -flow_0_1 * (1 - timestep)
        
        # Warp frames and depth maps
        warped1 = self.warping(frame1, flow1_t)
        warped2 = self.warping(frame2, flow2_t)
        warped_depth1 = self.warping(depth1, flow1_t)
        warped_depth2 = self.warping(depth2, flow2_t)
        
        # Depth-based occlusion detection
        depth_diff = torch.abs(warped_depth1 - warped_depth2)
        occlusion_mask = (depth_diff > 0.1).float()
        
        # Generate adaptive convolution kernels
        kernel_input = torch.cat([frame1, frame2, flow1_t, flow2_t], dim=1)
        adaptive_kernels = self.kernel_estimator(kernel_input)
        
        # Apply adaptive convolution to warped frames
        adapted1 = self.adaptive_convolution(warped1, adaptive_kernels)
        adapted2 = self.adaptive_convolution(warped2, adaptive_kernels)
        
        # Blend based on depth and occlusion
        alpha = 0.5 + 0.3 * (warped_depth1 - warped_depth2)  # Depth-based blending
        alpha = torch.clamp(alpha, 0, 1)
        alpha = alpha * (1 - occlusion_mask) + 0.5 * occlusion_mask  # Handle occlusions
        
        interpolated = alpha * adapted1 + (1 - alpha) * adapted2
        
        return {
            'interpolated_frame': interpolated,
            'depth1': depth1,
            'depth2': depth2,
            'flow_0_1': flow_0_1,
            'occlusion_mask': occlusion_mask,
            'alpha_map': alpha
        }


class AdaCOFModel(nn.Module):
    """
    AdaCoF (Adaptive Collaboration of Flows) Model
    
    Uses adaptive weighted collaboration of multiple optical flows
    """
    
    def __init__(self, num_flows: int = 4, kernel_size: int = 5):
        super().__init__()
        
        self.num_flows = num_flows
        self.kernel_size = kernel_size
        
        # Multiple flow estimators for different motions
        self.flow_estimators = nn.ModuleList([
            FlowEstimationBlock(6, num_levels=3) for _ in range(num_flows)
        ])
        
        # Flow collaboration network
        self.collaboration_net = nn.Sequential(
            ConvBlock(6 + num_flows * 2, 128, kernel_size=3, padding=1),  # frames + flows
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, num_flows, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        
        # Adaptive kernel prediction
        self.kernel_net = nn.Sequential(
            ConvBlock(6 + num_flows * 2, 128, kernel_size=3, padding=1),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, kernel_size * kernel_size * num_flows, kernel_size=3, padding=1)
        )
        
        # Warping layer
        self.warping = WarpingLayer()
    
    def forward(self, frame1, frame2, timestep=0.5):
        """
        Generate frame using adaptive collaboration of flows
        
        Args:
            frame1: First frame (B, 3, H, W)
            frame2: Second frame (B, 3, H, W)
            timestep: Temporal position (0.0 to 1.0)
        
        Returns:
            Interpolated frame using collaborative flows
        """
        B, C, H, W = frame1.shape
        input_frames = torch.cat([frame1, frame2], dim=1)
        
        # Estimate multiple flows
        all_flows = []
        warped_frames = []
        
        for flow_estimator in self.flow_estimators:
            flows = flow_estimator(input_frames)
            flow = flows[0] * timestep  # Scale by timestep
            all_flows.append(flow)
            
            # Warp both frames
            warped1 = self.warping(frame1, flow)
            warped2 = self.warping(frame2, -flow * (1 - timestep) / timestep)
            warped_frames.extend([warped1, warped2])
        
        # Concatenate all flows
        flows_tensor = torch.cat(all_flows, dim=1)  # (B, num_flows*2, H, W)
        
        # Compute collaboration weights
        collab_input = torch.cat([input_frames, flows_tensor], dim=1)
        collaboration_weights = self.collaboration_net(collab_input)  # (B, num_flows, H, W)
        
        # Generate adaptive kernels
        kernels = self.kernel_net(collab_input)  # (B, K*K*num_flows, H, W)
        kernels = kernels.view(B, self.num_flows, self.kernel_size * self.kernel_size, H, W)
        kernels = F.softmax(kernels, dim=2)
        
        # Apply collaborative warping
        collaborative_result = torch.zeros_like(frame1)
        
        for i in range(self.num_flows):
            # Get warped frames for this flow
            warped1 = warped_frames[i * 2]
            warped2 = warped_frames[i * 2 + 1]
            
            # Blend warped frames
            blended = 0.5 * warped1 + 0.5 * warped2
            
            # Apply adaptive kernel
            kernel = kernels[:, i]  # (B, K*K, H, W)
            adapted = self.adaptive_convolution(blended, kernel)
            
            # Weight by collaboration coefficient
            weight = collaboration_weights[:, i:i+1]  # (B, 1, H, W)
            collaborative_result += weight * adapted
        
        return {
            'interpolated_frame': collaborative_result,
            'collaboration_weights': collaboration_weights,
            'flows': all_flows,
            'warped_frames': warped_frames
        }
    
    def adaptive_convolution(self, input_tensor, kernel):
        """Apply adaptive convolution with given kernel"""
        B, C, H, W = input_tensor.shape
        K = self.kernel_size
        
        # Unfold input tensor
        unfolded = F.unfold(input_tensor, kernel_size=K, padding=K//2)
        unfolded = unfolded.view(B, C, K*K, H*W).permute(0, 3, 1, 2)  # (B, H*W, C, K*K)
        
        # Reshape kernel for batch matrix multiplication
        kernel = kernel.permute(0, 2, 3, 1).contiguous()  # (B, H, W, K*K)
        kernel = kernel.view(B, H*W, K*K, 1)  # (B, H*W, K*K, 1)
        
        # Apply kernels
        output = torch.matmul(unfolded, kernel).squeeze(-1)  # (B, H*W, C)
        output = output.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)
        
        return output


class SepConvModel(nn.Module):
    """
    SepConv (Separable Convolution) Model
    
    Uses separable convolution kernels for efficient frame interpolation
    """
    
    def __init__(self, kernel_size: int = 51):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            ConvBlock(6, 32, kernel_size=7, stride=1, padding=3),
            ConvBlock(32, 64, kernel_size=7, stride=1, padding=3),
            ConvBlock(64, 128, kernel_size=7, stride=1, padding=3),
            ConvBlock(128, 256, kernel_size=7, stride=1, padding=3),
        )
        
        # Separable kernel prediction - vertical
        self.vertical_kernel_net = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, kernel_size, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        
        # Separable kernel prediction - horizontal  
        self.horizontal_kernel_net = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, kernel_size, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
    
    def separable_convolution(self, input_tensor, v_kernel, h_kernel):
        """
        Apply separable convolution using vertical and horizontal kernels
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            v_kernel: Vertical kernel (B, K, H, W)
            h_kernel: Horizontal kernel (B, K, H, W)
        """
        B, C, H, W = input_tensor.shape
        K = v_kernel.shape[1]
        
        # Apply vertical convolution
        padding = K // 2
        v_result = torch.zeros_like(input_tensor)
        
        for i in range(K):
            shift = i - padding
            if shift < 0:
                padded = F.pad(input_tensor, (0, 0, -shift, 0))[:, :, :H, :]
            elif shift > 0:
                padded = F.pad(input_tensor, (0, 0, 0, shift))[:, :, shift:, :]
            else:
                padded = input_tensor
            
            weight = v_kernel[:, i:i+1, :, :].expand(-1, C, -1, -1)
            v_result += padded * weight
        
        # Apply horizontal convolution
        h_result = torch.zeros_like(v_result)
        
        for i in range(K):
            shift = i - padding
            if shift < 0:
                padded = F.pad(v_result, (-shift, 0, 0, 0))[:, :, :, :W]
            elif shift > 0:
                padded = F.pad(v_result, (0, shift, 0, 0))[:, :, :, shift:]
            else:
                padded = v_result
            
            weight = h_kernel[:, i:i+1, :, :].expand(-1, C, -1, -1)
            h_result += padded * weight
            
        return h_result
    
    def forward(self, frame1, frame2, timestep=0.5):
        """
        Generate interpolated frame using separable convolution
        
        Args:
            frame1: First frame (B, 3, H, W)
            frame2: Second frame (B, 3, H, W)
            timestep: Temporal position (0.0 to 1.0)
        
        Returns:
            Interpolated frame using separable convolution
        """
        # Concatenate input frames
        input_frames = torch.cat([frame1, frame2], dim=1)
        
        # Extract features
        features = self.feature_extractor(input_frames)
        
        # Predict separable kernels
        v_kernel = self.vertical_kernel_net(features)
        h_kernel = self.horizontal_kernel_net(features)
        
        # Apply separable convolution to both frames
        conv1 = self.separable_convolution(frame1, v_kernel, h_kernel)
        conv2 = self.separable_convolution(frame2, v_kernel, h_kernel)
        
        # Temporal blending
        interpolated = timestep * conv2 + (1 - timestep) * conv1
        
        return {
            'interpolated_frame': interpolated,
            'vertical_kernel': v_kernel,
            'horizontal_kernel': h_kernel,
            'conv1': conv1,
            'conv2': conv2
        }


class FrameInterpolationPipeline:
    """
    Complete frame interpolation pipeline with multiple models
    """
    
    def __init__(self, model_type: str = 'rife', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type.lower()
        
        # Initialize model
        if self.model_type == 'rife':
            self.model = RIFEModel()
        elif self.model_type == 'dain':
            self.model = DAINModel()
        elif self.model_type == 'adacof':
            self.model = AdaCOFModel()
        elif self.model_type == 'sepconv':
            self.model = SepConvModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),  # Adjust based on needs
            transforms.ToTensor()
        ])
        
        logger.info(f"Initialized {model_type.upper()} frame interpolation pipeline on {self.device}")
    
    def interpolate_frames(self, frame1: np.ndarray, frame2: np.ndarray, 
                          target_fps: int = 60, source_fps: int = 30) -> List[np.ndarray]:
        """
        Interpolate frames between two input frames
        
        Args:
            frame1: First frame as numpy array (H, W, 3)
            frame2: Second frame as numpy array (H, W, 3)
            target_fps: Target frame rate
            source_fps: Source frame rate
        
        Returns:
            List of interpolated frames
        """
        # Calculate number of intermediate frames needed
        interpolation_factor = target_fps // source_fps
        num_intermediate = interpolation_factor - 1
        
        if num_intermediate <= 0:
            return [frame1, frame2]
        
        # Convert to tensors
        tensor1 = self.transform(frame1).unsqueeze(0).to(self.device)
        tensor2 = self.transform(frame2).unsqueeze(0).to(self.device)
        
        interpolated_frames = [frame1]
        
        with torch.no_grad():
            for i in range(1, interpolation_factor):
                timestep = i / interpolation_factor
                
                # Generate intermediate frame
                result = self.model(tensor1, tensor2, timestep)
                interpolated_tensor = result['interpolated_frame']
                
                # Convert back to numpy
                interpolated_np = interpolated_tensor.squeeze(0).cpu().numpy()
                interpolated_np = np.transpose(interpolated_np, (1, 2, 0))
                interpolated_np = (interpolated_np * 255).astype(np.uint8)
                
                interpolated_frames.append(interpolated_np)
        
        interpolated_frames.append(frame2)
        return interpolated_frames
    
    def process_video(self, input_path: str, output_path: str, 
                      target_fps: int = 60, batch_size: int = 1) -> bool:
        """
        Process entire video for frame rate enhancement
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            target_fps: Target frame rate
            batch_size: Number of frame pairs to process at once
        
        Returns:
            Success status
        """
        try:
            import cv2
            
            # Open input video
            cap = cv2.VideoCapture(input_path)
            source_fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
            
            logger.info(f"Processing video: {source_fps}fps → {target_fps}fps")
            
            prev_frame = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    # Interpolate between previous and current frame
                    interpolated = self.interpolate_frames(
                        prev_frame, frame, target_fps, source_fps
                    )
                    
                    # Write all interpolated frames except the last one
                    for interp_frame in interpolated[:-1]:
                        out.write(interp_frame)
                
                prev_frame = frame
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
            
            # Write the last frame
            if prev_frame is not None:
                out.write(prev_frame)
            
            # Cleanup
            cap.release()
            out.release()
            
            logger.success(f"Video processing completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            return False


# Factory function for easy model creation
def create_frame_interpolator(model_type: str = 'rife', **kwargs) -> FrameInterpolationPipeline:
    """
    Factory function to create frame interpolation pipeline
    
    Args:
        model_type: Type of model ('rife', 'dain', 'adacof', 'sepconv')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Configured frame interpolation pipeline
    """
    return FrameInterpolationPipeline(model_type=model_type, **kwargs)


# Model information and capabilities
MODEL_INFO = {
    'rife': {
        'name': 'RIFE (Real-Time Intermediate Flow Estimation)',
        'description': 'High-performance real-time frame interpolation',
        'strengths': ['Real-time performance', 'Good motion handling', 'Low memory usage'],
        'best_for': 'General purpose, real-time applications'
    },
    'dain': {
        'name': 'DAIN (Depth-Aware Video Frame Interpolation)',
        'description': 'Depth-aware interpolation with occlusion handling',
        'strengths': ['Occlusion awareness', 'Depth information', 'High quality'],
        'best_for': 'Complex scenes with depth variations'
    },
    'adacof': {
        'name': 'AdaCoF (Adaptive Collaboration of Flows)',
        'description': 'Adaptive collaboration of multiple optical flows',
        'strengths': ['Multiple flow estimation', 'Adaptive blending', 'Robust to complex motion'],
        'best_for': 'Videos with complex or irregular motion patterns'
    },
    'sepconv': {
        'name': 'SepConv (Separable Convolution)',
        'description': 'Efficient separable convolution-based interpolation',
        'strengths': ['Computational efficiency', 'Good quality', 'Stable results'],
        'best_for': 'Balanced performance and quality'
    }
}


# Export main classes and functions
__all__ = [
    'RIFEModel',
    'DAINModel', 
    'AdaCOFModel',
    'SepConvModel',
    'FrameInterpolationPipeline',
    'create_frame_interpolator',
    'MODEL_INFO'
]