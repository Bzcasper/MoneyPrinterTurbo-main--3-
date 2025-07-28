"""
Training Data Management System for Neural Video Enhancement
===========================================================

Comprehensive data pipeline for managing video training datasets including:
- Video dataset loading and preprocessing
- Data augmentation for video sequences
- Training/validation split management
- Custom dataset integration
- Distributed data loading

Author: ML Model Developer
Version: 1.0.0
"""

import os
import json
import random
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
import queue
import time

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from loguru import logger


@dataclass
class DatasetConfig:
    """Configuration for dataset management"""
    name: str
    dataset_path: str
    dataset_type: str  # 'video_pairs', 'single_video', 'image_pairs', etc.
    
    # Data specifications
    input_size: Tuple[int, int] = (256, 256)
    sequence_length: int = 1  # For video sequences
    num_channels: int = 3
    
    # Split configuration
    train_split: float = 0.8
    val_split: float = 0.2
    test_split: float = 0.0
    
    # Augmentation settings
    enable_augmentation: bool = True
    augmentation_probability: float = 0.8
    
    # Loading settings
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    # Cache settings
    use_cache: bool = True
    cache_size: int = 1000  # Number of items to cache
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "name": self.name,
            "dataset_path": self.dataset_path,
            "dataset_type": self.dataset_type,
            "input_size": self.input_size,
            "sequence_length": self.sequence_length,
            "num_channels": self.num_channels,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "enable_augmentation": self.enable_augmentation,
            "augmentation_probability": self.augmentation_probability,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "shuffle": self.shuffle,
            "use_cache": self.use_cache,
            "cache_size": self.cache_size
        }


class VideoAugmentation:
    """Advanced video-specific data augmentation"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.prob = config.augmentation_probability
        
        # Spatial augmentations
        self.spatial_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            A.GaussNoise(
                var_limit=(10, 50),
                p=0.3
            ),
            A.GaussianBlur(
                blur_limit=3,
                p=0.2
            ),
            A.Resize(
                height=config.input_size[1],
                width=config.input_size[0],
                always_apply=True
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], p=self.prob)
        
        # Temporal augmentations for video sequences
        self.temporal_aug_enabled = config.sequence_length > 1
    
    def __call__(self, frames: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        """Apply augmentations to frame(s)"""
        if isinstance(frames, list):
            # Multiple frames (video sequence)
            augmented_frames = []
            
            # Apply same spatial transformation to all frames
            if random.random() < self.prob:
                # Get augmentation parameters for consistency
                augmented = self.spatial_aug(image=frames[0])
                augmented_frames.append(augmented['image'])
                
                # Apply same transformation to remaining frames
                for frame in frames[1:]:
                    transformed = A.ReplayCompose.replay(augmented, image=frame)
                    augmented_frames.append(transformed['image'])
            else:
                # No augmentation, just resize and normalize
                for frame in frames:
                    processed = A.Compose([
                        A.Resize(
                            height=self.config.input_size[1],
                            width=self.config.input_size[0]
                        ),
                        A.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        ),
                        ToTensorV2()
                    ])(image=frame)
                    augmented_frames.append(processed['image'])
            
            # Stack frames
            return torch.stack(augmented_frames)
        
        else:
            # Single frame
            if random.random() < self.prob:
                augmented = self.spatial_aug(image=frames)
                return augmented['image']
            else:
                # No augmentation
                processed = A.Compose([
                    A.Resize(
                        height=self.config.input_size[1],
                        width=self.config.input_size[0]
                    ),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2()
                ])(image=frames)
                return processed['image']


class DataCache:
    """LRU cache for dataset items"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self.lock:
            if key in self.cache:
                # Update existing item
                self.access_order.remove(key)
                self.access_order.append(key)
                self.cache[key] = value
            else:
                # Add new item
                if len(self.cache) >= self.max_size:
                    # Remove least recently used item
                    lru_key = self.access_order.pop(0)
                    del self.cache[lru_key]
                
                self.cache[key] = value
                self.access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)


class BaseVideoDataset(Dataset, ABC):
    """Base class for video datasets"""
    
    def __init__(self, config: DatasetConfig, split: str = 'train'):
        self.config = config
        self.split = split
        
        # Initialize augmentation
        self.augmentation = VideoAugmentation(config) if config.enable_augmentation and split == 'train' else None
        
        # Initialize cache
        self.cache = DataCache(config.cache_size) if config.use_cache else None
        
        # Load dataset metadata
        self.data_items = []
        self._load_dataset()
        
        # Split data
        self._create_splits()
        
        logger.info(f"Initialized {self.__class__.__name__} with {len(self.data_items)} items ({split} split)")
    
    @abstractmethod
    def _load_dataset(self):
        """Load dataset metadata - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _load_item(self, index: int) -> Dict[str, Any]:
        """Load a single data item - to be implemented by subclasses"""
        pass
    
    def _create_splits(self):
        """Create train/val/test splits"""
        total_items = len(self.data_items)
        
        # Calculate split sizes
        train_size = int(total_items * self.config.train_split)
        val_size = int(total_items * self.config.val_split)
        test_size = total_items - train_size - val_size
        
        # Create splits
        if self.split == 'train':
            self.data_items = self.data_items[:train_size]
        elif self.split == 'val':
            self.data_items = self.data_items[train_size:train_size + val_size]
        elif self.split == 'test':
            self.data_items = self.data_items[train_size + val_size:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def __len__(self) -> int:
        return len(self.data_items)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Create cache key
        cache_key = f"{self.split}_{index}_{hash(str(self.data_items[index]))}"
        
        # Try to get from cache
        if self.cache:
            cached_item = self.cache.get(cache_key)
            if cached_item is not None:
                return cached_item
        
        # Load item
        item = self._load_item(index)
        
        # Apply augmentation
        if self.augmentation:
            if 'input' in item:
                item['input'] = self.augmentation(item['input'])
            if 'target' in item:
                item['target'] = self.augmentation(item['target'])
        
        # Cache item
        if self.cache:
            self.cache.put(cache_key, item)
        
        return item


class VideoPairDataset(BaseVideoDataset):
    """Dataset for video pairs (e.g., low-res -> high-res)"""
    
    def _load_dataset(self):
        """Load video pair dataset"""
        dataset_path = Path(self.config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Look for paired structure: input/low_res and target/high_res
        input_dir = dataset_path / "input"
        target_dir = dataset_path / "target"
        
        if not (input_dir.exists() and target_dir.exists()):
            raise FileNotFoundError("Expected 'input' and 'target' directories in dataset path")
        
        # Find matching video pairs
        input_videos = sorted(list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi")))
        
        for input_video in input_videos:
            target_video = target_dir / input_video.name
            
            if target_video.exists():
                self.data_items.append({
                    'input_path': str(input_video),
                    'target_path': str(target_video),
                    'name': input_video.stem
                })
    
    def _load_item(self, index: int) -> Dict[str, Any]:
        """Load a video pair item"""
        item_info = self.data_items[index]
        
        # Load input video frames
        input_frames = self._load_video_frames(item_info['input_path'])
        target_frames = self._load_video_frames(item_info['target_path'])
        
        # Ensure same number of frames
        min_frames = min(len(input_frames), len(target_frames))
        if self.config.sequence_length > 1:
            # Select sequence of frames
            start_idx = random.randint(0, max(0, min_frames - self.config.sequence_length))
            input_sequence = input_frames[start_idx:start_idx + self.config.sequence_length]
            target_sequence = target_frames[start_idx:start_idx + self.config.sequence_length]
            
            return {
                'input': input_sequence,
                'target': target_sequence,
                'name': item_info['name']
            }
        else:
            # Select single frame
            frame_idx = random.randint(0, min_frames - 1)
            
            return {
                'input': input_frames[frame_idx],
                'target': target_frames[frame_idx],
                'name': item_info['name']
            }
    
    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load frames from video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                # Limit number of frames for memory efficiency
                if len(frames) >= 100:
                    break
        
        finally:
            cap.release()
        
        return frames


class SingleVideoDataset(BaseVideoDataset):
    """Dataset for single videos (for self-supervised learning)"""
    
    def _load_dataset(self):
        """Load single video dataset"""
        dataset_path = Path(self.config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(dataset_path.rglob(f"*{ext}"))
        
        for video_file in video_files:
            self.data_items.append({
                'video_path': str(video_file),
                'name': video_file.stem
            })
    
    def _load_item(self, index: int) -> Dict[str, Any]:
        """Load a single video item"""
        item_info = self.data_items[index]
        
        # Load video frames
        frames = self._load_video_frames(item_info['video_path'])
        
        if self.config.sequence_length > 1:
            # Select sequence of frames
            start_idx = random.randint(0, max(0, len(frames) - self.config.sequence_length))
            sequence = frames[start_idx:start_idx + self.config.sequence_length]
            
            return {
                'input': sequence,
                'name': item_info['name']
            }
        else:
            # Select single frame
            frame_idx = random.randint(0, len(frames) - 1)
            
            return {
                'input': frames[frame_idx],
                'name': item_info['name']
            }
    
    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load frames from video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                # Limit frames for memory efficiency
                if len(frames) >= 100:
                    break
        
        finally:
            cap.release()
        
        return frames


class ImagePairDataset(BaseVideoDataset):
    """Dataset for image pairs (for frame-based models)"""
    
    def _load_dataset(self):
        """Load image pair dataset"""
        dataset_path = Path(self.config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Look for paired structure
        input_dir = dataset_path / "input"
        target_dir = dataset_path / "target"
        
        if not (input_dir.exists() and target_dir.exists()):
            raise FileNotFoundError("Expected 'input' and 'target' directories in dataset path")
        
        # Find matching image pairs
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        input_images = []
        
        for ext in image_extensions:
            input_images.extend(input_dir.glob(f"*{ext}"))
        
        for input_image in input_images:
            target_image = target_dir / input_image.name
            
            if target_image.exists():
                self.data_items.append({
                    'input_path': str(input_image),
                    'target_path': str(target_image),
                    'name': input_image.stem
                })
    
    def _load_item(self, index: int) -> Dict[str, Any]:
        """Load an image pair item"""
        item_info = self.data_items[index]
        
        # Load images
        input_image = np.array(Image.open(item_info['input_path']).convert('RGB'))
        target_image = np.array(Image.open(item_info['target_path']).convert('RGB'))
        
        return {
            'input': input_image,
            'target': target_image,
            'name': item_info['name']
        }


class DatasetFactory:
    """Factory for creating datasets"""
    
    @staticmethod
    def create_dataset(config: DatasetConfig, split: str = 'train') -> BaseVideoDataset:
        """Create dataset based on configuration"""
        if config.dataset_type == 'video_pairs':
            return VideoPairDataset(config, split)
        elif config.dataset_type == 'single_video':
            return SingleVideoDataset(config, split)
        elif config.dataset_type == 'image_pairs':
            return ImagePairDataset(config, split)
        else:
            raise ValueError(f"Unknown dataset type: {config.dataset_type}")


class DataLoaderManager:
    """Manager for creating and managing data loaders"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.datasets = {}
        self.data_loaders = {}
    
    def create_datasets(self) -> Dict[str, BaseVideoDataset]:
        """Create train/val/test datasets"""
        splits = ['train']
        
        if self.config.val_split > 0:
            splits.append('val')
        
        if self.config.test_split > 0:
            splits.append('test')
        
        for split in splits:
            dataset = DatasetFactory.create_dataset(self.config, split)
            self.datasets[split] = dataset
        
        logger.info(f"Created datasets: {list(self.datasets.keys())}")
        return self.datasets
    
    def create_data_loaders(self, distributed: bool = False) -> Dict[str, DataLoader]:
        """Create data loaders for all datasets"""
        if not self.datasets:
            self.create_datasets()
        
        for split, dataset in self.datasets.items():
            # Configure sampler
            sampler = None
            shuffle = self.config.shuffle and split == 'train'
            
            if distributed:
                sampler = DistributedSampler(dataset, shuffle=shuffle)
                shuffle = False  # Sampler handles shuffling
            
            # Create data loader
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=split == 'train',
                persistent_workers=self.config.num_workers > 0
            )
            
            self.data_loaders[split] = data_loader
        
        logger.info(f"Created data loaders: {list(self.data_loaders.keys())}")
        return self.data_loaders
    
    def get_data_loader(self, split: str) -> Optional[DataLoader]:
        """Get data loader for specific split"""
        return self.data_loaders.get(split)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about datasets"""
        info = {
            'config': self.config.to_dict(),
            'datasets': {}
        }
        
        for split, dataset in self.datasets.items():
            info['datasets'][split] = {
                'size': len(dataset),
                'type': type(dataset).__name__
            }
        
        return info


def create_data_config(
    name: str,
    dataset_path: str,
    dataset_type: str,
    **kwargs
) -> DatasetConfig:
    """Factory function to create dataset configuration"""
    return DatasetConfig(
        name=name,
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        **kwargs
    )


def validate_dataset_structure(dataset_path: str, dataset_type: str) -> Dict[str, Any]:
    """Validate dataset structure and return information"""
    path = Path(dataset_path)
    validation_result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    if not path.exists():
        validation_result['errors'].append(f"Dataset path does not exist: {dataset_path}")
        return validation_result
    
    try:
        if dataset_type == 'video_pairs':
            input_dir = path / 'input'
            target_dir = path / 'target'
            
            if not input_dir.exists():
                validation_result['errors'].append("Missing 'input' directory")
            
            if not target_dir.exists():
                validation_result['errors'].append("Missing 'target' directory")
            
            if input_dir.exists() and target_dir.exists():
                input_videos = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi"))
                target_videos = list(target_dir.glob("*.mp4")) + list(target_dir.glob("*.avi"))
                
                validation_result['info']['input_videos'] = len(input_videos)
                validation_result['info']['target_videos'] = len(target_videos)
                
                # Check for matching pairs
                matched_pairs = 0
                for input_video in input_videos:
                    target_video = target_dir / input_video.name
                    if target_video.exists():
                        matched_pairs += 1
                
                validation_result['info']['matched_pairs'] = matched_pairs
                
                if matched_pairs == 0:
                    validation_result['errors'].append("No matching video pairs found")
                elif matched_pairs < len(input_videos):
                    validation_result['warnings'].append(f"Only {matched_pairs}/{len(input_videos)} videos have matching pairs")
        
        elif dataset_type == 'single_video':
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            total_videos = 0
            
            for ext in video_extensions:
                videos = list(path.rglob(f"*{ext}"))
                total_videos += len(videos)
            
            validation_result['info']['total_videos'] = total_videos
            
            if total_videos == 0:
                validation_result['errors'].append("No video files found")
        
        elif dataset_type == 'image_pairs':
            input_dir = path / 'input'
            target_dir = path / 'target'
            
            if not input_dir.exists():
                validation_result['errors'].append("Missing 'input' directory")
            
            if not target_dir.exists():
                validation_result['errors'].append("Missing 'target' directory")
            
            if input_dir.exists() and target_dir.exists():
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                input_images = []
                
                for ext in image_extensions:
                    input_images.extend(input_dir.glob(f"*{ext}"))
                
                matched_pairs = 0
                for input_image in input_images:
                    target_image = target_dir / input_image.name
                    if target_image.exists():
                        matched_pairs += 1
                
                validation_result['info']['input_images'] = len(input_images)
                validation_result['info']['matched_pairs'] = matched_pairs
                
                if matched_pairs == 0:
                    validation_result['errors'].append("No matching image pairs found")
        
        # Set valid flag
        validation_result['valid'] = len(validation_result['errors']) == 0
        
    except Exception as e:
        validation_result['errors'].append(f"Validation error: {str(e)}")
    
    return validation_result


# Export main classes and functions
__all__ = [
    'DatasetConfig',
    'VideoAugmentation',
    'DataCache',
    'BaseVideoDataset',
    'VideoPairDataset',
    'SingleVideoDataset',
    'ImagePairDataset',
    'DatasetFactory',
    'DataLoaderManager',
    'create_data_config',
    'validate_dataset_structure'
]