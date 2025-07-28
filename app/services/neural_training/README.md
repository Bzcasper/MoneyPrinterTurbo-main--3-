# Neural Training System for MoneyPrinterTurbo

This module provides comprehensive neural network training capabilities for video enhancement and generation tasks in MoneyPrinterTurbo.

## Overview

The neural training system consists of five main components:

1. **Training Infrastructure** - Job scheduling, resource management, experiment tracking
2. **Video Enhancement Models** - Neural networks for upscaling, quality enhancement, scene detection
3. **Training Data Management** - Data pipelines and augmentation systems
4. **Model Integration** - Integration with existing video processing pipeline
5. **Training Workflows** - Hyperparameter optimization and monitoring

## Features

### Training Infrastructure
- **Experiment Tracking**: Automatic versioning and metadata management
- **Model Registry**: Centralized storage and versioning of trained models
- **Job Scheduling**: Resource-aware training job scheduling
- **Multi-GPU Support**: Distributed training across multiple GPUs
- **TensorBoard Integration**: Real-time training monitoring

### Video Enhancement Models
- **Video Upscaler**: ESRGAN-inspired architecture for 2x/4x upscaling
- **Quality Enhancer**: U-Net based model for noise reduction and quality improvement
- **Scene Detector**: CNN-based scene classification and attention-based cropping
- **Audio-Visual Sync**: Model for detecting audio-visual synchronization
- **Style Transfer**: AdaIN-based style transfer for video aesthetics

### Training Data Management
- **Multiple Dataset Types**: Support for video pairs, single videos, and image pairs
- **Advanced Augmentation**: Spatial and temporal augmentations for video data
- **Efficient Loading**: Cached data loading with memory optimization
- **Validation Splits**: Automatic train/validation/test splitting

### Model Integration
- **Inference Server**: High-performance model serving with GPU acceleration
- **A/B Testing**: Framework for comparing different model versions
- **Performance Monitoring**: Real-time monitoring of model performance and drift detection
- **Fallback Support**: Graceful fallback to traditional methods if neural models fail

### Training Workflows
- **Hyperparameter Optimization**: Optuna-based automated hyperparameter tuning
- **Early Stopping**: Intelligent training termination to prevent overfitting
- **Progress Monitoring**: Comprehensive logging and visualization
- **Checkpoint Management**: Automatic model checkpointing and restoration

## Quick Start

### 1. Basic Training

```python
from app.services.neural_training import create_training_config, submit_training_job

# Create training configuration
config = create_training_config(
    model_type="video_upscaler",
    experiment_name="my_upscaler_experiment",
    dataset_path="/path/to/dataset",
    epochs=100,
    batch_size=8,
    learning_rate=1e-4
)

# Submit training job
job_id = submit_training_job(config)
print(f"Training job submitted: {job_id}")
```

### 2. Using Neural Enhancement

```python
import asyncio
from app.services.quality_enhancement import enhance_video_quality, create_quality_enhancement_config
from app.models.schema import VideoParams

async def enhance_video():
    # Configure neural enhancement
    config = create_quality_enhancement_config(
        enable_neural_models=True,
        neural_upscaling=True,
        neural_quality_enhancement=True
    )
    
    # Create video parameters
    params = VideoParams(
        voice_name="en-US-JennyNeural",
        subtitle_enabled=True
    )
    
    # Enhance video
    result = await enhance_video_quality(
        input_path="input_video.mp4",
        output_path="enhanced_video.mp4",
        params=params,
        config=config
    )
    
    print(f"Enhancement completed: {result['success']}")

# Run enhancement
asyncio.run(enhance_video())
```

### 3. Hyperparameter Optimization

```python
from app.services.neural_training.training_workflows import (
    run_hyperparameter_optimization,
    HyperparameterSpace,
    OptimizationObjective
)

# Define hyperparameter search space
space = HyperparameterSpace(
    learning_rate=(1e-5, 1e-2),
    batch_size=[4, 8, 16, 32],
    hidden_dim=(32, 256)
)

# Run optimization
results = run_hyperparameter_optimization(
    base_config=config,
    hyperparameter_space=space,
    n_trials=50,
    objective=OptimizationObjective.MINIMIZE_LOSS
)

print(f"Best parameters: {results['best_params']}")
```

### 4. Model Benchmarking

```python
from scripts.benchmark.neural_benchmarks import run_neural_benchmarks

# Run comprehensive benchmarks
results = await run_neural_benchmarks(
    test_dataset_path="/path/to/test_dataset",
    output_dir="benchmark_results",
    num_samples=100
)

print(f"Benchmarked {results['total_models_tested']} models")
```

## Architecture

### Model Types

The system supports the following neural model types:

1. **VIDEO_UPSCALER**: Super-resolution for video upscaling
2. **QUALITY_ENHANCER**: General quality improvement
3. **SCENE_DETECTOR**: Scene classification and cropping
4. **AUDIO_VISUAL_SYNC**: Audio-visual synchronization detection
5. **STYLE_TRANSFER**: Video style transfer and aesthetic enhancement

### Training Pipeline

```
Data Loading → Preprocessing → Model Training → Validation → Model Registration
     ↓              ↓             ↓              ↓              ↓
  Augmentation → GPU Allocation → Checkpointing → Metrics → Version Control
```

### Integration with MoneyPrinterTurbo

The neural training system seamlessly integrates with the existing MoneyPrinterTurbo pipeline:

1. **GPU Management**: Leverages existing GPU management system
2. **Quality Enhancement**: Extends the quality enhancement service
3. **Video Processing**: Integrates with video generation workflow
4. **Configuration**: Uses existing configuration management

## Configuration

### Training Configuration

```python
TrainingConfig(
    model_type=ModelType.VIDEO_UPSCALER,
    experiment_name="my_experiment",
    dataset_path="/path/to/dataset",
    
    # Training parameters
    batch_size=8,
    learning_rate=1e-4,
    epochs=100,
    optimizer="adam",
    
    # Model architecture
    hidden_dim=64,
    input_size=(256, 256),
    
    # Advanced settings
    use_distributed=True,
    mixed_precision=True,
    early_stopping_patience=20
)
```

### Dataset Configuration

```python
DatasetConfig(
    name="my_dataset",
    dataset_path="/path/to/dataset",
    dataset_type="video_pairs",  # or "single_video", "image_pairs"
    
    # Data settings
    input_size=(256, 256),
    batch_size=8,
    train_split=0.8,
    val_split=0.2,
    
    # Augmentation
    enable_augmentation=True,
    augmentation_probability=0.8
)
```

## Dataset Format

### Video Pairs Dataset
```
dataset/
├── input/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── target/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

### Image Pairs Dataset
```
dataset/
├── input/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── target/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Single Video Dataset
```
dataset/
├── video1.mp4
├── video2.mp4
├── video3.mp4
└── ...
```

## API Reference

### Training Infrastructure

```python
# Experiment tracking
tracker = get_experiment_tracker()
experiment_id = tracker.create_experiment(config)

# Model versioning
versioning = get_model_versioning()
version = versioning.register_model(model_path, model_type, experiment_id)

# Job scheduling
scheduler = get_job_scheduler()
job_id = scheduler.submit_job(config)
```

### Model Integration

```python
# Neural processor
processor = get_neural_processor()

# Load model
model_id = processor.model_server.load_model(ModelType.VIDEO_UPSCALER)

# Run inference
result = await processor.enhance_video_quality(frames, model_id)
```

### Training Workflows

```python
# Create trainer
trainer = ModelTrainer(config)

# Run training
result = trainer.train()

# Hyperparameter optimization
optimizer = HyperparameterOptimizer("study_name")
best_params = optimizer.optimize(config, space, n_trials=50)
```

## Performance Considerations

### Memory Management
- Use mixed precision training to reduce memory usage
- Configure batch sizes based on available GPU memory
- Enable gradient checkpointing for large models

### GPU Utilization
- Use distributed training for multi-GPU setups
- Monitor GPU utilization through the GPU manager
- Balance training jobs across available GPUs

### Data Loading
- Use multiple workers for data loading
- Enable pin memory for faster GPU transfers
- Cache frequently accessed data

## Monitoring and Debugging

### TensorBoard
Access TensorBoard logs at: `experiments/runs/{experiment_id}/logs/`

```bash
tensorboard --logdir experiments/runs/
```

### Logging
The system uses structured logging with different levels:
- `INFO`: General information and progress
- `WARNING`: Non-critical issues and fallbacks
- `ERROR`: Training failures and critical errors
- `DEBUG`: Detailed debugging information

### Performance Monitoring
- Model inference times and memory usage
- Training progress and loss curves
- GPU utilization and memory consumption
- Quality metrics (PSNR, SSIM, LPIPS)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable mixed precision training
   - Use gradient checkpointing

2. **Training Convergence Issues**
   - Adjust learning rate
   - Use learning rate scheduling
   - Check data quality and preprocessing

3. **Model Loading Errors**
   - Verify model compatibility
   - Check file paths and permissions
   - Ensure proper model registration

### Performance Optimization

1. **Slow Training**
   - Increase batch size if memory allows
   - Use multiple GPUs with distributed training
   - Optimize data loading pipeline

2. **Poor Model Quality**
   - Increase model capacity (hidden dimensions)
   - Add more training data
   - Improve data augmentation

3. **High Memory Usage**
   - Use mixed precision training
   - Reduce model size or batch size
   - Enable gradient checkpointing

## Contributing

When adding new model types or features:

1. **Model Implementation**: Add new models to `video_enhancement_models.py`
2. **Training Logic**: Update training workflows as needed
3. **Integration**: Ensure proper integration with existing pipeline
4. **Testing**: Add comprehensive tests and benchmarks
5. **Documentation**: Update this README and add docstrings

## Future Enhancements

Planned improvements include:

- **Model Quantization**: INT8 quantization for faster inference
- **Dynamic Batching**: Automatic batch size optimization
- **Cloud Training**: Support for cloud-based training
- **AutoML**: Automated model architecture search
- **Real-time Training**: Continuous learning from user feedback