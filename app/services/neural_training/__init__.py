"""
Neural Training Module for MoneyPrinterTurbo
============================================

This module provides comprehensive neural network training capabilities
for video enhancement and generation tasks.

Key Components:
- Training Infrastructure: Job scheduling, resource management, experiment tracking
- Video Enhancement Models: Neural networks for upscaling, quality enhancement, etc.
- Training Data Management: Data pipelines and augmentation systems
- Model Integration: Integration with existing video processing pipeline
- Training Workflows: Hyperparameter optimization and monitoring

Author: ML Model Developer
Version: 1.0.0
"""

from .training_infrastructure import (
    TrainingConfig,
    TrainingJob,
    TrainingStatus,
    ModelType,
    ExperimentTracker,
    ModelVersioning,
    TrainingJobScheduler,
    get_experiment_tracker,
    get_model_versioning,
    get_job_scheduler,
    create_training_config,
    submit_training_job
)

__all__ = [
    'TrainingConfig',
    'TrainingJob',
    'TrainingStatus', 
    'ModelType',
    'ExperimentTracker',
    'ModelVersioning',
    'TrainingJobScheduler',
    'get_experiment_tracker',
    'get_model_versioning',
    'get_job_scheduler',
    'create_training_config',
    'submit_training_job'
]