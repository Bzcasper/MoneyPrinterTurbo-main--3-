"""
Neural Training Infrastructure for MoneyPrinterTurbo
====================================================

Advanced neural network training infrastructure with multi-GPU support,
experiment tracking, and model versioning for video enhancement models.

Features:
- Distributed training across multiple GPUs
- Model versioning and experiment tracking
- Training job scheduling and resource allocation
- Real-time training monitoring and early stopping
- Integration with existing GPU management system

Author: ML Model Developer
Version: 1.0.0
"""

import os
import json
import time
import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import uuid
import pickle
import hashlib

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from loguru import logger
from app.services.gpu_manager import get_gpu_manager, GPUVendor, GPUInfo


class TrainingStatus(Enum):
    """Training job status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(Enum):
    """Neural model types for video enhancement"""
    VIDEO_UPSCALER = "video_upscaler"
    QUALITY_ENHANCER = "quality_enhancer"
    SCENE_DETECTOR = "scene_detector"
    AUDIO_VISUAL_SYNC = "audio_visual_sync"
    STYLE_TRANSFER = "style_transfer"
    NOISE_REDUCER = "noise_reducer"
    COLOR_CORRECTOR = "color_corrector"


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    model_type: ModelType
    experiment_name: str
    dataset_path: str
    output_dir: str
    
    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 100
    optimizer: str = "adam"
    loss_function: str = "mse"
    
    # Model architecture
    model_architecture: str = "resnet_unet"
    input_size: Tuple[int, int] = (256, 256)
    num_channels: int = 3
    hidden_dim: int = 64
    
    # Training settings
    validation_split: float = 0.2
    save_frequency: int = 10  # epochs
    validate_frequency: int = 5  # epochs
    early_stopping_patience: int = 20
    gradient_clip_norm: float = 1.0
    
    # Multi-GPU settings
    use_distributed: bool = True
    num_gpus: int = 0  # 0 = auto-detect
    mixed_precision: bool = True
    
    # Data augmentation
    data_augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "horizontal_flip": 0.5,
        "rotation": 15,
        "brightness": 0.2,
        "contrast": 0.2,
        "gaussian_noise": 0.01
    })
    
    # Advanced settings
    resume_from_checkpoint: Optional[str] = None
    pretrained_weights: Optional[str] = None
    scheduler_type: str = "cosine"
    warmup_epochs: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model_type": self.model_type.value,
            "experiment_name": self.experiment_name,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "loss_function": self.loss_function,
            "model_architecture": self.model_architecture,
            "input_size": self.input_size,
            "num_channels": self.num_channels,
            "hidden_dim": self.hidden_dim,
            "validation_split": self.validation_split,
            "save_frequency": self.save_frequency,
            "validate_frequency": self.validate_frequency,
            "early_stopping_patience": self.early_stopping_patience,
            "gradient_clip_norm": self.gradient_clip_norm,
            "use_distributed": self.use_distributed,
            "num_gpus": self.num_gpus,
            "mixed_precision": self.mixed_precision,
            "data_augmentation": self.data_augmentation,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "pretrained_weights": self.pretrained_weights,
            "scheduler_type": self.scheduler_type,
            "warmup_epochs": self.warmup_epochs
        }


@dataclass
class TrainingJob:
    """Training job representation"""
    job_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Training metrics
    current_epoch: int = 0
    best_loss: float = float('inf')
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    # Resource allocation
    allocated_gpus: List[int] = field(default_factory=list)
    memory_allocated: int = 0  # MB
    
    # Job metadata
    error_message: Optional[str] = None
    checkpoint_path: Optional[str] = None
    model_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary"""
        return {
            "job_id": self.job_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "allocated_gpus": self.allocated_gpus,
            "memory_allocated": self.memory_allocated,
            "error_message": self.error_message,
            "checkpoint_path": self.checkpoint_path,
            "model_path": self.model_path
        }


class ExperimentTracker:
    """Experiment tracking and versioning system"""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_dir / "runs").mkdir(exist_ok=True)
        (self.base_dir / "models").mkdir(exist_ok=True)
        (self.base_dir / "logs").mkdir(exist_ok=True)
        (self.base_dir / "checkpoints").mkdir(exist_ok=True)
        
        self.experiments_db = self.base_dir / "experiments.json"
        self._load_experiments()
    
    def _load_experiments(self):
        """Load experiments database"""
        if self.experiments_db.exists():
            with open(self.experiments_db, 'r') as f:
                self.experiments = json.load(f)
        else:
            self.experiments = {}
    
    def _save_experiments(self):
        """Save experiments database"""
        with open(self.experiments_db, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def create_experiment(self, config: TrainingConfig) -> str:
        """Create new experiment and return experiment ID"""
        experiment_id = str(uuid.uuid4())
        
        # Create experiment directory structure
        exp_dir = self.base_dir / "runs" / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "models").mkdir(exist_ok=True)
        
        # Save configuration
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Update experiments database
        self.experiments[experiment_id] = {
            "experiment_id": experiment_id,
            "name": config.experiment_name,
            "model_type": config.model_type.value,
            "created_at": datetime.now().isoformat(),
            "config_path": str(config_path),
            "status": "created"
        }
        
        self._save_experiments()
        logger.info(f"Created experiment: {experiment_id} ({config.experiment_name})")
        
        return experiment_id
    
    def get_experiment_dir(self, experiment_id: str) -> Path:
        """Get experiment directory path"""
        return self.base_dir / "runs" / experiment_id
    
    def get_tensorboard_dir(self, experiment_id: str) -> Path:
        """Get TensorBoard log directory"""
        return self.get_experiment_dir(experiment_id) / "logs"
    
    def get_checkpoint_dir(self, experiment_id: str) -> Path:
        """Get checkpoint directory"""
        return self.get_experiment_dir(experiment_id) / "checkpoints"
    
    def get_model_dir(self, experiment_id: str) -> Path:
        """Get model directory"""
        return self.get_experiment_dir(experiment_id) / "models"
    
    def list_experiments(self, model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """List all experiments, optionally filtered by model type"""
        experiments = list(self.experiments.values())
        
        if model_type:
            experiments = [exp for exp in experiments if exp["model_type"] == model_type.value]
        
        # Sort by creation date (newest first)
        experiments.sort(key=lambda x: x["created_at"], reverse=True)
        
        return experiments
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID"""
        return self.experiments.get(experiment_id)
    
    def update_experiment_status(self, experiment_id: str, status: str, metadata: Dict[str, Any] = None):
        """Update experiment status"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["status"] = status
            self.experiments[experiment_id]["updated_at"] = datetime.now().isoformat()
            
            if metadata:
                self.experiments[experiment_id].update(metadata)
            
            self._save_experiments()


class ModelVersioning:
    """Model versioning and registry system"""
    
    def __init__(self, base_dir: str = "model_registry"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.registry_file = self.base_dir / "model_registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
    
    def _save_registry(self):
        """Save model registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate hash of model file"""
        hasher = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def register_model(
        self,
        model_path: str,
        model_type: ModelType,
        experiment_id: str,
        version: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Register a trained model"""
        model_name = f"{model_type.value}"
        
        if model_name not in self.registry:
            self.registry[model_name] = {
                "model_type": model_type.value,
                "versions": {}
            }
        
        # Auto-generate version if not provided
        if version is None:
            existing_versions = list(self.registry[model_name]["versions"].keys())
            version_nums = [int(v.split('.')[-1]) for v in existing_versions if v.startswith('v')]
            next_num = max(version_nums, default=0) + 1
            version = f"v{next_num}"
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model_path)
        
        # Copy model to registry
        model_filename = f"{model_name}_{version}.pth"
        registry_path = self.base_dir / model_filename
        
        import shutil
        shutil.copy2(model_path, registry_path)
        
        # Register version
        self.registry[model_name]["versions"][version] = {
            "version": version,
            "model_path": str(registry_path),
            "experiment_id": experiment_id,
            "model_hash": model_hash,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self._save_registry()
        logger.info(f"Registered model: {model_name} {version}")
        
        return version
    
    def get_model_path(self, model_type: ModelType, version: str = "latest") -> Optional[str]:
        """Get path to registered model"""
        model_name = model_type.value
        
        if model_name not in self.registry:
            return None
        
        versions = self.registry[model_name]["versions"]
        
        if version == "latest":
            if not versions:
                return None
            # Get latest version by creation date
            latest_version = max(versions.keys(), key=lambda v: versions[v]["created_at"])
            return versions[latest_version]["model_path"]
        else:
            if version not in versions:
                return None
            return versions[version]["model_path"]
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """List all registered models"""
        models = []
        
        for model_name, model_data in self.registry.items():
            if model_type and model_data["model_type"] != model_type.value:
                continue
            
            for version, version_data in model_data["versions"].items():
                models.append({
                    "model_name": model_name,
                    "model_type": model_data["model_type"],
                    "version": version,
                    "created_at": version_data["created_at"],
                    "experiment_id": version_data["experiment_id"],
                    "model_path": version_data["model_path"],
                    "metadata": version_data["metadata"]
                })
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x["created_at"], reverse=True)
        
        return models


class TrainingJobScheduler:
    """Training job scheduler with resource management"""
    
    def __init__(self):
        self.gpu_manager = get_gpu_manager()
        self.experiment_tracker = ExperimentTracker()
        self.model_versioning = ModelVersioning()
        
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.job_queue: List[TrainingJob] = []
        self.job_history: List[TrainingJob] = []
        
        self._scheduler_running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        
        # Load job history
        self._load_job_history()
    
    def _load_job_history(self):
        """Load job history from disk"""
        history_file = Path("training_jobs.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                for job_data in history_data:
                    job = self._dict_to_job(job_data)
                    self.job_history.append(job)
                    
                logger.info(f"Loaded {len(self.job_history)} training jobs from history")
            except Exception as e:
                logger.warning(f"Failed to load job history: {e}")
    
    def _save_job_history(self):
        """Save job history to disk"""
        history_file = Path("training_jobs.json")
        try:
            history_data = [job.to_dict() for job in self.job_history]
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save job history: {e}")
    
    def _dict_to_job(self, job_data: Dict[str, Any]) -> TrainingJob:
        """Convert dictionary to TrainingJob object"""
        config_data = job_data["config"]
        config = TrainingConfig(
            model_type=ModelType(config_data["model_type"]),
            experiment_name=config_data["experiment_name"],
            dataset_path=config_data["dataset_path"],
            output_dir=config_data["output_dir"]
        )
        
        # Update config with all fields
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        job = TrainingJob(
            job_id=job_data["job_id"],
            config=config,
            status=TrainingStatus(job_data["status"])
        )
        
        # Update job with all fields
        for key, value in job_data.items():
            if hasattr(job, key) and key not in ["config", "status"]:
                if key in ["created_at", "started_at", "completed_at"] and value:
                    setattr(job, key, datetime.fromisoformat(value))
                else:
                    setattr(job, key, value)
        
        return job
    
    def submit_job(self, config: TrainingConfig) -> str:
        """Submit a new training job"""
        job_id = str(uuid.uuid4())
        
        # Create experiment
        experiment_id = self.experiment_tracker.create_experiment(config)
        config.output_dir = str(self.experiment_tracker.get_experiment_dir(experiment_id))
        
        # Create job
        job = TrainingJob(job_id=job_id, config=config)
        
        # Add to queue
        self.job_queue.append(job)
        
        logger.info(f"Submitted training job: {job_id} ({config.experiment_name})")
        
        # Start scheduler if not running
        if not self._scheduler_running:
            self.start_scheduler()
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        # Check active jobs
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].to_dict()
        
        # Check queue
        for job in self.job_queue:
            if job.job_id == job_id:
                return job.to_dict()
        
        # Check history
        for job in self.job_history:
            if job.job_id == job_id:
                return job.to_dict()
        
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        # Check if job is in queue
        for i, job in enumerate(self.job_queue):
            if job.job_id == job_id:
                job.status = TrainingStatus.CANCELLED
                self.job_queue.pop(i)
                self.job_history.append(job)
                self._save_job_history()
                logger.info(f"Cancelled queued job: {job_id}")
                return True
        
        # Check if job is active (would need to implement job termination)
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = TrainingStatus.CANCELLED
            # TODO: Implement job termination logic
            logger.info(f"Cancelled active job: {job_id}")
            return True
        
        return False
    
    def list_jobs(
        self,
        status: Optional[TrainingStatus] = None,
        model_type: Optional[ModelType] = None
    ) -> List[Dict[str, Any]]:
        """List training jobs"""
        all_jobs = list(self.active_jobs.values()) + self.job_queue + self.job_history
        
        # Filter by status
        if status:
            all_jobs = [job for job in all_jobs if job.status == status]
        
        # Filter by model type
        if model_type:
            all_jobs = [job for job in all_jobs if job.config.model_type == model_type]
        
        # Sort by creation date (newest first)
        all_jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return [job.to_dict() for job in all_jobs]
    
    def start_scheduler(self):
        """Start the job scheduler"""
        if self._scheduler_running:
            return
        
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        logger.info("Training job scheduler started")
    
    def stop_scheduler(self):
        """Stop the job scheduler"""
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        
        logger.info("Training job scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._scheduler_running:
            try:
                self._process_job_queue()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _process_job_queue(self):
        """Process jobs in the queue"""
        if not self.job_queue:
            return
        
        # Get available resources
        available_gpus = []
        for gpu in self.gpu_manager.available_gpus:
            if gpu.is_available and gpu.memory_free > 2048:  # At least 2GB free
                available_gpus.append(gpu)
        
        if not available_gpus:
            return
        
        # Try to start jobs
        jobs_to_start = []
        
        for job in self.job_queue[:]:  # Copy to avoid modification during iteration
            if job.status != TrainingStatus.PENDING:
                continue
            
            # Determine GPU requirements
            required_gpus = job.config.num_gpus if job.config.num_gpus > 0 else min(len(available_gpus), 2)
            required_memory = job.config.batch_size * 512  # Rough estimate: 512MB per batch item
            
            # Check if we have enough resources
            suitable_gpus = []
            for gpu in available_gpus:
                if gpu.memory_free >= required_memory:
                    suitable_gpus.append(gpu)
                    if len(suitable_gpus) >= required_gpus:
                        break
            
            if len(suitable_gpus) >= required_gpus:
                # Allocate resources
                job.allocated_gpus = [gpu.id for gpu in suitable_gpus[:required_gpus]]
                job.memory_allocated = required_memory * required_gpus
                
                # Remove from available
                for gpu in suitable_gpus[:required_gpus]:
                    if gpu in available_gpus:
                        available_gpus.remove(gpu)
                
                jobs_to_start.append(job)
                
                # Remove from queue
                self.job_queue.remove(job)
        
        # Start jobs
        for job in jobs_to_start:
            self._start_training_job(job)
    
    def _start_training_job(self, job: TrainingJob):
        """Start a training job"""
        try:
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            
            # Add to active jobs
            self.active_jobs[job.job_id] = job
            
            # Update experiment status
            self.experiment_tracker.update_experiment_status(
                job.job_id, "running",
                {"allocated_gpus": job.allocated_gpus}
            )
            
            logger.info(f"Started training job: {job.job_id} on GPUs {job.allocated_gpus}")
            
            # TODO: Implement actual training process launch
            # This would typically involve spawning a separate process or
            # using a job queue system like Slurm, Kubernetes, etc.
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            # Move to history
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            self.job_history.append(job)
            self._save_job_history()
            
            logger.error(f"Failed to start training job {job.job_id}: {e}")


# Global training infrastructure instances
experiment_tracker = ExperimentTracker()
model_versioning = ModelVersioning()
job_scheduler = TrainingJobScheduler()


def get_experiment_tracker() -> ExperimentTracker:
    """Get global experiment tracker instance"""
    return experiment_tracker


def get_model_versioning() -> ModelVersioning:
    """Get global model versioning instance"""
    return model_versioning


def get_job_scheduler() -> TrainingJobScheduler:
    """Get global job scheduler instance"""
    return job_scheduler


# Factory functions
def create_training_config(
    model_type: str,
    experiment_name: str,
    dataset_path: str,
    **kwargs
) -> TrainingConfig:
    """Factory function to create training configuration"""
    return TrainingConfig(
        model_type=ModelType(model_type),
        experiment_name=experiment_name,
        dataset_path=dataset_path,
        output_dir="",  # Will be set by scheduler
        **kwargs
    )


def submit_training_job(config: TrainingConfig) -> str:
    """Submit a training job to the scheduler"""
    return job_scheduler.submit_job(config)


# Export main classes and functions
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