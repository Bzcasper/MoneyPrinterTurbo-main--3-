"""
Training Workflows for Neural Video Enhancement
===============================================

Advanced training workflow system including hyperparameter optimization,
training progress monitoring, early stopping, and automated model evaluation.

Features:
- Hyperparameter optimization with Optuna
- Training progress monitoring and visualization
- Early stopping and checkpoint management
- Automated model evaluation and benchmarking
- Distributed training coordination

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
import pickle
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from loguru import logger
from app.services.neural_training.training_infrastructure import (
    TrainingConfig, TrainingJob, TrainingStatus, ModelType,
    get_experiment_tracker, get_model_versioning, get_job_scheduler
)
from app.services.neural_training.video_enhancement_models import ModelFactory, save_model
from app.services.neural_training.data_management import DataLoaderManager, create_data_config


class OptimizationObjective(Enum):
    """Optimization objectives for hyperparameter tuning"""
    MINIMIZE_LOSS = "minimize_loss"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_INFERENCE_TIME = "minimize_inference_time"
    BALANCE_QUALITY_SPEED = "balance_quality_speed"


@dataclass
class HyperparameterSpace:
    """Hyperparameter search space definition"""
    learning_rate: Tuple[float, float] = (1e-5, 1e-2)
    batch_size: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    hidden_dim: Tuple[int, int] = (32, 256)
    optimizer: List[str] = field(default_factory=lambda: ["adam", "adamw", "sgd"])
    scheduler_type: List[str] = field(default_factory=lambda: ["cosine", "step", "plateau"])
    dropout_rate: Tuple[float, float] = (0.0, 0.5)
    weight_decay: Tuple[float, float] = (1e-5, 1e-2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "optimizer": self.optimizer,
            "scheduler_type": self.scheduler_type,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay
        }


@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    accuracy: Optional[float] = None
    learning_rate: float = 0.0
    time_per_epoch: float = 0.0
    memory_usage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "accuracy": self.accuracy,
            "learning_rate": self.learning_rate,
            "time_per_epoch": self.time_per_epoch,
            "memory_usage": self.memory_usage
        }


class EarlyStopping:
    """Early stopping implementation"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.should_stop = False
        
        logger.info(f"Early stopping initialized (patience={patience}, mode={mode})")
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop"""
        
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered (no improvement for {self.patience} epochs)")
        
        return self.should_stop


class TrainingProgressMonitor:
    """Training progress monitoring and visualization"""
    
    def __init__(self, experiment_id: str, log_dir: str):
        self.experiment_id = experiment_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Metrics storage
        self.metrics_history: List[TrainingMetrics] = []
        self.best_metrics = {
            'best_train_loss': float('inf'),
            'best_val_loss': float('inf'),
            'best_accuracy': 0.0
        }
        
        logger.info(f"Training monitor initialized for experiment: {experiment_id}")
    
    def log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics"""
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/Train', metrics.train_loss, metrics.epoch)
        if metrics.val_loss is not None:
            self.writer.add_scalar('Loss/Validation', metrics.val_loss, metrics.epoch)
        if metrics.accuracy is not None:
            self.writer.add_scalar('Accuracy', metrics.accuracy, metrics.epoch)
        
        self.writer.add_scalar('Learning_Rate', metrics.learning_rate, metrics.epoch)
        self.writer.add_scalar('Time_Per_Epoch', metrics.time_per_epoch, metrics.epoch)
        self.writer.add_scalar('Memory_Usage', metrics.memory_usage, metrics.epoch)
        
        # Update best metrics
        if metrics.train_loss < self.best_metrics['best_train_loss']:
            self.best_metrics['best_train_loss'] = metrics.train_loss
        
        if metrics.val_loss is not None and metrics.val_loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = metrics.val_loss
        
        if metrics.accuracy is not None and metrics.accuracy > self.best_metrics['best_accuracy']:
            self.best_metrics['best_accuracy'] = metrics.accuracy
        
        # Log progress
        logger.info(
            f"Epoch {metrics.epoch}: "
            f"train_loss={metrics.train_loss:.4f}, "
            f"val_loss={metrics.val_loss:.4f if metrics.val_loss else 'N/A'}, "
            f"lr={metrics.learning_rate:.6f}"
        )
    
    def log_model_graph(self, model: nn.Module, input_shape: Tuple[int, ...]):
        """Log model architecture to TensorBoard"""
        try:
            dummy_input = torch.randn(1, *input_shape)
            self.writer.add_graph(model, dummy_input)
            logger.info("Model graph logged to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to log model graph: {str(e)}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters and final metrics"""
        self.writer.add_hparams(hparams, metrics)
        logger.info("Hyperparameters logged to TensorBoard")
    
    def save_metrics(self):
        """Save metrics to file"""
        metrics_file = self.log_dir / "metrics.json"
        
        with open(metrics_file, 'w') as f:
            metrics_data = {
                'experiment_id': self.experiment_id,
                'best_metrics': self.best_metrics,
                'history': [m.to_dict() for m in self.metrics_history]
            }
            json.dump(metrics_data, f, indent=2)
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()
        self.save_metrics()


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, study_name: str, objective: OptimizationObjective = OptimizationObjective.MINIMIZE_LOSS):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        self.study_name = study_name
        self.objective = objective
        
        # Create Optuna study
        self.study = optuna.create_study(
            study_name=study_name,
            direction="minimize" if "minimize" in objective.value else "maximize",
            sampler=TPESampler(),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        logger.info(f"Hyperparameter optimizer initialized: {study_name}")
    
    def suggest_hyperparameters(self, trial: optuna.Trial, space: HyperparameterSpace) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial"""
        
        suggested_params = {
            'learning_rate': trial.suggest_float(
                'learning_rate',
                space.learning_rate[0],
                space.learning_rate[1],
                log=True
            ),
            'batch_size': trial.suggest_categorical('batch_size', space.batch_size),
            'hidden_dim': trial.suggest_int('hidden_dim', space.hidden_dim[0], space.hidden_dim[1]),
            'optimizer': trial.suggest_categorical('optimizer', space.optimizer),
            'scheduler_type': trial.suggest_categorical('scheduler_type', space.scheduler_type),
            'dropout_rate': trial.suggest_float('dropout_rate', space.dropout_rate[0], space.dropout_rate[1]),
            'weight_decay': trial.suggest_float(
                'weight_decay',
                space.weight_decay[0],
                space.weight_decay[1],
                log=True
            )
        }
        
        return suggested_params
    
    def objective_function(
        self,
        trial: optuna.Trial,
        train_config: TrainingConfig,
        hyperparameter_space: HyperparameterSpace
    ) -> float:
        """Objective function for hyperparameter optimization"""
        
        # Get suggested hyperparameters
        suggested_params = self.suggest_hyperparameters(trial, hyperparameter_space)
        
        # Update training config with suggested parameters
        optimized_config = self._update_config_with_params(train_config, suggested_params)
        
        try:
            # Run training with suggested parameters
            trainer = ModelTrainer(optimized_config)
            result = trainer.train()
            
            # Return objective value based on optimization goal
            if self.objective == OptimizationObjective.MINIMIZE_LOSS:
                return result['best_val_loss']
            elif self.objective == OptimizationObjective.MAXIMIZE_ACCURACY:
                return result['best_accuracy']
            elif self.objective == OptimizationObjective.MINIMIZE_INFERENCE_TIME:
                return result['avg_inference_time']
            else:
                # Balance quality and speed
                return result['best_val_loss'] + (result['avg_inference_time'] * 0.1)
        
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            # Return worst possible value
            return float('inf') if "minimize" in self.objective.value else float('-inf')
    
    def _update_config_with_params(
        self,
        config: TrainingConfig,
        params: Dict[str, Any]
    ) -> TrainingConfig:
        """Update training config with suggested parameters"""
        
        updated_config = TrainingConfig(
            model_type=config.model_type,
            experiment_name=f"{config.experiment_name}_trial",
            dataset_path=config.dataset_path,
            output_dir=config.output_dir,
            
            # Updated hyperparameters
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            optimizer=params['optimizer'],
            scheduler_type=params['scheduler_type'],
            
            # Model architecture updates
            hidden_dim=params['hidden_dim'],
            
            # Other parameters from original config
            epochs=min(config.epochs, 20),  # Limit epochs for optimization
            model_architecture=config.model_architecture,
            input_size=config.input_size,
            num_channels=config.num_channels,
            validation_split=config.validation_split,
            early_stopping_patience=10,  # Reduced for optimization
            use_distributed=False,  # Disable for optimization trials
            mixed_precision=config.mixed_precision
        )
        
        return updated_config
    
    def optimize(
        self,
        train_config: TrainingConfig,
        hyperparameter_space: HyperparameterSpace,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        logger.info(f"Starting hyperparameter optimization ({n_trials} trials)")
        
        # Define objective function with fixed parameters
        def objective(trial):
            return self.objective_function(trial, train_config, hyperparameter_space)
        
        # Run optimization
        self.study.optimize(objective, n_trials=n_trials)
        
        # Get best results
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        logger.success(f"Optimization completed. Best value: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(self.study.trials),
            'study': self.study
        }


class ModelTrainer:
    """Main model training class"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.experiment_tracker = get_experiment_tracker()
        self.model_versioning = get_model_versioning()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.data_loaders = None
        self.progress_monitor = None
        self.early_stopping = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_interrupted = False
        
        logger.info(f"Model trainer initialized for {config.model_type.value}")
    
    def setup_training(self) -> str:
        """Setup training components"""
        
        # Create experiment
        experiment_id = self.experiment_tracker.create_experiment(self.config)
        self.experiment_id = experiment_id
        
        # Update output directory
        self.config.output_dir = str(self.experiment_tracker.get_experiment_dir(experiment_id))
        
        # Setup model
        self.model = self._create_model()
        
        # Setup data loaders
        self.data_loaders = self._create_data_loaders()
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_loss_function()
        
        # Setup monitoring
        log_dir = self.experiment_tracker.get_tensorboard_dir(experiment_id)
        self.progress_monitor = TrainingProgressMonitor(experiment_id, str(log_dir))
        
        # Setup early stopping
        if self.config.early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                mode="min"
            )
        
        # Log model graph
        self.progress_monitor.log_model_graph(self.model, self.config.input_size)
        
        logger.info(f"Training setup completed for experiment: {experiment_id}")
        return experiment_id
    
    def _create_model(self) -> nn.Module:
        """Create model instance"""
        
        model = ModelFactory.create_model(
            self.config.model_type.value,
            num_channels=self.config.num_channels,
            hidden_dim=self.config.hidden_dim
        )
        
        # Load pretrained weights if specified
        if self.config.pretrained_weights:
            checkpoint = torch.load(self.config.pretrained_weights, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded pretrained weights: {self.config.pretrained_weights}")
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup distributed training
        if self.config.use_distributed and torch.cuda.device_count() > 1:
            model = DDP(model)
            logger.info(f"Distributed training enabled on {torch.cuda.device_count()} GPUs")
        
        return model
    
    def _create_data_loaders(self) -> Dict[str, Any]:
        """Create data loaders"""
        
        # Create dataset config
        dataset_config = create_data_config(
            name=self.config.experiment_name,
            dataset_path=self.config.dataset_path,
            dataset_type="video_pairs",  # Default type
            input_size=self.config.input_size,
            batch_size=self.config.batch_size,
            train_split=1.0 - self.config.validation_split,
            val_split=self.config.validation_split,
            enable_augmentation=True,
            data_augmentation=self.config.data_augmentation
        )
        
        # Create data loader manager
        data_manager = DataLoaderManager(dataset_config)
        data_loaders = data_manager.create_data_loaders(distributed=self.config.use_distributed)
        
        return data_loaders
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        
        if self.config.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-4
            )
        elif self.config.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-4
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        
        if self.config.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            # No scheduler
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        
        return scheduler
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function"""
        
        if self.config.loss_function == "mse":
            return nn.MSELoss()
        elif self.config.loss_function == "l1":
            return nn.L1Loss()
        elif self.config.loss_function == "huber":
            return nn.SmoothL1Loss()
        elif self.config.loss_function == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        
        # Setup training
        experiment_id = self.setup_training()
        
        logger.info(f"Starting training for {self.config.epochs} epochs")
        start_time = time.time()
        
        try:
            for epoch in range(self.config.epochs):
                if self.training_interrupted:
                    break
                
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Training phase
                train_loss = self._train_epoch()
                
                # Validation phase
                val_loss = None
                if 'val' in self.data_loaders:
                    val_loss = self._validate_epoch()
                
                # Update scheduler
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss else train_loss)
                else:
                    self.scheduler.step()
                
                # Calculate metrics
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                
                # Log metrics
                metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=current_lr,
                    time_per_epoch=epoch_time,
                    memory_usage=memory_usage
                )
                
                self.progress_monitor.log_metrics(metrics)
                
                # Check for best model
                current_loss = val_loss if val_loss else train_loss
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self._save_checkpoint(epoch, is_best=True)
                
                # Save periodic checkpoint
                if (epoch + 1) % self.config.save_frequency == 0:
                    self._save_checkpoint(epoch, is_best=False)
                
                # Early stopping check
                if self.early_stopping and self.early_stopping(current_loss):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Training completed
            total_time = time.time() - start_time
            
            # Save final model
            final_model_path = self._save_final_model()
            
            # Register model
            model_version = self.model_versioning.register_model(
                final_model_path,
                self.config.model_type,
                experiment_id,
                metadata={
                    'epochs_trained': self.current_epoch + 1,
                    'best_loss': self.best_loss,
                    'training_time': total_time,
                    'config': self.config.to_dict()
                }
            )
            
            # Update experiment status
            self.experiment_tracker.update_experiment_status(
                experiment_id,
                "completed",
                {
                    'final_loss': self.best_loss,
                    'epochs_completed': self.current_epoch + 1,
                    'model_version': model_version,
                    'training_time': total_time
                }
            )
            
            # Log hyperparameters
            hparams = {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'optimizer': self.config.optimizer,
                'hidden_dim': self.config.hidden_dim
            }
            
            final_metrics = {
                'best_loss': self.best_loss,
                'training_time': total_time
            }
            
            self.progress_monitor.log_hyperparameters(hparams, final_metrics)
            
            # Close monitoring
            self.progress_monitor.close()
            
            logger.success(f"Training completed in {total_time:.2f}s")
            logger.success(f"Best loss: {self.best_loss:.4f}")
            logger.success(f"Model version: {model_version}")
            
            return {
                'success': True,
                'experiment_id': experiment_id,
                'model_version': model_version,
                'best_loss': self.best_loss,
                'best_val_loss': self.best_loss,  # For hyperparameter optimization
                'best_accuracy': 0.0,  # Would need to implement accuracy calculation
                'avg_inference_time': 0.0,  # Would need to benchmark inference
                'epochs_completed': self.current_epoch + 1,
                'training_time': total_time
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            
            # Update experiment status
            self.experiment_tracker.update_experiment_status(
                experiment_id,
                "failed",
                {'error': str(e)}
            )
            
            # Close monitoring
            if self.progress_monitor:
                self.progress_monitor.close()
            
            return {
                'success': False,
                'error': str(e),
                'experiment_id': experiment_id
            }
    
    def _train_epoch(self) -> float:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        train_loader = self.data_loaders['train']
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            inputs = batch['input'].to(next(self.model.parameters()).device)
            targets = batch['target'].to(next(self.model.parameters()).device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with gradient scaling
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                scaler.step(self.optimizer)
                scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self) -> float:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        val_loader = self.data_loaders['val']
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                inputs = batch['input'].to(next(self.model.parameters()).device)
                targets = batch['target'].to(next(self.model.parameters()).device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint_dir = self.experiment_tracker.get_checkpoint_dir(self.experiment_id)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")
    
    def _save_final_model(self) -> str:
        """Save final trained model"""
        
        model_dir = self.experiment_tracker.get_model_dir(self.experiment_id)
        model_path = model_dir / "final_model.pth"
        
        # Save model
        save_model(
            self.model,
            str(model_path),
            epoch=self.current_epoch,
            optimizer_state=self.optimizer.state_dict(),
            loss=self.best_loss,
            metadata={
                'config': self.config.to_dict(),
                'experiment_id': self.experiment_id
            }
        )
        
        return str(model_path)


# Factory functions
def create_training_workflow(
    model_type: str,
    experiment_name: str,
    dataset_path: str,
    **kwargs
) -> ModelTrainer:
    """Create training workflow"""
    
    from app.services.neural_training.training_infrastructure import create_training_config
    
    config = create_training_config(
        model_type=model_type,
        experiment_name=experiment_name,
        dataset_path=dataset_path,
        **kwargs
    )
    
    return ModelTrainer(config)


def run_hyperparameter_optimization(
    base_config: TrainingConfig,
    hyperparameter_space: HyperparameterSpace,
    n_trials: int = 50,
    objective: OptimizationObjective = OptimizationObjective.MINIMIZE_LOSS
) -> Dict[str, Any]:
    """Run hyperparameter optimization"""
    
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for hyperparameter optimization")
    
    study_name = f"{base_config.experiment_name}_optimization"
    optimizer = HyperparameterOptimizer(study_name, objective)
    
    return optimizer.optimize(base_config, hyperparameter_space, n_trials)


# Export main classes and functions
__all__ = [
    'OptimizationObjective',
    'HyperparameterSpace',
    'TrainingMetrics',
    'EarlyStopping',
    'TrainingProgressMonitor',
    'HyperparameterOptimizer',
    'ModelTrainer',
    'create_training_workflow',
    'run_hyperparameter_optimization'
]