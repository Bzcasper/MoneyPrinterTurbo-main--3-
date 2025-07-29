#!/usr/bin/env python3
"""
Workflow Execution Examples and Templates

Demonstrates various use cases for the workflow parameter validation system
including real-world scenarios, complex schemas, and integration patterns.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from workflow_execution import (
    WorkflowExecutor, ValidationLevel, WorkflowState
)


# =============================================================================
# Example 1: Video Processing Workflow
# =============================================================================

def create_video_processing_schema():
    """Schema for video processing workflow parameters."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "Video Processing Workflow",
        "description": "Parameters for automated video processing pipeline",
        "required": ["input_video", "output_format"],
        "properties": {
            "input_video": {
                "type": "string",
                "description": "Path or URL to input video file",
                "pattern": "^(https?://|/|\\\\|\./).*\\.(mp4|avi|mov|mkv|webm)$"
            },
            "output_format": {
                "type": "string",
                "enum": ["mp4", "webm", "avi", "mov"],
                "default": "mp4",
                "description": "Output video format"
            },
            "quality_settings": {
                "type": "object",
                "properties": {
                    "resolution": {
                        "type": "string",
                        "enum": ["480p", "720p", "1080p", "4k"],
                        "default": "1080p"
                    },
                    "bitrate": {
                        "type": "integer",
                        "minimum": 100,
                        "maximum": 50000,
                        "default": 2000,
                        "description": "Video bitrate in kbps"
                    },
                    "fps": {
                        "type": "integer",
                        "enum": [24, 30, 60],
                        "default": 30
                    }
                }
            },
            "processing_options": {
                "type": "object",
                "properties": {
                    "enable_gpu_acceleration": {
                        "type": "boolean",
                        "default": true
                    },
                    "audio_codec": {
                        "type": "string",
                        "enum": ["aac", "mp3", "opus"],
                        "default": "aac"
                    },
                    "threads": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 16,
                        "default": 4
                    },
                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "params": {"type": "object"}
                            },
                            "required": ["name"]
                        },
                        "maxItems": 10
                    }
                }
            },
            "output_settings": {
                "type": "object",
                "properties": {
                    "output_directory": {
                        "type": "string",
                        "default": "./output"
                    },
                    "filename_template": {
                        "type": "string",
                        "default": "{input_name}_{resolution}_{timestamp}",
                        "description": "Template for output filename"
                    },
                    "overwrite_existing": {
                        "type": "boolean",
                        "default": false
                    }
                }
            },
            "notification_settings": {
                "type": "object",
                "properties": {
                    "webhook_url": {
                        "type": "string",
                        "validator": "url",
                        "description": "Webhook URL for completion notifications"
                    },
                    "email": {
                        "type": "string",
                        "validator": "email",
                        "description": "Email address for notifications"
                    },
                    "notify_on_completion": {
                        "type": "boolean",
                        "default": true
                    },
                    "notify_on_error": {
                        "type": "boolean",
                        "default": true
                    }
                }
            }
        }
    }


def video_processing_workflow(context):
    """Example video processing workflow execution function."""
    params = context.parameters
    
    print(f"Processing video: {params['input_video']}")
    print(f"Output format: {params['output_format']}")
    
    # Simulate video processing steps
    steps = [
        "Loading input video",
        "Analyzing video properties",
        "Applying quality settings",
        "Processing filters",
        "Encoding output",
        "Finalizing output file"
    ]
    
    results = []
    for i, step in enumerate(steps, 1):
        print(f"Step {i}/{len(steps)}: {step}")
        # Simulate processing time
        import time
        time.sleep(0.5)
        results.append(f"{step} completed")
    
    output_file = f"{params.get('output_settings', {}).get('output_directory', './output')}/processed_video.{params['output_format']}"
    
    return {
        "status": "completed",
        "output_file": output_file,
        "processing_steps": results,
        "duration_seconds": len(steps) * 0.5,
        "metadata": {
            "resolution": params.get("quality_settings", {}).get("resolution", "1080p"),
            "bitrate": params.get("quality_settings", {}).get("bitrate", 2000),
            "audio_codec": params.get("processing_options", {}).get("audio_codec", "aac")
        }
    }


# =============================================================================
# Example 2: Data Pipeline Workflow
# =============================================================================

def create_data_pipeline_schema():
    """Schema for data processing pipeline parameters."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "Data Processing Pipeline",
        "required": ["data_source", "transformations"],
        "properties": {
            "data_source": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["database", "api", "file", "stream"]
                    },
                    "connection": {
                        "type": "object",
                        "properties": {
                            "host": {"type": "string"},
                            "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                            "database": {"type": "string"},
                            "username": {"type": "string"},
                            "password": {"type": "string"},
                            "ssl": {"type": "boolean", "default": true}
                        },
                        "required": ["host"]
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query or API endpoint"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to input file (for file type)"
                    }
                },
                "required": ["type"]
            },
            "transformations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["filter", "map", "aggregate", "join", "sort", "validate"]
                        },
                        "config": {
                            "type": "object",
                            "description": "Transformation-specific configuration"
                        },
                        "enabled": {
                            "type": "boolean",
                            "default": true
                        }
                    },
                    "required": ["type"]
                },
                "minItems": 1
            },
            "output_destination": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["database", "file", "api", "stream"]
                    },
                    "connection": {"type": "object"},
                    "format": {
                        "type": "string",
                        "enum": ["json", "csv", "parquet", "avro"],
                        "default": "json"
                    },
                    "compression": {
                        "type": "string",
                        "enum": ["none", "gzip", "snappy", "lz4"],
                        "default": "none"
                    }
                }
            },
            "performance_settings": {
                "type": "object",
                "properties": {
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100000,
                        "default": 1000
                    },
                    "parallel_workers": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 32,
                        "default": 4
                    },
                    "memory_limit_mb": {
                        "type": "integer",
                        "minimum": 128,
                        "maximum": 16384,
                        "default": 2048
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 30,
                        "maximum": 7200,
                        "default": 300
                    }
                }
            },
            "error_handling": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["fail_fast", "continue_on_error", "retry"],
                        "default": "retry"
                    },
                    "max_retries": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                        "default": 3
                    },
                    "retry_delay_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 300,
                        "default": 30
                    },
                    "dead_letter_queue": {
                        "type": "string",
                        "description": "Location for failed records"
                    }
                }
            }
        }
    }


def data_pipeline_workflow(context):
    """Example data pipeline workflow execution function."""
    params = context.parameters
    
    print(f"Starting data pipeline: {context.workflow_id}")
    print(f"Data source type: {params['data_source']['type']}")
    print(f"Transformations: {len(params['transformations'])}")
    
    # Simulate data processing
    processed_records = 0
    batch_size = params.get('performance_settings', {}).get('batch_size', 1000)
    
    for i, transformation in enumerate(params['transformations']):
        if transformation.get('enabled', True):
            print(f"Applying transformation {i+1}: {transformation['type']}")
            # Simulate processing
            import time
            time.sleep(0.3)
            processed_records += batch_size
    
    return {
        "status": "completed",
        "records_processed": processed_records,
        "transformations_applied": len([t for t in params['transformations'] if t.get('enabled', True)]),
        "output_format": params.get('output_destination', {}).get('format', 'json'),
        "performance_stats": {
            "batch_size": batch_size,
            "processing_time_seconds": len(params['transformations']) * 0.3
        }
    }


# =============================================================================
# Example 3: Machine Learning Training Workflow
# =============================================================================

def create_ml_training_schema():
    """Schema for ML model training workflow parameters."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "ML Model Training Workflow",
        "required": ["model_type", "training_data", "target_column"],
        "properties": {
            "model_type": {
                "type": "string",
                "enum": ["regression", "classification", "clustering", "neural_network"],
                "description": "Type of ML model to train"
            },
            "training_data": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "validator": "url",
                        "description": "URL to training dataset"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["csv", "json", "parquet", "arrow"],
                        "default": "csv"
                    },
                    "sample_size": {
                        "type": "integer",
                        "minimum": 100,
                        "description": "Number of samples for training"
                    },
                    "validation_split": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 0.5,
                        "default": 0.2,
                        "description": "Fraction of data for validation"
                    }
                },
                "required": ["source"]
            },
            "target_column": {
                "type": "string",
                "description": "Name of the target/label column"
            },
            "feature_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of feature column names (empty = all except target)"
            },
            "model_parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["random_forest", "gradient_boosting", "svm", "neural_network", "linear"]
                    },
                    "hyperparameters": {
                        "type": "object",
                        "description": "Algorithm-specific hyperparameters"
                    },
                    "cross_validation_folds": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 20,
                        "default": 5
                    }
                }
            },
            "training_options": {
                "type": "object",
                "properties": {
                    "max_iterations": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 10000,
                        "default": 1000
                    },
                    "early_stopping": {
                        "type": "boolean",
                        "default": true
                    },
                    "patience": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 100,
                        "default": 10
                    },
                    "use_gpu": {
                        "type": "boolean",
                        "default": false
                    },
                    "random_seed": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2147483647,
                        "default": 42
                    }
                }
            },
            "evaluation_metrics": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["accuracy", "precision", "recall", "f1", "mse", "mae", "r2", "auc"]
                },
                "minItems": 1,
                "default": ["accuracy"]
            },
            "output_settings": {
                "type": "object",
                "properties": {
                    "model_output_path": {
                        "type": "string",
                        "default": "./models/trained_model.pkl"
                    },
                    "save_predictions": {
                        "type": "boolean",
                        "default": true
                    },
                    "generate_report": {
                        "type": "boolean",
                        "default": true
                    },
                    "plot_metrics": {
                        "type": "boolean",
                        "default": true
                    }
                }
            }
        }
    }


def ml_training_workflow(context):
    """Example ML training workflow execution function."""
    params = context.parameters
    
    print(f"Training {params['model_type']} model")
    print(f"Target column: {params['target_column']}")
    
    # Simulate ML training pipeline
    steps = [
        "Loading training data",
        "Data preprocessing",
        "Feature engineering",
        "Model initialization",
        "Training model",
        "Model evaluation",
        "Saving model"
    ]
    
    metrics = {}
    for i, step in enumerate(steps, 1):
        print(f"Step {i}/{len(steps)}: {step}")
        import time
        time.sleep(0.4)
        
        if "evaluation" in step.lower():
            # Simulate evaluation metrics
            for metric in params.get('evaluation_metrics', ['accuracy']):
                if metric == 'accuracy':
                    metrics[metric] = 0.85 + (i * 0.02)  # Simulate improving accuracy
                elif metric in ['precision', 'recall', 'f1']:
                    metrics[metric] = 0.80 + (i * 0.015)
                elif metric in ['mse', 'mae']:
                    metrics[metric] = 0.15 - (i * 0.01)
                elif metric == 'r2':
                    metrics[metric] = 0.75 + (i * 0.02)
    
    return {
        "status": "completed",
        "model_type": params['model_type'],
        "training_samples": params['training_data'].get('sample_size', 'unknown'),
        "validation_split": params['training_data'].get('validation_split', 0.2),
        "evaluation_metrics": metrics,
        "model_output_path": params.get('output_settings', {}).get('model_output_path'),
        "training_duration_seconds": len(steps) * 0.4,
        "algorithm_used": params.get('model_parameters', {}).get('algorithm', 'default')
    }


# =============================================================================
# Usage Examples and Integration Patterns
# =============================================================================

def demonstrate_video_processing():
    """Demonstrate video processing workflow."""
    print("\n=== Video Processing Workflow Example ===")
    
    executor = WorkflowExecutor(
        validation_level=ValidationLevel.STRICT,
        enable_encryption=False
    )
    
    # Example parameters from different sources
    json_params = {
        "input_video": "/path/to/input.mp4",
        "output_format": "webm",
        "quality_settings": {
            "resolution": "720p",
            "bitrate": 1500,
            "fps": 30
        },
        "processing_options": {
            "enable_gpu_acceleration": True,
            "audio_codec": "opus",
            "threads": 8
        }
    }
    
    cli_params = {
        "output_settings": {
            "output_directory": "./processed_videos",
            "overwrite_existing": True
        }
    }
    
    env_params = {
        "notification_settings": {
            "notify_on_completion": True,
            "webhook_url": "https://webhook.example.com/video-complete"
        }
    }
    
    try:
        schema = create_video_processing_schema()
        workflow_id = "video_processing_001"
        
        context = executor.create_workflow(
            workflow_id=workflow_id,
            schema=schema,
            parameter_sources=[json_params, cli_params, env_params]
        )
        
        print(f"Created workflow: {workflow_id}")
        print(f"Parameters validated: {len(context.parameters)} parameters")
        
        result = executor.execute_workflow(workflow_id, video_processing_workflow)
        
        print(f"\nWorkflow completed successfully!")
        print(f"Output file: {result['output_file']}")
        print(f"Processing duration: {result['duration_seconds']}s")
        print(f"Resolution: {result['metadata']['resolution']}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if workflow_id in executor.workflows:
            executor.cleanup_workflow(workflow_id)


def demonstrate_data_pipeline():
    """Demonstrate data pipeline workflow."""
    print("\n=== Data Pipeline Workflow Example ===")
    
    executor = WorkflowExecutor(
        validation_level=ValidationLevel.LENIENT,
        enable_encryption=False
    )
    
    params = {
        "data_source": {
            "type": "database",
            "connection": {
                "host": "db.example.com",
                "port": 5432,
                "database": "analytics",
                "username": "pipeline_user",
                "ssl": True
            },
            "query": "SELECT * FROM user_events WHERE date >= '2023-01-01'"
        },
        "transformations": [
            {"type": "filter", "config": {"column": "status", "value": "active"}},
            {"type": "map", "config": {"function": "normalize_email"}},
            {"type": "aggregate", "config": {"group_by": "user_id", "metrics": ["count", "sum"]}}
        ],
        "output_destination": {
            "type": "file",
            "format": "parquet",
            "compression": "snappy"
        },
        "performance_settings": {
            "batch_size": 5000,
            "parallel_workers": 8,
            "memory_limit_mb": 4096
        }
    }
    
    try:
        schema = create_data_pipeline_schema()
        workflow_id = "data_pipeline_001"
        
        context = executor.create_workflow(
            workflow_id=workflow_id,
            schema=schema,
            parameter_sources=[params]
        )
        
        result = executor.execute_workflow(workflow_id, data_pipeline_workflow)
        
        print(f"Data pipeline completed!")
        print(f"Records processed: {result['records_processed']:,}")
        print(f"Transformations applied: {result['transformations_applied']}")
        print(f"Output format: {result['output_format']}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if workflow_id in executor.workflows:
            executor.cleanup_workflow(workflow_id)


def demonstrate_ml_training():
    """Demonstrate ML training workflow."""
    print("\n=== ML Training Workflow Example ===")
    
    executor = WorkflowExecutor(
        validation_level=ValidationLevel.STRICT,
        enable_encryption=False
    )
    
    params = {
        "model_type": "classification",
        "training_data": {
            "source": "https://example.com/dataset.csv",
            "format": "csv",
            "sample_size": 10000,
            "validation_split": 0.3
        },
        "target_column": "label",
        "feature_columns": ["feature1", "feature2", "feature3", "feature4"],
        "model_parameters": {
            "algorithm": "random_forest",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5
            },
            "cross_validation_folds": 5
        },
        "training_options": {
            "max_iterations": 500,
            "early_stopping": True,
            "patience": 20,
            "use_gpu": False,
            "random_seed": 42
        },
        "evaluation_metrics": ["accuracy", "precision", "recall", "f1"],
        "output_settings": {
            "model_output_path": "./models/classifier_v1.pkl",
            "save_predictions": True,
            "generate_report": True,
            "plot_metrics": True
        }
    }
    
    try:
        schema = create_ml_training_schema()
        workflow_id = "ml_training_001"
        
        context = executor.create_workflow(
            workflow_id=workflow_id,
            schema=schema,
            parameter_sources=[params]
        )
        
        result = executor.execute_workflow(workflow_id, ml_training_workflow)
        
        print(f"ML training completed!")
        print(f"Model type: {result['model_type']}")
        print(f"Training samples: {result['training_samples']:,}")
        print(f"Validation split: {result['validation_split']}")
        print(f"Algorithm: {result['algorithm_used']}")
        print(f"Training duration: {result['training_duration_seconds']}s")
        print("\nEvaluation Metrics:")
        for metric, value in result['evaluation_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if workflow_id in executor.workflows:
            executor.cleanup_workflow(workflow_id)


def demonstrate_complex_parameter_loading():
    """Demonstrate complex parameter loading from multiple sources."""
    print("\n=== Complex Parameter Loading Example ===")
    
    # Create temporary configuration files
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    
    # JSON config file
    json_config = {
        "global_settings": {
            "debug": True,
            "log_level": "INFO",
            "max_workers": 8
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "testdb"
        }
    }
    
    json_file = temp_dir / "config.json"
    with open(json_file, 'w') as f:
        json.dump(json_config, f, indent=2)
    
    # YAML config file
    yaml_config = """
    processing_options:
      batch_size: 1000
      timeout: 300
      retry_count: 3
    
    notification:
      enabled: true
      channels:
        - email
        - webhook
    """
    
    yaml_file = temp_dir / "config.yaml"
    with open(yaml_file, 'w') as f:
        f.write(yaml_config)
    
    # Set environment variables
    os.environ['WORKFLOW_API_KEY'] = 'secret_key_123'
    os.environ['WORKFLOW_ENVIRONMENT'] = 'production'
    os.environ['WORKFLOW_DATABASE_PASSWORD'] = 'secure_password'
    
    try:
        executor = WorkflowExecutor(enable_encryption=False)
        
        # Load from multiple sources
        json_params = executor.parameter_loader.load_from_json(json_file)
        yaml_params = executor.parameter_loader.load_from_yaml(yaml_file)
        env_params = executor.parameter_loader.load_from_env('WORKFLOW_')
        cli_params = executor.parameter_loader.load_from_cli([
            '--param', 'override_setting=cli_value',
            '--param', 'priority=high'
        ])
        
        # Merge all sources (later sources override earlier ones)
        merged_params = executor.parameter_loader.merge_parameters(
            json_params, yaml_params, env_params, cli_params
        )
        
        print("Parameter sources loaded:")
        print(f"- JSON config: {len(json_params)} parameters")
        print(f"- YAML config: {len(yaml_params)} parameters")
        print(f"- Environment variables: {len(env_params)} parameters")
        print(f"- CLI arguments: {len(cli_params)} parameters")
        print(f"- Merged total: {len(merged_params)} parameters")
        
        print("\nFinal merged parameters:")
        for key, value in merged_params.items():
            if 'password' in key.lower() or 'key' in key.lower():
                print(f"  {key}: [REDACTED]")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nParameter sources: {executor.parameter_loader.loaded_sources}")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        # Clean up environment variables
        for key in ['WORKFLOW_API_KEY', 'WORKFLOW_ENVIRONMENT', 'WORKFLOW_DATABASE_PASSWORD']:
            if key in os.environ:
                del os.environ[key]


def main():
    """Run all workflow examples."""
    print("Workflow Execution Examples")
    print("============================")
    
    try:
        demonstrate_video_processing()
        demonstrate_data_pipeline()
        demonstrate_ml_training()
        demonstrate_complex_parameter_loading()
        
        print("\n=== All Examples Completed Successfully! ===")
        
    except Exception as e:
        print(f"\nExample execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
