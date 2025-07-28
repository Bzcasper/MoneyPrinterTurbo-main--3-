# MoneyPrinterTurbo AI Coding Assistant Instructions

## 🎯 Project Overview
MoneyPrinterTurbo is an advanced AI-powered video generation platform featuring **swarm intelligence coordination**, **multi-vendor GPU acceleration**, and **modular video processing pipelines**. The architecture emphasizes enterprise-grade scalability with FastAPI backend, Streamlit WebUI, and sophisticated resource management.

## 🏗️ Key Architecture Components

### Core Service Layer (`app/services/`)
- **`hive_memory.py`**: SQL-based swarm coordination with SQLite backend, manages 8-agent parallel execution with thread-safe operations
- **`video_pipeline.py`**: Modular processing pipeline with configurable strategies (`sequential`, `parallel_threads`, `hybrid`, `gpu_accelerated`)
- **`gpu_manager.py`**: Multi-vendor GPU detection/allocation supporting NVIDIA NVENC, Intel QSV, AMD VCE with dynamic memory management
- **`video.py`**: Core video processing with hardware acceleration and codec optimization

### Configuration & State Management
- **`app/config/config.py`**: TOML-based configuration with auto-fallback from `config.example.toml`
- **`app/models/schema.py`**: Pydantic models for video parameters, material info, and API schemas
- **Hive Memory Database**: SQLite tables for sessions, agents, memory, events, and performance metrics

### Integration Patterns
- **`enhanced_integration.py`**: Custom CLI wrapper demonstrating proper subsystem initialization and coordination
- **WebUI**: Streamlit application in `webui/Main.py` with enhanced styling and real-time monitoring
- **FastAPI Backend**: RESTful API with auto-documentation and CORS middleware

## 🔧 Development Workflows

### Running the Application
```bash
# Start API server (port 8080)
./start_api.sh
# OR manually: cd app && python main.py

# Start WebUI (port 8501) 
./start_webui.sh
# OR manually: streamlit run webui/Main.py

# Health check
./health_check.sh
```

### Testing and Validation
```bash
# Run unit tests
python -m unittest discover -s test

# Performance validation
python performance_validator.py

# Setup and dependency check
./setup_and_test.sh
```

## 📝 Project-Specific Conventions

### Configuration Management
- Always check for `config.toml` existence; auto-copy from `config.example.toml` if missing
- Use `app.config.config` module for centralized configuration access
- Environment variables in `credentials.env` (chmod 400 for security)

### Video Processing Patterns
```python
# Standard pipeline initialization
pipeline_config = PipelineConfig(
    strategy="hybrid",
    gpu_enabled=True,
    hardware_acceleration=True,
    max_threads=8
)
pipeline = VideoProcessingPipeline(pipeline_config)

# Swarm coordination
hive_memory = HiveMemoryManager()
session_id = hive_memory.create_session(topology="star", max_agents=8)
```

### Error Handling & Resource Management
- Use context managers for database connections (`_get_cursor()`)
- Implement graceful fallbacks for GPU acceleration (hardware → software)
- Always call `close_clip()` for video resources to prevent memory leaks
- Thread-safe operations using `threading.local()` for database connections

### GPU Integration Best Practices
- Detect capabilities before allocation: `gpu_manager.get_best_gpu_for_task(task_type, memory_required)`
- Support vendor-specific codecs: `{'qsv', 'nvenc', 'vaapi', 'software'}` priority order
- Implement memory monitoring with automatic cleanup

## 🚀 Performance Considerations

### Parallel Processing
- Use `ThreadPoolExecutor` for I/O-bound operations
- Use `ProcessPoolExecutor` for CPU-intensive video processing
- Configure batch sizes based on available memory (default: 8 clips)

### Memory Management
- Monitor peak usage with `MemoryMonitor` class
- Implement LRU caching for frequently accessed resources
- Set memory limits in `PipelineConfig` (default: 2048MB)

### Database Operations
- Use connection pooling for high-concurrency scenarios
- Implement 30-day retention policy for hive memory cleanup
- Batch SQL operations for performance

## 🧪 Testing Patterns

### Service Testing
- Mock external dependencies (GPU detection, file system) for unit tests
- Use `test/resources/` for test video files and fixtures
- Follow naming convention: `test_<service_name>.py` in corresponding subdirectories

### Integration Testing
- Use `enhanced_integration.py` as reference for proper subsystem coordination
- Test GPU fallback scenarios with mock hardware unavailability
- Validate configuration loading with different TOML structures

## 🔌 Extension Points

### Adding New Video Processing Stages
1. Extend `PipelineStage` enum in `video_pipeline.py`
2. Implement stage processor class inheriting from base stage protocol
3. Register in pipeline configuration with appropriate resource requirements

### Custom GPU Vendor Support
1. Add vendor to `GPUVendor` enum in `gpu_manager.py`
2. Implement detection method following `_detect_nvidia_gpus()` pattern
3. Add codec capabilities mapping in `GPUCapability` enum

### Swarm Agent Types
1. Define agent behavior in hive memory schema
2. Implement coordination patterns in `HiveMemoryManager`
3. Add agent lifecycle management with proper cleanup

## 🐛 Common Debugging Approaches

### GPU Issues
- Check `get_gpu_manager().enumerate_gpus()` for detection
- Verify codec support with `ffmpeg -encoders | grep <codec>`
- Monitor GPU memory usage during processing

### Database Connection Issues
- Ensure SQLite file permissions and directory existence
- Check thread-local storage initialization
- Verify connection pool limits not exceeded

### Performance Problems
- Profile with `performance_validator.py` for bottleneck identification
- Monitor memory growth patterns with `MemoryMonitor`
- Check CPU/GPU utilization balance in hybrid processing mode
