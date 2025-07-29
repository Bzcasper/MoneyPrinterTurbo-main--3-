# Workflow Execution Engine

A comprehensive, production-ready workflow execution engine with advanced capabilities including dependency resolution, parallel execution strategies, rollback mechanisms, progress tracking, and Claude Flow MCP integration.

## Features

### 🚀 Core Capabilities
- **Multiple Execution Strategies**: Sequential, Parallel, Adaptive, and Hybrid execution modes
- **Dependency Resolution**: Automatic dependency graph analysis with circular dependency detection
- **Rollback Mechanisms**: Automatic checkpoint creation and sophisticated rollback for failed workflows
- **Progress Tracking**: Real-time progress monitoring with completion estimates
- **Dry-Run Mode**: Preview workflow execution without making actual changes

### 🔧 Advanced Features
- **Claude Flow Integration**: Enhanced coordination with Claude Flow MCP tools
- **Custom Actions**: Extensible action system with built-in actions for files, shell, HTTP, and variables
- **Workflow Templates**: Built-in templates and custom template support
- **Comprehensive Validation**: Multi-level validation with security checks
- **Performance Analytics**: Execution metrics and bottleneck identification

### 📊 Monitoring & Reporting
- **Status Reporting**: Multi-level status reporting with multiple output formats
- **Performance Metrics**: Detailed execution statistics and optimization suggestions
- **Event System**: Comprehensive event handling for integration with external systems

## Quick Start

### Basic Usage

```python
import asyncio
from workflow_engine import WorkflowEngine, Workflow, WorkflowStep

# Create a simple workflow
workflow = Workflow(
    id="hello_world",
    name="Hello World Workflow",
    description="A simple example workflow",
    steps=[
        WorkflowStep(
            id="greet",
            name="Say Hello",
            action="variable.set",
            parameters={"name": "message", "value": "Hello, World!"}
        ),
        WorkflowStep(
            id="display",
            name="Display Message",
            action="shell.execute",
            parameters={"command": "echo {message}"},
            dependencies=["greet"]
        )
    ]
)

# Execute the workflow
async def run_workflow():
    engine = WorkflowEngine()
    stats = await engine.execute_workflow(workflow)
    print(f"Completed {stats.completed_steps}/{stats.total_steps} steps")
    engine.cleanup()

asyncio.run(run_workflow())
```

### Loading from Templates

```python
from workflow_engine.utils import WorkflowLoader

loader = WorkflowLoader()

# Load a built-in template
workflow = loader.load_from_template(
    template_name="deployment_workflow",
    variables={
        "app_name": "myapp",
        "environment": "production"
    }
)

# Load from file
workflow = loader.load_from_file("my_workflow.json")
```

### Workflow Validation

```python
from workflow_engine.utils import WorkflowValidator

validator = WorkflowValidator()
result = validator.validate(workflow, strict=True)

if not result.is_valid:
    print("Validation errors:")
    for error in result.errors:
        print(f"  - {error}")
```

## Architecture

### Core Components

```
workflow_engine/
├── core/                    # Core execution components
│   ├── engine.py           # Main workflow engine
│   ├── workflow.py         # Workflow and step data models
│   ├── executor.py         # Step execution with retry logic
│   └── dependency_resolver.py  # Dependency analysis
├── strategies/             # Execution strategies
│   └── execution_strategy.py  # Sequential, Parallel, Adaptive strategies
├── rollback/              # Rollback management
│   └── rollback_manager.py   # Checkpoint and rollback system
├── monitoring/            # Progress tracking and reporting
│   ├── progress_tracker.py   # Real-time progress tracking
│   └── status_reporter.py    # Multi-format status reporting
├── integrations/          # External integrations
│   └── claude_flow_integration.py  # Claude Flow MCP integration
└── utils/                 # Utilities
    ├── workflow_loader.py     # Template and file loading
    └── workflow_validator.py  # Comprehensive validation
```

## Execution Strategies

### Sequential Strategy
Executes steps one by one in dependency order. Provides maximum control and predictability.

```python
engine = WorkflowEngine(default_strategy="sequential")
```

### Parallel Strategy
Executes independent steps concurrently for maximum throughput.

```python
engine = WorkflowEngine(default_strategy="parallel")
```

### Adaptive Strategy
Dynamically chooses between sequential and parallel based on workflow characteristics.

```python
engine = WorkflowEngine(default_strategy="adaptive")  # Default
```

### Hybrid Strategy
Combines sequential and parallel execution within the same workflow.

```python
engine = WorkflowEngine(default_strategy="hybrid")
```

## Available Actions

### File Operations
- `file.read` - Read file contents
- `file.write` - Write content to file
- `file.delete` - Delete file
- `file.copy` - Copy file from source to destination

### Shell Operations
- `shell.execute` - Execute shell commands with timeout and retry

### HTTP Operations
- `http.get` - HTTP GET request
- `http.post` - HTTP POST request

### Variable Operations
- `variable.set` - Set workflow variable
- `variable.get` - Get workflow variable value

### Utility Operations
- `wait` - Wait for specified time
- `validate.condition` - Validate conditions

### Claude Flow Integration
- `claude_flow.swarm_init` - Initialize coordination swarm
- `claude_flow.agent_spawn` - Spawn specialized agents
- `claude_flow.task_orchestrate` - Orchestrate complex tasks
- `claude_flow.memory_store` - Store data in shared memory

## Workflow Definition Format

### JSON Format
```json
{
  "id": "example_workflow",
  "name": "Example Workflow",
  "description": "A comprehensive example",
  "metadata": {
    "author": "Developer",
    "version": "1.0"
  },
  "steps": [
    {
      "id": "step1",
      "name": "First Step",
      "action": "file.write",
      "parameters": {
        "path": "/tmp/example.txt",
        "content": "Hello World"
      },
      "timeout": 30,
      "max_retries": 3,
      "critical": true
    },
    {
      "id": "step2",
      "name": "Second Step",
      "action": "shell.execute",
      "parameters": {
        "command": "cat /tmp/example.txt"
      },
      "dependencies": ["step1"],
      "rollback_action": "file.delete",
      "rollback_parameters": {
        "path": "/tmp/example.txt"
      }
    }
  ]
}
```

### YAML Format
```yaml
id: example_workflow
name: Example Workflow
description: A comprehensive example
steps:
  - id: step1
    name: First Step
    action: file.write
    parameters:
      path: /tmp/example.txt
      content: Hello World
    timeout: 30
    max_retries: 3
  - id: step2
    name: Second Step
    action: shell.execute
    parameters:
      command: cat /tmp/example.txt
    dependencies: [step1]
```

## Advanced Features

### Parallel Execution Groups
```python
WorkflowStep(
    id="parallel_task1",
    name="Parallel Task 1",
    action="some_action",
    parallel_group="group1"  # Execute with other steps in same group
)
```

### Conditional Execution
```python
WorkflowStep(
    id="conditional_step",
    name="Conditional Step",
    action="some_action",
    conditions={"environment": "production"}
)
```

### Rollback Actions
```python
WorkflowStep(
    id="file_operation",
    name="Write Important File",
    action="file.write",
    parameters={"path": "/important/file.txt", "content": "data"},
    rollback_action="file.delete",
    rollback_parameters={"path": "/important/file.txt"}
)
```

### Custom Actions
```python
def custom_action(parameters, context):
    # Custom logic here
    return StepResult(success=True, output="Custom result")

engine.register_custom_action("custom.action", custom_action)
```

## Claude Flow Integration

The workflow engine integrates seamlessly with Claude Flow MCP tools for enhanced coordination:

### Automatic Swarm Initialization
```python
engine = WorkflowEngine(enable_claude_flow=True)
# Automatically initializes swarms for complex workflows
```

### Intelligent Agent Spawning
- Analyzes workflow characteristics
- Spawns specialized agents based on step types
- Provides coordinated execution across agents

### Memory-Based Coordination
- Stores execution context and results
- Enables cross-step communication
- Provides persistent state across sessions

### Neural Pattern Learning
- Learns from execution patterns
- Optimizes future workflow execution
- Provides performance insights

## Error Handling and Recovery

### Automatic Rollback
```python
engine = WorkflowEngine(enable_rollback=True)
# Automatically creates checkpoints and rolls back on failure
```

### Retry Logic
```python
WorkflowStep(
    id="flaky_step",
    name="Potentially Flaky Step",
    action="http.get",
    parameters={"url": "https://api.example.com/data"},
    max_retries=5,  # Retry up to 5 times
    timeout=30      # 30 second timeout
)
```

### Critical vs Non-Critical Steps
```python
WorkflowStep(
    id="optional_cleanup",
    name="Optional Cleanup",
    action="file.delete",
    parameters={"path": "/tmp/temp_file"},
    critical=False  # Workflow continues if this fails
)
```

## Monitoring and Analytics

### Real-Time Progress Tracking
```python
def progress_callback(workflow_id, progress_data):
    print(f"Workflow {workflow_id}: {progress_data['completion_percentage']:.1f}% complete")

engine.progress_tracker.add_observer(progress_callback)
```

### Performance Metrics
```python
stats = await engine.execute_workflow(workflow)
print(f"Execution time: {stats.total_execution_time:.2f}s")
print(f"Parallel efficiency: {stats.parallel_efficiency:.2f}")
```

### Status Reporting
```python
from workflow_engine.monitoring import StatusReporter, ReportLevel

reporter = StatusReporter(log_file="workflow.log")
reporter.info("workflow_123", "Step completed successfully")
reporter.export_reports("report.json", format="json")
```

## Built-in Templates

### Available Templates
- `simple_file_workflow` - Basic file operations
- `api_integration_workflow` - API data fetching and processing
- `deployment_workflow` - Application deployment
- `data_processing_workflow` - Data transformation pipeline

### Template Variables
```python
workflow = loader.load_from_template(
    template_name="deployment_workflow",
    variables={
        "app_name": "myapp",
        "environment": "staging",
        "source_path": "/builds/myapp-v1.2.3"
    }
)
```

## Validation

### Comprehensive Validation
```python
validator = WorkflowValidator()
result = validator.validate(workflow, strict=True)

# Check validation results
if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error}")

for warning in result.warnings:
    print(f"Warning: {warning}")
```

### Security Validation
- Detects potentially dangerous shell commands
- Identifies hardcoded credentials
- Validates file operation permissions
- Checks for insecure HTTP usage

### Performance Optimization
```python
suggestions = validator.suggest_improvements(workflow)
for suggestion in suggestions:
    print(f"Suggestion: {suggestion}")
```

## Configuration

### Engine Configuration
```python
engine = WorkflowEngine(
    max_concurrent_steps=10,     # Maximum parallel step execution
    default_strategy="adaptive", # Default execution strategy
    enable_rollback=True,        # Enable automatic rollback
    enable_claude_flow=True      # Enable Claude Flow integration
)
```

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `usage_example.py` - Complete demonstration of all features
- `basic_examples.py` - Simple workflow examples
- `advanced_examples.py` - Complex workflow patterns

## API Reference

### WorkflowEngine
Main orchestration engine with comprehensive workflow execution capabilities.

**Methods:**
- `execute_workflow(workflow, execution_mode, strategy, variables, metadata)` - Execute workflow
- `validate_workflow(workflow)` - Validate workflow structure
- `cancel_workflow(workflow_id)` - Cancel running workflow
- `get_workflow_status(workflow_id)` - Get workflow status
- `register_custom_action(name, function)` - Register custom action

### Workflow
Data model representing a complete workflow definition.

**Properties:**
- `id` - Unique workflow identifier
- `name` - Human-readable workflow name
- `description` - Workflow description
- `steps` - List of workflow steps
- `metadata` - Additional workflow metadata

### WorkflowStep
Individual step within a workflow.

**Properties:**
- `id` - Unique step identifier
- `name` - Human-readable step name
- `action` - Action to execute
- `parameters` - Action parameters
- `dependencies` - List of dependency step IDs
- `timeout` - Execution timeout in seconds
- `max_retries` - Maximum retry attempts
- `critical` - Whether step failure should fail workflow
- `parallel_group` - Parallel execution group
- `rollback_action` - Action for rollback
- `rollback_parameters` - Rollback action parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For support, issues, or feature requests, please open an issue on the project repository.