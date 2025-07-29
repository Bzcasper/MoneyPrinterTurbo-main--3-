"""
Workflow Execution Engine

A comprehensive system for executing complex workflows with dependency resolution,
parallel/sequential execution strategies, rollback mechanisms, and integration
with Claude Flow MCP tools.
"""

from .core.engine import WorkflowEngine
from .core.workflow import Workflow, WorkflowStep
from .core.executor import StepExecutor
from .core.dependency_resolver import DependencyResolver
from .strategies.execution_strategy import ExecutionStrategy, ParallelStrategy, SequentialStrategy
from .rollback.rollback_manager import RollbackManager
from .monitoring.progress_tracker import ProgressTracker
from .monitoring.status_reporter import StatusReporter
from .integrations.claude_flow_integration import ClaudeFlowIntegration
from .utils.workflow_loader import WorkflowLoader
from .utils.workflow_validator import WorkflowValidator

__version__ = "1.0.0"
__author__ = "Claude Code Execution Engine"

__all__ = [
    "WorkflowEngine",
    "Workflow", 
    "WorkflowStep",
    "StepExecutor",
    "DependencyResolver",
    "ExecutionStrategy",
    "ParallelStrategy", 
    "SequentialStrategy",
    "RollbackManager",
    "ProgressTracker",
    "StatusReporter",
    "ClaudeFlowIntegration",
    "WorkflowLoader",
    "WorkflowValidator"
]