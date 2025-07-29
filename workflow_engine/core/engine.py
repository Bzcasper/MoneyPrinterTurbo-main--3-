"""
Main workflow execution engine that orchestrates all components.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum

from .workflow import Workflow, WorkflowStatus, StepStatus
from .executor import StepExecutor, ExecutionContext
from .dependency_resolver import DependencyResolver, CircularDependencyError
from ..strategies.execution_strategy import (
    ExecutionStrategy, SequentialStrategy, ParallelStrategy,
    AdaptiveStrategy, HybridStrategy, ExecutionStats
)
from ..rollback.rollback_manager import RollbackManager
from ..monitoring.progress_tracker import ProgressTracker
from ..monitoring.status_reporter import StatusReporter
from ..integrations.claude_flow_integration import ClaudeFlowIntegration


logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode enumeration."""
    NORMAL = "normal"
    DRY_RUN = "dry_run"
    VALIDATE_ONLY = "validate_only"


class WorkflowEngine:
    """
    Main workflow execution engine that coordinates all components.
    
    Features:
    - Multiple execution strategies (sequential, parallel, adaptive, hybrid)
    - Dependency resolution and validation
    - Progress tracking and status reporting
    - Rollback mechanism for failed workflows
    - Claude Flow MCP integration
    - Dry-run mode for workflow preview
    - Comprehensive error handling and logging
    """
    
    def __init__(self, 
                 max_concurrent_steps: int = 10,
                 default_strategy: str = "adaptive",
                 enable_rollback: bool = True,
                 enable_claude_flow: bool = True):
        """
        Initialize the workflow engine.
        
        Args:
            max_concurrent_steps: Maximum number of steps to execute concurrently
            default_strategy: Default execution strategy ("sequential", "parallel", "adaptive", "hybrid")
            enable_rollback: Whether to enable rollback functionality
            enable_claude_flow: Whether to enable Claude Flow MCP integration
        """
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.step_executor = StepExecutor(max_concurrent_steps)
        self.dependency_resolver = DependencyResolver()
        self.progress_tracker = ProgressTracker()
        self.status_reporter = StatusReporter()
        
        # Optional components
        self.rollback_manager = RollbackManager() if enable_rollback else None
        self.claude_flow = ClaudeFlowIntegration() if enable_claude_flow else None
        
        # Execution strategies
        self.strategies = {
            "sequential": SequentialStrategy(self.step_executor, self.dependency_resolver),
            "parallel": ParallelStrategy(self.step_executor, self.dependency_resolver, max_concurrent_steps // 2),
            "adaptive": AdaptiveStrategy(self.step_executor, self.dependency_resolver),
            "hybrid": HybridStrategy(self.step_executor, self.dependency_resolver)
        }
        self.default_strategy = default_strategy
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "workflow_started": [],
            "workflow_completed": [],
            "workflow_failed": [],
            "step_started": [],
            "step_completed": [],
            "step_failed": [],
            "rollback_started": [],
            "rollback_completed": []
        }
        
        # Runtime state
        self.active_workflows: Dict[str, Workflow] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"Workflow engine initialized with strategy '{default_strategy}'")
    
    def register_event_handler(self, event: str, handler: Callable):
        """Register an event handler."""
        if event in self.event_handlers:
            self.event_handlers[event].append(handler)
            self.logger.debug(f"Registered handler for event '{event}'")
        else:
            self.logger.warning(f"Unknown event type: {event}")
    
    def _emit_event(self, event: str, **kwargs):
        """Emit an event to all registered handlers."""
        for handler in self.event_handlers.get(event, []):
            try:
                handler(**kwargs)
            except Exception as e:
                self.logger.error(f"Error in event handler for '{event}': {e}")
    
    async def execute_workflow(self, 
                              workflow: Workflow,
                              execution_mode: ExecutionMode = ExecutionMode.NORMAL,
                              strategy: Optional[str] = None,
                              variables: Optional[Dict[str, Any]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> ExecutionStats:
        """
        Execute a workflow with the specified parameters.
        
        Args:
            workflow: The workflow to execute
            execution_mode: Execution mode (normal, dry_run, validate_only)
            strategy: Execution strategy to use (overrides default)
            variables: Initial workflow variables
            metadata: Execution metadata
            
        Returns:
            ExecutionStats with execution results
            
        Raises:
            CircularDependencyError: If circular dependencies are detected
            ValueError: If workflow validation fails
        """
        start_time = time.time()
        strategy_name = strategy or self.default_strategy
        
        self.logger.info(f"Starting execution of workflow '{workflow.name}' (mode: {execution_mode.value}, strategy: {strategy_name})")
        
        # Validate workflow
        validation_errors = self.validate_workflow(workflow)
        if validation_errors:
            error_msg = f"Workflow validation failed: {validation_errors}"
            self.logger.error(error_msg)
            if execution_mode == ExecutionMode.VALIDATE_ONLY:
                return ExecutionStats(total_steps=len(workflow.steps))
            raise ValueError(error_msg)
        
        if execution_mode == ExecutionMode.VALIDATE_ONLY:
            self.logger.info("Validation completed successfully")
            return ExecutionStats(total_steps=len(workflow.steps))
        
        # Prepare execution context
        context = ExecutionContext(
            workflow_id=workflow.id,
            step_id="",
            workflow_variables=variables or {},
            previous_results={},
            dry_run=(execution_mode == ExecutionMode.DRY_RUN),
            execution_metadata=metadata or {}
        )
        
        # Initialize tracking
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        self.active_workflows[workflow.id] = workflow
        
        # Start progress tracking
        self.progress_tracker.start_workflow(workflow.id, len(workflow.steps))
        
        # Initialize Claude Flow integration if enabled
        if self.claude_flow:
            await self.claude_flow.initialize_workflow(workflow, context)
        
        # Emit workflow started event
        self._emit_event("workflow_started", workflow=workflow, context=context)
        
        try:
            # Get execution strategy
            execution_strategy = self.strategies.get(strategy_name)
            if not execution_strategy:
                raise ValueError(f"Unknown execution strategy: {strategy_name}")
            
            # Execute workflow
            stats = await execution_strategy.execute_workflow(workflow, context)
            
            # Determine final status
            if workflow.has_failed_critical_step():
                workflow.status = WorkflowStatus.FAILED
                self._emit_event("workflow_failed", workflow=workflow, stats=stats)
                
                # Attempt rollback if enabled
                if self.rollback_manager and execution_mode == ExecutionMode.NORMAL:
                    self.logger.info("Starting rollback due to critical step failure")
                    await self._perform_rollback(workflow, context)
            else:
                workflow.status = WorkflowStatus.COMPLETED
                self._emit_event("workflow_completed", workflow=workflow, stats=stats)
            
            workflow.completed_at = datetime.now()
            
            # Finalize tracking
            self.progress_tracker.complete_workflow(workflow.id)
            
            # Store execution history
            execution_record = {
                "workflow_id": workflow.id,
                "workflow_name": workflow.name,
                "execution_mode": execution_mode.value,
                "strategy": strategy_name,
                "start_time": start_time,
                "end_time": time.time(),
                "status": workflow.status.value,
                "stats": stats.__dict__,
                "variables": context.workflow_variables
            }
            self.execution_history.append(execution_record)
            
            self.logger.info(f"Workflow '{workflow.name}' execution completed with status: {workflow.status.value}")
            return stats
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.now()
            
            self.logger.error(f"Workflow execution failed: {e}")
            self._emit_event("workflow_failed", workflow=workflow, error=str(e))
            
            # Attempt rollback on unexpected failure
            if self.rollback_manager and execution_mode == ExecutionMode.NORMAL:
                try:
                    await self._perform_rollback(workflow, context)
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed: {rollback_error}")
            
            raise
        finally:
            # Cleanup
            if workflow.id in self.active_workflows:
                del self.active_workflows[workflow.id]
            
            # Finalize Claude Flow integration
            if self.claude_flow:
                await self.claude_flow.finalize_workflow(workflow, context)
    
    async def _perform_rollback(self, workflow: Workflow, context: ExecutionContext):
        """Perform rollback for failed workflow."""
        if not self.rollback_manager:
            return
        
        self.logger.info(f"Starting rollback for workflow '{workflow.name}'")
        self._emit_event("rollback_started", workflow=workflow)
        
        try:
            success = await self.rollback_manager.rollback_workflow(workflow, context)
            
            if success:
                workflow.status = WorkflowStatus.ROLLED_BACK
                self.logger.info(f"Rollback completed successfully for workflow '{workflow.name}'")
            else:
                self.logger.error(f"Rollback failed for workflow '{workflow.name}'")
            
            self._emit_event("rollback_completed", workflow=workflow, success=success)
            
        except Exception as e:
            self.logger.error(f"Rollback error for workflow '{workflow.name}': {e}")
            self._emit_event("rollback_completed", workflow=workflow, success=False, error=str(e))
    
    def validate_workflow(self, workflow: Workflow) -> List[str]:
        """
        Comprehensive workflow validation.
        
        Args:
            workflow: The workflow to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Basic workflow validation
        workflow_errors = workflow.validate()
        errors.extend(workflow_errors)
        
        # Dependency validation
        dependency_errors = self.dependency_resolver.validate_dependencies(workflow)
        errors.extend(dependency_errors)
        
        # Strategy-specific validation
        try:
            self.dependency_resolver.resolve_execution_order(workflow)
        except CircularDependencyError as e:
            errors.append(str(e))
        
        # Action validation
        for step in workflow.steps:
            if not self.step_executor.action_registry.get(step.action):
                errors.append(f"Step '{step.id}' uses unknown action '{step.action}'")
        
        return errors
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a running workflow."""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return None
        
        progress = self.progress_tracker.get_progress(workflow_id)
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "progress": progress,
            "current_step": workflow.current_step,
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "status": step.status.value,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None
                }
                for step in workflow.steps
            ]
        }
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all currently active workflows."""
        return [
            {
                "workflow_id": workflow_id,
                "name": workflow.name,
                "status": workflow.status.value,
                "started_at": workflow.started_at.isoformat() if workflow.started_at else None
            }
            for workflow_id, workflow in self.active_workflows.items()
        ]
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history[-limit:]
    
    def register_custom_action(self, action_name: str, action_func: Callable):
        """Register a custom action for workflows."""
        self.step_executor.action_registry.register(action_name, action_func)
        self.logger.info(f"Registered custom action: {action_name}")
    
    def get_available_actions(self) -> List[str]:
        """Get list of available workflow actions."""
        return self.step_executor.action_registry.list_actions()
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return False
        
        self.logger.info(f"Cancelling workflow '{workflow.name}'")
        
        # Mark workflow as cancelled
        workflow.status = WorkflowStatus.CANCELLED
        workflow.completed_at = datetime.now()
        
        # Mark pending steps as skipped
        for step in workflow.steps:
            if step.status == StepStatus.PENDING:
                step.status = StepStatus.SKIPPED
        
        # Cleanup
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]
        
        self.progress_tracker.complete_workflow(workflow_id)
        
        return True
    
    def cleanup(self):
        """Cleanup engine resources."""
        self.logger.info("Cleaning up workflow engine")
        
        # Cancel all active workflows
        for workflow_id in list(self.active_workflows.keys()):
            asyncio.create_task(self.cancel_workflow(workflow_id))
        
        # Cleanup components
        if self.step_executor:
            self.step_executor.cleanup()
        
        if self.rollback_manager:
            self.rollback_manager.cleanup()
        
        if self.claude_flow:
            asyncio.create_task(self.claude_flow.cleanup())
        
        self.logger.info("Workflow engine cleanup completed")