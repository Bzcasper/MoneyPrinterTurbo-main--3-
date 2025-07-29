"""
Execution strategy implementations for different workflow execution patterns.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..core.workflow import Workflow, WorkflowStep, StepResult, StepStatus
from ..core.executor import StepExecutor, ExecutionContext
from ..core.dependency_resolver import DependencyResolver


logger = logging.getLogger(__name__)


@dataclass
class ExecutionStats:
    """Statistics for workflow execution."""
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    total_execution_time: float = 0.0
    parallel_efficiency: float = 0.0


class ExecutionStrategy(ABC):
    """
    Abstract base class for workflow execution strategies.
    Defines the interface for different execution approaches.
    """
    
    def __init__(self, step_executor: StepExecutor, dependency_resolver: DependencyResolver):
        self.step_executor = step_executor
        self.dependency_resolver = dependency_resolver
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def execute_workflow(self, 
                              workflow: Workflow, 
                              context: ExecutionContext) -> ExecutionStats:
        """
        Execute the workflow according to this strategy.
        
        Args:
            workflow: The workflow to execute
            context: Execution context with variables and metadata
            
        Returns:
            ExecutionStats with execution statistics
        """
        pass
    
    def _update_execution_stats(self, workflow: Workflow, start_time: float) -> ExecutionStats:
        """Calculate execution statistics."""
        import time
        total_time = time.time() - start_time
        
        completed = sum(1 for step in workflow.steps if step.status == StepStatus.COMPLETED)
        failed = sum(1 for step in workflow.steps if step.status == StepStatus.FAILED)
        skipped = sum(1 for step in workflow.steps if step.status == StepStatus.SKIPPED)
        
        return ExecutionStats(
            total_steps=len(workflow.steps),
            completed_steps=completed,
            failed_steps=failed,
            skipped_steps=skipped,
            total_execution_time=total_time
        )


class SequentialStrategy(ExecutionStrategy):
    """
    Sequential execution strategy - executes steps one by one in dependency order.
    Provides maximum control and predictability but no parallelism benefits.
    """
    
    async def execute_workflow(self, 
                              workflow: Workflow, 
                              context: ExecutionContext) -> ExecutionStats:
        """Execute workflow steps sequentially in dependency order."""
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting sequential execution of workflow '{workflow.name}'")
        
        # Get execution order
        execution_phases = self.dependency_resolver.resolve_execution_order(workflow)
        completed_steps: Set[str] = set()
        
        for phase_num, phase_steps in enumerate(execution_phases):
            self.logger.info(f"Executing phase {phase_num + 1}: {phase_steps}")
            
            # In sequential strategy, execute even parallel phases one by one
            for step_id in phase_steps:
                step = workflow.get_step(step_id)
                if not step:
                    self.logger.error(f"Step not found: {step_id}")
                    continue
                
                if step.status != StepStatus.PENDING:
                    continue
                
                # Check if dependencies are satisfied
                if not step.can_execute(completed_steps):
                    self.logger.warning(f"Skipping step '{step_id}' - dependencies not satisfied")
                    step.status = StepStatus.SKIPPED
                    continue
                
                # Execute step
                result = await self.step_executor.execute_step(step, context)
                
                if result.success:
                    completed_steps.add(step_id)
                    workflow.current_step = step_id
                else:
                    self.logger.error(f"Step '{step_id}' failed: {result.error}")
                    
                    # Handle critical step failure
                    if step.critical:
                        self.logger.error(f"Critical step '{step_id}' failed, stopping execution")
                        break
                
                # Update workflow variables with step results
                if result.output is not None:
                    context.workflow_variables[f"{step_id}_result"] = result.output
        
        stats = self._update_execution_stats(workflow, start_time)
        self.logger.info(f"Sequential execution completed: {stats.completed_steps}/{stats.total_steps} steps completed")
        
        return stats


class ParallelStrategy(ExecutionStrategy):
    """
    Parallel execution strategy - executes independent steps concurrently.
    Maximizes throughput by running compatible steps simultaneously.
    """
    
    def __init__(self, 
                 step_executor: StepExecutor, 
                 dependency_resolver: DependencyResolver,
                 max_concurrent_phases: int = 5):
        super().__init__(step_executor, dependency_resolver)
        self.max_concurrent_phases = max_concurrent_phases
    
    async def execute_workflow(self, 
                              workflow: Workflow, 
                              context: ExecutionContext) -> ExecutionStats:
        """Execute workflow steps with maximum parallelism."""
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting parallel execution of workflow '{workflow.name}'")
        
        # Get execution order
        execution_phases = self.dependency_resolver.resolve_execution_order(workflow)
        completed_steps: Set[str] = set()
        
        for phase_num, phase_steps in enumerate(execution_phases):
            self.logger.info(f"Executing phase {phase_num + 1} with {len(phase_steps)} parallel steps: {phase_steps}")
            
            # Execute all steps in this phase in parallel
            phase_results = await self._execute_phase_parallel(
                phase_steps, workflow, context, completed_steps
            )
            
            # Process results
            phase_successful = True
            for step_id, result in phase_results.items():
                if result.success:
                    completed_steps.add(step_id)
                    workflow.current_step = step_id
                    
                    # Update workflow variables
                    if result.output is not None:
                        context.workflow_variables[f"{step_id}_result"] = result.output
                else:
                    step = workflow.get_step(step_id)
                    self.logger.error(f"Step '{step_id}' failed: {result.error}")
                    
                    # Check if critical step failed
                    if step and step.critical:
                        self.logger.error(f"Critical step '{step_id}' failed")
                        phase_successful = False
            
            # Stop execution if critical step failed
            if not phase_successful:
                # Mark remaining steps as skipped
                for remaining_phase in execution_phases[phase_num + 1:]:
                    for step_id in remaining_phase:
                        step = workflow.get_step(step_id)
                        if step:
                            step.status = StepStatus.SKIPPED
                break
        
        stats = self._update_execution_stats(workflow, start_time)
        self.logger.info(f"Parallel execution completed: {stats.completed_steps}/{stats.total_steps} steps completed")
        
        return stats
    
    async def _execute_phase_parallel(self, 
                                     phase_steps: List[str], 
                                     workflow: Workflow,
                                     context: ExecutionContext, 
                                     completed_steps: Set[str]) -> Dict[str, StepResult]:
        """Execute a phase of steps in parallel."""
        tasks = []
        step_contexts = {}
        
        for step_id in phase_steps:
            step = workflow.get_step(step_id)
            if not step:
                self.logger.error(f"Step not found: {step_id}")
                continue
            
            if step.status != StepStatus.PENDING:
                continue
            
            # Check dependencies
            if not step.can_execute(completed_steps):
                self.logger.warning(f"Skipping step '{step_id}' - dependencies not satisfied")
                step.status = StepStatus.SKIPPED
                continue
            
            # Create separate context for each step to avoid race conditions
            step_context = ExecutionContext(
                workflow_id=context.workflow_id,
                step_id=step_id,
                workflow_variables=context.workflow_variables.copy(),
                previous_results=context.previous_results.copy(),
                dry_run=context.dry_run,
                execution_metadata=context.execution_metadata
            )
            step_contexts[step_id] = step_context
            
            # Create execution task
            task = asyncio.create_task(
                self.step_executor.execute_step(step, step_context),
                name=f"step_{step_id}"
            )
            tasks.append((step_id, task))
        
        # Wait for all tasks to complete
        results = {}
        if tasks:
            task_results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            for (step_id, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Task for step '{step_id}' raised exception: {result}")
                    results[step_id] = StepResult(
                        success=False,
                        error=str(result),
                        metadata={"exception_type": type(result).__name__}
                    )
                else:
                    results[step_id] = result
                    
                    # Merge step context variables back to main context
                    step_context = step_contexts[step_id]
                    context.workflow_variables.update(step_context.workflow_variables)
        
        return results


class AdaptiveStrategy(ExecutionStrategy):
    """
    Adaptive execution strategy - dynamically chooses between sequential and parallel
    execution based on workflow characteristics and system resources.
    """
    
    def __init__(self, 
                 step_executor: StepExecutor, 
                 dependency_resolver: DependencyResolver,
                 parallel_threshold: int = 3,
                 max_concurrent_phases: int = 3):
        super().__init__(step_executor, dependency_resolver)
        self.parallel_threshold = parallel_threshold
        self.max_concurrent_phases = max_concurrent_phases
        self.sequential_strategy = SequentialStrategy(step_executor, dependency_resolver)
        self.parallel_strategy = ParallelStrategy(step_executor, dependency_resolver, max_concurrent_phases)
    
    async def execute_workflow(self, 
                              workflow: Workflow, 
                              context: ExecutionContext) -> ExecutionStats:
        """Execute workflow using adaptive strategy selection."""
        # Analyze workflow characteristics
        execution_phases = self.dependency_resolver.resolve_execution_order(workflow)
        
        # Calculate parallelization potential
        total_parallel_steps = sum(len(phase) for phase in execution_phases if len(phase) > 1)
        parallel_ratio = total_parallel_steps / len(workflow.steps) if workflow.steps else 0
        
        # Decide strategy based on analysis
        if (parallel_ratio > 0.3 and  # At least 30% of steps can be parallelized
            len(workflow.steps) >= self.parallel_threshold and  # Enough steps to benefit
            not context.dry_run):  # Not in dry-run mode
            
            self.logger.info(f"Using parallel strategy (parallel ratio: {parallel_ratio:.2f})")
            return await self.parallel_strategy.execute_workflow(workflow, context)
        else:
            self.logger.info(f"Using sequential strategy (parallel ratio: {parallel_ratio:.2f})")
            return await self.sequential_strategy.execute_workflow(workflow, context)


class HybridStrategy(ExecutionStrategy):
    """
    Hybrid execution strategy - combines sequential and parallel execution
    within the same workflow based on step characteristics.
    """
    
    def __init__(self, 
                 step_executor: StepExecutor, 
                 dependency_resolver: DependencyResolver,
                 parallel_group_threshold: int = 2):
        super().__init__(step_executor, dependency_resolver)
        self.parallel_group_threshold = parallel_group_threshold
    
    async def execute_workflow(self, 
                              workflow: Workflow, 
                              context: ExecutionContext) -> ExecutionStats:
        """Execute workflow using hybrid sequential/parallel approach."""
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting hybrid execution of workflow '{workflow.name}'")
        
        # Get execution order
        execution_phases = self.dependency_resolver.resolve_execution_order(workflow)
        completed_steps: Set[str] = set()
        
        for phase_num, phase_steps in enumerate(execution_phases):
            phase_size = len(phase_steps)
            
            if phase_size >= self.parallel_group_threshold:
                # Execute large phases in parallel
                self.logger.info(f"Executing phase {phase_num + 1} in parallel: {phase_steps}")
                parallel_strategy = ParallelStrategy(self.step_executor, self.dependency_resolver)
                phase_results = await parallel_strategy._execute_phase_parallel(
                    phase_steps, workflow, context, completed_steps
                )
                
                # Process parallel results
                for step_id, result in phase_results.items():
                    if result.success:
                        completed_steps.add(step_id)
                        if result.output is not None:
                            context.workflow_variables[f"{step_id}_result"] = result.output
            else:
                # Execute small phases sequentially
                self.logger.info(f"Executing phase {phase_num + 1} sequentially: {phase_steps}")
                for step_id in phase_steps:
                    step = workflow.get_step(step_id)
                    if not step or step.status != StepStatus.PENDING:
                        continue
                    
                    if not step.can_execute(completed_steps):
                        step.status = StepStatus.SKIPPED
                        continue
                    
                    result = await self.step_executor.execute_step(step, context)
                    
                    if result.success:
                        completed_steps.add(step_id)
                        if result.output is not None:
                            context.workflow_variables[f"{step_id}_result"] = result.output
                    elif step.critical:
                        self.logger.error(f"Critical step '{step_id}' failed, stopping execution")
                        break
        
        stats = self._update_execution_stats(workflow, start_time)
        self.logger.info(f"Hybrid execution completed: {stats.completed_steps}/{stats.total_steps} steps completed")
        
        return stats