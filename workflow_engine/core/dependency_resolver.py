"""
Dependency resolution system for workflow steps.
Handles complex dependency graphs, circular dependency detection,
and optimal execution ordering.
"""

from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, deque
import logging
from .workflow import Workflow, WorkflowStep, StepStatus

logger = logging.getLogger(__name__)


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""
    pass


class DependencyResolver:
    """
    Resolves dependencies between workflow steps and determines optimal execution order.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def resolve_execution_order(self, workflow: Workflow) -> List[List[str]]:
        """
        Resolve execution order for workflow steps.
        Returns a list of lists, where each inner list contains step IDs
        that can be executed in parallel.
        
        Args:
            workflow: The workflow to resolve
            
        Returns:
            List of execution phases, each phase contains steps that can run in parallel
            
        Raises:
            CircularDependencyError: If circular dependencies are detected
        """
        # Build dependency graph
        graph = self._build_dependency_graph(workflow)
        
        # Check for circular dependencies
        if self._has_circular_dependencies(graph):
            cycles = self._find_cycles(graph)
            raise CircularDependencyError(f"Circular dependencies detected: {cycles}")
        
        # Perform topological sort to get execution order
        execution_phases = self._topological_sort_parallel(graph, workflow.steps)
        
        self.logger.info(f"Resolved execution order for workflow '{workflow.name}': {len(execution_phases)} phases")
        return execution_phases
    
    def _build_dependency_graph(self, workflow: Workflow) -> Dict[str, Set[str]]:
        """Build a dependency graph from workflow steps."""
        graph = defaultdict(set)
        
        # Initialize all step nodes
        for step in workflow.steps:
            graph[step.id] = set()
        
        # Add dependency edges
        for step in workflow.steps:
            for dependency in step.dependencies:
                if dependency in graph:
                    graph[step.id].add(dependency)
                else:
                    self.logger.warning(f"Step '{step.id}' depends on non-existent step '{dependency}'")
        
        return dict(graph)
    
    def _has_circular_dependencies(self, graph: Dict[str, Set[str]]) -> bool:
        """Check if the dependency graph has circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Find all cycles in the dependency graph."""
        visited = set()
        rec_stack = set()
        cycles = []
        current_path = []
        
        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            current_path.append(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = current_path.index(neighbor)
                    cycle = current_path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            current_path.pop()
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def _topological_sort_parallel(self, graph: Dict[str, Set[str]], 
                                   steps: List[WorkflowStep]) -> List[List[str]]:
        """
        Perform topological sort optimized for parallel execution.
        Groups steps that can be executed in parallel into the same phase.
        """
        # Calculate in-degrees
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for dependency in graph[node]:
                in_degree[node] += 1
        
        # Group steps by parallel execution groups
        parallel_groups = self._group_by_parallel_groups(steps)
        
        execution_phases = []
        available_steps = deque([node for node in in_degree if in_degree[node] == 0])
        
        while available_steps:
            # Current phase - steps that can execute now
            current_phase = []
            next_available = []
            
            # Process all currently available steps
            while available_steps:
                step_id = available_steps.popleft()
                current_phase.append(step_id)
                
                # Update dependencies for dependent steps
                for dependent in graph:
                    if step_id in graph[dependent]:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            next_available.append(dependent)
            
            # Group current phase by parallel groups for optimal execution
            grouped_phase = self._group_phase_by_parallel_constraints(
                current_phase, parallel_groups
            )
            
            execution_phases.extend(grouped_phase)
            available_steps.extend(next_available)
        
        return execution_phases
    
    def _group_by_parallel_groups(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
        """Group steps by their parallel execution groups."""
        groups = defaultdict(list)
        for step in steps:
            if step.parallel_group:
                groups[step.parallel_group].append(step.id)
            else:
                groups[f"single_{step.id}"].append(step.id)
        return dict(groups)
    
    def _group_phase_by_parallel_constraints(self, 
                                           phase_steps: List[str], 
                                           parallel_groups: Dict[str, List[str]]) -> List[List[str]]:
        """
        Group steps in a phase considering parallel execution constraints.
        Steps in the same parallel group should execute together when possible.
        """
        if not phase_steps:
            return []
        
        # Group steps by their parallel group membership
        grouped_steps = defaultdict(list)
        ungrouped_steps = []
        
        for step_id in phase_steps:
            found_group = False
            for group_name, group_steps in parallel_groups.items():
                if step_id in group_steps and not group_name.startswith("single_"):
                    grouped_steps[group_name].append(step_id)
                    found_group = True
                    break
            
            if not found_group:
                ungrouped_steps.append(step_id)
        
        # Create execution sub-phases
        sub_phases = []
        
        # Add grouped steps (each group becomes a sub-phase)
        for group_name, group_step_ids in grouped_steps.items():
            if group_step_ids:  # Only add non-empty groups
                sub_phases.append(group_step_ids)
        
        # Add ungrouped steps (each as individual sub-phase or combine if suitable)
        if ungrouped_steps:
            # For now, combine all ungrouped steps into one parallel phase
            # This could be optimized further based on resource constraints
            sub_phases.append(ungrouped_steps)
        
        return sub_phases if sub_phases else [[]]
    
    def get_ready_steps(self, workflow: Workflow, completed_steps: Set[str]) -> List[WorkflowStep]:
        """
        Get steps that are ready to execute based on completed dependencies.
        
        Args:
            workflow: The workflow to check
            completed_steps: Set of completed step IDs
            
        Returns:
            List of steps ready for execution
        """
        ready_steps = []
        
        for step in workflow.steps:
            if (step.status == StepStatus.PENDING and 
                self._are_dependencies_satisfied(step, completed_steps)):
                ready_steps.append(step)
        
        return ready_steps
    
    def _are_dependencies_satisfied(self, step: WorkflowStep, completed_steps: Set[str]) -> bool:
        """Check if all dependencies for a step are satisfied."""
        return all(dep_id in completed_steps for dep_id in step.dependencies)
    
    def get_blocked_steps(self, workflow: Workflow, failed_steps: Set[str]) -> List[WorkflowStep]:
        """
        Get steps that are blocked due to failed dependencies.
        
        Args:
            workflow: The workflow to check
            failed_steps: Set of failed step IDs
            
        Returns:
            List of blocked steps
        """
        blocked_steps = []
        
        for step in workflow.steps:
            if (step.status == StepStatus.PENDING and
                any(dep_id in failed_steps for dep_id in step.dependencies)):
                blocked_steps.append(step)
        
        return blocked_steps
    
    def calculate_critical_path(self, workflow: Workflow) -> Tuple[List[str], float]:
        """
        Calculate the critical path through the workflow.
        
        Args:
            workflow: The workflow to analyze
            
        Returns:
            Tuple of (critical_path_step_ids, estimated_duration)
        """
        # Build dependency graph with durations
        graph = self._build_dependency_graph(workflow)
        step_durations = {step.id: step.timeout or 60 for step in workflow.steps}  # Default 60s
        
        # Calculate longest path (critical path)
        memo = {}
        
        def longest_path(step_id: str) -> Tuple[List[str], float]:
            if step_id in memo:
                return memo[step_id]
            
            if not graph[step_id]:  # No dependencies
                path = [step_id]
                duration = step_durations[step_id]
                memo[step_id] = (path, duration)
                return path, duration
            
            max_path = []
            max_duration = 0
            
            for dep_id in graph[step_id]:
                dep_path, dep_duration = longest_path(dep_id)
                total_duration = dep_duration + step_durations[step_id]
                
                if total_duration > max_duration:
                    max_duration = total_duration
                    max_path = dep_path + [step_id]
            
            memo[step_id] = (max_path, max_duration)
            return max_path, max_duration
        
        # Find the overall critical path
        critical_path = []
        critical_duration = 0
        
        for step in workflow.steps:
            path, duration = longest_path(step.id)
            if duration > critical_duration:
                critical_duration = duration
                critical_path = path
        
        self.logger.info(f"Critical path calculated: {critical_path}, duration: {critical_duration}s")
        return critical_path, critical_duration
    
    def validate_dependencies(self, workflow: Workflow) -> List[str]:
        """
        Validate workflow dependencies and return list of validation errors.
        
        Args:
            workflow: The workflow to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        step_ids = {step.id for step in workflow.steps}
        
        # Check for invalid dependency references
        for step in workflow.steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    errors.append(f"Step '{step.id}' references non-existent dependency '{dep_id}'")
        
        # Check for circular dependencies
        try:
            graph = self._build_dependency_graph(workflow)
            if self._has_circular_dependencies(graph):
                cycles = self._find_cycles(graph)
                errors.append(f"Circular dependencies detected: {cycles}")
        except Exception as e:
            errors.append(f"Error analyzing dependencies: {str(e)}")
        
        # Check for self-dependencies
        for step in workflow.steps:
            if step.id in step.dependencies:
                errors.append(f"Step '{step.id}' cannot depend on itself")
        
        return errors