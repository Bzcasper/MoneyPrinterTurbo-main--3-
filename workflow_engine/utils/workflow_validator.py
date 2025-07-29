"""
Workflow validation utilities for comprehensive workflow checking.
"""

import logging
import re
from typing import Dict, Any, List, Set, Optional, Tuple
from datetime import datetime

from ..core.workflow import Workflow, WorkflowStep
from ..core.dependency_resolver import DependencyResolver, CircularDependencyError


logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of workflow validation."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.is_valid: bool = True
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_info(self, message: str):
        """Add an info message."""
        self.info.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.info)
        }


class WorkflowValidator:
    """
    Comprehensive workflow validation system.
    
    Validation categories:
    - Structure validation (required fields, data types)
    - Dependency validation (circular dependencies, invalid references)
    - Action validation (unknown actions, parameter validation)
    - Logic validation (unreachable steps, inefficient patterns)
    - Performance validation (potential bottlenecks, optimization suggestions)
    - Security validation (unsafe operations, permission issues)
    """
    
    def __init__(self, available_actions: Optional[List[str]] = None):
        """
        Initialize the workflow validator.
        
        Args:
            available_actions: List of available workflow actions for validation
        """
        self.available_actions = set(available_actions or [])
        self.dependency_resolver = DependencyResolver()
        self.logger = logging.getLogger(__name__)
        
        # Validation rules configuration
        self.max_workflow_steps = 1000
        self.max_step_dependencies = 50
        self.max_parallel_groups = 20
        self.recommended_step_timeout = 300  # 5 minutes
        self.max_step_timeout = 3600  # 1 hour
        
        # Security patterns
        self.unsafe_shell_patterns = [
            r'rm\s+-rf\s+/',
            r'del\s+/[sq]',
            r'format\s+[cd]:',
            r'>\s*/dev/null\s+2>&1',
            r'curl.*\|\s*sh',
            r'wget.*\|\s*sh'
        ]
        
        self.logger.info(f"Workflow validator initialized with {len(self.available_actions)} known actions")
    
    def validate(self, workflow: Workflow, strict: bool = False) -> ValidationResult:
        """
        Perform comprehensive workflow validation.
        
        Args:
            workflow: The workflow to validate
            strict: Whether to apply strict validation rules
            
        Returns:
            ValidationResult with detailed validation information
        """
        self.logger.info(f"Validating workflow '{workflow.name}' (strict: {strict})")
        
        result = ValidationResult()
        
        # Basic structure validation
        self._validate_structure(workflow, result)
        
        # Step validation
        self._validate_steps(workflow, result, strict)
        
        # Dependency validation
        self._validate_dependencies(workflow, result)
        
        # Action validation
        self._validate_actions(workflow, result)
        
        # Logic validation
        self._validate_logic(workflow, result)
        
        # Performance validation
        self._validate_performance(workflow, result)
        
        # Security validation
        self._validate_security(workflow, result, strict)
        
        # Additional checks for strict mode
        if strict:
            self._validate_strict_rules(workflow, result)
        
        validation_level = "PASSED" if result.is_valid else "FAILED"
        self.logger.info(f"Workflow validation {validation_level}: {len(result.errors)} errors, {len(result.warnings)} warnings")
        
        return result
    
    def _validate_structure(self, workflow: Workflow, result: ValidationResult):
        """Validate basic workflow structure."""
        # Required fields
        if not workflow.id:
            result.add_error("Workflow ID is required")
        elif not isinstance(workflow.id, str) or len(workflow.id.strip()) == 0:
            result.add_error("Workflow ID must be a non-empty string")
        
        if not workflow.name:
            result.add_error("Workflow name is required")
        elif not isinstance(workflow.name, str) or len(workflow.name.strip()) == 0:
            result.add_error("Workflow name must be a non-empty string")
        
        # ID format validation
        if workflow.id and not re.match(r'^[a-zA-Z0-9_-]+$', workflow.id):
            result.add_error(f"Workflow ID '{workflow.id}' contains invalid characters. Use only alphanumeric, underscore, and dash.")
        
        # Steps validation
        if not workflow.steps:
            result.add_error("Workflow must contain at least one step")
        elif len(workflow.steps) > self.max_workflow_steps:
            result.add_error(f"Workflow has too many steps ({len(workflow.steps)}). Maximum allowed: {self.max_workflow_steps}")
        
        # Metadata validation
        if workflow.metadata and not isinstance(workflow.metadata, dict):
            result.add_error("Workflow metadata must be a dictionary")
        
        # Version validation
        if workflow.version and not isinstance(workflow.version, str):
            result.add_error("Workflow version must be a string")
    
    def _validate_steps(self, workflow: Workflow, result: ValidationResult, strict: bool):
        """Validate individual workflow steps."""
        step_ids = set()
        
        for i, step in enumerate(workflow.steps):
            step_prefix = f"Step {i+1} ('{step.id}')"
            
            # Required fields
            if not step.id:
                result.add_error(f"{step_prefix}: Step ID is required")
            elif not isinstance(step.id, str) or len(step.id.strip()) == 0:
                result.add_error(f"{step_prefix}: Step ID must be a non-empty string")
            
            if not step.name:
                result.add_error(f"{step_prefix}: Step name is required")
            elif not isinstance(step.name, str) or len(step.name.strip()) == 0:
                result.add_error(f"{step_prefix}: Step name must be a non-empty string")
            
            if not step.action:
                result.add_error(f"{step_prefix}: Step action is required")
            elif not isinstance(step.action, str) or len(step.action.strip()) == 0:
                result.add_error(f"{step_prefix}: Step action must be a non-empty string")
            
            # ID uniqueness
            if step.id:
                if step.id in step_ids:
                    result.add_error(f"{step_prefix}: Duplicate step ID '{step.id}'")
                else:
                    step_ids.add(step.id)
                
                # ID format validation
                if not re.match(r'^[a-zA-Z0-9_-]+$', step.id):
                    result.add_error(f"{step_prefix}: Step ID contains invalid characters. Use only alphanumeric, underscore, and dash.")
            
            # Parameters validation
            if not isinstance(step.parameters, dict):
                result.add_error(f"{step_prefix}: Step parameters must be a dictionary")
            
            # Dependencies validation
            if not isinstance(step.dependencies, list):
                result.add_error(f"{step_prefix}: Step dependencies must be a list")
            elif len(step.dependencies) > self.max_step_dependencies:
                result.add_error(f"{step_prefix}: Too many dependencies ({len(step.dependencies)}). Maximum: {self.max_step_dependencies}")
            
            # Timeout validation
            if step.timeout is not None:
                if not isinstance(step.timeout, (int, float)) or step.timeout <= 0:
                    result.add_error(f"{step_prefix}: Timeout must be a positive number")
                elif step.timeout > self.max_step_timeout:
                    result.add_error(f"{step_prefix}: Timeout too large ({step.timeout}s). Maximum: {self.max_step_timeout}s")
                elif step.timeout > self.recommended_step_timeout:
                    result.add_warning(f"{step_prefix}: Long timeout ({step.timeout}s). Consider if this is necessary.")
            
            # Retry count validation
            if not isinstance(step.max_retries, int) or step.max_retries < 0:
                result.add_error(f"{step_prefix}: Max retries must be a non-negative integer")
            elif step.max_retries > 10:
                result.add_warning(f"{step_prefix}: High retry count ({step.max_retries}). Consider if this is appropriate.")
            
            # Critical flag validation
            if not isinstance(step.critical, bool):
                result.add_error(f"{step_prefix}: Critical flag must be boolean")
            
            # Parallel group validation
            if step.parallel_group and not isinstance(step.parallel_group, str):
                result.add_error(f"{step_prefix}: Parallel group must be a string")
            
            # Conditions validation
            if step.conditions and not isinstance(step.conditions, dict):
                result.add_error(f"{step_prefix}: Conditions must be a dictionary")
            
            # Rollback validation
            if step.rollback_action and not isinstance(step.rollback_action, str):
                result.add_error(f"{step_prefix}: Rollback action must be a string")
            
            if step.rollback_parameters and not isinstance(step.rollback_parameters, dict):
                result.add_error(f"{step_prefix}: Rollback parameters must be a dictionary")
            
            # Strict mode additional checks
            if strict:
                if not step.rollback_action and step.action.startswith(('file.write', 'file.delete', 'shell.execute')):
                    result.add_warning(f"{step_prefix}: Consider adding rollback action for potentially destructive operation")
                
                if step.timeout is None:
                    result.add_info(f"{step_prefix}: No timeout specified. Consider adding one for better error handling.")
    
    def _validate_dependencies(self, workflow: Workflow, result: ValidationResult):
        """Validate step dependencies."""
        step_ids = {step.id for step in workflow.steps if step.id}
        
        # Check dependency references
        for step in workflow.steps:
            if not step.id:
                continue
                
            step_prefix = f"Step '{step.id}'"
            
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    result.add_error(f"{step_prefix}: References non-existent dependency '{dep_id}'")
                elif dep_id == step.id:
                    result.add_error(f"{step_prefix}: Cannot depend on itself")
        
        # Check for circular dependencies
        try:
            self.dependency_resolver.resolve_execution_order(workflow)
        except CircularDependencyError as e:
            result.add_error(f"Circular dependency detected: {str(e)}")
        
        # Check for orphaned steps (no dependencies and not depended upon)
        depended_upon = set()
        for step in workflow.steps:
            depended_upon.update(step.dependencies)
        
        for step in workflow.steps:
            if not step.id:
                continue
            if not step.dependencies and step.id not in depended_upon:
                if len(workflow.steps) > 1:  # Only warn if there are multiple steps
                    result.add_warning(f"Step '{step.id}' appears to be isolated (no dependencies and not depended upon)")
    
    def _validate_actions(self, workflow: Workflow, result: ValidationResult):
        """Validate workflow actions."""
        action_counts = {}
        
        for step in workflow.steps:
            if not step.action:
                continue
                
            step_prefix = f"Step '{step.id}'"
            
            # Count action usage
            action_counts[step.action] = action_counts.get(step.action, 0) + 1
            
            # Check if action is known
            if self.available_actions and step.action not in self.available_actions:
                result.add_error(f"{step_prefix}: Unknown action '{step.action}'")
            
            # Action-specific parameter validation
            self._validate_action_parameters(step, result)
        
        # Report action usage statistics
        if len(action_counts) > 20:
            result.add_warning(f"Workflow uses {len(action_counts)} different action types. Consider consolidation.")
        
        for action, count in action_counts.items():
            if count > 50:
                result.add_info(f"Action '{action}' is used {count} times. Consider optimization.")
    
    def _validate_action_parameters(self, step: WorkflowStep, result: ValidationResult):
        """Validate parameters for specific actions."""
        step_prefix = f"Step '{step.id}'"
        action = step.action
        params = step.parameters
        
        # File operations
        if action in ['file.read', 'file.write', 'file.delete']:
            if 'path' not in params:
                result.add_error(f"{step_prefix}: Missing required parameter 'path' for {action}")
            elif not isinstance(params['path'], str):
                result.add_error(f"{step_prefix}: Parameter 'path' must be a string")
            elif params['path'].startswith('/tmp/') and action == 'file.delete':
                result.add_warning(f"{step_prefix}: Deleting file in /tmp directory - ensure this is intentional")
        
        if action == 'file.write':
            if 'content' not in params:
                result.add_error(f"{step_prefix}: Missing required parameter 'content' for file.write")
        
        if action == 'file.copy':
            for required_param in ['src', 'dst']:
                if required_param not in params:
                    result.add_error(f"{step_prefix}: Missing required parameter '{required_param}' for file.copy")
        
        # HTTP operations
        if action in ['http.get', 'http.post']:
            if 'url' not in params:
                result.add_error(f"{step_prefix}: Missing required parameter 'url' for {action}")
            elif not isinstance(params['url'], str):
                result.add_error(f"{step_prefix}: Parameter 'url' must be a string")
            elif not params['url'].startswith(('http://', 'https://')):
                result.add_warning(f"{step_prefix}: URL should start with http:// or https://")
        
        # Shell operations
        if action == 'shell.execute':
            if 'command' not in params:
                result.add_error(f"{step_prefix}: Missing required parameter 'command' for shell.execute")
            elif not isinstance(params['command'], str):
                result.add_error(f"{step_prefix}: Parameter 'command' must be a string")
        
        # Variable operations
        if action == 'variable.set':
            if 'name' not in params:
                result.add_error(f"{step_prefix}: Missing required parameter 'name' for variable.set")
            elif not isinstance(params['name'], str):
                result.add_error(f"{step_prefix}: Parameter 'name' must be a string")
        
        if action == 'variable.get':
            if 'name' not in params:
                result.add_error(f"{step_prefix}: Missing required parameter 'name' for variable.get")
        
        # Wait operation
        if action == 'wait':
            if 'seconds' in params and (not isinstance(params['seconds'], (int, float)) or params['seconds'] <= 0):
                result.add_error(f"{step_prefix}: Parameter 'seconds' must be a positive number")
    
    def _validate_logic(self, workflow: Workflow, result: ValidationResult):
        """Validate workflow logic and flow."""
        # Check for unreachable steps
        try:
            execution_phases = self.dependency_resolver.resolve_execution_order(workflow)
            reachable_steps = set()
            for phase in execution_phases:
                reachable_steps.update(phase)
            
            all_step_ids = {step.id for step in workflow.steps if step.id}
            unreachable = all_step_ids - reachable_steps
            
            for step_id in unreachable:
                result.add_warning(f"Step '{step_id}' may be unreachable due to dependency configuration")
        
        except Exception as e:
            result.add_error(f"Error analyzing workflow logic: {str(e)}")
        
        # Check parallel groups
        parallel_groups = {}
        for step in workflow.steps:
            if step.parallel_group:
                if step.parallel_group not in parallel_groups:
                    parallel_groups[step.parallel_group] = []
                parallel_groups[step.parallel_group].append(step.id)
        
        if len(parallel_groups) > self.max_parallel_groups:
            result.add_warning(f"Many parallel groups ({len(parallel_groups)}). Consider consolidation.")
        
        for group_name, step_ids in parallel_groups.items():
            if len(step_ids) == 1:
                result.add_info(f"Parallel group '{group_name}' has only one step. Consider removing the group.")
    
    def _validate_performance(self, workflow: Workflow, result: ValidationResult):
        """Validate workflow for performance considerations."""
        # Check for potential bottlenecks
        step_count = len(workflow.steps)
        
        if step_count > 100:
            result.add_warning(f"Large workflow ({step_count} steps). Consider breaking into smaller workflows.")
        
        # Check dependency chains
        try:
            critical_path, duration = self.dependency_resolver.calculate_critical_path(workflow)
            if duration > 3600:  # 1 hour
                result.add_warning(f"Critical path is long ({duration:.0f}s). Consider optimization.")
                result.add_info(f"Critical path: {' -> '.join(critical_path)}")
        except Exception:
            pass  # Critical path calculation failed, skip this check
        
        # Check for sequential file operations that could be parallelized
        file_operations = [step for step in workflow.steps if step.action.startswith('file.')]
        sequential_file_ops = []
        
        for i, step in enumerate(file_operations[:-1]):
            next_step = file_operations[i + 1]
            if step.id in next_step.dependencies:
                sequential_file_ops.append((step.id, next_step.id))
        
        if len(sequential_file_ops) > 5:
            result.add_info("Many sequential file operations detected. Consider parallelization where possible.")
    
    def _validate_security(self, workflow: Workflow, result: ValidationResult, strict: bool):
        """Validate workflow for security considerations."""
        for step in workflow.steps:
            if not step.id:
                continue
                
            step_prefix = f"Step '{step.id}'"
            
            # Check shell commands for dangerous patterns
            if step.action == 'shell.execute' and 'command' in step.parameters:
                command = step.parameters['command']
                
                for pattern in self.unsafe_shell_patterns:
                    if re.search(pattern, command, re.IGNORECASE):
                        result.add_error(f"{step_prefix}: Potentially dangerous shell command detected: {command}")
                        break
                
                # Check for other security concerns
                if 'sudo' in command:
                    result.add_warning(f"{step_prefix}: Command uses sudo. Ensure proper permissions are configured.")
                
                if '|' in command and 'sh' in command:
                    result.add_warning(f"{step_prefix}: Command pipes to shell. Verify this is safe.")
                
                if command.strip().startswith('curl') and 'http://' in command:
                    result.add_warning(f"{step_prefix}: Using HTTP instead of HTTPS for curl command.")
            
            # Check file operations for security
            if step.action in ['file.write', 'file.delete'] and 'path' in step.parameters:
                path = step.parameters['path']
                
                if path.startswith('/etc/'):
                    result.add_warning(f"{step_prefix}: Modifying system configuration files. Ensure proper permissions.")
                
                if path.startswith('/'):
                    result.add_info(f"{step_prefix}: Using absolute path. Consider if relative path is more appropriate.")
            
            # Check HTTP operations for security
            if step.action in ['http.get', 'http.post'] and 'url' in step.parameters:
                url = step.parameters['url']
                
                if url.startswith('http://'):
                    result.add_warning(f"{step_prefix}: Using unencrypted HTTP. Consider HTTPS.")
                
                if 'localhost' in url or '127.0.0.1' in url:
                    result.add_info(f"{step_prefix}: Making request to localhost. Ensure this is intentional.")
            
            # Check for hardcoded credentials
            param_str = str(step.parameters)
            if any(keyword in param_str.lower() for keyword in ['password', 'secret', 'key', 'token']):
                if strict:
                    result.add_error(f"{step_prefix}: Possible hardcoded credentials detected. Use environment variables or secure storage.")
                else:
                    result.add_warning(f"{step_prefix}: Possible hardcoded credentials detected. Consider using environment variables.")
    
    def _validate_strict_rules(self, workflow: Workflow, result: ValidationResult):
        """Apply strict validation rules."""
        # All steps must have descriptions
        for step in workflow.steps:
            if not step.name or len(step.name.strip()) < 10:
                result.add_warning(f"Step '{step.id}': Step name should be descriptive (at least 10 characters)")
        
        # Workflow must have description
        if not workflow.description or len(workflow.description.strip()) < 20:
            result.add_warning("Workflow description should be comprehensive (at least 20 characters)")
        
        # All file operations should have rollback actions
        for step in workflow.steps:
            if step.action in ['file.write', 'file.delete', 'file.copy'] and not step.rollback_action:
                result.add_error(f"Step '{step.id}': File operation requires rollback action in strict mode")
        
        # All shell commands should have timeouts
        for step in workflow.steps:
            if step.action == 'shell.execute' and step.timeout is None:
                result.add_error(f"Step '{step.id}': Shell command requires timeout in strict mode")
    
    def validate_step_parameters(self, action: str, parameters: Dict[str, Any]) -> List[str]:
        """
        Validate parameters for a specific action.
        
        Args:
            action: The action name
            parameters: The parameters to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Create a temporary step for validation
        temp_step = WorkflowStep(
            id="temp_validation_step",
            name="Temporary Validation Step",
            action=action,
            parameters=parameters
        )
        
        result = ValidationResult()
        self._validate_action_parameters(temp_step, result)
        
        return result.errors
    
    def suggest_improvements(self, workflow: Workflow) -> List[str]:
        """
        Suggest improvements for the workflow.
        
        Args:
            workflow: The workflow to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Analyze parallel execution opportunities
        try:
            execution_phases = self.dependency_resolver.resolve_execution_order(workflow)
            
            # Look for phases with only one step that could be parallelized
            single_step_phases = [phase for phase in execution_phases if len(phase) == 1]
            if len(single_step_phases) > len(execution_phases) * 0.7:
                suggestions.append("Consider adding parallel execution groups to improve performance")
        
        except Exception:
            pass
        
        # Analyze action patterns
        action_counts = {}
        for step in workflow.steps:
            action_counts[step.action] = action_counts.get(step.action, 0) + 1
        
        # Suggest consolidation for repeated actions
        for action, count in action_counts.items():
            if count > 10:
                suggestions.append(f"Consider consolidating {count} instances of '{action}' action")
        
        # Suggest timeout optimization
        steps_without_timeout = [step for step in workflow.steps if step.timeout is None and step.action == 'shell.execute']
        if len(steps_without_timeout) > 5:
            suggestions.append("Add timeouts to shell commands to prevent hanging")
        
        # Suggest error handling improvements
        steps_without_rollback = [
            step for step in workflow.steps 
            if step.action in ['file.write', 'file.delete'] and not step.rollback_action
        ]
        if len(steps_without_rollback) > 3:
            suggestions.append("Add rollback actions to file operations for better error recovery")
        
        return suggestions
    
    def get_validation_summary(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Get a comprehensive validation summary.
        
        Args:
            workflow: The workflow to analyze
            
        Returns:
            Dictionary with validation summary
        """
        result = self.validate(workflow)
        suggestions = self.suggest_improvements(workflow)
        
        return {
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "validation_result": result.to_dict(),
            "suggestions": suggestions,
            "complexity_score": len(workflow.steps) + sum(len(step.dependencies) for step in workflow.steps),
            "parallel_potential": len([step for step in workflow.steps if not step.dependencies]) / len(workflow.steps) if workflow.steps else 0,
            "validation_timestamp": datetime.now().isoformat()
        }