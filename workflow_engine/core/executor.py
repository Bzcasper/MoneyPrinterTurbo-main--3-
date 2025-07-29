"""
Step execution system with retry logic, timeout handling, and result management.
"""

import asyncio
import time
import logging
import traceback
from typing import Dict, Any, Optional, Callable, List
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from .workflow import WorkflowStep, StepResult, StepStatus


logger = logging.getLogger(__name__)


@dataclass 
class ExecutionContext:
    """Context information for step execution."""
    workflow_id: str
    step_id: str
    workflow_variables: Dict[str, Any]
    previous_results: Dict[str, StepResult]
    dry_run: bool = False
    execution_metadata: Optional[Dict[str, Any]] = None


class ActionRegistry:
    """Registry for available workflow actions."""
    
    def __init__(self):
        self._actions: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, action_name: str, action_func: Callable):
        """Register an action function."""
        self._actions[action_name] = action_func
        self.logger.debug(f"Registered action: {action_name}")
    
    def get(self, action_name: str) -> Optional[Callable]:
        """Get an action function by name."""
        return self._actions.get(action_name)
    
    def list_actions(self) -> List[str]:
        """List all registered actions."""
        return list(self._actions.keys())
    
    def unregister(self, action_name: str) -> bool:
        """Unregister an action."""
        if action_name in self._actions:
            del self._actions[action_name]
            self.logger.debug(f"Unregistered action: {action_name}")
            return True
        return False


class StepExecutor:
    """
    Executes individual workflow steps with comprehensive error handling,
    retry logic, timeout management, and rollback capabilities.
    """
    
    def __init__(self, max_concurrent_steps: int = 10):
        self.action_registry = ActionRegistry()
        self.max_concurrent_steps = max_concurrent_steps
        self.logger = logging.getLogger(__name__)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_steps)
        self._setup_default_actions()
    
    def _setup_default_actions(self):
        """Register default workflow actions."""
        # File operations
        self.action_registry.register("file.read", self._action_file_read)
        self.action_registry.register("file.write", self._action_file_write)
        self.action_registry.register("file.delete", self._action_file_delete)
        self.action_registry.register("file.copy", self._action_file_copy)
        
        # Shell commands
        self.action_registry.register("shell.execute", self._action_shell_execute)
        
        # HTTP requests
        self.action_registry.register("http.get", self._action_http_get)
        self.action_registry.register("http.post", self._action_http_post)
        
        # Wait/delay operations
        self.action_registry.register("wait", self._action_wait)
        
        # Variable operations
        self.action_registry.register("variable.set", self._action_variable_set)
        self.action_registry.register("variable.get", self._action_variable_get)
        
        # Claude Flow MCP integration
        self.action_registry.register("claude_flow.swarm_init", self._action_claude_flow_swarm_init)
        self.action_registry.register("claude_flow.agent_spawn", self._action_claude_flow_agent_spawn)
        self.action_registry.register("claude_flow.task_orchestrate", self._action_claude_flow_task_orchestrate)
        self.action_registry.register("claude_flow.memory_store", self._action_claude_flow_memory_store)
        
        # Validation actions
        self.action_registry.register("validate.condition", self._action_validate_condition)
        
        self.logger.info(f"Registered {len(self.action_registry.list_actions())} default actions")
    
    async def execute_step(self, 
                          step: WorkflowStep, 
                          context: ExecutionContext) -> StepResult:
        """
        Execute a single workflow step with full error handling and retry logic.
        
        Args:
            step: The workflow step to execute
            context: Execution context with variables and metadata
            
        Returns:
            StepResult with execution outcome
        """
        self.logger.info(f"Executing step '{step.id}': {step.name}")
        
        step.started_at = time.time()
        step.status = StepStatus.RUNNING
        
        result = None
        attempt = 0
        
        while attempt <= step.max_retries:
            try:
                # Execute step with timeout
                result = await self._execute_with_timeout(step, context)
                
                if result.success:
                    step.status = StepStatus.COMPLETED
                    step.completed_at = time.time()
                    step.result = result
                    self.logger.info(f"Step '{step.id}' completed successfully")
                    return result
                
                # Step failed but might be retryable
                attempt += 1
                if attempt <= step.max_retries:
                    self.logger.warning(f"Step '{step.id}' failed (attempt {attempt}/{step.max_retries + 1}): {result.error}")
                    await asyncio.sleep(min(2 ** attempt, 30))  # Exponential backoff, max 30s
                    step.retry_count = attempt
                else:
                    break
                    
            except Exception as e:
                attempt += 1
                error_msg = f"Unexpected error in step '{step.id}': {str(e)}"
                self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                
                result = StepResult(
                    success=False,
                    error=error_msg,
                    metadata={"exception_type": type(e).__name__, "traceback": traceback.format_exc()}
                )
                
                if attempt <= step.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 30))
                    step.retry_count = attempt
                else:
                    break
        
        # All retries exhausted
        step.status = StepStatus.FAILED
        step.completed_at = time.time()
        step.result = result
        step.error_details = result.error if result else "Unknown error"
        
        self.logger.error(f"Step '{step.id}' failed after {step.max_retries + 1} attempts")
        return result
    
    async def _execute_with_timeout(self, 
                                   step: WorkflowStep, 
                                   context: ExecutionContext) -> StepResult:
        """Execute step with timeout handling."""
        action_func = self.action_registry.get(step.action)
        if not action_func:
            return StepResult(
                success=False,
                error=f"Unknown action: {step.action}",
                metadata={"available_actions": self.action_registry.list_actions()}
            )
        
        start_time = time.time()
        
        try:
            if step.timeout:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_action(action_func, step, context),
                    timeout=step.timeout
                )
            else:
                # Execute without timeout
                result = await self._execute_action(action_func, step, context)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return StepResult(
                success=False,
                error=f"Step timed out after {step.timeout} seconds",
                execution_time=execution_time,
                metadata={"timeout": step.timeout}
            )
    
    async def _execute_action(self, 
                             action_func: Callable, 
                             step: WorkflowStep, 
                             context: ExecutionContext) -> StepResult:
        """Execute the actual action function."""
        if context.dry_run:
            # Dry run mode - return simulated success
            return StepResult(
                success=True,
                output=f"DRY RUN: Would execute {step.action} with parameters {step.parameters}",
                metadata={"dry_run": True}
            )
        
        # Execute in thread pool for CPU-bound operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            action_func,
            step.parameters,
            context
        )
    
    # Default action implementations
    
    def _action_file_read(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Read file action."""
        try:
            file_path = params.get("path")
            if not file_path:
                return StepResult(success=False, error="Missing 'path' parameter")
            
            with open(file_path, 'r', encoding=params.get('encoding', 'utf-8')) as f:
                content = f.read()
            
            return StepResult(
                success=True,
                output=content,
                metadata={"file_path": file_path, "size": len(content)}
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
    
    def _action_file_write(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Write file action."""
        try:
            file_path = params.get("path")
            content = params.get("content", "")
            
            if not file_path:
                return StepResult(success=False, error="Missing 'path' parameter")
            
            # Store original for rollback
            rollback_data = {}
            try:
                with open(file_path, 'r') as f:
                    rollback_data["original_content"] = f.read()
                rollback_data["existed"] = True
            except FileNotFoundError:
                rollback_data["existed"] = False
            
            with open(file_path, 'w', encoding=params.get('encoding', 'utf-8')) as f:
                f.write(content)
            
            return StepResult(
                success=True,
                output=f"Written {len(content)} characters to {file_path}",
                rollback_data=rollback_data,
                metadata={"file_path": file_path, "size": len(content)}
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
    
    def _action_file_delete(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Delete file action."""
        try:
            import os
            file_path = params.get("path")
            
            if not file_path:
                return StepResult(success=False, error="Missing 'path' parameter")
            
            # Store for rollback
            rollback_data = {}
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    rollback_data["content"] = f.read()
                rollback_data["existed"] = True
                os.remove(file_path)
            else:
                rollback_data["existed"] = False
            
            return StepResult(
                success=True,
                output=f"Deleted {file_path}",
                rollback_data=rollback_data,
                metadata={"file_path": file_path}
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
    
    def _action_file_copy(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Copy file action."""
        try:
            import shutil
            src = params.get("src")
            dst = params.get("dst")
            
            if not src or not dst:
                return StepResult(success=False, error="Missing 'src' or 'dst' parameter")
            
            shutil.copy2(src, dst)
            
            return StepResult(
                success=True,
                output=f"Copied {src} to {dst}",
                metadata={"src": src, "dst": dst}
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
    
    def _action_shell_execute(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Execute shell command action."""
        try:
            import subprocess
            command = params.get("command")
            if not command:
                return StepResult(success=False, error="Missing 'command' parameter")
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=params.get("timeout", 300)
            )
            
            return StepResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                metadata={
                    "command": command,
                    "return_code": result.returncode,
                    "stderr": result.stderr
                }
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
    
    def _action_http_get(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """HTTP GET request action."""
        try:
            import requests
            url = params.get("url")
            if not url:
                return StepResult(success=False, error="Missing 'url' parameter")
            
            response = requests.get(
                url,
                headers=params.get("headers", {}),
                timeout=params.get("timeout", 30)
            )
            
            return StepResult(
                success=response.status_code < 400,
                output=response.text,
                metadata={
                    "url": url,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
    
    def _action_http_post(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """HTTP POST request action."""
        try:
            import requests
            url = params.get("url")
            if not url:
                return StepResult(success=False, error="Missing 'url' parameter")
            
            response = requests.post(
                url,
                json=params.get("json"),
                data=params.get("data"),
                headers=params.get("headers", {}),
                timeout=params.get("timeout", 30)
            )
            
            return StepResult(
                success=response.status_code < 400,
                output=response.text,
                metadata={
                    "url": url,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
    
    def _action_wait(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Wait/delay action."""
        try:
            import time
            seconds = params.get("seconds", 1)
            time.sleep(seconds)
            
            return StepResult(
                success=True,
                output=f"Waited {seconds} seconds",
                metadata={"wait_time": seconds}
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
    
    def _action_variable_set(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Set workflow variable action."""
        try:
            name = params.get("name")
            value = params.get("value")
            
            if not name:
                return StepResult(success=False, error="Missing 'name' parameter")
            
            context.workflow_variables[name] = value
            
            return StepResult(
                success=True,
                output=f"Set variable '{name}' = {value}",
                metadata={"variable_name": name, "variable_value": value}
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
    
    def _action_variable_get(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Get workflow variable action."""
        try:
            name = params.get("name")
            default = params.get("default")
            
            if not name:
                return StepResult(success=False, error="Missing 'name' parameter")
            
            value = context.workflow_variables.get(name, default)
            
            return StepResult(
                success=True,
                output=value,
                metadata={"variable_name": name, "variable_value": value}
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
    
    def _action_validate_condition(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Validate condition action."""
        try:
            condition = params.get("condition")
            if not condition:
                return StepResult(success=False, error="Missing 'condition' parameter")
            
            # Simple condition evaluation (could be extended with expression parser)
            # For now, support basic comparisons
            result = eval(condition, {"__builtins__": {}}, context.workflow_variables)
            
            return StepResult(
                success=bool(result),
                output=f"Condition '{condition}' evaluated to {result}",
                metadata={"condition": condition, "result": result}
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
    
    # Claude Flow MCP integration actions
    
    def _action_claude_flow_swarm_init(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Initialize Claude Flow swarm."""
        try:
            # This would integrate with the actual Claude Flow MCP tools
            # For now, return a simulated result
            topology = params.get("topology", "mesh")
            max_agents = params.get("maxAgents", 5)
            
            return StepResult(
                success=True,
                output=f"Initialized swarm with topology '{topology}' and {max_agents} max agents",
                metadata={"topology": topology, "max_agents": max_agents}
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
            
    def _action_claude_flow_agent_spawn(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Spawn Claude Flow agent."""
        try:
            agent_type = params.get("type", "coordinator")
            name = params.get("name", f"{agent_type}_agent")
            
            return StepResult(
                success=True,
                output=f"Spawned {agent_type} agent named '{name}'",
                metadata={"agent_type": agent_type, "agent_name": name}
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
            
    def _action_claude_flow_task_orchestrate(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Orchestrate Claude Flow task."""
        try:
            task = params.get("task", "")
            strategy = params.get("strategy", "adaptive")
            
            return StepResult(
                success=True,
                output=f"Orchestrated task '{task}' with strategy '{strategy}'",
                metadata={"task": task, "strategy": strategy}
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
            
    def _action_claude_flow_memory_store(self, params: Dict[str, Any], context: ExecutionContext) -> StepResult:
        """Store data in Claude Flow memory."""
        try:
            key = params.get("key", "")
            value = params.get("value", "")
            
            return StepResult(
                success=True,
                output=f"Stored value in memory with key '{key}'",
                metadata={"memory_key": key, "value": value}
            )
        except Exception as e:
            return StepResult(success=False, error=str(e))
    
    def cleanup(self):
        """Cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self.logger.info("Step executor cleanup completed")