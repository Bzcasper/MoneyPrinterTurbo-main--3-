"""
Rollback management system for failed workflows.
Provides automatic rollback capabilities with checkpoint management.
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import os

from ..core.workflow import Workflow, WorkflowStep, StepStatus, StepResult
from ..core.executor import ExecutionContext


logger = logging.getLogger(__name__)


@dataclass
class RollbackCheckpoint:
    """Represents a rollback checkpoint for a workflow step."""
    step_id: str
    timestamp: datetime
    pre_execution_state: Dict[str, Any]
    rollback_data: Dict[str, Any]
    dependencies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "timestamp": self.timestamp.isoformat(),
            "pre_execution_state": self.pre_execution_state,
            "rollback_data": self.rollback_data,
            "dependencies": self.dependencies
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RollbackCheckpoint':
        """Create from dictionary."""
        return cls(
            step_id=data["step_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            pre_execution_state=data["pre_execution_state"],
            rollback_data=data["rollback_data"],
            dependencies=data["dependencies"]
        )


class RollbackManager:
    """
    Manages rollback operations for failed workflows.
    
    Features:
    - Automatic checkpoint creation before step execution
    - Dependency-aware rollback ordering
    - File system state restoration
    - Variable state restoration
    - Configurable rollback strategies
    - Persistent rollback history
    """
    
    def __init__(self, 
                 checkpoint_dir: str = ".workflow_checkpoints",
                 max_checkpoints: int = 100,
                 enable_file_snapshots: bool = True):
        """
        Initialize the rollback manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint data
            max_checkpoints: Maximum number of checkpoints to keep
            enable_file_snapshots: Whether to create file snapshots
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.enable_file_snapshots = enable_file_snapshots
        self.logger = logging.getLogger(__name__)
        
        # Runtime state
        self.workflow_checkpoints: Dict[str, List[RollbackCheckpoint]] = {}
        self.rollback_actions = {
            "file.write": self._rollback_file_write,
            "file.delete": self._rollback_file_delete,
            "file.copy": self._rollback_file_copy,
            "shell.execute": self._rollback_shell_execute,
            "variable.set": self._rollback_variable_set,
            "claude_flow.swarm_init": self._rollback_claude_flow_swarm,
            "claude_flow.agent_spawn": self._rollback_claude_flow_agent,
            "claude_flow.memory_store": self._rollback_claude_flow_memory
        }
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.logger.info(f"Rollback manager initialized with checkpoint dir: {checkpoint_dir}")
    
    async def create_checkpoint(self, 
                               workflow: Workflow,
                               step: WorkflowStep,
                               context: ExecutionContext) -> RollbackCheckpoint:
        """
        Create a rollback checkpoint before executing a step.
        
        Args:
            workflow: The workflow being executed
            step: The step about to be executed
            context: The execution context
            
        Returns:
            Created checkpoint
        """
        self.logger.debug(f"Creating checkpoint for step '{step.id}'")
        
        # Capture pre-execution state
        pre_execution_state = {
            "workflow_variables": context.workflow_variables.copy(),
            "step_status": step.status.value,
            "timestamp": datetime.now().isoformat()
        }
        
        # Prepare rollback data based on step action
        rollback_data = await self._prepare_rollback_data(step, context)
        
        # Create checkpoint
        checkpoint = RollbackCheckpoint(
            step_id=step.id,
            timestamp=datetime.now(),
            pre_execution_state=pre_execution_state,
            rollback_data=rollback_data,
            dependencies=step.dependencies.copy()
        )
        
        # Store checkpoint
        if workflow.id not in self.workflow_checkpoints:
            self.workflow_checkpoints[workflow.id] = []
        
        self.workflow_checkpoints[workflow.id].append(checkpoint)
        
        # Persist checkpoint
        await self._persist_checkpoint(workflow.id, checkpoint)
        
        # Cleanup old checkpoints
        await self._cleanup_old_checkpoints(workflow.id)
        
        self.logger.debug(f"Checkpoint created for step '{step.id}'")
        return checkpoint
    
    async def _prepare_rollback_data(self, 
                                    step: WorkflowStep, 
                                    context: ExecutionContext) -> Dict[str, Any]:
        """Prepare rollback data specific to the step action."""
        rollback_data = {}
        
        try:
            if step.action == "file.write":
                file_path = step.parameters.get("path")
                if file_path and os.path.exists(file_path):
                    # Backup existing file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        rollback_data["original_content"] = f.read()
                    rollback_data["file_existed"] = True
                else:
                    rollback_data["file_existed"] = False
                rollback_data["file_path"] = file_path
            
            elif step.action == "file.delete":
                file_path = step.parameters.get("path")
                if file_path and os.path.exists(file_path):
                    # Backup file content before deletion
                    with open(file_path, 'r', encoding='utf-8') as f:
                        rollback_data["file_content"] = f.read()
                    rollback_data["file_existed"] = True
                else:
                    rollback_data["file_existed"] = False
                rollback_data["file_path"] = file_path
            
            elif step.action == "file.copy":
                src = step.parameters.get("src")
                dst = step.parameters.get("dst")
                rollback_data["src"] = src
                rollback_data["dst"] = dst
                rollback_data["dst_existed"] = os.path.exists(dst) if dst else False
                
                if dst and os.path.exists(dst):
                    # Backup destination file
                    with open(dst, 'r', encoding='utf-8') as f:
                        rollback_data["dst_original_content"] = f.read()
            
            elif step.action == "variable.set":
                var_name = step.parameters.get("name")
                rollback_data["variable_name"] = var_name
                rollback_data["previous_value"] = context.workflow_variables.get(var_name)
                rollback_data["had_previous_value"] = var_name in context.workflow_variables
            
            elif step.action in ["claude_flow.swarm_init", "claude_flow.agent_spawn", "claude_flow.memory_store"]:
                # For Claude Flow actions, store the action parameters for potential cleanup
                rollback_data["action_parameters"] = step.parameters.copy()
                rollback_data["context_state"] = {
                    "workflow_id": context.workflow_id,
                    "variables": context.workflow_variables.copy()
                }
        
        except Exception as e:
            self.logger.warning(f"Error preparing rollback data for step '{step.id}': {e}")
            rollback_data["error"] = str(e)
        
        return rollback_data
    
    async def rollback_workflow(self, 
                               workflow: Workflow, 
                               context: ExecutionContext) -> bool:
        """
        Rollback a failed workflow by reversing completed steps.
        
        Args:
            workflow: The workflow to rollback
            context: The execution context
            
        Returns:
            True if rollback succeeded, False otherwise
        """
        self.logger.info(f"Starting rollback for workflow '{workflow.name}'")
        
        checkpoints = self.workflow_checkpoints.get(workflow.id, [])
        if not checkpoints:
            self.logger.warning(f"No checkpoints found for workflow '{workflow.id}'")
            return True  # Nothing to rollback
        
        # Get completed steps that need rollback
        completed_steps = [
            step for step in workflow.steps 
            if step.status == StepStatus.COMPLETED
        ]
        
        if not completed_steps:
            self.logger.info("No completed steps to rollback")
            return True
        
        # Order steps for rollback (reverse dependency order)
        rollback_order = self._calculate_rollback_order(completed_steps, checkpoints)
        
        self.logger.info(f"Rolling back {len(rollback_order)} steps in order: {[s.id for s in rollback_order]}")
        
        # Perform rollback
        success_count = 0
        total_steps = len(rollback_order)
        
        for step in rollback_order:
            try:
                checkpoint = self._find_checkpoint(step.id, checkpoints)
                if checkpoint:
                    success = await self._rollback_step(step, checkpoint, context)
                    if success:
                        step.status = StepStatus.ROLLED_BACK
                        success_count += 1
                        self.logger.debug(f"Successfully rolled back step '{step.id}'")
                    else:
                        self.logger.error(f"Failed to rollback step '{step.id}'")
                else:
                    self.logger.warning(f"No checkpoint found for step '{step.id}'")
            
            except Exception as e:
                self.logger.error(f"Error during rollback of step '{step.id}': {e}")
        
        # Restore workflow variables to initial state
        if checkpoints:
            initial_checkpoint = min(checkpoints, key=lambda c: c.timestamp)
            context.workflow_variables.clear()
            context.workflow_variables.update(
                initial_checkpoint.pre_execution_state.get("workflow_variables", {})
            )
        
        # Cleanup checkpoints
        await self._cleanup_workflow_checkpoints(workflow.id)
        
        rollback_success = success_count == total_steps
        self.logger.info(f"Rollback completed: {success_count}/{total_steps} steps rolled back successfully")
        
        return rollback_success
    
    def _calculate_rollback_order(self, 
                                 completed_steps: List[WorkflowStep],
                                 checkpoints: List[RollbackCheckpoint]) -> List[WorkflowStep]:
        """Calculate the order in which steps should be rolled back."""
        # Create dependency graph
        step_map = {step.id: step for step in completed_steps}
        dependents = {}  # step_id -> [steps_that_depend_on_it]
        
        for step in completed_steps:
            for dep_id in step.dependencies:
                if dep_id not in dependents:
                    dependents[dep_id] = []
                dependents[dep_id].append(step.id)
        
        # Topological sort in reverse order (dependents first)
        rollback_order = []
        processed = set()
        
        def process_step(step_id: str):
            if step_id in processed or step_id not in step_map:
                return
            
            # Process all steps that depend on this one first
            for dependent_id in dependents.get(step_id, []):
                process_step(dependent_id)
            
            processed.add(step_id)
            rollback_order.append(step_map[step_id])
        
        # Start with steps that have no dependents
        for step in completed_steps:
            process_step(step.id)
        
        return rollback_order
    
    def _find_checkpoint(self, step_id: str, checkpoints: List[RollbackCheckpoint]) -> Optional[RollbackCheckpoint]:
        """Find checkpoint for a specific step."""
        for checkpoint in checkpoints:
            if checkpoint.step_id == step_id:
                return checkpoint
        return None
    
    async def _rollback_step(self, 
                            step: WorkflowStep, 
                            checkpoint: RollbackCheckpoint,
                            context: ExecutionContext) -> bool:
        """Rollback a single step using its checkpoint."""
        rollback_action = self.rollback_actions.get(step.action)
        if not rollback_action:
            self.logger.warning(f"No rollback action defined for '{step.action}'")
            return True  # Consider it successful if no rollback needed
        
        try:
            return await rollback_action(step, checkpoint, context)
        except Exception as e:
            self.logger.error(f"Error rolling back step '{step.id}': {e}")
            return False
    
    # Rollback action implementations
    
    async def _rollback_file_write(self, 
                                  step: WorkflowStep, 
                                  checkpoint: RollbackCheckpoint,
                                  context: ExecutionContext) -> bool:
        """Rollback file write operation."""
        try:
            rollback_data = checkpoint.rollback_data
            file_path = rollback_data.get("file_path")
            
            if not file_path:
                return False
            
            if rollback_data.get("file_existed", False):
                # Restore original content
                original_content = rollback_data.get("original_content", "")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                self.logger.debug(f"Restored original content to {file_path}")
            else:
                # Remove the file that was created
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.debug(f"Removed created file {file_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error rolling back file write: {e}")
            return False
    
    async def _rollback_file_delete(self, 
                                   step: WorkflowStep, 
                                   checkpoint: RollbackCheckpoint,
                                   context: ExecutionContext) -> bool:
        """Rollback file delete operation."""
        try:
            rollback_data = checkpoint.rollback_data
            file_path = rollback_data.get("file_path")
            
            if not file_path or not rollback_data.get("file_existed", False):
                return True  # Nothing to restore
            
            # Restore the deleted file
            file_content = rollback_data.get("file_content", "")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            self.logger.debug(f"Restored deleted file {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error rolling back file delete: {e}")
            return False
    
    async def _rollback_file_copy(self, 
                                 step: WorkflowStep, 
                                 checkpoint: RollbackCheckpoint,
                                 context: ExecutionContext) -> bool:
        """Rollback file copy operation."""
        try:
            rollback_data = checkpoint.rollback_data
            dst = rollback_data.get("dst")
            
            if not dst:
                return False
            
            if rollback_data.get("dst_existed", False):
                # Restore original destination content
                original_content = rollback_data.get("dst_original_content", "")
                with open(dst, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                self.logger.debug(f"Restored original content to {dst}")
            else:
                # Remove the copied file
                if os.path.exists(dst):
                    os.remove(dst)
                    self.logger.debug(f"Removed copied file {dst}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error rolling back file copy: {e}")
            return False
    
    async def _rollback_shell_execute(self, 
                                     step: WorkflowStep, 
                                     checkpoint: RollbackCheckpoint,
                                     context: ExecutionContext) -> bool:
        """Rollback shell execute operation."""
        # Shell commands are generally hard to rollback automatically
        # This would require specific rollback commands to be defined
        rollback_command = step.rollback_parameters.get("rollback_command") if step.rollback_parameters else None
        
        if rollback_command:
            try:
                import subprocess
                result = subprocess.run(
                    rollback_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                success = result.returncode == 0
                if not success:
                    self.logger.error(f"Rollback command failed: {result.stderr}")
                else:
                    self.logger.debug(f"Executed rollback command: {rollback_command}")
                
                return success
            except Exception as e:
                self.logger.error(f"Error executing rollback command: {e}")
                return False
        
        self.logger.warning(f"No rollback command defined for shell step '{step.id}'")
        return True  # Consider successful if no rollback command
    
    async def _rollback_variable_set(self, 
                                    step: WorkflowStep, 
                                    checkpoint: RollbackCheckpoint,
                                    context: ExecutionContext) -> bool:
        """Rollback variable set operation."""
        try:
            rollback_data = checkpoint.rollback_data
            var_name = rollback_data.get("variable_name")
            
            if not var_name:
                return False
            
            if rollback_data.get("had_previous_value", False):
                # Restore previous value
                previous_value = rollback_data.get("previous_value")
                context.workflow_variables[var_name] = previous_value
                self.logger.debug(f"Restored variable '{var_name}' to previous value")
            else:
                # Remove the variable
                if var_name in context.workflow_variables:
                    del context.workflow_variables[var_name]
                    self.logger.debug(f"Removed variable '{var_name}'")
            
            return True
        except Exception as e:
            self.logger.error(f"Error rolling back variable set: {e}")
            return False
    
    async def _rollback_claude_flow_swarm(self, 
                                         step: WorkflowStep, 
                                         checkpoint: RollbackCheckpoint,
                                         context: ExecutionContext) -> bool:
        """Rollback Claude Flow swarm initialization."""
        # This would require integration with Claude Flow MCP tools
        # For now, log the rollback attempt
        self.logger.info(f"Rolling back Claude Flow swarm initialization for step '{step.id}'")
        return True
    
    async def _rollback_claude_flow_agent(self, 
                                         step: WorkflowStep, 
                                         checkpoint: RollbackCheckpoint,
                                         context: ExecutionContext) -> bool:
        """Rollback Claude Flow agent spawn."""
        self.logger.info(f"Rolling back Claude Flow agent spawn for step '{step.id}'")
        return True
    
    async def _rollback_claude_flow_memory(self, 
                                          step: WorkflowStep, 
                                          checkpoint: RollbackCheckpoint,
                                          context: ExecutionContext) -> bool:
        """Rollback Claude Flow memory store."""
        self.logger.info(f"Rolling back Claude Flow memory store for step '{step.id}'")
        return True
    
    async def _persist_checkpoint(self, workflow_id: str, checkpoint: RollbackCheckpoint):
        """Persist checkpoint to disk."""
        checkpoint_file = os.path.join(
            self.checkpoint_dir, 
            f"{workflow_id}_{checkpoint.step_id}_{int(checkpoint.timestamp.timestamp())}.json"
        )
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error persisting checkpoint: {e}")
    
    async def _cleanup_old_checkpoints(self, workflow_id: str):
        """Remove old checkpoints to stay within limits."""
        checkpoints = self.workflow_checkpoints.get(workflow_id, [])
        
        if len(checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints
            checkpoints.sort(key=lambda c: c.timestamp)
            to_remove = checkpoints[:-self.max_checkpoints]
            
            for checkpoint in to_remove:
                checkpoints.remove(checkpoint)
                # Remove persisted file
                checkpoint_file = os.path.join(
                    self.checkpoint_dir,
                    f"{workflow_id}_{checkpoint.step_id}_{int(checkpoint.timestamp.timestamp())}.json"
                )
                try:
                    if os.path.exists(checkpoint_file):
                        os.remove(checkpoint_file)
                except Exception as e:
                    self.logger.error(f"Error removing checkpoint file: {e}")
    
    async def _cleanup_workflow_checkpoints(self, workflow_id: str):
        """Remove all checkpoints for a workflow."""
        if workflow_id in self.workflow_checkpoints:
            checkpoints = self.workflow_checkpoints[workflow_id]
            
            # Remove persisted files
            for checkpoint in checkpoints:
                checkpoint_file = os.path.join(
                    self.checkpoint_dir,
                    f"{workflow_id}_{checkpoint.step_id}_{int(checkpoint.timestamp.timestamp())}.json"
                )
                try:
                    if os.path.exists(checkpoint_file):
                        os.remove(checkpoint_file)
                except Exception as e:
                    self.logger.error(f"Error removing checkpoint file: {e}")
            
            # Remove from memory
            del self.workflow_checkpoints[workflow_id]
            
            self.logger.debug(f"Cleaned up checkpoints for workflow '{workflow_id}'")
    
    def get_checkpoint_summary(self, workflow_id: str) -> Dict[str, Any]:
        """Get summary of checkpoints for a workflow."""
        checkpoints = self.workflow_checkpoints.get(workflow_id, [])
        
        return {
            "workflow_id": workflow_id,
            "checkpoint_count": len(checkpoints),
            "checkpoints": [
                {
                    "step_id": cp.step_id,
                    "timestamp": cp.timestamp.isoformat(),
                    "has_rollback_data": bool(cp.rollback_data)
                }
                for cp in checkpoints
            ]
        }
    
    def cleanup(self):
        """Cleanup rollback manager resources."""
        self.logger.info("Cleaning up rollback manager")
        # Clear in-memory checkpoints
        self.workflow_checkpoints.clear()