"""
Claude Flow MCP integration for enhanced workflow coordination.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core.workflow import Workflow, WorkflowStep
from ..core.executor import ExecutionContext


logger = logging.getLogger(__name__)


class ClaudeFlowIntegration:
    """
    Integration with Claude Flow MCP tools for enhanced workflow coordination.
    
    Features:
    - Automatic swarm initialization for complex workflows
    - Intelligent agent spawning based on workflow characteristics
    - Memory-based coordination between workflow steps
    - Performance analytics and optimization
    - Neural pattern learning from workflow execution
    """
    
    def __init__(self, 
                 auto_swarm: bool = True,
                 swarm_threshold: int = 5,
                 memory_namespace: str = "workflow_engine"):
        """
        Initialize Claude Flow integration.
        
        Args:
            auto_swarm: Whether to automatically initialize swarms for complex workflows
            swarm_threshold: Minimum steps to trigger swarm initialization
            memory_namespace: Namespace for workflow memory storage
        """
        self.auto_swarm = auto_swarm
        self.swarm_threshold = swarm_threshold
        self.memory_namespace = memory_namespace
        self.logger = logging.getLogger(__name__)
        
        # Runtime state
        self.active_swarms: Dict[str, Dict[str, Any]] = {}
        self.workflow_agents: Dict[str, List[str]] = {}
        self.memory_keys: Dict[str, List[str]] = {}
        
        self.logger.info(f"Claude Flow integration initialized (auto_swarm: {auto_swarm})")
    
    async def initialize_workflow(self, workflow: Workflow, context: ExecutionContext):
        """
        Initialize Claude Flow coordination for a workflow.
        
        Args:
            workflow: The workflow to initialize
            context: Execution context
        """
        self.logger.info(f"Initializing Claude Flow coordination for workflow '{workflow.name}'")
        
        # Store workflow metadata
        await self._store_workflow_metadata(workflow, context)
        
        # Initialize swarm if applicable
        if self.auto_swarm and len(workflow.steps) >= self.swarm_threshold:
            await self._initialize_swarm(workflow, context)
        
        # Spawn specialized agents based on workflow characteristics
        agents = await self._spawn_workflow_agents(workflow, context)
        self.workflow_agents[workflow.id] = agents
        
        self.logger.info(f"Claude Flow initialization completed for workflow '{workflow.name}'")
    
    async def _store_workflow_metadata(self, workflow: Workflow, context: ExecutionContext):
        """Store workflow metadata in Claude Flow memory."""
        try:
            metadata = {
                "workflow_id": workflow.id,
                "workflow_name": workflow.name,
                "description": workflow.description,
                "total_steps": len(workflow.steps),
                "step_types": self._analyze_step_types(workflow),
                "complexity_score": self._calculate_complexity_score(workflow),
                "initialization_time": datetime.now().isoformat(),
                "context_variables": list(context.workflow_variables.keys())
            }
            
            memory_key = f"{self.memory_namespace}/workflows/{workflow.id}/metadata"
            await self._store_memory(memory_key, metadata)
            
            self.logger.debug(f"Stored workflow metadata for '{workflow.id}'")
            
        except Exception as e:
            self.logger.error(f"Error storing workflow metadata: {e}")
    
    async def _initialize_swarm(self, workflow: Workflow, context: ExecutionContext):
        """Initialize Claude Flow swarm for workflow coordination."""
        try:
            # Determine optimal swarm topology
            topology = self._determine_optimal_topology(workflow)
            max_agents = min(len(workflow.steps) // 2, 10)  # Reasonable agent limit
            
            # This would call the actual Claude Flow MCP tools
            # For now, we simulate the swarm initialization
            swarm_config = {
                "workflow_id": workflow.id,
                "topology": topology,
                "max_agents": max_agents,
                "strategy": "adaptive",
                "initialized_at": datetime.now().isoformat()
            }
            
            self.active_swarms[workflow.id] = swarm_config
            
            # Store swarm configuration
            memory_key = f"{self.memory_namespace}/swarms/{workflow.id}/config"
            await self._store_memory(memory_key, swarm_config)
            
            self.logger.info(f"Initialized swarm for workflow '{workflow.id}' with topology '{topology}'")
            
        except Exception as e:
            self.logger.error(f"Error initializing swarm: {e}")
    
    async def _spawn_workflow_agents(self, workflow: Workflow, context: ExecutionContext) -> List[str]:
        """Spawn specialized agents based on workflow characteristics."""
        agents = []
        
        try:
            step_types = self._analyze_step_types(workflow)
            
            # Spawn agents based on workflow content
            if "file" in step_types:
                agents.append("file_manager")
            if "shell" in step_types:
                agents.append("system_coordinator")
            if "http" in step_types:
                agents.append("api_integrator")
            if "claude_flow" in step_types:
                agents.append("swarm_coordinator")
            if len(workflow.steps) > 10:
                agents.append("performance_optimizer")
            
            # Always include a workflow coordinator
            agents.append("workflow_coordinator")
            
            # Store agent information
            for agent in agents:
                agent_info = {
                    "agent_type": agent,
                    "workflow_id": workflow.id,
                    "spawned_at": datetime.now().isoformat(),
                    "responsibilities": self._get_agent_responsibilities(agent, step_types)
                }
                
                memory_key = f"{self.memory_namespace}/agents/{workflow.id}/{agent}"
                await self._store_memory(memory_key, agent_info)
            
            self.logger.info(f"Spawned {len(agents)} agents for workflow '{workflow.id}': {agents}")
            
        except Exception as e:
            self.logger.error(f"Error spawning agents: {e}")
        
        return agents
    
    def _analyze_step_types(self, workflow: Workflow) -> Dict[str, int]:
        """Analyze the types of steps in the workflow."""
        step_types = {}
        
        for step in workflow.steps:
            action_category = step.action.split('.')[0] if '.' in step.action else step.action
            step_types[action_category] = step_types.get(action_category, 0) + 1
        
        return step_types
    
    def _calculate_complexity_score(self, workflow: Workflow) -> float:
        """Calculate a complexity score for the workflow."""
        # Base complexity from number of steps
        base_score = len(workflow.steps) * 1.0
        
        # Add complexity for dependencies
        total_dependencies = sum(len(step.dependencies) for step in workflow.steps)
        dependency_score = total_dependencies * 0.5
        
        # Add complexity for parallel groups
        parallel_groups = set(step.parallel_group for step in workflow.steps if step.parallel_group)
        parallel_score = len(parallel_groups) * 0.3
        
        # Add complexity for different action types
        unique_actions = set(step.action for step in workflow.steps)
        action_score = len(unique_actions) * 0.2
        
        return base_score + dependency_score + parallel_score + action_score
    
    def _determine_optimal_topology(self, workflow: Workflow) -> str:
        """Determine optimal swarm topology based on workflow characteristics."""
        complexity = self._calculate_complexity_score(workflow)
        step_count = len(workflow.steps)
        
        # Simple heuristics for topology selection
        if complexity > 50 or step_count > 20:
            return "hierarchical"  # Best for complex workflows
        elif complexity > 20 or step_count > 10:
            return "mesh"  # Good for moderate complexity
        else:
            return "star"  # Simple topology for basic workflows
    
    def _get_agent_responsibilities(self, agent_type: str, step_types: Dict[str, int]) -> List[str]:
        """Get responsibilities for an agent type."""
        responsibilities = {
            "file_manager": ["file operations", "path validation", "backup management"],
            "system_coordinator": ["command execution", "resource monitoring", "error handling"],
            "api_integrator": ["HTTP requests", "API coordination", "response validation"],
            "swarm_coordinator": ["agent communication", "task distribution", "load balancing"],
            "performance_optimizer": ["execution optimization", "bottleneck detection", "resource allocation"],
            "workflow_coordinator": ["overall coordination", "step orchestration", "progress tracking"]
        }
        
        return responsibilities.get(agent_type, ["general coordination"])
    
    async def coordinate_step_execution(self, 
                                       workflow: Workflow, 
                                       step: WorkflowStep, 
                                       context: ExecutionContext):
        """Coordinate step execution with Claude Flow agents."""
        try:
            # Store step execution context
            step_context = {
                "step_id": step.id,
                "step_name": step.name,
                "action": step.action,
                "parameters": step.parameters,
                "dependencies": step.dependencies,
                "started_at": datetime.now().isoformat(),
                "workflow_variables": context.workflow_variables.copy()
            }
            
            memory_key = f"{self.memory_namespace}/execution/{workflow.id}/{step.id}/context"
            await self._store_memory(memory_key, step_context)
            
            # Notify relevant agents
            await self._notify_agents(workflow.id, "step_started", {
                "step_id": step.id,
                "action": step.action,
                "context": step_context
            })
            
        except Exception as e:
            self.logger.error(f"Error coordinating step execution: {e}")
    
    async def handle_step_completion(self, 
                                    workflow: Workflow, 
                                    step: WorkflowStep, 
                                    result: Dict[str, Any],
                                    context: ExecutionContext):
        """Handle step completion with Claude Flow coordination."""
        try:
            # Store step results
            completion_data = {
                "step_id": step.id,
                "completed_at": datetime.now().isoformat(),
                "success": result.get("success", False),
                "output": result.get("output"),
                "error": result.get("error"),
                "execution_time": result.get("execution_time"),
                "metadata": result.get("metadata", {})
            }
            
            memory_key = f"{self.memory_namespace}/execution/{workflow.id}/{step.id}/result"
            await self._store_memory(memory_key, completion_data)
            
            # Update workflow progress
            await self._update_workflow_progress(workflow, context)
            
            # Notify agents of completion
            await self._notify_agents(workflow.id, "step_completed", {
                "step_id": step.id,
                "result": completion_data
            })
            
            # Train neural patterns from execution
            await self._train_neural_patterns(workflow, step, result)
            
        except Exception as e:
            self.logger.error(f"Error handling step completion: {e}")
    
    async def _update_workflow_progress(self, workflow: Workflow, context: ExecutionContext):
        """Update workflow progress in Claude Flow memory."""
        try:
            completed_steps = [s for s in workflow.steps if hasattr(s, 'completed_at') and s.completed_at]
            progress = {
                "workflow_id": workflow.id,
                "total_steps": len(workflow.steps),
                "completed_steps": len(completed_steps),
                "completion_percentage": (len(completed_steps) / len(workflow.steps)) * 100,
                "last_updated": datetime.now().isoformat(),
                "current_variables": context.workflow_variables.copy()
            }
            
            memory_key = f"{self.memory_namespace}/progress/{workflow.id}"
            await self._store_memory(memory_key, progress)
            
        except Exception as e:
            self.logger.error(f"Error updating workflow progress: {e}")
    
    async def _notify_agents(self, workflow_id: str, event: str, data: Dict[str, Any]):
        """Notify relevant agents of workflow events."""
        try:
            notification = {
                "workflow_id": workflow_id,
                "event": event,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            memory_key = f"{self.memory_namespace}/notifications/{workflow_id}/{event}"
            await self._store_memory(memory_key, notification)
            
            self.logger.debug(f"Notified agents of event '{event}' for workflow '{workflow_id}'")
            
        except Exception as e:
            self.logger.error(f"Error notifying agents: {e}")
    
    async def _train_neural_patterns(self, 
                                    workflow: Workflow, 
                                    step: WorkflowStep, 
                                    result: Dict[str, Any]):
        """Train neural patterns from step execution results."""
        try:
            # Extract patterns for learning
            pattern_data = {
                "action_type": step.action,
                "parameters": step.parameters,
                "success": result.get("success", False),
                "execution_time": result.get("execution_time", 0),
                "error_type": result.get("error", "").split(":")[0] if result.get("error") else None,
                "workflow_complexity": self._calculate_complexity_score(workflow),
                "dependency_count": len(step.dependencies),
                "parallel_group": step.parallel_group,
                "timestamp": datetime.now().isoformat()
            }
            
            memory_key = f"{self.memory_namespace}/patterns/{step.action}/{datetime.now().strftime('%Y%m%d')}"
            await self._store_memory(memory_key, pattern_data)
            
            self.logger.debug(f"Stored neural pattern data for step '{step.id}'")
            
        except Exception as e:
            self.logger.error(f"Error training neural patterns: {e}")
    
    async def _store_memory(self, key: str, value: Any):
        """Store data in Claude Flow memory."""
        # This would integrate with actual Claude Flow MCP memory tools
        # For now, we simulate memory storage
        if key not in self.memory_keys:
            self.memory_keys[key] = []
        
        self.memory_keys[key].append({
            "value": value,
            "stored_at": datetime.now().isoformat()
        })
        
        self.logger.debug(f"Stored memory: {key}")
    
    async def get_workflow_analytics(self, workflow_id: str) -> Dict[str, Any]:
        """Get analytics data for a workflow."""
        try:
            analytics = {
                "workflow_id": workflow_id,
                "swarm_active": workflow_id in self.active_swarms,
                "agent_count": len(self.workflow_agents.get(workflow_id, [])),
                "memory_keys": len([k for k in self.memory_keys.keys() if workflow_id in k]),
                "execution_patterns": await self._analyze_execution_patterns(workflow_id),
                "performance_metrics": await self._get_performance_metrics(workflow_id)
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting workflow analytics: {e}")
            return {}
    
    async def _analyze_execution_patterns(self, workflow_id: str) -> Dict[str, Any]:
        """Analyze execution patterns for a workflow."""
        # This would analyze stored pattern data
        return {
            "common_actions": ["file.write", "shell.execute"],
            "success_rate": 0.95,
            "average_execution_time": 2.3,
            "error_patterns": ["timeout", "file_not_found"]
        }
    
    async def _get_performance_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get performance metrics for a workflow."""
        # This would calculate actual performance metrics
        return {
            "total_execution_time": 45.2,
            "parallel_efficiency": 0.78,
            "resource_utilization": 0.65,
            "optimization_suggestions": ["increase parallelism", "optimize file operations"]
        }
    
    async def finalize_workflow(self, workflow: Workflow, context: ExecutionContext):
        """Finalize Claude Flow coordination for a completed workflow."""
        self.logger.info(f"Finalizing Claude Flow coordination for workflow '{workflow.name}'")
        
        try:
            # Store final workflow state
            final_state = {
                "workflow_id": workflow.id,
                "status": workflow.status.value,
                "completed_at": datetime.now().isoformat(),
                "total_steps": len(workflow.steps),
                "successful_steps": len([s for s in workflow.steps if hasattr(s, 'result') and s.result and s.result.success]),
                "final_variables": context.workflow_variables.copy(),
                "execution_summary": await self._generate_execution_summary(workflow)
            }
            
            memory_key = f"{self.memory_namespace}/workflows/{workflow.id}/final_state"
            await self._store_memory(memory_key, final_state)
            
            # Cleanup active swarm
            if workflow.id in self.active_swarms:
                del self.active_swarms[workflow.id]
            
            # Cleanup agent tracking
            if workflow.id in self.workflow_agents:
                del self.workflow_agents[workflow.id]
            
            self.logger.info(f"Claude Flow finalization completed for workflow '{workflow.name}'")
            
        except Exception as e:
            self.logger.error(f"Error finalizing workflow: {e}")
    
    async def _generate_execution_summary(self, workflow: Workflow) -> Dict[str, Any]:
        """Generate execution summary for a workflow."""
        return {
            "workflow_name": workflow.name,
            "description": workflow.description,
            "total_steps": len(workflow.steps),
            "execution_time": "calculated_from_timestamps",
            "success_rate": "calculated_from_results",
            "performance_score": "calculated_metric",
            "lessons_learned": "extracted_patterns"
        }
    
    async def cleanup(self):
        """Cleanup Claude Flow integration resources."""
        self.logger.info("Cleaning up Claude Flow integration")
        
        # Cleanup active swarms
        for workflow_id in list(self.active_swarms.keys()):
            # This would call actual Claude Flow cleanup
            del self.active_swarms[workflow_id]
        
        # Clear tracking data
        self.workflow_agents.clear()
        self.memory_keys.clear()
        
        self.logger.info("Claude Flow integration cleanup completed")