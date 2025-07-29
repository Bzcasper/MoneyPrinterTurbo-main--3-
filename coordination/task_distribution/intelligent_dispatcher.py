"""
Intelligent Task Distribution System for Hive-Mind Coordination
Optimizes task assignment based on agent capabilities, workload, and performance
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Task:
    """Task definition with metadata"""
    task_id: str
    task_type: str
    priority: TaskPriority
    requirements: Dict[str, Any]
    data: Dict[str, Any]
    deadline: Optional[datetime] = None
    max_retries: int = 3
    retry_count: int = 0
    estimated_duration: int = 300  # seconds
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class AgentCapability:
    """Agent capability profile"""
    agent_id: str
    agent_type: str
    specializations: List[str]
    current_load: float  # 0.0 to 1.0
    performance_score: float  # 0.0 to 1.0
    avg_completion_time: float  # seconds
    success_rate: float  # 0.0 to 1.0
    last_task_completed: Optional[datetime] = None
    preferred_task_types: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3


class IntelligentTaskDispatcher:
    """Advanced task distribution system with ML-inspired optimization"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.task_queue: Dict[str, Task] = {}
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.task_history: List[Task] = []
        self.assignment_lock = asyncio.Lock()
        
        # Performance tracking
        self.assignment_metrics = {
            "total_assigned": 0,
            "successful_completions": 0,
            "failed_assignments": 0,
            "avg_assignment_time": 0.0,
            "load_balance_score": 1.0
        }
        
        # Learning parameters for optimization
        self.learning_rate = 0.1
        self.assignment_weights = {
            "capability_match": 0.4,
            "current_load": 0.3,
            "performance_score": 0.2,
            "completion_time": 0.1
        }
    
    async def register_agent(self, agent_id: str, agent_type: str, 
                           specializations: List[str], max_concurrent: int = 3):
        """Register an agent with the dispatcher"""
        try:
            capability = AgentCapability(
                agent_id=agent_id,
                agent_type=agent_type,
                specializations=specializations,
                current_load=0.0,
                performance_score=0.8,  # Start with good score
                avg_completion_time=300.0,
                success_rate=1.0,
                max_concurrent_tasks=max_concurrent,
                preferred_task_types=specializations
            )
            
            self.agent_capabilities[agent_id] = capability
            logger.info(f"Registered agent {agent_id} with specializations: {specializations}")
            
        except Exception as e:
            logger.error(f"Error registering agent {agent_id}: {e}")
    
    async def submit_task(self, task: Task) -> bool:
        """Submit a new task for processing"""
        try:
            async with self.assignment_lock:
                self.task_queue[task.task_id] = task
                logger.info(f"Submitted task {task.task_id} with priority {task.priority.value}")
                
                # Attempt immediate assignment if possible
                await self._attempt_immediate_assignment(task)
                
                return True
                
        except Exception as e:
            logger.error(f"Error submitting task {task.task_id}: {e}")
            return False
    
    async def _attempt_immediate_assignment(self, task: Task):
        """Try to assign task immediately if suitable agent available"""
        try:
            best_agent = await self._find_best_agent_for_task(task)
            if best_agent and self._can_agent_accept_task(best_agent, task):
                await self._assign_task_to_agent(task, best_agent)
        except Exception as e:
            logger.debug(f"Could not immediately assign task {task.task_id}: {e}")
    
    async def _find_best_agent_for_task(self, task: Task) -> Optional[str]:
        """Find the best agent for a given task using scoring algorithm"""
        try:
            if not self.agent_capabilities:
                return None
            
            best_agent = None
            best_score = -1.0
            
            for agent_id, capability in self.agent_capabilities.items():
                if not self._can_agent_accept_task(capability, task):
                    continue
                
                score = await self._calculate_assignment_score(capability, task)
                
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
            
            logger.debug(f"Best agent for task {task.task_id}: {best_agent} (score: {best_score:.3f})")
            return best_agent
            
        except Exception as e:
            logger.error(f"Error finding best agent for task {task.task_id}: {e}")
            return None
    
    def _can_agent_accept_task(self, capability: AgentCapability, task: Task) -> bool:
        """Check if agent can accept the task based on capacity and requirements"""
        try:
            # Check load capacity
            if capability.current_load >= 1.0:
                return False
            
            # Check if agent has required specializations
            required_specs = task.requirements.get("specializations", [])
            if required_specs and not any(spec in capability.specializations for spec in required_specs):
                return False
            
            # Check if agent type matches if specified
            required_type = task.requirements.get("agent_type")
            if required_type and capability.agent_type != required_type:
                return False
            
            # Check deadline constraints
            if task.deadline:
                estimated_completion = datetime.now() + timedelta(seconds=capability.avg_completion_time)
                if estimated_completion > task.deadline:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking agent capability: {e}")
            return False
    
    async def _calculate_assignment_score(self, capability: AgentCapability, task: Task) -> float:
        """Calculate assignment score for agent-task pair"""
        try:
            # Capability match score
            capability_score = 0.0
            required_specs = task.requirements.get("specializations", [])
            if required_specs:
                matches = sum(1 for spec in required_specs if spec in capability.specializations)
                capability_score = matches / len(required_specs)
            else:
                # If no specific requirements, check preferred types
                if task.task_type in capability.preferred_task_types:
                    capability_score = 1.0
                else:
                    capability_score = 0.5
            
            # Load balancing score (prefer less loaded agents)
            load_score = 1.0 - capability.current_load
            
            # Performance score (use historical performance)
            performance_score = capability.performance_score
            
            # Completion time score (prefer faster agents for urgent tasks)
            if task.priority == TaskPriority.CRITICAL:
                time_score = 1.0 / max(1.0, capability.avg_completion_time / 300.0)
            else:
                time_score = 0.5  # Less weight for non-critical tasks
            
            # Priority boost for critical tasks
            priority_multiplier = 1.0
            if task.priority == TaskPriority.CRITICAL:
                priority_multiplier = 1.5
            elif task.priority == TaskPriority.HIGH:
                priority_multiplier = 1.2
            
            # Weighted combination
            final_score = (
                capability_score * self.assignment_weights["capability_match"] +
                load_score * self.assignment_weights["current_load"] +
                performance_score * self.assignment_weights["performance_score"] +
                time_score * self.assignment_weights["completion_time"]
            ) * priority_multiplier
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error(f"Error calculating assignment score: {e}")
            return 0.0
    
    async def _assign_task_to_agent(self, task: Task, agent_id: str):
        """Assign task to specific agent"""
        try:
            capability = self.agent_capabilities[agent_id]
            
            # Update task status
            task.status = TaskStatus.ASSIGNED
            task.assigned_agent = agent_id
            task.assigned_at = datetime.now()
            
            # Update agent load
            task_load = min(0.5, task.estimated_duration / 600.0)  # Cap at 0.5 load per task
            capability.current_load = min(1.0, capability.current_load + task_load)
            
            # Update metrics
            self.assignment_metrics["total_assigned"] += 1
            
            logger.info(f"Assigned task {task.task_id} to agent {agent_id} (load: {capability.current_load:.2f})")
            
            # Start task execution monitoring
            asyncio.create_task(self._monitor_task_execution(task))
            
        except Exception as e:
            logger.error(f"Error assigning task {task.task_id} to agent {agent_id}: {e}")
            task.status = TaskStatus.PENDING
    
    async def _monitor_task_execution(self, task: Task):
        """Monitor task execution and handle completion/failure"""
        try:
            # Simulate task execution monitoring
            start_time = time.time()
            timeout = task.estimated_duration * 2  # Allow 2x estimated time
            
            while task.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Check for timeout
                if time.time() - start_time > timeout:
                    await self._handle_task_timeout(task)
                    break
            
            # Handle completion
            if task.status == TaskStatus.COMPLETED:
                await self._handle_task_completion(task)
            elif task.status == TaskStatus.FAILED:
                await self._handle_task_failure(task)
                
        except Exception as e:
            logger.error(f"Error monitoring task {task.task_id}: {e}")
    
    async def _handle_task_completion(self, task: Task):
        """Handle successful task completion"""
        try:
            if not task.assigned_agent:
                return
            
            capability = self.agent_capabilities[task.assigned_agent]
            
            # Calculate actual completion time
            if task.assigned_at and task.completed_at:
                actual_time = (task.completed_at - task.assigned_at).total_seconds()
                
                # Update agent performance metrics
                capability.avg_completion_time = (
                    capability.avg_completion_time * 0.8 + actual_time * 0.2
                )
                capability.last_task_completed = task.completed_at
                
                # Update success rate
                capability.success_rate = min(1.0, capability.success_rate * 0.95 + 0.05)
                
                # Update performance score
                time_efficiency = task.estimated_duration / max(1.0, actual_time)
                capability.performance_score = min(1.0, 
                    capability.performance_score * 0.9 + time_efficiency * 0.1
                )
            
            # Release agent load
            task_load = min(0.5, task.estimated_duration / 600.0)
            capability.current_load = max(0.0, capability.current_load - task_load)
            
            # Update global metrics
            self.assignment_metrics["successful_completions"] += 1
            
            # Move to history
            self.task_history.append(task)
            if task.task_id in self.task_queue:
                del self.task_queue[task.task_id]
            
            logger.info(f"Task {task.task_id} completed successfully by agent {task.assigned_agent}")
            
            # Trigger rebalancing if needed
            await self._rebalance_workload()
            
        except Exception as e:
            logger.error(f"Error handling task completion {task.task_id}: {e}")
    
    async def _handle_task_failure(self, task: Task):
        """Handle task failure and potential retry"""
        try:
            if not task.assigned_agent:
                return
            
            capability = self.agent_capabilities[task.assigned_agent]
            
            # Update agent failure metrics
            capability.success_rate = max(0.1, capability.success_rate * 0.9)
            capability.performance_score = max(0.1, capability.performance_score * 0.9)
            
            # Release agent load
            task_load = min(0.5, task.estimated_duration / 600.0)
            capability.current_load = max(0.0, capability.current_load - task_load)
            
            # Update global metrics
            self.assignment_metrics["failed_assignments"] += 1
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                task.assigned_agent = None
                task.assigned_at = None
                
                logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                
                # Wait before retry and attempt reassignment
                await asyncio.sleep(min(60, task.retry_count * 10))
                await self._attempt_immediate_assignment(task)
            else:
                logger.error(f"Task {task.task_id} failed permanently after {task.max_retries} retries")
                self.task_history.append(task)
                if task.task_id in self.task_queue:
                    del self.task_queue[task.task_id]
            
        except Exception as e:
            logger.error(f"Error handling task failure {task.task_id}: {e}")
    
    async def _handle_task_timeout(self, task: Task):
        """Handle task timeout"""
        try:
            logger.warning(f"Task {task.task_id} timed out")
            task.status = TaskStatus.FAILED
            task.error_message = "Task execution timeout"
            await self._handle_task_failure(task)
        except Exception as e:
            logger.error(f"Error handling task timeout {task.task_id}: {e}")
    
    async def _rebalance_workload(self):
        """Rebalance workload across agents"""
        try:
            # Check for overloaded and underloaded agents
            overloaded = [cap for cap in self.agent_capabilities.values() if cap.current_load > 0.8]
            underloaded = [cap for cap in self.agent_capabilities.values() if cap.current_load < 0.3]
            
            if overloaded and underloaded:
                logger.info("Performing workload rebalancing")
                # Implementation would move tasks from overloaded to underloaded agents
                # For now, just log the potential rebalancing
            
            # Update load balance score
            loads = [cap.current_load for cap in self.agent_capabilities.values()]
            if loads:
                load_variance = sum((load - sum(loads)/len(loads))**2 for load in loads) / len(loads)
                self.assignment_metrics["load_balance_score"] = max(0.0, 1.0 - load_variance)
            
        except Exception as e:
            logger.error(f"Error in workload rebalancing: {e}")
    
    async def get_dispatcher_status(self) -> Dict[str, Any]:
        """Get comprehensive dispatcher status"""
        try:
            return {
                "session_id": self.session_id,
                "task_queue_size": len(self.task_queue),
                "registered_agents": len(self.agent_capabilities),
                "task_history_size": len(self.task_history),
                "assignment_metrics": self.assignment_metrics.copy(),
                "agent_loads": {
                    agent_id: {
                        "current_load": cap.current_load,
                        "performance_score": cap.performance_score,
                        "success_rate": cap.success_rate
                    }
                    for agent_id, cap in self.agent_capabilities.items()
                },
                "pending_tasks": len([t for t in self.task_queue.values() if t.status == TaskStatus.PENDING]),
                "active_tasks": len([t for t in self.task_queue.values() if t.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]])
            }
        except Exception as e:
            logger.error(f"Error getting dispatcher status: {e}")
            return {"error": str(e)}


# Global dispatcher instance
_task_dispatcher: Optional[IntelligentTaskDispatcher] = None


def get_task_dispatcher(session_id: str) -> IntelligentTaskDispatcher:
    """Get or create the global task dispatcher"""
    global _task_dispatcher
    if _task_dispatcher is None:
        _task_dispatcher = IntelligentTaskDispatcher(session_id)
    return _task_dispatcher