"""
Workflow and WorkflowStep data models for the execution engine.
"""

import uuid
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import json
from datetime import datetime


class StepStatus(Enum):
    """Status enumeration for workflow steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


class WorkflowStatus(Enum):
    """Status enumeration for workflows."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


@dataclass
class StepResult:
    """Result data from step execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    rollback_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata or {},
            "execution_time": self.execution_time,
            "rollback_data": self.rollback_data or {}
        }


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    id: str
    name: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str] = None
    rollback_action: Optional[str] = None
    rollback_parameters: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    critical: bool = True
    parallel_group: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.dependencies is None:
            self.dependencies = []
        if self.rollback_parameters is None:
            self.rollback_parameters = {}
        
        # Runtime attributes
        self.status: StepStatus = StepStatus.PENDING
        self.result: Optional[StepResult] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_details: Optional[str] = None

    def can_execute(self, completed_steps: Set[str]) -> bool:
        """Check if step can be executed based on dependencies."""
        if self.conditions:
            # Check conditional execution
            if not self._evaluate_conditions():
                return False
        
        return all(dep in completed_steps for dep in self.dependencies)

    def _evaluate_conditions(self) -> bool:
        """Evaluate step conditions."""
        # Placeholder for condition evaluation logic
        # Could be extended to support complex conditions
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "action": self.action,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "rollback_action": self.rollback_action,
            "rollback_parameters": self.rollback_parameters,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "critical": self.critical,
            "parallel_group": self.parallel_group,
            "conditions": self.conditions,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_details": self.error_details
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowStep':
        """Create WorkflowStep from dictionary."""
        step = cls(
            id=data["id"],
            name=data["name"],
            action=data["action"],
            parameters=data["parameters"],
            dependencies=data.get("dependencies", []),
            rollback_action=data.get("rollback_action"),
            rollback_parameters=data.get("rollback_parameters", {}),
            timeout=data.get("timeout"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            critical=data.get("critical", True),
            parallel_group=data.get("parallel_group"),
            conditions=data.get("conditions")
        )
        
        # Set runtime attributes if present
        if "status" in data:
            step.status = StepStatus(data["status"])
        if "started_at" in data and data["started_at"]:
            step.started_at = datetime.fromisoformat(data["started_at"])
        if "completed_at" in data and data["completed_at"]:
            step.completed_at = datetime.fromisoformat(data["completed_at"])
        if "error_details" in data:
            step.error_details = data["error_details"]
            
        return step


@dataclass
class Workflow:
    """Complete workflow definition with steps and metadata."""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    metadata: Dict[str, Any] = None
    version: str = "1.0"
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
            
        # Runtime attributes
        self.status: WorkflowStatus = WorkflowStatus.CREATED
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.current_step: Optional[str] = None
        self.execution_log: List[Dict[str, Any]] = []

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_executable_steps(self, completed_steps: Set[str]) -> List[WorkflowStep]:
        """Get all steps that can be executed now."""
        return [
            step for step in self.steps
            if step.status == StepStatus.PENDING and step.can_execute(completed_steps)
        ]

    def get_parallel_groups(self) -> Dict[str, List[WorkflowStep]]:
        """Group steps by parallel execution group."""
        groups = {}
        for step in self.steps:
            if step.parallel_group:
                if step.parallel_group not in groups:
                    groups[step.parallel_group] = []
                groups[step.parallel_group].append(step)
        return groups

    def get_completion_percentage(self) -> float:
        """Calculate workflow completion percentage."""
        if not self.steps:
            return 0.0
        
        completed = sum(1 for step in self.steps if step.status == StepStatus.COMPLETED)
        return (completed / len(self.steps)) * 100.0

    def has_failed_critical_step(self) -> bool:
        """Check if any critical step has failed."""
        return any(
            step.critical and step.status == StepStatus.FAILED
            for step in self.steps
        )

    def validate(self) -> List[str]:
        """Validate workflow structure and return error messages."""
        errors = []
        
        # Check for duplicate step IDs
        step_ids = [step.id for step in self.steps]
        duplicates = set([x for x in step_ids if step_ids.count(x) > 1])
        if duplicates:
            errors.append(f"Duplicate step IDs found: {duplicates}")
        
        # Check dependency references
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step '{step.id}' has invalid dependency '{dep}'")
        
        # Check for circular dependencies (simplified check)
        # A more comprehensive check would use topological sorting
        for step in self.steps:
            if step.id in step.dependencies:
                errors.append(f"Step '{step.id}' has circular dependency on itself")
        
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_step": self.current_step,
            "execution_log": self.execution_log
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workflow':
        """Create Workflow from dictionary."""
        steps = [WorkflowStep.from_dict(step_data) for step_data in data["steps"]]
        
        workflow = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            steps=steps,
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
        )
        
        # Set runtime attributes if present
        if "status" in data:
            workflow.status = WorkflowStatus(data["status"])
        if "started_at" in data and data["started_at"]:
            workflow.started_at = datetime.fromisoformat(data["started_at"])
        if "completed_at" in data and data["completed_at"]:
            workflow.completed_at = datetime.fromisoformat(data["completed_at"])
        if "current_step" in data:
            workflow.current_step = data["current_step"]
        if "execution_log" in data:
            workflow.execution_log = data["execution_log"]
            
        return workflow

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Workflow':
        """Create Workflow from JSON string."""
        return cls.from_dict(json.loads(json_str))