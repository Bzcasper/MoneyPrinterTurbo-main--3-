"""
Comprehensive Workflow Monitoring and Progress Tracking System

This module provides real-time workflow monitoring with progress tracking,
performance metrics, error detection, and visual CLI indicators.
Integrates with Claude Flow memory system for persistence.

Features:
- Real-time progress tracking with percentage completion
- Step-by-step execution status reporting  
- Performance metrics collection (timing, resource usage)
- Error detection and alerting mechanisms
- Workflow execution history and logs
- Integration with Claude Flow memory system
- Visual progress indicators for CLI output
"""

import time
import threading
import uuid
import json
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import statistics
import os
import sqlite3
from pathlib import Path

from loguru import logger
import psutil

from app.config import config
from app.services.hive_memory import HiveMemory


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Individual step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class WorkflowStep:
    """Individual workflow step"""
    step_id: str
    name: str
    description: str
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    progress: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkflowAlert:
    """Workflow monitoring alert"""
    alert_id: str
    workflow_id: str
    step_id: Optional[str]
    severity: AlertSeverity
    message: str
    timestamp: float
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance measurement point"""
    metric_id: str
    workflow_id: str
    step_id: Optional[str]
    metric_name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Complete workflow execution state"""
    workflow_id: str
    name: str
    description: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    progress: float = 0.0
    steps: Dict[str, WorkflowStep] = field(default_factory=dict)
    alerts: List[WorkflowAlert] = field(default_factory=list)
    metrics: List[PerformanceMetric] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow"""
        self.steps[step.step_id] = step
        self.total_steps = len(self.steps)
        self._update_progress()
    
    def update_step_status(self, step_id: str, status: StepStatus, 
                          progress: float = None, error_message: str = None) -> None:
        """Update step status and recalculate workflow progress"""
        if step_id not in self.steps:
            return
            
        step = self.steps[step_id]
        old_status = step.status
        step.status = status
        
        if progress is not None:
            step.progress = min(100.0, max(0.0, progress))
            
        if error_message:
            step.error_message = error_message
            
        # Update timing
        current_time = time.time()
        if status == StepStatus.RUNNING and old_status == StepStatus.PENDING:
            step.start_time = current_time
        elif status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED]:
            if step.start_time:
                step.end_time = current_time
                step.duration = current_time - step.start_time
            
        self._update_progress()
    
    def _update_progress(self) -> None:
        """Update overall workflow progress"""
        if not self.steps:
            self.progress = 0.0
            return
            
        total_progress = sum(step.progress for step in self.steps.values())
        self.progress = total_progress / len(self.steps)
        
        # Update counts
        self.completed_steps = sum(1 for step in self.steps.values() 
                                 if step.status == StepStatus.COMPLETED)
        self.failed_steps = sum(1 for step in self.steps.values() 
                              if step.status == StepStatus.FAILED)


class WorkflowMonitor:
    """
    Comprehensive workflow monitoring system with real-time tracking,
    performance metrics, and alert management.
    """
    
    def __init__(self, memory_system: Optional[HiveMemory] = None):
        """Initialize workflow monitor"""
        self.workflows: Dict[str, WorkflowExecution] = {}
        self.alerts: deque = deque(maxlen=1000)
        self.metrics: deque = deque(maxlen=10000)
        
        # Memory system for persistence
        self.memory = memory_system or HiveMemory()
        
        # Performance tracking
        self.system_metrics: Dict[str, float] = {}
        self.resource_thresholds = {
            'memory_percent': 85.0,
            'cpu_percent': 90.0,
            'disk_percent': 95.0
        }
        
        # Alert handlers
        self.alert_handlers: List[Callable[[WorkflowAlert], None]] = []
        
        # Background monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._lock = threading.RLock()
        
        # Database for persistent storage
        self.db_path = Path("workflow_monitor.db")
        self._init_database()
        
        logger.info("WorkflowMonitor initialized successfully")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for workflow persistence"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Workflows table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    duration REAL,
                    total_steps INTEGER,
                    completed_steps INTEGER,
                    failed_steps INTEGER,
                    progress REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Steps table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_steps (
                    step_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    duration REAL,
                    progress REAL,
                    dependencies TEXT,
                    metadata TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_alerts (
                    alert_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    step_id TEXT,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
                )
            """)
            
            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_metrics (
                    metric_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    step_id TEXT,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    timestamp REAL NOT NULL,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread"""
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Workflow monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        logger.info("Workflow monitoring stopped")
    
    def create_workflow(self, name: str, description: str = "", 
                       metadata: Dict[str, Any] = None) -> str:
        """Create a new workflow for monitoring"""
        workflow_id = str(uuid.uuid4())
        
        workflow = WorkflowExecution(
            workflow_id=workflow_id,
            name=name,
            description=description,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.workflows[workflow_id] = workflow
            
        # Store in memory system
        self._store_workflow_in_memory(workflow)
        
        # Persist to database
        self._persist_workflow(workflow)
        
        logger.info(f"Created workflow: {name} ({workflow_id})")
        return workflow_id
    
    def add_workflow_step(self, workflow_id: str, step_name: str, 
                         description: str = "", dependencies: List[str] = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Add a step to an existing workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        step_id = str(uuid.uuid4())
        step = WorkflowStep(
            step_id=step_id,
            name=step_name,
            description=description,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        with self._lock:
            workflow = self.workflows[workflow_id]
            workflow.add_step(step)
            
        # Store in memory
        self._store_step_in_memory(workflow_id, step)
        
        # Persist to database
        self._persist_step(workflow_id, step)
        
        logger.debug(f"Added step '{step_name}' to workflow {workflow_id}")
        return step_id
    
    def start_workflow(self, workflow_id: str) -> None:
        """Start workflow execution"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        with self._lock:
            workflow = self.workflows[workflow_id]
            workflow.status = WorkflowStatus.RUNNING
            workflow.start_time = time.time()
            
        self._persist_workflow(workflow)
        
        # Create start alert
        self._create_alert(
            workflow_id=workflow_id,
            severity=AlertSeverity.INFO,
            message=f"Workflow '{workflow.name}' started"
        )
        
        logger.info(f"Started workflow: {workflow.name} ({workflow_id})")
    
    def start_step(self, workflow_id: str, step_id: str) -> None:
        """Start execution of a specific step"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflows[workflow_id]
        if step_id not in workflow.steps:
            raise ValueError(f"Step {step_id} not found in workflow")
            
        with self._lock:
            workflow.update_step_status(step_id, StepStatus.RUNNING, 0.0)
            
        # Store in memory
        self._store_step_in_memory(workflow_id, workflow.steps[step_id])
        
        # Persist to database
        self._persist_step(workflow_id, workflow.steps[step_id])
        
        step_name = workflow.steps[step_id].name
        logger.info(f"Started step: {step_name} ({step_id})")
    
    def update_step_progress(self, workflow_id: str, step_id: str, 
                           progress: float, message: str = None) -> None:
        """Update progress of a running step"""
        if workflow_id not in self.workflows:
            return
            
        workflow = self.workflows[workflow_id]
        if step_id not in workflow.steps:
            return
            
        with self._lock:
            workflow.update_step_status(step_id, StepStatus.RUNNING, progress)
            if message:
                workflow.steps[step_id].metadata['last_message'] = message
                
        # Store progress in memory
        self._store_step_progress(workflow_id, step_id, progress, message)
        
        # Persist to database
        self._persist_step(workflow_id, workflow.steps[step_id])
        
        logger.debug(f"Step {step_id} progress: {progress:.1f}%")
    
    def complete_step(self, workflow_id: str, step_id: str, 
                     result: Dict[str, Any] = None) -> None:
        """Mark a step as completed"""
        if workflow_id not in self.workflows:
            return
            
        workflow = self.workflows[workflow_id]
        if step_id not in workflow.steps:
            return
            
        with self._lock:
            workflow.update_step_status(step_id, StepStatus.COMPLETED, 100.0)
            if result:
                workflow.steps[step_id].metadata['result'] = result
                
        # Store completion in memory
        self._store_step_completion(workflow_id, step_id, True, result)
        
        # Persist to database
        self._persist_step(workflow_id, workflow.steps[step_id])
        
        # Check if workflow is complete
        self._check_workflow_completion(workflow_id)
        
        step_name = workflow.steps[step_id].name
        logger.info(f"Completed step: {step_name} ({step_id})")
    
    def fail_step(self, workflow_id: str, step_id: str, 
                  error_message: str, retry: bool = True) -> None:
        """Mark a step as failed"""
        if workflow_id not in self.workflows:
            return
            
        workflow = self.workflows[workflow_id]
        if step_id not in workflow.steps:
            return
            
        step = workflow.steps[step_id]
        
        with self._lock:
            # Check if we should retry
            if retry and step.retry_count < step.max_retries:
                step.retry_count += 1
                step.status = StepStatus.PENDING
                step.error_message = None
                logger.warning(f"Step {step_id} failed, retrying ({step.retry_count}/{step.max_retries})")
            else:
                workflow.update_step_status(step_id, StepStatus.FAILED, error_message=error_message)
                
                # Create failure alert
                self._create_alert(
                    workflow_id=workflow_id,
                    step_id=step_id,
                    severity=AlertSeverity.ERROR,
                    message=f"Step '{step.name}' failed: {error_message}"
                )
                
        # Store failure in memory
        self._store_step_completion(workflow_id, step_id, False, {"error": error_message})
        
        # Persist to database
        self._persist_step(workflow_id, step)
        
        # Check if workflow should fail
        self._check_workflow_failure(workflow_id)
        
        logger.error(f"Failed step: {step.name} ({step_id}) - {error_message}")
    
    def record_metric(self, workflow_id: str, metric_name: str, value: float,
                     unit: str = "", step_id: str = None, tags: Dict[str, str] = None) -> None:
        """Record a performance metric"""
        metric = PerformanceMetric(
            metric_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            step_id=step_id,
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        with self._lock:
            self.metrics.append(metric)
            if workflow_id in self.workflows:
                self.workflows[workflow_id].metrics.append(metric)
                
        # Store in memory
        self._store_metric_in_memory(metric)
        
        # Persist to database
        self._persist_metric(metric)
        
        logger.debug(f"Recorded metric: {metric_name} = {value} {unit}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        if workflow_id not in self.workflows:
            return None
            
        workflow = self.workflows[workflow_id]
        
        return {
            'workflow_id': workflow_id,
            'name': workflow.name,
            'status': workflow.status.value,
            'progress': workflow.progress,
            'total_steps': workflow.total_steps,
            'completed_steps': workflow.completed_steps,
            'failed_steps': workflow.failed_steps,
            'start_time': workflow.start_time,
            'duration': time.time() - workflow.start_time if workflow.start_time else None,
            'steps': {
                step_id: {
                    'name': step.name,
                    'status': step.status.value,
                    'progress': step.progress,
                    'duration': step.duration
                }
                for step_id, step in workflow.steps.items()
            }
        }
    
    def get_workflow_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get execution history for a workflow"""
        history = []
        
        # Get from memory system
        memory_key = f"workflow_history_{workflow_id}"
        stored_history = self.memory.retrieve_swarm_memory(memory_key)
        if stored_history:
            history.extend(json.loads(stored_history))
            
        # Get from database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT w.*, GROUP_CONCAT(s.name) as step_names
                FROM workflows w
                LEFT JOIN workflow_steps s ON w.workflow_id = s.workflow_id
                WHERE w.workflow_id = ?
                GROUP BY w.workflow_id
                ORDER BY w.created_at DESC
            """, (workflow_id,))
            
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                db_record = dict(zip(columns, row))
                history.append(db_record)
                
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to get workflow history: {str(e)}")
            
        return history
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all currently active workflows"""
        active = []
        
        with self._lock:
            for workflow_id, workflow in self.workflows.items():
                if workflow.status == WorkflowStatus.RUNNING:
                    active.append(self.get_workflow_status(workflow_id))
                    
        return active
    
    def get_recent_alerts(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent workflow alerts"""
        with self._lock:
            recent_alerts = list(self.alerts)[-count:]
            
        return [asdict(alert) for alert in recent_alerts]
    
    def get_performance_summary(self, workflow_id: str = None) -> Dict[str, Any]:
        """Get performance metrics summary"""
        relevant_metrics = []
        
        with self._lock:
            if workflow_id:
                relevant_metrics = [m for m in self.metrics if m.workflow_id == workflow_id]
            else:
                relevant_metrics = list(self.metrics)
                
        if not relevant_metrics:
            return {}
            
        # Group by metric name
        grouped_metrics = defaultdict(list)
        for metric in relevant_metrics:
            grouped_metrics[metric.metric_name].append(metric.value)
            
        summary = {}
        for name, values in grouped_metrics.items():
            summary[name] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'latest': values[-1] if values else 0
            }
            
            if len(values) > 1:
                summary[name]['std_dev'] = statistics.stdev(values)
                
        return summary
    
    def add_alert_handler(self, handler: Callable[[WorkflowAlert], None]) -> None:
        """Add a custom alert handler"""
        self.alert_handlers.append(handler)
    
    def generate_progress_report(self, workflow_id: str) -> str:
        """Generate a visual progress report for CLI display"""
        if workflow_id not in self.workflows:
            return f"Workflow {workflow_id} not found"
            
        workflow = self.workflows[workflow_id]
        
        # Header
        report = f"\n{'='*60}\n"
        report += f"📊 WORKFLOW PROGRESS REPORT\n"
        report += f"{'='*60}\n\n"
        
        # Workflow overview
        report += f"🎯 Name: {workflow.name}\n"
        report += f"📋 Status: {workflow.status.value.upper()}\n"
        report += f"📈 Progress: {workflow.progress:.1f}%\n"
        report += f"⏱️  Duration: {workflow.duration or 0:.1f}s\n"
        report += f"📦 Steps: {workflow.completed_steps}/{workflow.total_steps}\n\n"
        
        # Progress bar
        progress_bar_width = 40
        filled_width = int(progress_bar_width * workflow.progress / 100)
        progress_bar = "█" * filled_width + "░" * (progress_bar_width - filled_width)
        report += f"Progress: [{progress_bar}] {workflow.progress:.1f}%\n\n"
        
        # Step details
        report += f"📋 STEP DETAILS\n"
        report += f"{'-'*40}\n"
        
        for step_id, step in workflow.steps.items():
            status_icon = {
                StepStatus.PENDING: "⭕",
                StepStatus.RUNNING: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️"
            }[step.status]
            
            report += f"{status_icon} {step.name}\n"
            report += f"   Status: {step.status.value} ({step.progress:.1f}%)\n"
            if step.duration:
                report += f"   Duration: {step.duration:.1f}s\n"
            if step.error_message:
                report += f"   Error: {step.error_message}\n"
            report += "\n"
            
        # Recent alerts
        recent_alerts = [a for a in self.alerts if a.workflow_id == workflow_id][-5:]
        if recent_alerts:
            report += f"🚨 RECENT ALERTS\n"
            report += f"{'-'*40}\n"
            for alert in recent_alerts:
                severity_icon = {
                    AlertSeverity.INFO: "ℹ️",
                    AlertSeverity.WARNING: "⚠️",
                    AlertSeverity.ERROR: "❌",
                    AlertSeverity.CRITICAL: "🚨"
                }[alert.severity]
                
                timestamp = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
                report += f"{severity_icon} [{timestamp}] {alert.message}\n"
                
        report += f"\n{'='*60}\n"
        
        return report
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check for performance alerts
                self._check_performance_alerts()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(30)  # Wait longer on error
    
    def _update_system_metrics(self) -> None:
        """Update system performance metrics"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            self.system_metrics = {
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'cpu_percent': cpu_percent,
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {str(e)}")
    
    def _check_performance_alerts(self) -> None:
        """Check for performance-related alerts"""
        for metric_name, threshold in self.resource_thresholds.items():
            current_value = self.system_metrics.get(metric_name, 0)
            
            if current_value > threshold:
                # Create alert for active workflows
                for workflow_id, workflow in self.workflows.items():
                    if workflow.status == WorkflowStatus.RUNNING:
                        self._create_alert(
                            workflow_id=workflow_id,
                            severity=AlertSeverity.WARNING,
                            message=f"High {metric_name}: {current_value:.1f}% (threshold: {threshold}%)"
                        )
    
    def _create_alert(self, workflow_id: str, severity: AlertSeverity, 
                     message: str, step_id: str = None, metadata: Dict[str, Any] = None) -> None:
        """Create and store a new alert"""
        alert = WorkflowAlert(
            alert_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            step_id=step_id,
            severity=severity,
            message=message,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.alerts.append(alert)
            if workflow_id in self.workflows:
                self.workflows[workflow_id].alerts.append(alert)
                
        # Store in memory
        self._store_alert_in_memory(alert)
        
        # Persist to database
        self._persist_alert(alert)
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}")
                
        logger.warning(f"Alert created: {message}")
    
    def _check_workflow_completion(self, workflow_id: str) -> None:
        """Check if workflow is complete"""
        if workflow_id not in self.workflows:
            return
            
        workflow = self.workflows[workflow_id]
        
        # Check if all steps are completed or failed
        pending_steps = [s for s in workflow.steps.values() 
                        if s.status in [StepStatus.PENDING, StepStatus.RUNNING]]
        
        if not pending_steps:
            # All steps are done
            if workflow.failed_steps > 0:
                workflow.status = WorkflowStatus.FAILED
            else:
                workflow.status = WorkflowStatus.COMPLETED
                
            workflow.end_time = time.time()
            if workflow.start_time:
                workflow.duration = workflow.end_time - workflow.start_time
                
            # Create completion alert
            self._create_alert(
                workflow_id=workflow_id,
                severity=AlertSeverity.INFO if workflow.status == WorkflowStatus.COMPLETED else AlertSeverity.ERROR,
                message=f"Workflow '{workflow.name}' {workflow.status.value}"
            )
            
            # Store completion in memory
            self._store_workflow_completion(workflow_id, workflow.status == WorkflowStatus.COMPLETED)
            
            # Persist to database
            self._persist_workflow(workflow)
            
            logger.info(f"Workflow {workflow_id} {workflow.status.value}")
    
    def _check_workflow_failure(self, workflow_id: str) -> None:
        """Check if workflow should be marked as failed"""
        if workflow_id not in self.workflows:
            return
            
        workflow = self.workflows[workflow_id]
        
        # If too many steps failed, fail the workflow
        failure_threshold = 0.5  # 50% of steps
        if workflow.failed_steps / max(workflow.total_steps, 1) > failure_threshold:
            workflow.status = WorkflowStatus.FAILED
            workflow.end_time = time.time()
            if workflow.start_time:
                workflow.duration = workflow.end_time - workflow.start_time
                
            self._create_alert(
                workflow_id=workflow_id,
                severity=AlertSeverity.CRITICAL,
                message=f"Workflow '{workflow.name}' failed due to too many step failures"
            )
    
    def _store_workflow_in_memory(self, workflow: WorkflowExecution) -> None:
        """Store workflow state in memory system"""
        try:
            memory_key = f"workflow_{workflow.workflow_id}"
            workflow_data = {
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'description': workflow.description,
                'status': workflow.status.value,
                'progress': workflow.progress,
                'metadata': workflow.metadata,
                'timestamp': time.time()
            }
            
            self.memory.store_swarm_memory(memory_key, json.dumps(workflow_data))
            
        except Exception as e:
            logger.error(f"Failed to store workflow in memory: {str(e)}")
    
    def _store_step_in_memory(self, workflow_id: str, step: WorkflowStep) -> None:
        """Store step state in memory system"""
        try:
            memory_key = f"step_{workflow_id}_{step.step_id}"
            step_data = {
                'step_id': step.step_id,
                'name': step.name,
                'status': step.status.value,
                'progress': step.progress,
                'duration': step.duration,
                'metadata': step.metadata,
                'timestamp': time.time()
            }
            
            self.memory.store_swarm_memory(memory_key, json.dumps(step_data))
            
        except Exception as e:
            logger.error(f"Failed to store step in memory: {str(e)}")
    
    def _store_step_progress(self, workflow_id: str, step_id: str, 
                           progress: float, message: str = None) -> None:
        """Store step progress in memory"""
        try:
            memory_key = f"progress_{workflow_id}_{step_id}"
            progress_data = {
                'step_id': step_id,
                'progress': progress,
                'message': message,
                'timestamp': time.time()
            }
            
            self.memory.store_swarm_memory(memory_key, json.dumps(progress_data))
            
        except Exception as e:
            logger.error(f"Failed to store step progress: {str(e)}")
    
    def _store_step_completion(self, workflow_id: str, step_id: str, 
                             success: bool, result: Dict[str, Any] = None) -> None:
        """Store step completion in memory"""
        try:
            memory_key = f"completion_{workflow_id}_{step_id}"
            completion_data = {
                'step_id': step_id,
                'success': success,
                'result': result,
                'timestamp': time.time()
            }
            
            self.memory.store_swarm_memory(memory_key, json.dumps(completion_data))
            
        except Exception as e:
            logger.error(f"Failed to store step completion: {str(e)}")
    
    def _store_workflow_completion(self, workflow_id: str, success: bool) -> None:
        """Store workflow completion in memory"""
        try:
            memory_key = f"workflow_completion_{workflow_id}"
            completion_data = {
                'workflow_id': workflow_id,
                'success': success,
                'timestamp': time.time()
            }
            
            self.memory.store_swarm_memory(memory_key, json.dumps(completion_data))
            
        except Exception as e:
            logger.error(f"Failed to store workflow completion: {str(e)}")
    
    def _store_metric_in_memory(self, metric: PerformanceMetric) -> None:
        """Store performance metric in memory"""
        try:
            memory_key = f"metric_{metric.workflow_id}_{metric.metric_name}"
            metric_data = {
                'metric_name': metric.metric_name,
                'value': metric.value,
                'unit': metric.unit,
                'tags': metric.tags,
                'timestamp': metric.timestamp
            }
            
            self.memory.store_swarm_memory(memory_key, json.dumps(metric_data))
            
        except Exception as e:
            logger.error(f"Failed to store metric in memory: {str(e)}")
    
    def _store_alert_in_memory(self, alert: WorkflowAlert) -> None:
        """Store alert in memory"""
        try:
            memory_key = f"alert_{alert.workflow_id}_{alert.alert_id}"
            alert_data = {
                'alert_id': alert.alert_id,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.timestamp,
                'metadata': alert.metadata
            }
            
            self.memory.store_swarm_memory(memory_key, json.dumps(alert_data))
            
        except Exception as e:
            logger.error(f"Failed to store alert in memory: {str(e)}")
    
    def _persist_workflow(self, workflow: WorkflowExecution) -> None:
        """Persist workflow to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO workflows 
                (workflow_id, name, description, status, start_time, end_time, duration,
                 total_steps, completed_steps, failed_steps, progress, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                workflow.workflow_id,
                workflow.name,
                workflow.description,
                workflow.status.value,
                workflow.start_time,
                workflow.end_time,
                workflow.duration,
                workflow.total_steps,
                workflow.completed_steps,
                workflow.failed_steps,
                workflow.progress,
                json.dumps(workflow.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist workflow: {str(e)}")
    
    def _persist_step(self, workflow_id: str, step: WorkflowStep) -> None:
        """Persist step to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO workflow_steps 
                (step_id, workflow_id, name, description, status, start_time, end_time, 
                 duration, progress, dependencies, metadata, error_message, retry_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                step.step_id,
                workflow_id,
                step.name,
                step.description,
                step.status.value,
                step.start_time,
                step.end_time,
                step.duration,
                step.progress,
                json.dumps(step.dependencies),
                json.dumps(step.metadata),
                step.error_message,
                step.retry_count
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist step: {str(e)}")
    
    def _persist_alert(self, alert: WorkflowAlert) -> None:
        """Persist alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO workflow_alerts 
                (alert_id, workflow_id, step_id, severity, message, timestamp, resolved, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.workflow_id,
                alert.step_id,
                alert.severity.value,
                alert.message,
                alert.timestamp,
                alert.resolved,
                json.dumps(alert.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist alert: {str(e)}")
    
    def _persist_metric(self, metric: PerformanceMetric) -> None:
        """Persist metric to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO workflow_metrics 
                (metric_id, workflow_id, step_id, metric_name, value, unit, timestamp, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.metric_id,
                metric.workflow_id,
                metric.step_id,
                metric.metric_name,
                metric.value,
                metric.unit,
                metric.timestamp,
                json.dumps(metric.tags)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to persist metric: {str(e)}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data from memory and database"""
        try:
            # Clean up database - keep only last 30 days
            cutoff_time = time.time() - (30 * 24 * 3600)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean old workflows
            cursor.execute("""
                DELETE FROM workflows 
                WHERE created_at < datetime(?, 'unixepoch')
                AND status IN ('completed', 'failed', 'cancelled')
            """, (cutoff_time,))
            
            # Clean old alerts
            cursor.execute("""
                DELETE FROM workflow_alerts 
                WHERE created_at < datetime(?, 'unixepoch')
            """, (cutoff_time,))
            
            # Clean old metrics
            cursor.execute("""
                DELETE FROM workflow_metrics 
                WHERE created_at < datetime(?, 'unixepoch')
            """, (cutoff_time,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
    
    def __del__(self):
        """Cleanup when monitor is destroyed"""
        self.stop_monitoring()


# Global workflow monitor instance
workflow_monitor = WorkflowMonitor()


# Convenience functions for easy usage
def create_workflow(name: str, description: str = "") -> str:
    """Create a new workflow"""
    return workflow_monitor.create_workflow(name, description)


def add_step(workflow_id: str, step_name: str, description: str = "") -> str:
    """Add a step to a workflow"""
    return workflow_monitor.add_workflow_step(workflow_id, step_name, description)


def start_workflow(workflow_id: str) -> None:
    """Start a workflow"""
    workflow_monitor.start_workflow(workflow_id)


def start_step(workflow_id: str, step_id: str) -> None:
    """Start a step"""
    workflow_monitor.start_step(workflow_id, step_id)


def update_progress(workflow_id: str, step_id: str, progress: float, message: str = None) -> None:
    """Update step progress"""
    workflow_monitor.update_step_progress(workflow_id, step_id, progress, message)


def complete_step(workflow_id: str, step_id: str, result: Dict[str, Any] = None) -> None:
    """Complete a step"""
    workflow_monitor.complete_step(workflow_id, step_id, result)


def fail_step(workflow_id: str, step_id: str, error_message: str) -> None:
    """Fail a step"""
    workflow_monitor.fail_step(workflow_id, step_id, error_message)


def record_metric(workflow_id: str, metric_name: str, value: float, unit: str = "") -> None:
    """Record a performance metric"""
    workflow_monitor.record_metric(workflow_id, metric_name, value, unit)


def get_status(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow status"""
    return workflow_monitor.get_workflow_status(workflow_id)


def generate_report(workflow_id: str) -> str:
    """Generate progress report"""
    return workflow_monitor.generate_progress_report(workflow_id)