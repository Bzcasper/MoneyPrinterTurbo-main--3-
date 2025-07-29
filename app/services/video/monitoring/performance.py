"""
Performance Monitor - Real-time performance tracking and optimization

This module provides comprehensive performance monitoring for video processing
with real-time metrics, bottleneck detection, and adaptive optimization.

Author: MoneyPrinterTurbo Enhanced System
Version: 1.0.0
"""

import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from loguru import logger
import psutil


class MetricType(Enum):
    """Performance metric types"""
    PROCESSING_TIME = "processing_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    metric_type: MetricType
    value: float
    timestamp: float
    workflow_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert"""
    metric_type: MetricType
    threshold_exceeded: str
    current_value: float
    threshold_value: float
    timestamp: float
    severity: str = "warning"  # warning, critical


class MetricsBuffer:
    """Thread-safe circular buffer for metrics"""
    
    def __init__(self, max_size: int = 1000):
        """Initialize metrics buffer"""
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self._lock = threading.RLock()
    
    def add(self, metric: PerformanceMetric):
        """Add metric to buffer"""
        with self._lock:
            self.buffer.append(metric)
    
    def get_recent(self, count: int = 100) -> List[PerformanceMetric]:
        """Get recent metrics"""
        with self._lock:
            return list(self.buffer)[-count:]
    
    def get_by_type(self, metric_type: MetricType, count: int = 100) -> List[PerformanceMetric]:
        """Get metrics by type"""
        with self._lock:
            filtered = [m for m in self.buffer if m.metric_type == metric_type]
            return filtered[-count:]
    
    def clear(self):
        """Clear buffer"""
        with self._lock:
            self.buffer.clear()


class PerformanceMonitor:
    """
    Real-time performance monitoring and optimization
    
    Provides comprehensive performance tracking with metrics collection,
    alert generation, and optimization recommendations.
    """
    
    def __init__(self):
        """Initialize performance monitor"""
        self.metrics_buffer = MetricsBuffer()
        self.workflow_metrics: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[PerformanceAlert] = []
        
        # Thresholds
        self.thresholds = {
            MetricType.PROCESSING_TIME: 300.0,  # 5 minutes max
            MetricType.MEMORY_USAGE: 85.0,      # 85% memory usage
            MetricType.CPU_USAGE: 90.0,         # 90% CPU usage
            MetricType.ERROR_RATE: 10.0,        # 10% error rate
            MetricType.QUEUE_SIZE: 50,          # 50 items in queue
        }
        
        # State tracking
        self._lock = threading.RLock()
        self._monitoring_enabled = True
        self._last_system_check = time.time()
        self.system_check_interval = 30  # 30 seconds
        
        logger.info("PerformanceMonitor initialized successfully")
    
    def start_workflow_monitoring(self, workflow_id: str) -> Dict[str, float]:
        """Start monitoring a workflow"""
        with self._lock:
            start_time = time.time()
            
            self.workflow_metrics[workflow_id] = {
                'start_time': start_time,
                'last_update': start_time,
                'processing_time': 0.0,
                'memory_start': self._get_memory_usage(),
                'cpu_start': self._get_cpu_usage(),
                'operations_count': 0,
                'errors_count': 0,
                'status': 'running'
            }
            
            logger.debug(f"Started monitoring workflow: {workflow_id}")
            
            return {
                'start_time': start_time,
                'memory_baseline': self.workflow_metrics[workflow_id]['memory_start'],
                'cpu_baseline': self.workflow_metrics[workflow_id]['cpu_start']
            }
    
    def record_operation(
        self, 
        workflow_id: str, 
        operation_name: str, 
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record an operation for a workflow"""
        with self._lock:
            if workflow_id not in self.workflow_metrics:
                self.start_workflow_monitoring(workflow_id)
            
            # Update workflow metrics
            workflow = self.workflow_metrics[workflow_id]
            workflow['last_update'] = time.time()
            workflow['operations_count'] += 1
            workflow['processing_time'] += duration
            
            if not success:
                workflow['errors_count'] += 1
            
            # Record metric
            metric = PerformanceMetric(
                metric_type=MetricType.PROCESSING_TIME,
                value=duration,
                timestamp=time.time(),
                workflow_id=workflow_id,
                metadata={
                    'operation': operation_name,
                    'success': success,
                    **(metadata or {})
                }
            )
            
            self.metrics_buffer.add(metric)
            
            # Check for performance issues
            self._check_thresholds(workflow_id, duration)
    
    def finish_workflow_monitoring(self, workflow_id: str, success: bool = True) -> Dict[str, Any]:
        """Finish monitoring a workflow and return summary"""
        with self._lock:
            if workflow_id not in self.workflow_metrics:
                return {}
            
            workflow = self.workflow_metrics[workflow_id]
            end_time = time.time()
            total_time = end_time - workflow['start_time']
            
            # Calculate final metrics
            error_rate = (workflow['errors_count'] / max(workflow['operations_count'], 1)) * 100
            throughput = workflow['operations_count'] / max(total_time, 0.001)
            
            summary = {
                'workflow_id': workflow_id,
                'total_time': total_time,
                'processing_time': workflow['processing_time'],
                'operations_count': workflow['operations_count'],
                'errors_count': workflow['errors_count'],
                'error_rate': error_rate,
                'throughput': throughput,
                'success': success,
                'memory_peak': self._get_memory_usage(),
                'cpu_peak': self._get_cpu_usage()
            }
            
            # Record final metrics
            self._record_workflow_summary(workflow_id, summary)
            
            # Update status
            workflow['status'] = 'completed' if success else 'failed'
            workflow['end_time'] = end_time
            
            logger.info(f"Workflow {workflow_id} completed: "
                       f"{total_time:.2f}s, {workflow['operations_count']} ops, "
                       f"{error_rate:.1f}% error rate")
            
            return summary
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            metrics = {
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'cpu_percent': cpu_percent,
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
            
            # Record system metrics
            for metric_name, value in metrics.items():
                if 'memory' in metric_name:
                    metric_type = MetricType.MEMORY_USAGE
                elif 'cpu' in metric_name:
                    metric_type = MetricType.CPU_USAGE
                else:
                    continue
                
                metric = PerformanceMetric(
                    metric_type=metric_type,
                    value=value,
                    timestamp=time.time(),
                    metadata={'system_metric': metric_name}
                )
                self.metrics_buffer.add(metric)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {str(e)}")
            return {}
    
    def get_workflow_summary(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific workflow"""
        with self._lock:
            if workflow_id not in self.workflow_metrics:
                return None
            
            workflow = self.workflow_metrics[workflow_id]
            current_time = time.time()
            
            if workflow['status'] == 'running':
                elapsed_time = current_time - workflow['start_time']
                error_rate = (workflow['errors_count'] / max(workflow['operations_count'], 1)) * 100
                throughput = workflow['operations_count'] / max(elapsed_time, 0.001)
            else:
                elapsed_time = workflow.get('end_time', current_time) - workflow['start_time']
                error_rate = (workflow['errors_count'] / max(workflow['operations_count'], 1)) * 100
                throughput = workflow['operations_count'] / max(elapsed_time, 0.001)
            
            return {
                'workflow_id': workflow_id,
                'status': workflow['status'],
                'elapsed_time': elapsed_time,
                'processing_time': workflow['processing_time'],
                'operations_count': workflow['operations_count'],
                'errors_count': workflow['errors_count'],
                'error_rate': error_rate,
                'throughput': throughput,
                'last_update': workflow['last_update']
            }
    
    def get_performance_alerts(self, clear_after_read: bool = True) -> List[PerformanceAlert]:
        """Get current performance alerts"""
        with self._lock:
            alerts = self.alerts.copy()
            if clear_after_read:
                self.alerts.clear()
            return alerts
    
    def get_metrics_summary(self, time_window: int = 300) -> Dict[str, Any]:
        """Get metrics summary for the last time window (seconds)"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Get recent metrics
        recent_metrics = [
            m for m in self.metrics_buffer.get_recent(1000)
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate summaries by type
        summary = {}
        
        for metric_type in MetricType:
            type_metrics = [m for m in recent_metrics if m.metric_type == metric_type]
            
            if type_metrics:
                values = [m.value for m in type_metrics]
                summary[metric_type.value] = {
                    'count': len(values),
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1] if values else 0
                }
        
        # Add system metrics
        summary['system'] = self.get_system_metrics()
        summary['time_window'] = time_window
        summary['timestamp'] = current_time
        
        return summary
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Analyze performance and provide optimization recommendations"""
        system_metrics = self.get_system_metrics()
        metrics_summary = self.get_metrics_summary()
        
        recommendations = []
        
        # Memory optimization
        if system_metrics.get('memory_percent', 0) > 80:
            recommendations.append({
                'type': 'memory',
                'severity': 'high',
                'message': 'High memory usage detected - consider reducing batch sizes',
                'current_value': system_metrics['memory_percent']
            })
        
        # CPU optimization
        if system_metrics.get('cpu_percent', 0) > 85:
            recommendations.append({
                'type': 'cpu',
                'severity': 'high',
                'message': 'High CPU usage - consider reducing concurrent processing',
                'current_value': system_metrics['cpu_percent']
            })
        
        # Processing time optimization
        processing_metrics = metrics_summary.get(MetricType.PROCESSING_TIME.value, {})
        if processing_metrics.get('average', 0) > 60:  # 1 minute average
            recommendations.append({
                'type': 'processing_time',
                'severity': 'medium',
                'message': 'Long processing times - consider optimizing algorithms',
                'current_value': processing_metrics['average']
            })
        
        # Error rate optimization
        error_metrics = metrics_summary.get(MetricType.ERROR_RATE.value, {})
        if error_metrics.get('latest', 0) > 5:  # 5% error rate
            recommendations.append({
                'type': 'error_rate',
                'severity': 'high',
                'message': 'High error rate detected - check input validation',
                'current_value': error_metrics['latest']
            })
        
        return {
            'recommendations': recommendations,
            'system_health': self._calculate_health_score(system_metrics, metrics_summary),
            'timestamp': time.time()
        }
    
    def is_healthy(self) -> bool:
        """Check if performance monitor is healthy"""
        try:
            # Check if monitoring is enabled and working
            if not self._monitoring_enabled:
                return False
            
            # Check system resources
            system_metrics = self.get_system_metrics()
            if not system_metrics:
                return False
            
            # Check if memory usage is reasonable
            if system_metrics.get('memory_percent', 100) > 95:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            return psutil.virtual_memory().percent
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def _check_thresholds(self, workflow_id: str, value: float):
        """Check if metrics exceed thresholds and generate alerts"""
        # Check processing time threshold
        if value > self.thresholds[MetricType.PROCESSING_TIME]:
            alert = PerformanceAlert(
                metric_type=MetricType.PROCESSING_TIME,
                threshold_exceeded="processing_time",
                current_value=value,
                threshold_value=self.thresholds[MetricType.PROCESSING_TIME],
                timestamp=time.time(),
                severity="warning" if value < self.thresholds[MetricType.PROCESSING_TIME] * 1.5 else "critical"
            )
            
            self.alerts.append(alert)
            logger.warning(f"Performance alert: {alert.threshold_exceeded} = {value:.2f}")
        
        # Check system metrics periodically
        current_time = time.time()
        if current_time - self._last_system_check > self.system_check_interval:
            self._check_system_thresholds()
            self._last_system_check = current_time
    
    def _check_system_thresholds(self):
        """Check system resource thresholds"""
        system_metrics = self.get_system_metrics()
        
        # Check memory threshold
        memory_percent = system_metrics.get('memory_percent', 0)
        if memory_percent > self.thresholds[MetricType.MEMORY_USAGE]:
            alert = PerformanceAlert(
                metric_type=MetricType.MEMORY_USAGE,
                threshold_exceeded="memory_usage",
                current_value=memory_percent,
                threshold_value=self.thresholds[MetricType.MEMORY_USAGE],
                timestamp=time.time(),
                severity="critical" if memory_percent > 95 else "warning"
            )
            self.alerts.append(alert)
        
        # Check CPU threshold
        cpu_percent = system_metrics.get('cpu_percent', 0)
        if cpu_percent > self.thresholds[MetricType.CPU_USAGE]:
            alert = PerformanceAlert(
                metric_type=MetricType.CPU_USAGE,
                threshold_exceeded="cpu_usage",
                current_value=cpu_percent,
                threshold_value=self.thresholds[MetricType.CPU_USAGE],
                timestamp=time.time(),
                severity="critical" if cpu_percent > 98 else "warning"
            )
            self.alerts.append(alert)
    
    def _record_workflow_summary(self, workflow_id: str, summary: Dict[str, Any]):
        """Record workflow summary metrics"""
        # Record error rate
        error_rate_metric = PerformanceMetric(
            metric_type=MetricType.ERROR_RATE,
            value=summary['error_rate'],
            timestamp=time.time(),
            workflow_id=workflow_id
        )
        self.metrics_buffer.add(error_rate_metric)
        
        # Record throughput
        throughput_metric = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=summary['throughput'],
            timestamp=time.time(),
            workflow_id=workflow_id
        )
        self.metrics_buffer.add(throughput_metric)
    
    def _calculate_health_score(
        self, system_metrics: Dict[str, float], metrics_summary: Dict[str, Any]
    ) -> float:
        """Calculate overall system health score (0-100)"""
        score = 100.0
        
        # Deduct for high memory usage
        memory_percent = system_metrics.get('memory_percent', 0)
        if memory_percent > 70:
            score -= (memory_percent - 70) * 2
        
        # Deduct for high CPU usage
        cpu_percent = system_metrics.get('cpu_percent', 0)
        if cpu_percent > 80:
            score -= (cpu_percent - 80) * 1.5
        
        # Deduct for high error rates
        error_metrics = metrics_summary.get(MetricType.ERROR_RATE.value, {})
        error_rate = error_metrics.get('latest', 0)
        if error_rate > 1:
            score -= error_rate * 5
        
        return max(0.0, min(100.0, score))