"""
MCP Monitoring and Analytics

Provides comprehensive monitoring, metrics collection, alerting, and analytics
for MCP servers and clients, including performance tracking and error analysis.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import statistics
from loguru import logger
import redis

from app.config import config


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str]
    metric_type: MetricType


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: str
    condition: str  # e.g., "> 100", "< 0.5"
    duration: int   # seconds
    cooldown: int   # seconds
    severity: str   # critical, warning, info
    enabled: bool = True


class MCPMetricsCollector:
    """Collects and stores MCP metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Redis for distributed metrics (optional)
        self.redis_client = None
        if config.app.get("enable_redis", False):
            try:
                redis_url = f"redis://:{config.app.get('redis_password', '')}@{config.app.get('redis_host', 'redis')}:{config.app.get('redis_port', 6379)}/{config.app.get('redis_db', 0)}"
                self.redis_client = redis.Redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for metrics: {e}")
                
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        self.counters[name] += value
        self._record_metric(name, value, MetricType.COUNTER, labels or {})
        
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        self.gauges[name] = value
        self._record_metric(name, value, MetricType.GAUGE, labels or {})
        
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        self.histograms[name].append(value)
        # Keep only recent values
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]
        self._record_metric(name, value, MetricType.HISTOGRAM, labels or {})
        
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timer duration"""
        self.timers[name].append(duration)
        # Keep only recent values
        if len(self.timers[name]) > 1000:
            self.timers[name] = self.timers[name][-1000:]
        self._record_metric(name, duration, MetricType.TIMER, labels or {})
        
    def _record_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str]):
        """Record a metric point"""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels,
            metric_type=metric_type
        )
        
        self.metrics[name].append(point)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                key = f"mcp_metrics:{name}"
                data = asdict(point)
                self.redis_client.lpush(key, json.dumps(data))
                self.redis_client.ltrim(key, 0, 9999)  # Keep last 10k points
                self.redis_client.expire(key, 86400)   # Expire after 24 hours
            except Exception as e:
                logger.warning(f"Failed to store metric in Redis: {e}")
                
    def get_metric_summary(self, name: str, duration_seconds: int = 3600) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        cutoff_time = time.time() - duration_seconds
        points = [p for p in self.metrics[name] if p.timestamp >= cutoff_time]
        
        if not points:
            return {"name": name, "count": 0}
            
        values = [p.value for p in points]
        
        summary = {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "sum": sum(values)
        }
        
        if len(values) > 1:
            summary["std_dev"] = statistics.stdev(values)
            summary["median"] = statistics.median(values)
            summary["p95"] = self._percentile(values, 0.95)
            summary["p99"] = self._percentile(values, 0.99)
            
        return summary
        
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
        
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values"""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {k: self.get_metric_summary(k) for k in self.histograms.keys()},
            "timers": {k: self.get_metric_summary(k) for k in self.timers.keys()}
        }


class MCPPerformanceTracker:
    """Tracks performance metrics for MCP operations"""
    
    def __init__(self, metrics_collector: MCPMetricsCollector):
        self.metrics = metrics_collector
        self.active_operations: Dict[str, float] = {}
        
    def start_operation(self, operation_id: str, operation_type: str):
        """Start tracking an operation"""
        self.active_operations[operation_id] = time.time()
        self.metrics.increment_counter(f"mcp.operations.started", labels={"type": operation_type})
        
    def end_operation(self, operation_id: str, operation_type: str, success: bool = True):
        """End tracking an operation"""
        if operation_id not in self.active_operations:
            return
            
        start_time = self.active_operations.pop(operation_id)
        duration = time.time() - start_time
        
        self.metrics.record_timer(f"mcp.operations.duration", duration, labels={"type": operation_type})
        
        if success:
            self.metrics.increment_counter(f"mcp.operations.success", labels={"type": operation_type})
        else:
            self.metrics.increment_counter(f"mcp.operations.error", labels={"type": operation_type})
            
    def track_connection(self, connection_id: str, action: str):
        """Track connection events"""
        self.metrics.increment_counter(f"mcp.connections.{action}")
        if action == "connected":
            self.metrics.increment_counter("mcp.connections.active")
        elif action == "disconnected":
            self.metrics.increment_counter("mcp.connections.active", -1)
            
    def track_tool_usage(self, tool_name: str, success: bool, duration: float):
        """Track tool usage statistics"""
        labels = {"tool": tool_name}
        self.metrics.increment_counter("mcp.tools.calls", labels=labels)
        self.metrics.record_timer("mcp.tools.duration", duration, labels=labels)
        
        if success:
            self.metrics.increment_counter("mcp.tools.success", labels=labels)
        else:
            self.metrics.increment_counter("mcp.tools.error", labels=labels)


class MCPAlertManager:
    """Manages alerts based on metric thresholds"""
    
    def __init__(self, metrics_collector: MCPMetricsCollector):
        self.metrics = metrics_collector
        self.alert_rules: List[AlertRule] = []
        self.alert_state: Dict[str, Dict] = {}
        self.alert_handlers: List[Callable] = []
        
        # Load default alert rules
        self._load_default_rules()
        
    def _load_default_rules(self):
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                metric="mcp.operations.error",
                condition="> 10",
                duration=300,  # 5 minutes
                cooldown=900,  # 15 minutes
                severity="warning"
            ),
            AlertRule(
                name="low_success_rate",
                metric="mcp.operations.success_rate",
                condition="< 0.95",
                duration=300,
                cooldown=600,
                severity="critical"
            ),
            AlertRule(
                name="high_response_time",
                metric="mcp.operations.duration.p95",
                condition="> 5.0",
                duration=300,
                cooldown=600,
                severity="warning"
            ),
            AlertRule(
                name="too_many_connections",
                metric="mcp.connections.active",
                condition="> 100",
                duration=60,
                cooldown=300,
                severity="warning"
            )
        ]
        
        self.alert_rules.extend(default_rules)
        
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules.append(rule)
        
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler function"""
        self.alert_handlers.append(handler)
        
    async def check_alerts(self):
        """Check all alert rules and trigger alerts if needed"""
        current_time = time.time()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            try:
                # Get metric value
                metric_value = self._get_metric_value(rule.metric)
                if metric_value is None:
                    continue
                    
                # Check condition
                condition_met = self._evaluate_condition(metric_value, rule.condition)
                
                # Update alert state
                rule_state = self.alert_state.get(rule.name, {
                    "triggered": False,
                    "first_triggered": None,
                    "last_triggered": None,
                    "last_resolved": None
                })
                
                if condition_met:
                    if not rule_state["triggered"]:
                        rule_state["first_triggered"] = current_time
                    
                    # Check if condition has been met for required duration
                    if (rule_state["first_triggered"] and 
                        current_time - rule_state["first_triggered"] >= rule.duration):
                        
                        # Check cooldown
                        if (not rule_state["last_triggered"] or 
                            current_time - rule_state["last_triggered"] >= rule.cooldown):
                            
                            # Trigger alert
                            await self._trigger_alert(rule, metric_value)
                            rule_state["last_triggered"] = current_time
                            rule_state["triggered"] = True
                else:
                    if rule_state["triggered"]:
                        # Resolve alert
                        await self._resolve_alert(rule, metric_value)
                        rule_state["last_resolved"] = current_time
                        rule_state["triggered"] = False
                        rule_state["first_triggered"] = None
                        
                self.alert_state[rule.name] = rule_state
                
            except Exception as e:
                logger.error(f"Error checking alert rule {rule.name}: {str(e)}")
                
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric"""
        if metric_name in self.metrics.counters:
            return self.metrics.counters[metric_name]
        elif metric_name in self.metrics.gauges:
            return self.metrics.gauges[metric_name]
        elif metric_name.endswith(".p95") or metric_name.endswith(".p99"):
            base_name = metric_name.rsplit(".", 1)[0]
            percentile = float(metric_name.split(".")[-1][1:]) / 100
            if base_name in self.metrics.timers:
                values = self.metrics.timers[base_name]
                if values:
                    return self.metrics._percentile(values, percentile)
        elif metric_name.endswith("_rate"):
            base_name = metric_name[:-5]
            # Calculate rate (simplified)
            if base_name in self.metrics.counters:
                return self.metrics.counters[base_name] / 60.0  # per minute
                
        return None
        
    def _evaluate_condition(self, value: float, condition: str) -> bool:
        """Evaluate alert condition"""
        try:
            if condition.startswith("> "):
                threshold = float(condition[2:])
                return value > threshold
            elif condition.startswith("< "):
                threshold = float(condition[2:])
                return value < threshold
            elif condition.startswith(">= "):
                threshold = float(condition[3:])
                return value >= threshold
            elif condition.startswith("<= "):
                threshold = float(condition[3:])
                return value <= threshold
            elif condition.startswith("== "):
                threshold = float(condition[3:])
                return value == threshold
            elif condition.startswith("!= "):
                threshold = float(condition[3:])
                return value != threshold
        except ValueError:
            pass
            
        return False
        
    async def _trigger_alert(self, rule: AlertRule, metric_value: float):
        """Trigger an alert"""
        alert_data = {
            "rule_name": rule.name,
            "metric": rule.metric,
            "value": metric_value,
            "condition": rule.condition,
            "severity": rule.severity,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "triggered"
        }
        
        logger.warning(f"Alert triggered: {rule.name} - {rule.metric} = {metric_value}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}")
                
    async def _resolve_alert(self, rule: AlertRule, metric_value: float):
        """Resolve an alert"""
        alert_data = {
            "rule_name": rule.name,
            "metric": rule.metric,
            "value": metric_value,
            "condition": rule.condition,
            "severity": rule.severity,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "resolved"
        }
        
        logger.info(f"Alert resolved: {rule.name} - {rule.metric} = {metric_value}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}")


class MCPMonitor:
    """Main monitoring coordinator"""
    
    def __init__(self):
        self.metrics_collector = MCPMetricsCollector()
        self.performance_tracker = MCPPerformanceTracker(self.metrics_collector)
        self.alert_manager = MCPAlertManager(self.metrics_collector)
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._alert_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self):
        """Start monitoring background tasks"""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._alert_task = asyncio.create_task(self._alert_loop())
        logger.info("MCP monitoring started")
        
    async def stop_monitoring(self):
        """Stop monitoring background tasks"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._alert_task:
            self._alert_task.cancel()
        logger.info("MCP monitoring stopped")
        
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Update system metrics
                self.metrics_collector.set_gauge("mcp.monitoring.timestamp", time.time())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                
    async def _alert_loop(self):
        """Background alert checking loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check alerts every minute
                await self.alert_manager.check_alerts()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert loop: {str(e)}")
                
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics_collector.get_all_metrics(),
            "alerts": {
                "active": [
                    name for name, state in self.alert_manager.alert_state.items()
                    if state.get("triggered", False)
                ],
                "rules": [asdict(rule) for rule in self.alert_manager.alert_rules]
            },
            "performance": {
                "active_operations": len(self.performance_tracker.active_operations),
                "operation_summaries": {
                    "tool_calls": self.metrics_collector.get_metric_summary("mcp.tools.calls"),
                    "connection_events": self.metrics_collector.get_metric_summary("mcp.connections.connected"),
                    "response_times": self.metrics_collector.get_metric_summary("mcp.operations.duration")
                }
            }
        }


# Global monitoring instance
mcp_monitor = MCPMonitor()


# Convenience functions for easy usage
def track_operation(operation_type: str):
    """Decorator to track operation performance"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            operation_id = f"{func.__name__}_{int(time.time() * 1000)}"
            mcp_monitor.performance_tracker.start_operation(operation_id, operation_type)
            
            try:
                result = await func(*args, **kwargs)
                mcp_monitor.performance_tracker.end_operation(operation_id, operation_type, True)
                return result
            except Exception as e:
                mcp_monitor.performance_tracker.end_operation(operation_id, operation_type, False)
                raise
                
        return wrapper
    return decorator


def increment_counter(name: str, value: float = 1.0, **labels):
    """Convenience function to increment counter"""
    mcp_monitor.metrics_collector.increment_counter(name, value, labels)


def set_gauge(name: str, value: float, **labels):
    """Convenience function to set gauge"""
    mcp_monitor.metrics_collector.set_gauge(name, value, labels)


def record_duration(name: str, duration: float, **labels):
    """Convenience function to record duration"""
    mcp_monitor.metrics_collector.record_timer(name, duration, labels)