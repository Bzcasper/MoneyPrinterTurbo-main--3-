"""
Video Service Logging Configuration
=================================

Centralized, structured logging configuration for video services with:
- Configurable log levels and formats
- Contextual logging utilities
- Performance tracking capabilities
- Environment-based configuration
- No hardcoded values (SPARC compliant)

Author: VideoEngineer Agent
Version: 1.0.0
"""

import os
import sys
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

from loguru import logger


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Log format enumeration"""
    STRUCTURED = "structured"
    HUMAN_READABLE = "human"
    JSON = "json"


@dataclass
class LoggingConfig:
    """Logging configuration parameters"""
    level: LogLevel = LogLevel.INFO
    format_type: LogFormat = LogFormat.STRUCTURED
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    log_directory: str = "logs"
    max_file_size: str = "10MB"
    retention_days: int = 30
    enable_performance_logging: bool = True
    enable_context_logging: bool = True
    enable_error_tracking: bool = True
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Create configuration from environment variables"""
        return cls(
            level=LogLevel(os.getenv('LOG_LEVEL', 'INFO')),
            format_type=LogFormat(os.getenv('LOG_FORMAT', 'structured')),
            enable_file_logging=os.getenv('ENABLE_FILE_LOGGING', 'true').lower() == 'true',
            enable_console_logging=os.getenv('ENABLE_CONSOLE_LOGGING', 'true').lower() == 'true',
            log_directory=os.getenv('LOG_DIRECTORY', 'logs'),
            max_file_size=os.getenv('LOG_MAX_FILE_SIZE', '10MB'),
            retention_days=int(os.getenv('LOG_RETENTION_DAYS', '30')),
            enable_performance_logging=os.getenv('ENABLE_PERFORMANCE_LOGGING', 'true').lower() == 'true',
            enable_context_logging=os.getenv('ENABLE_CONTEXT_LOGGING', 'true').lower() == 'true',
            enable_error_tracking=os.getenv('ENABLE_ERROR_TRACKING', 'true').lower() == 'true'
        )


@dataclass
class LogContext:
    """Contextual information for structured logging"""
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    service: str = "video_service"
    component: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging"""
        return {
            'user_id': self.user_id,
            'task_id': self.task_id,
            'request_id': self.request_id,
            'session_id': self.session_id,
            'operation': self.operation,
            'service': self.service,
            'component': self.component,
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id,
            'iso_timestamp': datetime.fromtimestamp(self.timestamp).isoformat()
        }


class VideoServiceLogger:
    """Enhanced logger for video services with contextual and performance logging"""
    
    def __init__(self, config: LoggingConfig = None):
        self.config = config or LoggingConfig.from_env()
        self._setup_logging()
        self._performance_timers: Dict[str, float] = {}
        
    def _setup_logging(self):
        """Configure loguru logger based on configuration"""
        # Remove default logger
        logger.remove()
        
        # Create log directory if needed
        if self.config.enable_file_logging:
            Path(self.config.log_directory).mkdir(parents=True, exist_ok=True)
        
        # Setup console logging
        if self.config.enable_console_logging:
            console_format = self._get_log_format()
            logger.add(
                sys.stdout,
                level=self.config.level.value,
                format=console_format,
                colorize=True
            )
        
        # Setup file logging
        if self.config.enable_file_logging:
            file_format = self._get_log_format(for_file=True)
            
            # Main log file
            logger.add(
                f"{self.config.log_directory}/video_service.log",
                level=self.config.level.value,
                format=file_format,
                rotation=self.config.max_file_size,
                retention=f"{self.config.retention_days} days",
                compression="gz"
            )
            
            # Error-only log file
            logger.add(
                f"{self.config.log_directory}/video_service_errors.log",
                level="ERROR",
                format=file_format,
                rotation=self.config.max_file_size,
                retention=f"{self.config.retention_days} days",
                compression="gz",
                filter=lambda record: record["level"].name in ["ERROR", "CRITICAL"]
            )
            
            # Performance log file (if enabled)
            if self.config.enable_performance_logging:
                logger.add(
                    f"{self.config.log_directory}/video_service_performance.log",
                    level="INFO",
                    format=file_format,
                    rotation=self.config.max_file_size,
                    retention=f"{self.config.retention_days} days",
                    compression="gz",
                    filter=lambda record: record.get("extra", {}).get("log_type") == "performance"
                )
    
    def _get_log_format(self, for_file: bool = False) -> str:
        """Get log format string based on configuration"""
        if self.config.format_type == LogFormat.JSON:
            return "{time} | {level} | {message} | {extra}"
        elif self.config.format_type == LogFormat.STRUCTURED:
            if for_file:
                return "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}"
            else:
                return "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> | {message} | {extra}"
        else:  # HUMAN_READABLE
            return "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
    
    def with_context(self, context: LogContext) -> 'ContextualLogger':
        """Create a contextual logger with embedded context"""
        return ContextualLogger(self, context)
    
    def log_performance_start(self, operation: str, context: LogContext = None) -> str:
        """Start performance timing for an operation"""
        timer_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        self._performance_timers[timer_id] = time.time()
        
        extra_data = {"log_type": "performance", "operation": operation, "phase": "start"}
        if context and self.config.enable_context_logging:
            extra_data.update(context.to_dict())
        
        if self.config.enable_performance_logging:
            logger.info(f"🚀 Performance tracking started: {operation}", extra=extra_data)
        
        return timer_id
    
    def log_performance_end(
        self, 
        timer_id: str, 
        operation: str, 
        context: LogContext = None,
        additional_metrics: Dict[str, Any] = None
    ):
        """End performance timing and log results"""
        if timer_id not in self._performance_timers:
            logger.warning(f"Performance timer not found: {timer_id}")
            return
        
        duration = time.time() - self._performance_timers[timer_id]
        del self._performance_timers[timer_id]
        
        extra_data = {
            "log_type": "performance",
            "operation": operation,
            "phase": "end",
            "duration_seconds": round(duration, 3),
            "duration_ms": round(duration * 1000, 1)
        }
        
        if additional_metrics:
            extra_data.update(additional_metrics)
        
        if context and self.config.enable_context_logging:
            extra_data.update(context.to_dict())
        
        if self.config.enable_performance_logging:
            logger.info(f"⏱️ Performance tracking completed: {operation} ({duration:.3f}s)", extra=extra_data)
    
    def log_error_with_context(
        self,
        message: str,
        error: Exception = None,
        context: LogContext = None,
        additional_data: Dict[str, Any] = None
    ):
        """Log error with full context and error tracking"""
        extra_data = {"log_type": "error"}
        
        if error:
            extra_data.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_args": getattr(error, 'args', [])
            })
        
        if context and self.config.enable_context_logging:
            extra_data.update(context.to_dict())
        
        if additional_data:
            extra_data.update(additional_data)
        
        if self.config.enable_error_tracking:
            logger.error(f"❌ {message}", extra=extra_data)
        else:
            logger.error(message)


class ContextualLogger:
    """Logger wrapper that automatically includes context in all log messages"""
    
    def __init__(self, video_logger: VideoServiceLogger, context: LogContext):
        self.video_logger = video_logger
        self.context = context
        self._logger = logger
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with embedded context"""
        extra_data = {}
        if self.video_logger.config.enable_context_logging:
            extra_data.update(self.context.to_dict())
        
        if 'extra' in kwargs:
            extra_data.update(kwargs['extra'])
        
        kwargs['extra'] = extra_data
        getattr(self._logger, level.lower())(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self._log_with_context("DEBUG", f"🔍 {message}", **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self._log_with_context("INFO", f"ℹ️ {message}", **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self._log_with_context("WARNING", f"⚠️ {message}", **kwargs)
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error message with context"""
        extra_data = kwargs.get('extra', {})
        
        if error:
            extra_data.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_args": getattr(error, 'args', [])
            })
        
        kwargs['extra'] = extra_data
        self._log_with_context("ERROR", f"❌ {message}", **kwargs)
    
    def critical(self, message: str, error: Exception = None, **kwargs):
        """Log critical message with context"""
        extra_data = kwargs.get('extra', {})
        
        if error:
            extra_data.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_args": getattr(error, 'args', [])
            })
        
        kwargs['extra'] = extra_data
        self._log_with_context("CRITICAL", f"🚨 {message}", **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message with context"""
        self._log_with_context("INFO", f"✅ {message}", **kwargs)


# Global logger instance
_global_logger: Optional[VideoServiceLogger] = None


def get_video_logger() -> VideoServiceLogger:
    """Get or create global video service logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = VideoServiceLogger()
    return _global_logger


def create_context(
    user_id: str = None,
    task_id: str = None,
    request_id: str = None,
    operation: str = None,
    component: str = None,
    **kwargs
) -> LogContext:
    """Create logging context with common parameters"""
    return LogContext(
        user_id=user_id,
        task_id=task_id,
        request_id=request_id,
        operation=operation,
        component=component,
        **kwargs
    )


# Convenience functions for common logging patterns
def log_video_operation_start(
    operation: str,
    user_id: str = None,
    task_id: str = None,
    **context_kwargs
) -> tuple[str, ContextualLogger]:
    """Start logging for a video operation"""
    context = create_context(
        user_id=user_id,
        task_id=task_id,
        operation=operation,
        **context_kwargs
    )
    
    video_logger = get_video_logger()
    timer_id = video_logger.log_performance_start(operation, context)
    contextual_logger = video_logger.with_context(context)
    
    return timer_id, contextual_logger


def log_video_operation_end(
    timer_id: str,
    operation: str,
    success: bool = True,
    error: Exception = None,
    metrics: Dict[str, Any] = None
):
    """End logging for a video operation"""
    video_logger = get_video_logger()
    
    # Add success/failure metrics
    operation_metrics = {"success": success}
    if metrics:
        operation_metrics.update(metrics)
    
    video_logger.log_performance_end(timer_id, operation, additional_metrics=operation_metrics)
    
    if not success and error:
        video_logger.log_error_with_context(
            f"Video operation failed: {operation}",
            error=error,
            additional_data=operation_metrics
        )


# Export main classes and functions
__all__ = [
    'VideoServiceLogger',
    'LoggingConfig',
    'LogContext',
    'ContextualLogger',
    'LogLevel',
    'LogFormat',
    'get_video_logger',
    'create_context',
    'log_video_operation_start',
    'log_video_operation_end'
]