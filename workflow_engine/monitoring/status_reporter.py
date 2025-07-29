"""
Status reporting system for workflow execution.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import os


logger = logging.getLogger(__name__)


class ReportLevel(Enum):
    """Report level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ReportFormat(Enum):
    """Report format enumeration."""
    JSON = "json"
    TEXT = "text"
    HTML = "html"
    CSV = "csv"


@dataclass
class StatusReport:
    """Individual status report entry."""
    timestamp: datetime
    workflow_id: str
    level: ReportLevel
    message: str
    details: Dict[str, Any]
    step_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "level": self.level.value,
            "message": self.message,
            "details": self.details
        }


class StatusReporter:
    """
    Comprehensive status reporting system for workflow execution.
    
    Features:
    - Multi-level status reporting (debug, info, warning, error, critical)
    - Multiple output formats (JSON, text, HTML, CSV)
    - File and console output
    - Real-time status updates
    - Report filtering and querying
    - Report archiving and rotation
    - Custom report handlers
    """
    
    def __init__(self, 
                 log_file: Optional[str] = None,
                 console_output: bool = True,
                 max_reports: int = 10000,
                 min_level: ReportLevel = ReportLevel.INFO):
        """
        Initialize the status reporter.
        
        Args:
            log_file: Path to log file for reports
            console_output: Whether to output to console
            max_reports: Maximum number of reports to keep in memory
            min_level: Minimum report level to process
        """
        self.log_file = log_file
        self.console_output = console_output
        self.max_reports = max_reports
        self.min_level = min_level
        self.logger = logging.getLogger(__name__)
        
        # Runtime state
        self.reports: List[StatusReport] = []
        self.custom_handlers: List[Callable[[StatusReport], None]] = []
        
        # Setup file logging if specified
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        self.logger.info(f"Status reporter initialized (min_level: {min_level.value})")
    
    def report(self, 
               workflow_id: str, 
               level: ReportLevel, 
               message: str,
               details: Optional[Dict[str, Any]] = None,
               step_id: Optional[str] = None):
        """
        Submit a status report.
        
        Args:
            workflow_id: ID of the workflow
            level: Report level
            message: Report message
            details: Additional details
            step_id: Optional step ID
        """
        # Check minimum level
        level_priority = {
            ReportLevel.DEBUG: 0,
            ReportLevel.INFO: 1,
            ReportLevel.WARNING: 2,
            ReportLevel.ERROR: 3,
            ReportLevel.CRITICAL: 4
        }
        
        if level_priority[level] < level_priority[self.min_level]:
            return
        
        # Create report
        report = StatusReport(
            timestamp=datetime.now(),
            workflow_id=workflow_id,
            level=level,
            message=message,
            details=details or {},
            step_id=step_id
        )
        
        # Store report
        self.reports.append(report)
        
        # Maintain size limit
        if len(self.reports) > self.max_reports:
            self.reports.pop(0)
        
        # Output report
        self._output_report(report)
        
        # Call custom handlers
        for handler in self.custom_handlers:
            try:
                handler(report)
            except Exception as e:
                self.logger.error(f"Error in custom report handler: {e}")
    
    def debug(self, workflow_id: str, message: str, details: Optional[Dict[str, Any]] = None, step_id: Optional[str] = None):
        """Submit a debug report."""
        self.report(workflow_id, ReportLevel.DEBUG, message, details, step_id)
    
    def info(self, workflow_id: str, message: str, details: Optional[Dict[str, Any]] = None, step_id: Optional[str] = None):
        """Submit an info report."""
        self.report(workflow_id, ReportLevel.INFO, message, details, step_id)
    
    def warning(self, workflow_id: str, message: str, details: Optional[Dict[str, Any]] = None, step_id: Optional[str] = None):
        """Submit a warning report."""
        self.report(workflow_id, ReportLevel.WARNING, message, details, step_id)
    
    def error(self, workflow_id: str, message: str, details: Optional[Dict[str, Any]] = None, step_id: Optional[str] = None):
        """Submit an error report."""
        self.report(workflow_id, ReportLevel.ERROR, message, details, step_id)
    
    def critical(self, workflow_id: str, message: str, details: Optional[Dict[str, Any]] = None, step_id: Optional[str] = None):
        """Submit a critical report."""
        self.report(workflow_id, ReportLevel.CRITICAL, message, details, step_id)
    
    def _output_report(self, report: StatusReport):
        """Output report to configured destinations."""
        # Console output
        if self.console_output:
            console_msg = self._format_report_text(report)
            print(console_msg)
        
        # File output
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(report.to_dict()) + '\n')
            except Exception as e:
                self.logger.error(f"Error writing to log file: {e}")
    
    def _format_report_text(self, report: StatusReport) -> str:
        """Format report as text."""
        timestamp = report.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        level_str = report.level.value.upper()
        
        base_msg = f"[{timestamp}] [{level_str}] {report.workflow_id}"
        if report.step_id:
            base_msg += f":{report.step_id}"
        base_msg += f" - {report.message}"
        
        if report.details:
            details_str = ", ".join(f"{k}={v}" for k, v in report.details.items())
            base_msg += f" ({details_str})"
        
        return base_msg
    
    def get_reports(self, 
                   workflow_id: Optional[str] = None,
                   level: Optional[ReportLevel] = None,
                   step_id: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get reports with optional filtering.
        
        Args:
            workflow_id: Filter by workflow ID
            level: Filter by report level
            step_id: Filter by step ID
            limit: Maximum number of reports to return
            
        Returns:
            List of matching reports
        """
        reports = self.reports
        
        # Apply filters
        if workflow_id:
            reports = [r for r in reports if r.workflow_id == workflow_id]
        
        if level:
            reports = [r for r in reports if r.level == level]
        
        if step_id:
            reports = [r for r in reports if r.step_id == step_id]
        
        # Apply limit
        if limit:
            reports = reports[-limit:]
        
        return [r.to_dict() for r in reports]
    
    def get_workflow_summary(self, workflow_id: str) -> Dict[str, Any]:
        """Get summary of reports for a specific workflow."""
        workflow_reports = [r for r in self.reports if r.workflow_id == workflow_id]
        
        if not workflow_reports:
            return {
                "workflow_id": workflow_id,
                "total_reports": 0,
                "level_counts": {},
                "first_report": None,
                "last_report": None
            }
        
        # Count by level
        level_counts = {}
        for level in ReportLevel:
            level_counts[level.value] = sum(1 for r in workflow_reports if r.level == level)
        
        # Sort by timestamp
        workflow_reports.sort(key=lambda r: r.timestamp)
        
        return {
            "workflow_id": workflow_id,
            "total_reports": len(workflow_reports),
            "level_counts": level_counts,
            "first_report": workflow_reports[0].to_dict(),
            "last_report": workflow_reports[-1].to_dict()
        }
    
    def export_reports(self, 
                      file_path: str, 
                      format: ReportFormat = ReportFormat.JSON,
                      workflow_id: Optional[str] = None,
                      level: Optional[ReportLevel] = None) -> bool:
        """
        Export reports to file in specified format.
        
        Args:
            file_path: Output file path
            format: Export format
            workflow_id: Filter by workflow ID
            level: Filter by report level
            
        Returns:
            True if export succeeded
        """
        try:
            reports = self.get_reports(workflow_id=workflow_id, level=level)
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if format == ReportFormat.JSON:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(reports, f, indent=2)
            
            elif format == ReportFormat.TEXT:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for report_dict in reports:
                        report = StatusReport(
                            timestamp=datetime.fromisoformat(report_dict["timestamp"]),
                            workflow_id=report_dict["workflow_id"],
                            level=ReportLevel(report_dict["level"]),
                            message=report_dict["message"],
                            details=report_dict["details"],
                            step_id=report_dict.get("step_id")
                        )
                        f.write(self._format_report_text(report) + '\n')
            
            elif format == ReportFormat.HTML:
                html_content = self._generate_html_report(reports)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            elif format == ReportFormat.CSV:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    if reports:
                        writer = csv.DictWriter(f, fieldnames=reports[0].keys())
                        writer.writeheader()
                        writer.writerows(reports)
            
            self.logger.info(f"Exported {len(reports)} reports to {file_path} ({format.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting reports: {e}")
            return False
    
    def _generate_html_report(self, reports: List[Dict[str, Any]]) -> str:
        """Generate HTML report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Workflow Status Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .report { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
                .debug { border-left-color: #999; }
                .info { border-left-color: #2196F3; }
                .warning { border-left-color: #FF9800; }
                .error { border-left-color: #F44336; }
                .critical { border-left-color: #9C27B0; }
                .timestamp { color: #666; font-size: 0.9em; }
                .details { margin-top: 5px; font-size: 0.9em; color: #555; }
            </style>
        </head>
        <body>
            <h1>Workflow Status Report</h1>
            <p>Generated: {}</p>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        for report in reports:
            level_class = report["level"]
            html += f"""
            <div class="report {level_class}">
                <div class="timestamp">{report["timestamp"]}</div>
                <strong>[{report["level"].upper()}] {report["workflow_id"]}</strong>
                {f":{report['step_id']}" if report.get("step_id") else ""} - {report["message"]}
                {f'<div class="details">{report["details"]}</div>' if report["details"] else ""}
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def add_custom_handler(self, handler: Callable[[StatusReport], None]):
        """Add a custom report handler."""
        self.custom_handlers.append(handler)
        self.logger.debug(f"Added custom report handler: {handler.__name__}")
    
    def remove_custom_handler(self, handler: Callable[[StatusReport], None]):
        """Remove a custom report handler."""
        if handler in self.custom_handlers:
            self.custom_handlers.remove(handler)
            self.logger.debug(f"Removed custom report handler: {handler.__name__}")
    
    def clear_reports(self, workflow_id: Optional[str] = None):
        """Clear reports, optionally for a specific workflow."""
        if workflow_id:
            self.reports = [r for r in self.reports if r.workflow_id != workflow_id]
            self.logger.info(f"Cleared reports for workflow '{workflow_id}'")
        else:
            self.reports.clear()
            self.logger.info("Cleared all reports")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reporting statistics."""
        if not self.reports:
            return {
                "total_reports": 0,
                "level_distribution": {},
                "workflows": [],
                "date_range": None
            }
        
        # Level distribution
        level_dist = {}
        for level in ReportLevel:
            level_dist[level.value] = sum(1 for r in self.reports if r.level == level)
        
        # Unique workflows
        workflows = list(set(r.workflow_id for r in self.reports))
        
        # Date range
        timestamps = [r.timestamp for r in self.reports]
        date_range = {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat()
        }
        
        return {
            "total_reports": len(self.reports),
            "level_distribution": level_dist,
            "workflows": workflows,
            "date_range": date_range
        }
    
    def cleanup(self):
        """Cleanup reporter resources."""
        self.reports.clear()
        self.custom_handlers.clear()
        self.logger.info("Status reporter cleanup completed")