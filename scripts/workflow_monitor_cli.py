#!/usr/bin/env python3
"""
Workflow Monitor CLI - Command-line interface for workflow monitoring

This script provides a comprehensive CLI for interacting with the workflow
monitoring system, displaying real-time progress, generating reports,
and managing workflow execution.

Usage:
    python workflow_monitor_cli.py create --name "My Workflow" --description "Test workflow"
    python workflow_monitor_cli.py status --workflow-id <id>
    python workflow_monitor_cli.py monitor --workflow-id <id>
    python workflow_monitor_cli.py report --workflow-id <id>
    python workflow_monitor_cli.py list --active
"""

import argparse
import sys
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.workflow_monitor import (
    workflow_monitor, 
    WorkflowStatus, 
    StepStatus,
    create_workflow,
    add_step,
    start_workflow,
    get_status,
    generate_report
)
from loguru import logger


class WorkflowMonitorCLI:
    """Command-line interface for workflow monitoring"""
    
    def __init__(self):
        self.monitor = workflow_monitor
        
        # Start monitoring if not already started
        self.monitor.start_monitoring()
        
        # Colors for terminal output
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m'
        }
    
    def colorize(self, text: str, color: str) -> str:
        """Add color to text for terminal output"""
        return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"
    
    def create_workflow_command(self, args) -> None:
        """Create a new workflow"""
        try:
            workflow_id = create_workflow(args.name, args.description or "")
            
            print(self.colorize("✅ Workflow Created Successfully!", "green"))
            print(f"📋 Name: {args.name}")
            print(f"🆔 ID: {workflow_id}")
            print(f"📝 Description: {args.description or 'No description'}")
            
            # Add steps if provided
            if args.steps:
                steps = args.steps.split(',')
                for i, step_name in enumerate(steps, 1):
                    step_id = add_step(workflow_id, step_name.strip(), f"Step {i}")
                    print(f"   📌 Added step: {step_name.strip()} ({step_id[:8]}...)")
            
            print(f"\n💡 Use this command to monitor: {sys.argv[0]} monitor --workflow-id {workflow_id}")
            
        except Exception as e:
            print(self.colorize(f"❌ Error creating workflow: {str(e)}", "red"))
            sys.exit(1)
    
    def add_step_command(self, args) -> None:
        """Add a step to an existing workflow"""
        try:
            step_id = add_step(args.workflow_id, args.name, args.description or "")
            
            print(self.colorize("✅ Step Added Successfully!", "green"))
            print(f"📋 Name: {args.name}")
            print(f"🆔 Step ID: {step_id}")
            print(f"🔗 Workflow ID: {args.workflow_id}")
            
        except Exception as e:
            print(self.colorize(f"❌ Error adding step: {str(e)}", "red"))
            sys.exit(1)
    
    def start_workflow_command(self, args) -> None:
        """Start workflow execution"""
        try:
            start_workflow(args.workflow_id)
            
            print(self.colorize("🚀 Workflow Started!", "green"))
            print(f"🆔 Workflow ID: {args.workflow_id}")
            print(f"\n💡 Monitor progress: {sys.argv[0]} monitor --workflow-id {args.workflow_id}")
            
        except Exception as e:
            print(self.colorize(f"❌ Error starting workflow: {str(e)}", "red"))
            sys.exit(1)
    
    def status_command(self, args) -> None:
        """Show workflow status"""
        try:
            status = get_status(args.workflow_id)
            
            if not status:
                print(self.colorize(f"❌ Workflow {args.workflow_id} not found", "red"))
                return
            
            # Header
            print(self.colorize("📊 WORKFLOW STATUS", "bold"))
            print("=" * 50)
            
            # Basic info
            print(f"📋 Name: {self.colorize(status['name'], 'cyan')}")
            print(f"🆔 ID: {status['workflow_id']}")
            print(f"📈 Status: {self._format_status(status['status'])}")
            print(f"📊 Progress: {self._format_progress(status['progress'])}")
            print(f"📦 Steps: {status['completed_steps']}/{status['total_steps']}")
            
            if status['duration']:
                print(f"⏱️  Duration: {status['duration']:.1f}s")
            
            # Steps details
            if status['steps']:
                print(f"\n📋 STEPS ({len(status['steps'])})")
                print("-" * 30)
                
                for step_id, step_info in status['steps'].items():
                    step_status = self._format_step_status(step_info['status'])
                    progress_bar = self._create_progress_bar(step_info['progress'], 20)
                    
                    print(f"  {step_status} {step_info['name']}")
                    print(f"    [{progress_bar}] {step_info['progress']:.1f}%")
                    
                    if step_info['duration']:
                        print(f"    ⏱️  {step_info['duration']:.1f}s")
                    print()
            
        except Exception as e:
            print(self.colorize(f"❌ Error getting status: {str(e)}", "red"))
            sys.exit(1)
    
    def monitor_command(self, args) -> None:
        """Real-time monitoring of workflow"""
        try:
            print(self.colorize("🔄 Starting Real-Time Monitoring...", "yellow"))
            print(f"🆔 Workflow ID: {args.workflow_id}")
            print("Press Ctrl+C to stop monitoring\n")
            
            last_status = None
            
            while True:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Get current status
                current_status = get_status(args.workflow_id)
                
                if not current_status:
                    print(self.colorize(f"❌ Workflow {args.workflow_id} not found", "red"))
                    break
                
                # Display status
                self._display_monitoring_status(current_status)
                
                # Check if workflow is complete
                if current_status['status'] in ['completed', 'failed', 'cancelled']:
                    print(self.colorize(f"\n🏁 Workflow {current_status['status']}!", "green" if current_status['status'] == 'completed' else "red"))
                    
                    if args.generate_report:
                        print("\n" + generate_report(args.workflow_id))
                    
                    break
                
                # Wait before next update
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print(self.colorize("\n⏹️  Monitoring stopped by user", "yellow"))
        except Exception as e:
            print(self.colorize(f"❌ Error during monitoring: {str(e)}", "red"))
    
    def report_command(self, args) -> None:
        """Generate workflow report"""
        try:
            report = generate_report(args.workflow_id)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(self.colorize(f"📄 Report saved to: {args.output}", "green"))
            else:
                print(report)
                
        except Exception as e:
            print(self.colorize(f"❌ Error generating report: {str(e)}", "red"))
            sys.exit(1)
    
    def list_command(self, args) -> None:
        """List workflows"""
        try:
            if args.active:
                workflows = self.monitor.get_active_workflows()
                title = "🔄 ACTIVE WORKFLOWS"
            else:
                # Get all workflows from recent history
                workflows = []
                for workflow_id in list(self.monitor.workflows.keys())[-10:]:  # Last 10
                    status = get_status(workflow_id)
                    if status:
                        workflows.append(status)
                title = "📋 RECENT WORKFLOWS"
            
            print(self.colorize(title, "bold"))
            print("=" * 50)
            
            if not workflows:
                print(self.colorize("No workflows found", "yellow"))
                return
                
            for workflow in workflows:
                status_icon = self._get_status_icon(workflow['status'])
                progress_bar = self._create_progress_bar(workflow['progress'], 15)
                
                print(f"{status_icon} {workflow['name']}")
                print(f"  🆔 {workflow['workflow_id'][:8]}...")
                print(f"  📊 [{progress_bar}] {workflow['progress']:.1f}%")
                print(f"  📦 {workflow['completed_steps']}/{workflow['total_steps']} steps")
                
                if workflow['duration']:
                    print(f"  ⏱️  {workflow['duration']:.1f}s")
                print()
                
        except Exception as e:
            print(self.colorize(f"❌ Error listing workflows: {str(e)}", "red"))
            sys.exit(1)
    
    def alerts_command(self, args) -> None:
        """Show recent alerts"""
        try:
            alerts = self.monitor.get_recent_alerts(args.count)
            
            print(self.colorize("🚨 RECENT ALERTS", "bold"))
            print("=" * 50)
            
            if not alerts:
                print(self.colorize("No recent alerts", "green"))
                return
                
            for alert in reversed(alerts[-args.count:]):  # Show most recent first
                severity_icon = self._get_severity_icon(alert['severity'])
                timestamp = time.strftime("%H:%M:%S", time.localtime(alert['timestamp']))
                
                print(f"{severity_icon} [{timestamp}] {alert['message']}")
                print(f"  🆔 Workflow: {alert['workflow_id'][:8]}...")
                
                if alert['step_id']:
                    print(f"  📌 Step: {alert['step_id'][:8]}...")
                print()
                
        except Exception as e:
            print(self.colorize(f"❌ Error getting alerts: {str(e)}", "red"))
            sys.exit(1)
    
    def metrics_command(self, args) -> None:
        """Show performance metrics"""
        try:
            metrics = self.monitor.get_performance_summary(args.workflow_id)
            
            print(self.colorize("📈 PERFORMANCE METRICS", "bold"))
            print("=" * 50)
            
            if not metrics:
                print(self.colorize("No metrics available", "yellow"))
                return
                
            for metric_name, stats in metrics.items():
                print(f"📊 {self.colorize(metric_name, 'cyan')}")
                print(f"  Count: {stats['count']}")
                print(f"  Latest: {stats['latest']:.3f}")
                print(f"  Mean: {stats['mean']:.3f}")
                print(f"  Min/Max: {stats['min']:.3f} / {stats['max']:.3f}")
                
                if 'std_dev' in stats:
                    print(f"  Std Dev: {stats['std_dev']:.3f}")
                print()
                
        except Exception as e:
            print(self.colorize(f"❌ Error getting metrics: {str(e)}", "red"))
            sys.exit(1)
    
    def _display_monitoring_status(self, status: Dict[str, Any]) -> None:
        """Display status for real-time monitoring"""
        # Header
        print(self.colorize("🔄 LIVE MONITORING", "bold"))
        print("=" * 60)
        print(f"🕐 {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Workflow info
        print(f"📋 {self.colorize(status['name'], 'cyan')}")
        print(f"🆔 {status['workflow_id']}")
        print(f"📈 Status: {self._format_status(status['status'])}")
        
        # Overall progress
        overall_progress_bar = self._create_progress_bar(status['progress'], 40)
        print(f"📊 Progress: [{overall_progress_bar}] {status['progress']:.1f}%")
        print(f"📦 Steps: {status['completed_steps']}/{status['total_steps']}")
        
        if status['duration']:
            print(f"⏱️  Duration: {status['duration']:.1f}s")
        
        print()
        
        # Steps details
        if status['steps']:
            print(self.colorize("📋 STEP DETAILS", "bold"))
            print("-" * 40)
            
            for step_id, step_info in status['steps'].items():
                step_status = self._format_step_status(step_info['status'])
                progress_bar = self._create_progress_bar(step_info['progress'], 25)
                
                print(f"{step_status} {step_info['name']}")
                print(f"   [{progress_bar}] {step_info['progress']:.1f}%")
                
                if step_info['duration']:
                    print(f"   ⏱️  {step_info['duration']:.1f}s")
                print()
    
    def _format_status(self, status: str) -> str:
        """Format workflow status with colors"""
        status_colors = {
            'pending': 'yellow',
            'running': 'blue', 
            'completed': 'green',
            'failed': 'red',
            'cancelled': 'magenta'
        }
        
        color = status_colors.get(status, 'white')
        return self.colorize(status.upper(), color)
    
    def _format_step_status(self, status: str) -> str:
        """Format step status with icons"""
        status_icons = {
            'pending': '⭕',
            'running': '🔄',
            'completed': '✅',
            'failed': '❌',
            'skipped': '⏭️'
        }
        
        return status_icons.get(status, '❓')
    
    def _format_progress(self, progress: float) -> str:
        """Format progress with color coding"""
        if progress >= 100:
            return self.colorize(f"{progress:.1f}%", "green")
        elif progress >= 75:
            return self.colorize(f"{progress:.1f}%", "cyan")
        elif progress >= 50:
            return self.colorize(f"{progress:.1f}%", "yellow")
        else:
            return self.colorize(f"{progress:.1f}%", "red")
    
    def _create_progress_bar(self, progress: float, width: int = 20) -> str:
        """Create a visual progress bar"""
        filled_width = int(width * progress / 100)
        bar = "█" * filled_width + "░" * (width - filled_width)
        
        # Color code the progress bar
        if progress >= 100:
            return self.colorize(bar, "green")
        elif progress >= 75:
            return self.colorize(bar, "cyan")
        elif progress >= 50:
            return self.colorize(bar, "yellow")
        else:
            return self.colorize(bar, "red")
    
    def _get_status_icon(self, status: str) -> str:
        """Get icon for workflow status"""
        status_icons = {
            'pending': '⏳',
            'running': '🔄',
            'completed': '✅',
            'failed': '❌',
            'cancelled': '⏹️'
        }
        
        return status_icons.get(status, '❓')
    
    def _get_severity_icon(self, severity: str) -> str:
        """Get icon for alert severity"""
        severity_icons = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }
        
        return severity_icons.get(severity, '❓')


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Workflow Monitor CLI - Monitor and manage workflow execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new workflow
  %(prog)s create --name "Video Processing" --description "Process video files"
  
  # Add steps to a workflow
  %(prog)s add-step --workflow-id abc123 --name "Download Videos"
  
  # Start workflow execution
  %(prog)s start --workflow-id abc123
  
  # Monitor workflow in real-time
  %(prog)s monitor --workflow-id abc123 --interval 5
  
  # Get workflow status
  %(prog)s status --workflow-id abc123
  
  # Generate progress report
  %(prog)s report --workflow-id abc123 --output report.txt
  
  # List active workflows
  %(prog)s list --active
  
  # Show recent alerts
  %(prog)s alerts --count 10
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create workflow command
    create_parser = subparsers.add_parser('create', help='Create a new workflow')
    create_parser.add_argument('--name', required=True, help='Workflow name')
    create_parser.add_argument('--description', help='Workflow description')
    create_parser.add_argument('--steps', help='Comma-separated list of step names')
    
    # Add step command
    add_step_parser = subparsers.add_parser('add-step', help='Add step to workflow')
    add_step_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    add_step_parser.add_argument('--name', required=True, help='Step name')
    add_step_parser.add_argument('--description', help='Step description')
    
    # Start workflow command
    start_parser = subparsers.add_parser('start', help='Start workflow execution')
    start_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show workflow status')
    status_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Real-time monitoring')
    monitor_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    monitor_parser.add_argument('--interval', type=int, default=2, help='Update interval in seconds')
    monitor_parser.add_argument('--generate-report', action='store_true', help='Generate report when complete')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate workflow report')
    report_parser.add_argument('--workflow-id', required=True, help='Workflow ID')
    report_parser.add_argument('--output', help='Output file (default: stdout)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List workflows')
    list_parser.add_argument('--active', action='store_true', help='Show only active workflows')
    
    # Alerts command
    alerts_parser = subparsers.add_parser('alerts', help='Show recent alerts')
    alerts_parser.add_argument('--count', type=int, default=20, help='Number of alerts to show')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show performance metrics')
    metrics_parser.add_argument('--workflow-id', help='Workflow ID (optional)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize CLI
    cli = WorkflowMonitorCLI()
    
    # Execute command
    try:
        if args.command == 'create':
            cli.create_workflow_command(args)
        elif args.command == 'add-step':
            cli.add_step_command(args)
        elif args.command == 'start':
            cli.start_workflow_command(args)
        elif args.command == 'status':
            cli.status_command(args)
        elif args.command == 'monitor':
            cli.monitor_command(args)
        elif args.command == 'report':
            cli.report_command(args)
        elif args.command == 'list':
            cli.list_command(args)
        elif args.command == 'alerts':
            cli.alerts_command(args)
        elif args.command == 'metrics':
            cli.metrics_command(args)
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        logger.exception("CLI error")
        sys.exit(1)


if __name__ == "__main__":
    main()