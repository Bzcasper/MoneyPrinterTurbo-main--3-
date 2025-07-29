#!/usr/bin/env python3
"""
Workflow Monitor Demo - Demonstration script showing how to use the workflow monitoring system

This script demonstrates the complete workflow monitoring capabilities:
- Creating workflows with multiple steps
- Real-time progress tracking
- Performance metrics collection
- Error handling and alerting
- Report generation

Run this script to see the workflow monitor in action.
"""

import sys
import time
import random
import threading
from pathlib import Path
from typing import List, Dict, Any

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.workflow_monitor import (
    workflow_monitor,
    create_workflow,
    add_step,
    start_workflow,
    start_step,
    update_progress,
    complete_step,
    fail_step,
    record_metric,
    get_status,
    generate_report
)
from loguru import logger


class WorkflowDemo:
    """Demonstration of workflow monitoring capabilities"""
    
    def __init__(self):
        self.workflow_id = None
        self.step_ids = []
        
        # Start monitoring system
        workflow_monitor.start_monitoring()
        
        # Add custom alert handler
        workflow_monitor.add_alert_handler(self._alert_handler)
    
    def _alert_handler(self, alert) -> None:
        """Custom alert handler for demo"""
        severity_icons = {
            'info': 'ℹ️',
            'warning': '⚠️', 
            'error': '❌',
            'critical': '🚨'
        }
        
        icon = severity_icons.get(alert.severity.value, '❓')
        print(f"\n{icon} ALERT: {alert.message}")
    
    def create_demo_workflow(self) -> str:
        """Create a demo workflow with multiple steps"""
        print("🚀 Creating Demo Workflow...")
        
        # Create workflow
        self.workflow_id = create_workflow(
            name="Video Processing Pipeline Demo",
            description="Demonstration of video processing workflow with monitoring"
        )
        
        # Add steps
        steps = [
            ("Initialize Environment", "Set up processing environment and validate dependencies"),
            ("Download Source Material", "Download video files from external sources"),
            ("Process Audio", "Extract and process audio tracks"),
            ("Apply Video Effects", "Apply filters and effects to video content"),
            ("Encode Output", "Encode final video with optimal settings"),
            ("Upload Results", "Upload processed videos to storage"),
            ("Send Notifications", "Notify users of completion")
        ]
        
        for step_name, step_desc in steps:
            step_id = add_step(self.workflow_id, step_name, step_desc)
            self.step_ids.append(step_id)
        
        print(f"✅ Created workflow: {self.workflow_id}")
        print(f"📦 Added {len(steps)} steps")
        
        return self.workflow_id
    
    def simulate_workflow_execution(self) -> None:
        """Simulate workflow execution with realistic timing and progress"""
        print("\n🎬 Starting Workflow Execution...")
        
        # Start the workflow
        start_workflow(self.workflow_id)
        
        # Execute each step with realistic simulation
        step_configs = [
            {"name": "Initialize Environment", "duration": 3, "failure_rate": 0.1},
            {"name": "Download Source Material", "duration": 8, "failure_rate": 0.15},
            {"name": "Process Audio", "duration": 5, "failure_rate": 0.05},
            {"name": "Apply Video Effects", "duration": 12, "failure_rate": 0.2},
            {"name": "Encode Output", "duration": 15, "failure_rate": 0.1},
            {"name": "Upload Results", "duration": 6, "failure_rate": 0.15},
            {"name": "Send Notifications", "duration": 2, "failure_rate": 0.02}
        ]
        
        for i, (step_id, config) in enumerate(zip(self.step_ids, step_configs)):
            self._execute_step_with_monitoring(step_id, config, i + 1)
            
            # Small delay between steps
            time.sleep(0.5)
        
        print("\n🏁 Workflow Execution Complete!")
    
    def _execute_step_with_monitoring(self, step_id: str, config: Dict[str, Any], step_number: int) -> None:
        """Execute a single step with detailed monitoring"""
        step_name = config["name"]
        duration = config["duration"]
        failure_rate = config.get("failure_rate", 0.1)
        
        print(f"\n📌 Step {step_number}: {step_name}")
        
        # Start the step
        start_step(self.workflow_id, step_id)
        
        # Record start metric
        record_metric(self.workflow_id, "step_start_time", time.time(), "timestamp", step_id)
        
        # Simulate step execution with progress updates
        start_time = time.time()
        
        # Simulate failure chance
        if random.random() < failure_rate:
            print(f"   ❌ Step failed!")
            error_message = f"Simulated failure in {step_name} - random error for demo"
            fail_step(self.workflow_id, step_id, error_message, retry=True)
            
            # Wait a bit and retry
            time.sleep(1)
            print(f"   🔄 Retrying step...")
            start_step(self.workflow_id, step_id)
        
        # Progressive execution with updates
        progress_points = [10, 25, 40, 60, 75, 90, 100]
        
        for progress in progress_points:
            elapsed = time.time() - start_time
            remaining_time = duration - elapsed
            
            if remaining_time > 0:
                sleep_time = min(remaining_time / len(progress_points), 1.0)
                time.sleep(sleep_time)
            
            # Update progress
            message = self._get_progress_message(step_name, progress)
            update_progress(self.workflow_id, step_id, progress, message)
            
            # Record performance metrics
            self._record_step_metrics(step_id, progress, elapsed)
            
            print(f"   📊 {progress}% - {message}")
        
        # Complete the step
        execution_time = time.time() - start_time
        result = {
            "execution_time": execution_time,
            "files_processed": random.randint(1, 10),
            "memory_used_mb": random.randint(100, 500),
            "success": True
        }
        
        complete_step(self.workflow_id, step_id, result)
        
        # Record completion metrics
        record_metric(self.workflow_id, "step_execution_time", execution_time, "seconds", step_id)
        record_metric(self.workflow_id, "memory_usage", result["memory_used_mb"], "MB", step_id)
        
        print(f"   ✅ Completed in {execution_time:.1f}s")
    
    def _get_progress_message(self, step_name: str, progress: float) -> str:
        """Generate realistic progress messages"""
        messages = {
            "Initialize Environment": [
                "Checking dependencies...",
                "Loading configuration...",
                "Initializing GPU drivers...",
                "Setting up temp directories...",
                "Validating licenses...",
                "Preparing workspace...",
                "Environment ready"
            ],
            "Download Source Material": [
                "Connecting to source...",
                "Authenticating...",
                "Starting download...",
                "Downloading files...",
                "Verifying checksums...",
                "Processing manifests...",
                "Download complete"
            ],
            "Process Audio": [
                "Loading audio tracks...",
                "Analyzing audio format...",
                "Applying noise reduction...",
                "Normalizing levels...",
                "Encoding audio...",
                "Validating output...",
                "Audio processing complete"
            ],
            "Apply Video Effects": [
                "Loading video frames...",
                "Analyzing content...",
                "Applying filters...",
                "Processing effects...",
                "Optimizing quality...",
                "Rendering frames...",
                "Effects applied"
            ],
            "Encode Output": [
                "Initializing encoder...",
                "Setting encoding parameters...",
                "Starting first pass...",
                "Running second pass...",
                "Optimizing compression...",
                "Finalizing encoding...",
                "Encoding complete"
            ],
            "Upload Results": [
                "Connecting to storage...",
                "Preparing upload...",
                "Uploading files...",
                "Verifying upload...",
                "Updating metadata...",
                "Cleaning temp files...",
                "Upload successful"
            ],
            "Send Notifications": [
                "Preparing notifications...",
                "Sending emails...",
                "Updating status...",
                "Logging completion...",
                "Archiving results...",
                "Cleanup complete...",
                "Notifications sent"
            ]
        }
        
        step_messages = messages.get(step_name, ["Processing..."] * 7)
        message_index = min(int(progress / 100 * len(step_messages)), len(step_messages) - 1)
        
        return step_messages[message_index]
    
    def _record_step_metrics(self, step_id: str, progress: float, elapsed_time: float) -> None:
        """Record realistic performance metrics during step execution"""
        
        # CPU usage (simulated)
        cpu_usage = 40 + random.randint(-20, 40)  # 20-80%
        record_metric(self.workflow_id, "cpu_usage", cpu_usage, "percent", step_id)
        
        # Memory usage (simulated)
        base_memory = 200
        memory_usage = base_memory + (progress / 100) * random.randint(100, 300)
        record_metric(self.workflow_id, "memory_usage", memory_usage, "MB", step_id)
        
        # Processing rate (simulated)
        if progress > 0:
            processing_rate = progress / max(elapsed_time, 0.1)
            record_metric(self.workflow_id, "processing_rate", processing_rate, "percent/second", step_id)
        
        # Disk I/O (simulated)
        disk_io = random.randint(10, 100)
        record_metric(self.workflow_id, "disk_io", disk_io, "MB/s", step_id)
    
    def display_real_time_monitoring(self, duration: int = 30) -> None:
        """Display real-time monitoring information"""
        print(f"\n📺 Real-time Monitoring (for {duration} seconds)...")
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Clear screen (simplified)
                print("\n" + "="*60)
                print("🔄 LIVE WORKFLOW MONITORING")
                print("="*60)
                
                # Get current status
                status = get_status(self.workflow_id)
                if status:
                    print(f"📋 Workflow: {status['name']}")
                    print(f"📈 Status: {status['status'].upper()}")
                    print(f"📊 Progress: {status['progress']:.1f}%")
                    print(f"📦 Steps: {status['completed_steps']}/{status['total_steps']}")
                    
                    if status['duration']:
                        print(f"⏱️  Duration: {status['duration']:.1f}s")
                    
                    # Show active steps
                    running_steps = [
                        step for step in status['steps'].values()
                        if step['status'] == 'running'
                    ]
                    
                    if running_steps:
                        print(f"\n🔄 Currently Running:")
                        for step in running_steps:
                            print(f"  📌 {step['name']}: {step['progress']:.1f}%")
                
                # Show recent alerts
                recent_alerts = workflow_monitor.get_recent_alerts(3)
                if recent_alerts:
                    print(f"\n🚨 Recent Alerts:")
                    for alert in recent_alerts[-3:]:
                        timestamp = time.strftime("%H:%M:%S", time.localtime(alert['timestamp']))
                        print(f"  [{timestamp}] {alert['message']}")
                
                # Show system metrics
                system_metrics = workflow_monitor.system_metrics
                if system_metrics:
                    print(f"\n💻 System Resources:")
                    print(f"  CPU: {system_metrics.get('cpu_percent', 0):.1f}%")
                    print(f"  Memory: {system_metrics.get('memory_percent', 0):.1f}%")
                    print(f"  Disk: {system_metrics.get('disk_percent', 0):.1f}%")
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n⏹️  Real-time monitoring stopped")
    
    def generate_final_report(self) -> None:
        """Generate and display final workflow report"""
        print("\n📊 Generating Final Report...")
        
        report = generate_report(self.workflow_id)
        print(report)
        
        # Additional analytics
        print("\n📈 PERFORMANCE ANALYTICS")
        print("="*50)
        
        performance_summary = workflow_monitor.get_performance_summary(self.workflow_id)
        
        if performance_summary:
            for metric_name, stats in performance_summary.items():
                print(f"📊 {metric_name.replace('_', ' ').title()}:")
                print(f"   Average: {stats['mean']:.2f}")
                print(f"   Min/Max: {stats['min']:.2f} / {stats['max']:.2f}")
                print(f"   Count: {stats['count']}")
                print()
        
        # Show workflow history
        history = workflow_monitor.get_workflow_history(self.workflow_id)
        if history:
            print("📋 EXECUTION HISTORY")
            print("-"*30)
            for record in history:
                if isinstance(record, dict):
                    print(f"⏰ Execution: {record.get('created_at', 'Unknown time')}")
                    print(f"📊 Status: {record.get('status', 'Unknown')}")
                    print(f"⏱️  Duration: {record.get('duration', 0):.1f}s")
                    print()
    
    def run_complete_demo(self) -> None:
        """Run the complete workflow monitoring demonstration"""
        print("🎯 WORKFLOW MONITORING SYSTEM DEMONSTRATION")
        print("="*60)
        print("This demo shows comprehensive workflow monitoring capabilities:")
        print("• Real-time progress tracking")
        print("• Performance metrics collection")
        print("• Error detection and alerting")
        print("• Visual progress indicators")
        print("• Detailed reporting")
        print("="*60)
        
        try:
            # Create demo workflow
            self.create_demo_workflow()
            
            # Show initial status
            print(f"\n📋 Initial Status:")
            status = get_status(self.workflow_id)
            if status:
                print(f"   📦 Steps: {status['total_steps']}")
                print(f"   📊 Progress: {status['progress']:.1f}%")
            
            # Start execution in background thread
            execution_thread = threading.Thread(target=self.simulate_workflow_execution)
            execution_thread.daemon = True
            execution_thread.start()
            
            # Show real-time monitoring
            self.display_real_time_monitoring(45)
            
            # Wait for execution to complete
            execution_thread.join(timeout=60)
            
            # Generate final report
            time.sleep(1)  # Allow final updates
            self.generate_final_report()
            
            print("\n🎉 Demo Complete!")
            print("The workflow monitoring system successfully demonstrated:")
            print("✅ Real-time progress tracking")
            print("✅ Performance metrics collection") 
            print("✅ Error handling and retry logic")
            print("✅ Alert generation and handling")
            print("✅ Comprehensive reporting")
            print("✅ Database persistence")
            print("✅ Memory system integration")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {str(e)}")
            logger.exception("Demo execution error")
        
        finally:
            # Stop monitoring
            workflow_monitor.stop_monitoring()


def main():
    """Main demo entry point"""
    print("🚀 Starting Workflow Monitor Demo...")
    
    try:
        demo = WorkflowDemo()
        demo.run_complete_demo()
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        logger.exception("Demo failed")


if __name__ == "__main__":
    main()