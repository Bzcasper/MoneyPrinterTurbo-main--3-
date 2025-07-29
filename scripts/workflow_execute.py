#!/usr/bin/env python3
"""
Workflow Execute Command Implementation
Provides robust workflow execution capabilities with Claude Flow integration.
"""

import os
import sys
import json
import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app.services.workflow_monitor import (
        WorkflowMonitor, create_workflow, add_step, 
        start_workflow, complete_step, fail_step
    )
    from app.services.video.workflow_integration import start_monitored_video_task
except ImportError:
    print("Warning: MoneyPrinterTurbo modules not available. Using standalone mode.")
    WorkflowMonitor = None


@dataclass
class WorkflowExecutionResult:
    """Results from workflow execution"""
    success: bool
    workflow_id: str
    total_steps: int
    completed_steps: int
    failed_steps: int
    execution_time: float
    output_data: Dict[str, Any]
    error_message: Optional[str] = None


class WorkflowExecutor:
    """
    Comprehensive workflow execution engine with Claude Flow integration
    """
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.workflows_cache = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_workflow(self, name: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Load workflow definition by name"""
        # Try to load from workflows directory
        workflows_dir = Path(__file__).parent.parent / "workflows"
        workflow_file = workflows_dir / f"{name}.json"
        
        if workflow_file.exists():
            with open(workflow_file, 'r') as f:
                workflow = json.load(f)
        else:
            # Use built-in workflow templates
            workflow = self._get_builtin_workflow(name)
        
        # Apply parameters
        if params:
            workflow = self._apply_parameters(workflow, params)
            
        return workflow
    
    def _get_builtin_workflow(self, name: str) -> Dict[str, Any]:
        """Get built-in workflow templates"""
        templates = {
            "deploy-api": {
                "name": "API Deployment",
                "description": "Deploy API to production environment",
                "steps": [
                    {
                        "id": "build",
                        "name": "Build Application",
                        "action": "shell",
                        "command": "npm run build",
                        "timeout": 300
                    },
                    {
                        "id": "test",
                        "name": "Run Tests",
                        "action": "shell", 
                        "command": "npm test",
                        "timeout": 600
                    },
                    {
                        "id": "deploy",
                        "name": "Deploy to Production",
                        "action": "shell",
                        "command": "npm run deploy",
                        "timeout": 900
                    },
                    {
                        "id": "verify",
                        "name": "Verify Deployment",
                        "action": "http",
                        "url": "https://api.example.com/health",
                        "method": "GET",
                        "expected_status": 200
                    }
                ]
            },
            "test-suite": {
                "name": "Test Suite Execution",
                "description": "Run comprehensive test suite",
                "steps": [
                    {
                        "id": "unit-tests",
                        "name": "Unit Tests",
                        "action": "shell",
                        "command": "npm run test:unit",
                        "timeout": 300
                    },
                    {
                        "id": "integration-tests", 
                        "name": "Integration Tests",
                        "action": "shell",
                        "command": "npm run test:integration",
                        "timeout": 600
                    },
                    {
                        "id": "e2e-tests",
                        "name": "End-to-End Tests",
                        "action": "shell",
                        "command": "npm run test:e2e",
                        "timeout": 900
                    }
                ]
            },
            "video-generation": {
                "name": "Video Generation Pipeline",
                "description": "Generate AI video content",
                "steps": [
                    {
                        "id": "script",
                        "name": "Generate Script",
                        "action": "video_task",
                        "phase": "script_generation"
                    },
                    {
                        "id": "audio",
                        "name": "Generate Audio",
                        "action": "video_task", 
                        "phase": "audio_generation"
                    },
                    {
                        "id": "video",
                        "name": "Generate Video",
                        "action": "video_task",
                        "phase": "video_generation"
                    }
                ]
            }
        }
        
        if name not in templates:
            raise ValueError(f"Unknown workflow template: {name}")
            
        return templates[name]
    
    def _apply_parameters(self, workflow: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameters to workflow template"""
        workflow_str = json.dumps(workflow)
        
        # Replace parameter placeholders
        for key, value in params.items():
            placeholder = f"{{{key}}}"
            workflow_str = workflow_str.replace(placeholder, str(value))
            
        return json.loads(workflow_str)
    
    def validate_workflow(self, workflow: Dict[str, Any]) -> List[str]:
        """Validate workflow structure"""
        errors = []
        
        # Check required fields
        if 'name' not in workflow:
            errors.append("Workflow must have a 'name' field")
        if 'steps' not in workflow:
            errors.append("Workflow must have a 'steps' field")
        elif not isinstance(workflow['steps'], list):
            errors.append("Workflow 'steps' must be a list")
        
        # Validate steps
        for i, step in enumerate(workflow.get('steps', [])):
            if not isinstance(step, dict):
                errors.append(f"Step {i} must be a dictionary")
                continue
                
            if 'id' not in step:
                errors.append(f"Step {i} must have an 'id' field")
            if 'name' not in step:
                errors.append(f"Step {i} must have a 'name' field")
            if 'action' not in step:
                errors.append(f"Step {i} must have an 'action' field")
                
        return errors
    
    def execute_workflow(self, workflow: Dict[str, Any]) -> WorkflowExecutionResult:
        """Execute workflow with monitoring and error handling"""
        start_time = time.time()
        workflow_name = workflow.get('name', 'Unknown Workflow')
        
        self.logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Starting workflow: {workflow_name}")
        
        # Validate workflow
        errors = self.validate_workflow(workflow)
        if errors:
            error_msg = f"Workflow validation failed: {'; '.join(errors)}"
            self.logger.error(error_msg)
            return WorkflowExecutionResult(
                success=False,
                workflow_id="",
                total_steps=0,
                completed_steps=0,
                failed_steps=0,
                execution_time=time.time() - start_time,
                output_data={},
                error_message=error_msg
            )
        
        # Create monitored workflow if available
        workflow_id = ""
        if WorkflowMonitor and not self.dry_run:
            try:
                workflow_id = create_workflow(
                    workflow_name,
                    workflow.get('description', 'Workflow execution')
                )
                self.logger.info(f"Created monitored workflow: {workflow_id}")
            except Exception as e:
                self.logger.warning(f"Failed to create monitored workflow: {e}")
        
        steps = workflow.get('steps', [])
        total_steps = len(steps)
        completed_steps = 0
        failed_steps = 0
        output_data = {}
        
        # Execute steps
        for step in steps:
            step_id = step.get('id', f"step_{completed_steps}")
            step_name = step.get('name', step_id)
            
            self.logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Executing step: {step_name}")
            
            # Add step to monitor if available
            if WorkflowMonitor and workflow_id and not self.dry_run:
                try:
                    add_step(workflow_id, step_name, step.get('description', ''))
                except Exception as e:
                    self.logger.warning(f"Failed to add monitored step: {e}")
            
            try:
                if not self.dry_run:
                    step_result = self._execute_step(step)
                    output_data[step_id] = step_result
                else:
                    # Dry run - just validate step
                    self._validate_step(step)
                    output_data[step_id] = {"dry_run": True, "status": "would_execute"}
                
                completed_steps += 1
                
                # Mark step complete in monitor
                if WorkflowMonitor and workflow_id and not self.dry_run:
                    try:
                        complete_step(workflow_id, step_id)
                    except Exception as e:
                        self.logger.warning(f"Failed to complete monitored step: {e}")
                        
                self.logger.info(f"✅ Step completed: {step_name}")
                
            except Exception as e:
                failed_steps += 1
                error_msg = f"Step failed: {step_name} - {str(e)}"
                self.logger.error(error_msg)
                
                # Mark step failed in monitor
                if WorkflowMonitor and workflow_id and not self.dry_run:
                    try:
                        fail_step(workflow_id, step_id, str(e))
                    except Exception as monitor_e:
                        self.logger.warning(f"Failed to mark monitored step as failed: {monitor_e}")
                
                output_data[step_id] = {"error": str(e), "status": "failed"}
                
                # Stop execution on failure (could be made configurable)
                break
        
        execution_time = time.time() - start_time
        success = failed_steps == 0
        
        result = WorkflowExecutionResult(
            success=success,
            workflow_id=workflow_id,
            total_steps=total_steps,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            execution_time=execution_time,
            output_data=output_data,
            error_message=None if success else f"Workflow failed with {failed_steps} failed steps"
        )
        
        self.logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Workflow completed: "
                        f"{completed_steps}/{total_steps} steps successful "
                        f"in {execution_time:.2f}s")
        
        return result
    
    def _validate_step(self, step: Dict[str, Any]) -> None:
        """Validate step in dry-run mode"""
        action = step.get('action')
        
        if action == 'shell':
            command = step.get('command')
            if not command:
                raise ValueError("Shell action requires 'command' field")
        elif action == 'http':
            url = step.get('url')
            if not url:
                raise ValueError("HTTP action requires 'url' field")
        elif action == 'video_task':
            phase = step.get('phase')
            if not phase:
                raise ValueError("Video task action requires 'phase' field")
    
    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual workflow step"""
        action = step.get('action')
        
        if action == 'shell':
            return self._execute_shell_step(step)
        elif action == 'http':
            return self._execute_http_step(step)
        elif action == 'video_task':
            return self._execute_video_step(step)
        else:
            raise ValueError(f"Unknown action type: {action}")
    
    def _execute_shell_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell command step"""
        import subprocess
        
        command = step.get('command')
        timeout = step.get('timeout', 300)  # 5 minute default
        cwd = step.get('cwd')
        
        self.logger.debug(f"Executing command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )
            
            return {
                "status": "success" if result.returncode == 0 else "error",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command
            }
            
        except subprocess.TimeoutExpired:
            raise Exception(f"Command timed out after {timeout} seconds")
        except Exception as e:
            raise Exception(f"Command execution failed: {str(e)}")
    
    def _execute_http_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP request step"""
        import requests
        
        url = step.get('url')
        method = step.get('method', 'GET').upper()
        headers = step.get('headers', {})
        data = step.get('data')
        timeout = step.get('timeout', 30)
        expected_status = step.get('expected_status', 200)
        
        self.logger.debug(f"Making {method} request to: {url}")
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                timeout=timeout
            )
            
            success = response.status_code == expected_status
            
            return {
                "status": "success" if success else "error",
                "status_code": response.status_code,
                "expected_status": expected_status,
                "response_time": response.elapsed.total_seconds(),
                "url": url,
                "method": method
            }
            
        except Exception as e:
            raise Exception(f"HTTP request failed: {str(e)}")
    
    def _execute_video_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute video processing step"""
        if not WorkflowMonitor:
            raise Exception("Video processing requires MoneyPrinterTurbo modules")
            
        phase = step.get('phase')
        task_id = step.get('task_id', 'workflow_task')
        video_params = step.get('params', {})
        
        self.logger.debug(f"Executing video task phase: {phase}")
        
        try:
            # This would integrate with the actual video processing system
            result = start_monitored_video_task(task_id, video_params)
            
            return {
                "status": "success",
                "phase": phase,
                "result": result,
                "task_id": task_id
            }
            
        except Exception as e:
            raise Exception(f"Video processing failed: {str(e)}")


def main():
    """Main CLI interface for workflow execution"""
    parser = argparse.ArgumentParser(description='Execute Claude Flow workflows')
    parser.add_argument('--name', required=True, help='Workflow name')
    parser.add_argument('--params', help='Workflow parameters (JSON string)')
    parser.add_argument('--dry-run', action='store_true', help='Preview execution without running')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Parse parameters
    params = {}
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"Error parsing parameters: {e}")
            sys.exit(1)
    
    # Initialize executor
    executor = WorkflowExecutor(dry_run=args.dry_run, verbose=args.verbose)
    
    try:
        # Load and execute workflow
        workflow = executor.load_workflow(args.name, params)
        result = executor.execute_workflow(workflow)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"Results written to: {args.output}")
        
        # Print summary
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        print(f"\n{status}")
        print(f"Workflow: {workflow.get('name', 'Unknown')}")
        print(f"Steps: {result.completed_steps}/{result.total_steps} completed")
        print(f"Time: {result.execution_time:.2f}s")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
        
        if args.verbose and result.output_data:
            print("\nStep Results:")
            for step_id, step_result in result.output_data.items():
                print(f"  {step_id}: {step_result.get('status', 'unknown')}")
        
        sys.exit(0 if result.success else 1)
        
    except Exception as e:
        print(f"❌ Execution failed: {str(e)}")
        if args.verbose:
            print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()