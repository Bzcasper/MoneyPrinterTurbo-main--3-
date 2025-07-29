"""
Workflow loading and parsing utilities.
"""

import json
import yaml
import os
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from ..core.workflow import Workflow, WorkflowStep


logger = logging.getLogger(__name__)


class WorkflowLoadError(Exception):
    """Raised when workflow loading fails."""
    pass


class WorkflowLoader:
    """
    Comprehensive workflow loading system with support for multiple formats.
    
    Supported formats:
    - JSON (.json)
    - YAML (.yaml, .yml)
    - Python dictionary
    - Template-based workflows with variable substitution
    """
    
    def __init__(self, template_dirs: Optional[List[str]] = None):
        """
        Initialize the workflow loader.
        
        Args:
            template_dirs: List of directories to search for workflow templates
        """
        self.template_dirs = template_dirs or []
        self.logger = logging.getLogger(__name__)
        
        # Built-in workflow templates
        self.built_in_templates = {
            "simple_file_workflow": self._create_simple_file_template,
            "api_integration_workflow": self._create_api_integration_template,
            "deployment_workflow": self._create_deployment_template,
            "data_processing_workflow": self._create_data_processing_template
        }
        
        self.logger.info(f"Workflow loader initialized with {len(self.template_dirs)} template directories")
    
    def load_from_file(self, file_path: str, variables: Optional[Dict[str, Any]] = None) -> Workflow:
        """
        Load workflow from file.
        
        Args:
            file_path: Path to workflow file
            variables: Variables for template substitution
            
        Returns:
            Loaded workflow
            
        Raises:
            WorkflowLoadError: If loading fails
        """
        self.logger.info(f"Loading workflow from file: {file_path}")
        
        if not os.path.exists(file_path):
            raise WorkflowLoadError(f"Workflow file not found: {file_path}")
        
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.json':
                workflow_data = self._load_json_file(file_path)
            elif file_extension in ['.yaml', '.yml']:
                workflow_data = self._load_yaml_file(file_path)
            else:
                raise WorkflowLoadError(f"Unsupported file format: {file_extension}")
            
            # Apply variable substitution if provided
            if variables:
                workflow_data = self._substitute_variables(workflow_data, variables)
            
            # Create workflow from data
            workflow = self._create_workflow_from_data(workflow_data)
            
            self.logger.info(f"Successfully loaded workflow '{workflow.name}' with {len(workflow.steps)} steps")
            return workflow
            
        except Exception as e:
            raise WorkflowLoadError(f"Error loading workflow from {file_path}: {str(e)}")
    
    def load_from_dict(self, workflow_data: Dict[str, Any], variables: Optional[Dict[str, Any]] = None) -> Workflow:
        """
        Load workflow from dictionary.
        
        Args:
            workflow_data: Workflow data dictionary
            variables: Variables for template substitution
            
        Returns:
            Loaded workflow
        """
        self.logger.debug("Loading workflow from dictionary")
        
        try:
            # Apply variable substitution if provided
            if variables:
                workflow_data = self._substitute_variables(workflow_data, variables)
            
            # Create workflow from data
            workflow = self._create_workflow_from_data(workflow_data)
            
            self.logger.debug(f"Successfully loaded workflow '{workflow.name}' from dictionary")
            return workflow
            
        except Exception as e:
            raise WorkflowLoadError(f"Error loading workflow from dictionary: {str(e)}")
    
    def load_from_template(self, template_name: str, variables: Optional[Dict[str, Any]] = None) -> Workflow:
        """
        Load workflow from built-in or custom template.
        
        Args:
            template_name: Name of the template
            variables: Variables for template customization
            
        Returns:
            Loaded workflow
        """
        self.logger.info(f"Loading workflow from template: {template_name}")
        
        # Check built-in templates first
        if template_name in self.built_in_templates:
            template_func = self.built_in_templates[template_name]
            workflow_data = template_func(variables or {})
            return self._create_workflow_from_data(workflow_data)
        
        # Search template directories
        for template_dir in self.template_dirs:
            template_path = self._find_template_file(template_dir, template_name)
            if template_path:
                return self.load_from_file(template_path, variables)
        
        raise WorkflowLoadError(f"Template not found: {template_name}")
    
    def _load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load JSON workflow file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """Load YAML workflow file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise WorkflowLoadError("PyYAML is required to load YAML files. Install with: pip install PyYAML")
    
    def _substitute_variables(self, data: Any, variables: Dict[str, Any]) -> Any:
        """Recursively substitute variables in workflow data."""
        if isinstance(data, dict):
            return {key: self._substitute_variables(value, variables) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._substitute_variables(item, variables) for item in data]
        elif isinstance(data, str):
            # Simple variable substitution using format strings
            try:
                return data.format(**variables)
            except (KeyError, ValueError):
                # Return original string if substitution fails
                return data
        else:
            return data
    
    def _create_workflow_from_data(self, workflow_data: Dict[str, Any]) -> Workflow:
        """Create Workflow object from data dictionary."""
        # Validate required fields
        required_fields = ['id', 'name', 'steps']
        for field in required_fields:
            if field not in workflow_data:
                raise WorkflowLoadError(f"Missing required field: {field}")
        
        # Create workflow steps
        steps = []
        for step_data in workflow_data['steps']:
            step = self._create_step_from_data(step_data)
            steps.append(step)
        
        # Create workflow
        workflow = Workflow(
            id=workflow_data['id'],
            name=workflow_data['name'],
            description=workflow_data.get('description', ''),
            steps=steps,
            metadata=workflow_data.get('metadata', {}),
            version=workflow_data.get('version', '1.0')
        )
        
        return workflow
    
    def _create_step_from_data(self, step_data: Dict[str, Any]) -> WorkflowStep:
        """Create WorkflowStep object from data dictionary."""
        # Validate required step fields
        required_fields = ['id', 'name', 'action']
        for field in required_fields:
            if field not in step_data:
                raise WorkflowLoadError(f"Missing required step field: {field}")
        
        step = WorkflowStep(
            id=step_data['id'],
            name=step_data['name'],
            action=step_data['action'],
            parameters=step_data.get('parameters', {}),
            dependencies=step_data.get('dependencies', []),
            rollback_action=step_data.get('rollback_action'),
            rollback_parameters=step_data.get('rollback_parameters', {}),
            timeout=step_data.get('timeout'),
            retry_count=0,
            max_retries=step_data.get('max_retries', 3),
            critical=step_data.get('critical', True),
            parallel_group=step_data.get('parallel_group'),
            conditions=step_data.get('conditions')
        )
        
        return step
    
    def _find_template_file(self, template_dir: str, template_name: str) -> Optional[str]:
        """Find template file in directory."""
        possible_extensions = ['.json', '.yaml', '.yml']
        
        for ext in possible_extensions:
            template_path = os.path.join(template_dir, f"{template_name}{ext}")
            if os.path.exists(template_path):
                return template_path
        
        return None
    
    # Built-in template generators
    
    def _create_simple_file_template(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simple file operation workflow template."""
        source_file = variables.get('source_file', '/path/to/source.txt')
        dest_file = variables.get('dest_file', '/path/to/dest.txt')
        
        return {
            "id": f"simple_file_workflow_{variables.get('suffix', '001')}",
            "name": "Simple File Operations",
            "description": "Basic file read, process, and write workflow",
            "steps": [
                {
                    "id": "read_file",
                    "name": "Read Source File",
                    "action": "file.read",
                    "parameters": {"path": source_file}
                },
                {
                    "id": "process_content",
                    "name": "Process File Content",
                    "action": "variable.set",
                    "parameters": {
                        "name": "processed_content",
                        "value": "Processed: {read_file_result}"
                    },
                    "dependencies": ["read_file"]
                },
                {
                    "id": "write_file",
                    "name": "Write Destination File",
                    "action": "file.write",
                    "parameters": {
                        "path": dest_file,
                        "content": "{processed_content}"
                    },
                    "dependencies": ["process_content"]
                }
            ]
        }
    
    def _create_api_integration_template(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Create an API integration workflow template."""
        api_url = variables.get('api_url', 'https://api.example.com')
        
        return {
            "id": f"api_integration_{variables.get('suffix', '001')}",
            "name": "API Integration Workflow",
            "description": "Fetch data from API and process results",
            "steps": [
                {
                    "id": "fetch_data",
                    "name": "Fetch Data from API",
                    "action": "http.get",
                    "parameters": {
                        "url": f"{api_url}/data",
                        "headers": {"Accept": "application/json"}
                    }
                },
                {
                    "id": "validate_response",
                    "name": "Validate API Response",
                    "action": "validate.condition",
                    "parameters": {
                        "condition": "fetch_data_result is not None"
                    },
                    "dependencies": ["fetch_data"]
                },
                {
                    "id": "save_results",
                    "name": "Save API Results",
                    "action": "file.write",
                    "parameters": {
                        "path": variables.get('output_file', '/tmp/api_results.json'),
                        "content": "{fetch_data_result}"
                    },
                    "dependencies": ["validate_response"]
                }
            ]
        }
    
    def _create_deployment_template(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deployment workflow template."""
        app_name = variables.get('app_name', 'myapp')
        environment = variables.get('environment', 'production')
        
        return {
            "id": f"deployment_{app_name}_{environment}",
            "name": f"Deploy {app_name} to {environment}",
            "description": f"Complete deployment workflow for {app_name}",
            "steps": [
                {
                    "id": "backup_current",
                    "name": "Backup Current Version",
                    "action": "shell.execute",
                    "parameters": {
                        "command": f"cp -r /apps/{app_name} /backups/{app_name}_$(date +%Y%m%d_%H%M%S)"
                    }
                },
                {
                    "id": "stop_services",
                    "name": "Stop Application Services",
                    "action": "shell.execute",
                    "parameters": {
                        "command": f"systemctl stop {app_name}"
                    },
                    "dependencies": ["backup_current"]
                },
                {
                    "id": "deploy_code",
                    "name": "Deploy New Code",
                    "action": "shell.execute",
                    "parameters": {
                        "command": f"rsync -av {variables.get('source_path', '/tmp/deploy/')} /apps/{app_name}/"
                    },
                    "dependencies": ["stop_services"]
                },
                {
                    "id": "update_config",
                    "name": "Update Configuration",
                    "action": "file.write",
                    "parameters": {
                        "path": f"/apps/{app_name}/config.env",
                        "content": f"ENVIRONMENT={environment}\nAPP_NAME={app_name}\n"
                    },
                    "dependencies": ["deploy_code"]
                },
                {
                    "id": "start_services",
                    "name": "Start Application Services",
                    "action": "shell.execute",
                    "parameters": {
                        "command": f"systemctl start {app_name}"
                    },
                    "dependencies": ["update_config"]
                },
                {
                    "id": "health_check",
                    "name": "Verify Deployment Health",
                    "action": "http.get",
                    "parameters": {
                        "url": f"http://localhost:8080/health",
                        "timeout": 30
                    },
                    "dependencies": ["start_services"]
                }
            ]
        }
    
    def _create_data_processing_template(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Create a data processing workflow template."""
        input_file = variables.get('input_file', '/data/input.csv')
        output_file = variables.get('output_file', '/data/output.csv')
        
        return {
            "id": f"data_processing_{variables.get('suffix', '001')}",
            "name": "Data Processing Workflow",
            "description": "Process and transform data files",
            "steps": [
                {
                    "id": "validate_input",
                    "name": "Validate Input File",
                    "action": "validate.condition",
                    "parameters": {
                        "condition": f"os.path.exists('{input_file}')"
                    }
                },
                {
                    "id": "read_data",
                    "name": "Read Input Data",
                    "action": "file.read",
                    "parameters": {"path": input_file},
                    "dependencies": ["validate_input"]
                },
                {
                    "id": "process_data",
                    "name": "Process Data",
                    "action": "shell.execute",
                    "parameters": {
                        "command": f"python /scripts/process_data.py {input_file} {output_file}"
                    },
                    "dependencies": ["read_data"],
                    "parallel_group": "processing"
                },
                {
                    "id": "validate_output",
                    "name": "Validate Output",
                    "action": "validate.condition",
                    "parameters": {
                        "condition": f"os.path.exists('{output_file}')"
                    },
                    "dependencies": ["process_data"]
                },
                {
                    "id": "cleanup_temp",
                    "name": "Cleanup Temporary Files",
                    "action": "shell.execute",
                    "parameters": {
                        "command": "rm -f /tmp/processing_*"
                    },
                    "dependencies": ["validate_output"],
                    "critical": False
                }
            ]
        }
    
    def list_built_in_templates(self) -> List[str]:
        """List available built-in templates."""
        return list(self.built_in_templates.keys())
    
    def save_workflow(self, workflow: Workflow, file_path: str, format: str = 'json'):
        """
        Save workflow to file.
        
        Args:
            workflow: Workflow to save
            file_path: Output file path
            format: Output format ('json' or 'yaml')
        """
        self.logger.info(f"Saving workflow '{workflow.name}' to {file_path}")
        
        workflow_data = workflow.to_dict()
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(workflow_data, f, indent=2)
            elif format.lower() in ['yaml', 'yml']:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(workflow_data, f, default_flow_style=False, indent=2)
                except ImportError:
                    raise WorkflowLoadError("PyYAML is required to save YAML files. Install with: pip install PyYAML")
            else:
                raise WorkflowLoadError(f"Unsupported save format: {format}")
            
            self.logger.info(f"Successfully saved workflow to {file_path}")
            
        except Exception as e:
            raise WorkflowLoadError(f"Error saving workflow to {file_path}: {str(e)}")
    
    def create_template_from_workflow(self, workflow: Workflow, template_name: str, variables: List[str]):
        """
        Create a reusable template from an existing workflow.
        
        Args:
            workflow: Source workflow
            template_name: Name for the template
            variables: List of parameter names to make configurable
        """
        # This would convert specific values to template variables
        # and save as a template for future use
        self.logger.info(f"Creating template '{template_name}' from workflow '{workflow.name}'")
        
        # Implementation would involve identifying parameterizable values
        # and replacing them with template placeholders
        pass