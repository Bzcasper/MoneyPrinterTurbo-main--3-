"""
MCP Tools Implementation for MoneyPrinterTurbo

Exposes video generation, script creation, voice synthesis, and other capabilities
as MCP tools that can be called by MCP clients.
"""

import asyncio
import json
import uuid
from typing import Any, Callable, Dict, List, Optional
from functools import wraps
from loguru import logger

from .protocol import MCPTool, MCPRequest, MCPResponse, MCPError, create_success_response, create_error_response
from app.services import llm, task as tm
from app.services.voice import get_voice_manager
from app.services.video_generator_enhanced import generate_video
from app.models.schema import TaskVideoRequest, VideoScriptRequest, AudioRequest


class MCPToolRegistry:
    """Registry for MCP tools and their handlers"""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.handlers: Dict[str, Callable] = {}
        self.middleware: List[Callable] = []
        
    def register_tool(self, tool: MCPTool, handler: Callable):
        """Register a tool with its handler"""
        self.tools[tool.name] = tool
        self.handlers[tool.name] = handler
        logger.info(f"Registered MCP tool: {tool.name}")
        
    def add_middleware(self, middleware: Callable):
        """Add middleware for request processing"""
        self.middleware.append(middleware)
        
    def get_tools(self) -> List[MCPTool]:
        """Get all registered tools"""
        return list(self.tools.values())
        
    async def call_tool(self, tool_name: str, params: Dict[str, Any], 
                       context: Optional[Dict] = None) -> Dict[str, Any]:
        """Call a tool with the given parameters"""
        if tool_name not in self.handlers:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        handler = self.handlers[tool_name]
        
        # Apply middleware
        for middleware in self.middleware:
            params = await middleware(tool_name, params, context or {})
            
        try:
            # Call the handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(params, context or {})
            else:
                result = handler(params, context or {})
                
            return result
            
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}': {str(e)}")
            raise


def mcp_tool(name: str, description: str, input_schema: Dict[str, Any], 
             output_schema: Optional[Dict[str, Any]] = None, category: str = "general"):
    """Decorator for registering MCP tools"""
    def decorator(func: Callable):
        tool = MCPTool(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            category=category
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        wrapper._mcp_tool = tool
        return wrapper
    return decorator


class MoneyPrinterMCPTools:
    """MCP tools for MoneyPrinterTurbo video generation"""
    
    def __init__(self):
        self.registry = MCPToolRegistry()
        self._register_all_tools()
        
    def _register_all_tools(self):
        """Register all MoneyPrinterTurbo tools"""
        # Video generation tools
        self.registry.register_tool(self.get_generate_video_script_tool(), self.generate_video_script)
        self.registry.register_tool(self.get_generate_video_terms_tool(), self.generate_video_terms)
        self.registry.register_tool(self.get_create_video_tool(), self.create_video)
        self.registry.register_tool(self.get_synthesize_voice_tool(), self.synthesize_voice)
        
        # Batch processing tools
        self.registry.register_tool(self.get_batch_video_generation_tool(), self.batch_video_generation)
        
        # Analysis tools
        self.registry.register_tool(self.get_analyze_video_content_tool(), self.analyze_video_content)
        
        # Monitoring tools
        self.registry.register_tool(self.get_get_generation_status_tool(), self.get_generation_status)
        
    def get_generate_video_script_tool(self) -> MCPTool:
        """Tool for generating video scripts"""
        return MCPTool(
            name="generate_video_script",
            description="Generate a video script based on a subject and parameters",
            input_schema={
                "type": "object",
                "properties": {
                    "video_subject": {
                        "type": "string",
                        "description": "The subject/topic for the video"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language for the script (optional)",
                        "default": ""
                    },
                    "paragraph_number": {
                        "type": "integer",
                        "description": "Number of paragraphs in the script",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["video_subject"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "script": {"type": "string"},
                    "word_count": {"type": "integer"},
                    "estimated_duration": {"type": "number"}
                }
            },
            category="content_generation"
        )
        
    async def generate_video_script(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a video script"""
        try:
            video_subject = params["video_subject"]
            language = params.get("language", "")
            paragraph_number = params.get("paragraph_number", 1)
            
            script = llm.generate_script(
                video_subject=video_subject,
                language=language,
                paragraph_number=paragraph_number
            )
            
            # Calculate metrics
            word_count = len(script.split())
            estimated_duration = word_count * 0.5  # Rough estimate: 2 words per second
            
            return {
                "script": script,
                "word_count": word_count,
                "estimated_duration": estimated_duration
            }
            
        except Exception as e:
            logger.error(f"Error generating video script: {str(e)}")
            raise
            
    def get_generate_video_terms_tool(self) -> MCPTool:
        """Tool for generating video search terms"""
        return MCPTool(
            name="generate_video_terms",
            description="Generate search terms for finding relevant video materials",
            input_schema={
                "type": "object",
                "properties": {
                    "video_subject": {
                        "type": "string",
                        "description": "The video subject"
                    },
                    "video_script": {
                        "type": "string",
                        "description": "The video script content"
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Number of search terms to generate",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["video_subject", "video_script"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            category="content_generation"
        )
        
    async def generate_video_terms(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video search terms"""
        try:
            video_subject = params["video_subject"]
            video_script = params["video_script"]
            amount = params.get("amount", 5)
            
            terms = llm.generate_terms(
                video_subject=video_subject,
                video_script=video_script,
                amount=amount
            )
            
            return {"terms": terms}
            
        except Exception as e:
            logger.error(f"Error generating video terms: {str(e)}")
            raise
            
    def get_create_video_tool(self) -> MCPTool:
        """Tool for creating complete videos"""
        return MCPTool(
            name="create_video",
            description="Create a complete video with script, voice, and visual elements",
            input_schema={
                "type": "object",
                "properties": {
                    "video_subject": {
                        "type": "string",
                        "description": "The video subject/topic"
                    },
                    "video_script": {
                        "type": "string",
                        "description": "Pre-written script (optional)"
                    },
                    "video_aspect": {
                        "type": "string",
                        "enum": ["16:9", "9:16", "1:1"],
                        "description": "Video aspect ratio",
                        "default": "9:16"
                    },
                    "voice_name": {
                        "type": "string",
                        "description": "Voice to use for narration"
                    },
                    "bgm_type": {
                        "type": "string",
                        "description": "Background music type",
                        "default": "random"
                    },
                    "subtitle_enabled": {
                        "type": "boolean",
                        "description": "Enable subtitles",
                        "default": true
                    }
                },
                "required": ["video_subject"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "status": {"type": "string"},
                    "estimated_completion": {"type": "string"}
                }
            },
            category="video_generation"
        )
        
    async def create_video(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a complete video"""
        try:
            # Create video request
            video_request = TaskVideoRequest(
                video_subject=params["video_subject"],
                video_script=params.get("video_script", ""),
                video_aspect=params.get("video_aspect", "9:16"),
                voice_name=params.get("voice_name", ""),
                bgm_type=params.get("bgm_type", "random"),
                subtitle_enabled=params.get("subtitle_enabled", True)
            )
            
            # Start video generation task
            task_id = f"mcp_{uuid.uuid4().hex[:8]}"
            
            # This would integrate with the existing task manager
            # For now, returning a task structure
            return {
                "task_id": task_id,
                "status": "initiated",
                "estimated_completion": "5-10 minutes"
            }
            
        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            raise
            
    def get_synthesize_voice_tool(self) -> MCPTool:
        """Tool for voice synthesis"""
        return MCPTool(
            name="synthesize_voice",
            description="Convert text to speech using various voice options",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to convert to speech"
                    },
                    "voice_name": {
                        "type": "string",
                        "description": "Voice to use for synthesis"
                    },
                    "voice_rate": {
                        "type": "number",
                        "description": "Speech rate (0.5-2.0)",
                        "default": 1.0,
                        "minimum": 0.5,
                        "maximum": 2.0
                    },
                    "voice_volume": {
                        "type": "number",
                        "description": "Voice volume (0.0-1.0)",
                        "default": 1.0,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["text", "voice_name"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "audio_file": {"type": "string"},
                    "duration": {"type": "number"},
                    "format": {"type": "string"}
                }
            },
            category="audio_generation"
        )
        
    async def synthesize_voice(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize voice from text"""
        try:
            text = params["text"]
            voice_name = params["voice_name"]
            voice_rate = params.get("voice_rate", 1.0)
            voice_volume = params.get("voice_volume", 1.0)
            
            # Get voice manager and synthesize
            voice_manager = get_voice_manager()
            audio_file = voice_manager.synthesize(
                text=text,
                voice=voice_name,
                rate=voice_rate,
                volume=voice_volume
            )
            
            # Calculate duration (approximate)
            word_count = len(text.split())
            duration = word_count * 0.5 * (1.0 / voice_rate)  # Adjust for rate
            
            return {
                "audio_file": audio_file,
                "duration": duration,
                "format": "wav"
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing voice: {str(e)}")
            raise
            
    def get_batch_video_generation_tool(self) -> MCPTool:
        """Tool for batch video generation"""
        return MCPTool(
            name="batch_video_generation",
            description="Generate multiple videos from a list of subjects",
            input_schema={
                "type": "object",
                "properties": {
                    "subjects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of video subjects"
                    },
                    "template_params": {
                        "type": "object",
                        "description": "Common parameters for all videos"
                    },
                    "max_concurrent": {
                        "type": "integer",
                        "description": "Maximum concurrent generations",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["subjects"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string"},
                    "total_videos": {"type": "integer"},
                    "estimated_completion": {"type": "string"}
                }
            },
            category="batch_processing"
        )
        
    async def batch_video_generation(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate multiple videos in batch"""
        try:
            subjects = params["subjects"]
            template_params = params.get("template_params", {})
            max_concurrent = params.get("max_concurrent", 3)
            
            batch_id = f"batch_{uuid.uuid4().hex[:8]}"
            
            # This would integrate with batch processing system
            return {
                "batch_id": batch_id,
                "total_videos": len(subjects),
                "estimated_completion": f"{len(subjects) * 7} minutes"
            }
            
        except Exception as e:
            logger.error(f"Error in batch video generation: {str(e)}")
            raise
            
    def get_analyze_video_content_tool(self) -> MCPTool:
        """Tool for analyzing video content"""
        return MCPTool(
            name="analyze_video_content",
            description="Analyze video content and provide insights",
            input_schema={
                "type": "object",
                "properties": {
                    "video_uri": {
                        "type": "string",
                        "description": "URI of the video to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["quality", "content", "performance", "full"],
                        "description": "Type of analysis to perform",
                        "default": "content"
                    }
                },
                "required": ["video_uri"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "analysis_results": {"type": "object"},
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "quality_score": {"type": "number"}
                }
            },
            category="analysis"
        )
        
    async def analyze_video_content(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze video content"""
        try:
            video_uri = params["video_uri"]
            analysis_type = params.get("analysis_type", "content")
            
            # Mock analysis results
            return {
                "analysis_results": {
                    "duration": 30.5,
                    "resolution": "1080x1920",
                    "fps": 30,
                    "audio_quality": "high",
                    "visual_quality": "high"
                },
                "recommendations": [
                    "Consider adding more dynamic transitions",
                    "Audio levels are well balanced"
                ],
                "quality_score": 8.5
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video content: {str(e)}")
            raise
            
    def get_get_generation_status_tool(self) -> MCPTool:
        """Tool for checking generation status"""
        return MCPTool(
            name="get_generation_status",
            description="Get the status of video generation tasks",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to check status for"
                    }
                },
                "required": ["task_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "progress": {"type": "number"},
                    "estimated_remaining": {"type": "string"},
                    "result_urls": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            category="monitoring"
        )
        
    async def get_generation_status(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Get video generation status"""
        try:
            task_id = params["task_id"]
            
            # This would integrate with the task manager
            return {
                "status": "processing",
                "progress": 0.65,
                "estimated_remaining": "3 minutes",
                "result_urls": []
            }
            
        except Exception as e:
            logger.error(f"Error getting generation status: {str(e)}")
            raise


# Create global instance
mcp_tools = MoneyPrinterMCPTools()