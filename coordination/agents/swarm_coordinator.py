"""
Swarm Coordinator for Hive-Mind Agent Management
Handles spawning, coordination, and lifecycle management of specialized agents
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
import logging
from contextlib import asynccontextmanager

from app.services.hive_memory import get_hive_memory, log_swarm_event

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Enumeration of available agent types"""
    RESEARCHER = "researcher"
    CODER = "coder" 
    ANALYST = "analyst"
    TESTER = "tester"
    VIDEO_PROCESSOR = "video_processor"
    TTS_COORDINATOR = "tts_coordinator"
    DATABASE_MANAGER = "database_manager"
    API_COORDINATOR = "api_coordinator"


class AgentStatus(Enum):
    """Agent lifecycle status"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class AgentConfig:
    """Configuration for agent initialization"""
    agent_type: AgentType
    agent_name: str
    specialization: Dict[str, Any]
    resource_limits: Dict[str, Any]
    communication_channels: List[str]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['agent_type'] = self.agent_type.value
        return data


@dataclass
class AgentInstance:
    """Runtime agent instance"""
    agent_id: str
    config: AgentConfig
    status: AgentStatus
    session_id: str
    created_at: datetime
    last_heartbeat: datetime
    task_queue: List[str]
    performance_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage and communication"""
        return {
            'agent_id': self.agent_id,
            'config': self.config.to_dict(),
            'status': self.status.value,
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'task_queue': self.task_queue,
            'performance_metrics': self.performance_metrics
        }


class SwarmCoordinator:
    """Central coordinator for managing agent swarms"""
    
    def __init__(self, session_id: str, max_agents: int = 8):
        self.session_id = session_id
        self.max_agents = max_agents
        self.agents: Dict[str, AgentInstance] = {}
        self.agent_configs: Dict[AgentType, AgentConfig] = {}
        self.hive_memory = get_hive_memory()
        self.heartbeat_interval = 30  # seconds
        self.task_distribution_lock = asyncio.Lock()
        
        # Initialize with default agent configurations
        self._setup_default_configs()
        
        # Create session in hive memory if it doesn't exist
        self._initialize_session()
    
    def _setup_default_configs(self):
        """Setup default configurations for each agent type"""
        
        # Video Processor Agent Configuration
        self.agent_configs[AgentType.VIDEO_PROCESSOR] = AgentConfig(
            agent_type=AgentType.VIDEO_PROCESSOR,
            agent_name="Video Processing Specialist",
            specialization={
                "video_formats": ["mp4", "avi", "mov", "webm"],
                "codecs": ["h264", "h265", "vp9"],
                "quality_enhancement": True,
                "batch_processing": True,
                "gpu_acceleration": True
            },
            resource_limits={
                "max_concurrent_tasks": 3,
                "memory_limit_mb": 2048,
                "processing_timeout": 1800  # 30 minutes
            },
            communication_channels=["video_queue", "status_updates", "error_reports"]
        )
        
        # TTS Coordinator Agent Configuration
        self.agent_configs[AgentType.TTS_COORDINATOR] = AgentConfig(
            agent_type=AgentType.TTS_COORDINATOR,
            agent_name="Text-to-Speech Coordinator",
            specialization={
                "tts_engines": ["edge_tts", "google_tts", "gpt_sovits", "characterbox"],
                "voice_cloning": True,
                "multilingual": True,
                "audio_enhancement": True,
                "batch_synthesis": True
            },
            resource_limits={
                "max_concurrent_requests": 5,
                "memory_limit_mb": 1024,
                "request_timeout": 300  # 5 minutes
            },
            communication_channels=["tts_queue", "voice_status", "audio_delivery"]
        )
        
        # Database Manager Agent Configuration
        self.agent_configs[AgentType.DATABASE_MANAGER] = AgentConfig(
            agent_type=AgentType.DATABASE_MANAGER,
            agent_name="Database Operations Manager",
            specialization={
                "databases": ["sqlite", "postgresql", "redis"],
                "operations": ["crud", "migrations", "backup", "optimization"],
                "analytics": True,
                "real_time_sync": True,
                "data_validation": True
            },
            resource_limits={
                "max_connections": 20,
                "memory_limit_mb": 512,
                "query_timeout": 60  # 1 minute
            },
            communication_channels=["db_operations", "data_events", "maintenance_alerts"]
        )
        
        # API Coordinator Agent Configuration
        self.agent_configs[AgentType.API_COORDINATOR] = AgentConfig(
            agent_type=AgentType.API_COORDINATOR,
            agent_name="API Request Coordinator",
            specialization={
                "protocols": ["http", "websocket", "grpc"],
                "rate_limiting": True,
                "load_balancing": True,
                "circuit_breaker": True,
                "request_routing": True
            },
            resource_limits={
                "max_concurrent_requests": 100,
                "memory_limit_mb": 256,
                "request_timeout": 30  # 30 seconds
            },
            communication_channels=["api_requests", "response_delivery", "system_metrics"]
        )
    
    def _initialize_session(self):
        """Initialize session in hive memory"""
        try:
            # Check if session exists
            session_status = self.hive_memory.get_session_status(self.session_id)
            if not session_status:
                # Create new session
                success = self.hive_memory.create_session(
                    session_id=self.session_id,
                    topology="distributed_swarm",
                    max_agents=self.max_agents,
                    strategy="adaptive_coordination",
                    metadata={
                        "coordinator_version": "1.0.0",
                        "created_by": "SwarmCoordinator",
                        "agent_types": [agent_type.value for agent_type in AgentType]
                    }
                )
                if success:
                    logger.info(f"Created new hive session: {self.session_id}")
                else:
                    logger.error(f"Failed to create session: {self.session_id}")
            else:
                logger.info(f"Using existing session: {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
    
    async def spawn_agent(self, agent_type: AgentType, 
                         custom_config: Optional[AgentConfig] = None) -> Optional[str]:
        """Spawn a new agent of the specified type"""
        try:
            # Check if we've reached max agents
            if len(self.agents) >= self.max_agents:
                logger.warning(f"Maximum agents ({self.max_agents}) reached")
                return None
            
            # Generate unique agent ID
            agent_id = f"worker-{self.session_id}-{len(self.agents)}"
            
            # Use custom config or default
            config = custom_config or self.agent_configs.get(agent_type)
            if not config:
                logger.error(f"No configuration found for agent type: {agent_type}")
                return None
            
            # Create agent instance
            now = datetime.now()
            agent = AgentInstance(
                agent_id=agent_id,
                config=config,
                status=AgentStatus.INITIALIZING,
                session_id=self.session_id,
                created_at=now,
                last_heartbeat=now,
                task_queue=[],
                performance_metrics={
                    "tasks_completed": 0,
                    "tasks_failed": 0,
                    "average_response_time": 0,
                    "uptime_seconds": 0
                }
            )
            
            # Register agent in hive memory
            success = self.hive_memory.register_agent(
                agent_id=agent_id,
                session_id=self.session_id,
                agent_type=agent_type.value,
                agent_name=config.agent_name,
                metadata=config.to_dict()
            )
            
            if not success:
                logger.error(f"Failed to register agent {agent_id} in hive memory")
                return None
            
            # Add to local registry
            self.agents[agent_id] = agent
            
            # Log spawn event
            log_swarm_event(
                session_id=self.session_id,
                agent_id=agent_id,
                event_type="agent_spawned",
                event_data={
                    "agent_type": agent_type.value,
                    "config": config.to_dict()
                }
            )
            
            # Initialize agent (start its processes)
            await self._initialize_agent(agent)
            
            logger.info(f"Successfully spawned agent {agent_id} of type {agent_type.value}")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to spawn agent {agent_type.value}: {e}")
            return None
    
    async def _initialize_agent(self, agent: AgentInstance):
        """Initialize agent processes and set to active status"""
        try:
            agent.status = AgentStatus.ACTIVE
            agent.last_heartbeat = datetime.now()
            
            # Start heartbeat task
            asyncio.create_task(self._agent_heartbeat(agent.agent_id))
            
            # Initialize agent-specific processes
            await self._start_agent_processes(agent)
            
            logger.debug(f"Agent {agent.agent_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {agent.agent_id}: {e}")
            agent.status = AgentStatus.ERROR
    
    async def _start_agent_processes(self, agent: AgentInstance):
        """Start agent-specific background processes"""
        agent_type = agent.config.agent_type
        
        if agent_type == AgentType.VIDEO_PROCESSOR:
            asyncio.create_task(self._video_processor_loop(agent.agent_id))
        elif agent_type == AgentType.TTS_COORDINATOR:
            asyncio.create_task(self._tts_coordinator_loop(agent.agent_id))
        elif agent_type == AgentType.DATABASE_MANAGER:
            asyncio.create_task(self._database_manager_loop(agent.agent_id))
        elif agent_type == AgentType.API_COORDINATOR:
            asyncio.create_task(self._api_coordinator_loop(agent.agent_id))
    
    async def _agent_heartbeat(self, agent_id: str):
        """Maintain agent heartbeat"""
        while agent_id in self.agents:
            try:
                agent = self.agents[agent_id]
                if agent.status == AgentStatus.TERMINATED:
                    break
                
                agent.last_heartbeat = datetime.now()
                
                # Update performance metrics
                uptime = (agent.last_heartbeat - agent.created_at).total_seconds()
                agent.performance_metrics["uptime_seconds"] = uptime
                
                # Log heartbeat event
                log_swarm_event(
                    session_id=self.session_id,
                    agent_id=agent_id,
                    event_type="heartbeat",
                    event_data={
                        "status": agent.status.value,
                        "uptime": uptime,
                        "task_queue_size": len(agent.task_queue)
                    }
                )
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat failed for agent {agent_id}: {e}")
                if agent_id in self.agents:
                    self.agents[agent_id].status = AgentStatus.ERROR
                break
    
    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status"""
        try:
            agent_statuses = {}
            for agent_id, agent in self.agents.items():
                agent_statuses[agent_id] = {
                    "type": agent.config.agent_type.value,
                    "name": agent.config.agent_name,
                    "status": agent.status.value,
                    "uptime": (datetime.now() - agent.created_at).total_seconds(),
                    "task_queue_size": len(agent.task_queue),
                    "performance": agent.performance_metrics
                }
            
            return {
                "session_id": self.session_id,
                "total_agents": len(self.agents),
                "max_agents": self.max_agents,
                "agent_statuses": agent_statuses,
                "coordinator_status": "active",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get swarm status: {e}")
            return {"error": str(e)}
    
    # Placeholder methods for agent-specific processing loops
    async def _video_processor_loop(self, agent_id: str):
        """Video processor agent main loop"""
        logger.info(f"Video processor {agent_id} started")
        # Implementation will be added in subsequent agents
        await asyncio.sleep(1)
    
    async def _tts_coordinator_loop(self, agent_id: str):
        """TTS coordinator agent main loop"""
        logger.info(f"TTS coordinator {agent_id} started")
        # Implementation will be added in subsequent agents
        await asyncio.sleep(1)
    
    async def _database_manager_loop(self, agent_id: str):
        """Database manager agent main loop"""
        logger.info(f"Database manager {agent_id} started")
        # Implementation will be added in subsequent agents
        await asyncio.sleep(1)
    
    async def _api_coordinator_loop(self, agent_id: str):
        """API coordinator agent main loop"""
        logger.info(f"API coordinator {agent_id} started")
        # Implementation will be added in subsequent agents
        await asyncio.sleep(1)


# Global swarm coordinator instance
_swarm_coordinator: Optional[SwarmCoordinator] = None


def get_swarm_coordinator(session_id: str, max_agents: int = 8) -> SwarmCoordinator:
    """Get or create the global swarm coordinator"""
    global _swarm_coordinator
    if _swarm_coordinator is None:
        _swarm_coordinator = SwarmCoordinator(session_id, max_agents)
    return _swarm_coordinator


async def spawn_remaining_agents(session_id: str) -> Dict[str, str]:
    """Spawn the remaining 4 agents to complete the 8-agent swarm"""
    coordinator = get_swarm_coordinator(session_id)
    
    # Define the missing agent types based on the session analysis
    missing_agents = [
        AgentType.VIDEO_PROCESSOR,
        AgentType.TTS_COORDINATOR, 
        AgentType.DATABASE_MANAGER,
        AgentType.API_COORDINATOR
    ]
    
    spawned_agents = {}
    
    for agent_type in missing_agents:
        try:
            agent_id = await coordinator.spawn_agent(agent_type)
            if agent_id:
                spawned_agents[agent_type.value] = agent_id
                logger.info(f"Successfully spawned {agent_type.value}: {agent_id}")
            else:
                logger.error(f"Failed to spawn {agent_type.value}")
        except Exception as e:
            logger.error(f"Error spawning {agent_type.value}: {e}")
    
    return spawned_agents