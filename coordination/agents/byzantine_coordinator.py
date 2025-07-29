"""
Byzantine Fault Tolerant Extensions for Swarm Coordinator
Enhances the base coordinator with Byzantine fault detection and recovery
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from .swarm_coordinator import SwarmCoordinator, AgentInstance, AgentStatus, AgentType
from app.services.hive_memory import log_swarm_event

logger = logging.getLogger(__name__)


@dataclass
class ByzantineMetrics:
    """Metrics for detecting Byzantine behavior"""
    health_score: float
    consecutive_failures: int
    last_anomaly_time: Optional[datetime]
    response_time_history: List[float]
    task_success_rate: float
    anomaly_count: int


class ByzantineFaultTolerantCoordinator(SwarmCoordinator):
    """Enhanced coordinator with Byzantine fault tolerance"""
    
    def __init__(self, session_id: str, max_agents: int = 8):
        super().__init__(session_id, max_agents)
        self.byzantine_metrics: Dict[str, ByzantineMetrics] = {}
        self.fault_tolerance_threshold = 0.3  # Health score threshold
        self.max_consecutive_failures = 3
        self.quarantine_agents: Dict[str, datetime] = {}
        self.consensus_quorum = max(2, (max_agents // 2) + 1)
        
    async def calculate_agent_health(self, agent: AgentInstance) -> float:
        """Calculate comprehensive agent health score (0.0 - 1.0)"""
        try:
            metrics = agent.performance_metrics
            
            # Task success rate
            total_tasks = metrics.get("tasks_completed", 0) + metrics.get("tasks_failed", 0)
            task_success_rate = 1.0 if total_tasks == 0 else metrics["tasks_completed"] / total_tasks
            
            # Response time health (healthy if under 5 seconds average)
            avg_response = metrics.get("average_response_time", 1.0)
            response_time_health = min(1.0, 5.0 / max(1.0, avg_response))
            
            # Uptime health (penalize recent restarts)
            uptime_seconds = metrics.get("uptime_seconds", 0)
            uptime_health = min(1.0, uptime_seconds / 3600)  # 1 hour = full health
            
            # Queue health (penalize overloaded agents)
            queue_size = len(agent.task_queue)
            queue_health = max(0.0, 1.0 - (queue_size / 10.0))
            
            # Memory consistency health (check for data corruption)
            memory_health = await self._check_memory_consistency(agent)
            
            # Weighted health score
            health_score = (
                task_success_rate * 0.3 +
                response_time_health * 0.25 +
                uptime_health * 0.15 +
                queue_health * 0.15 +
                memory_health * 0.15
            )
            
            return min(1.0, max(0.0, health_score))
            
        except Exception as e:
            logger.error(f"Error calculating health for agent {agent.agent_id}: {e}")
            return 0.5  # Neutral health on error
    
    async def _check_memory_consistency(self, agent: AgentInstance) -> float:
        """Check agent's memory and data consistency"""
        try:
            # Verify agent's internal state consistency
            if hasattr(agent, 'last_heartbeat') and agent.last_heartbeat:
                time_since_heartbeat = (datetime.now() - agent.last_heartbeat).total_seconds()
                if time_since_heartbeat > self.heartbeat_interval * 2:
                    return 0.5  # Stale heartbeat indicates potential issues
            
            # Check for data corruption indicators
            metrics = agent.performance_metrics
            if any(value < 0 for value in metrics.values() if isinstance(value, (int, float))):
                return 0.2  # Negative metrics indicate corruption
            
            return 1.0  # Memory appears consistent
            
        except Exception as e:
            logger.warning(f"Memory consistency check failed for {agent.agent_id}: {e}")
            return 0.7
    
    async def detect_byzantine_behavior(self, agent_id: str) -> bool:
        """Detect if agent is exhibiting Byzantine behavior"""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # Initialize metrics if not present
        if agent_id not in self.byzantine_metrics:
            self.byzantine_metrics[agent_id] = ByzantineMetrics(
                health_score=1.0,
                consecutive_failures=0,
                last_anomaly_time=None,
                response_time_history=[],
                task_success_rate=1.0,
                anomaly_count=0
            )
        
        metrics = self.byzantine_metrics[agent_id]
        
        # Calculate current health
        current_health = await self.calculate_agent_health(agent)
        metrics.health_score = current_health
        
        # Check for Byzantine patterns
        is_byzantine = await self._analyze_byzantine_patterns(agent_id, metrics)
        
        if is_byzantine:
            metrics.anomaly_count += 1
            metrics.last_anomaly_time = datetime.now()
            
            logger.warning(f"Byzantine behavior detected in agent {agent_id}")
            
        return is_byzantine
    
    async def _analyze_byzantine_patterns(self, agent_id: str, metrics: ByzantineMetrics) -> bool:
        """Analyze patterns that indicate Byzantine behavior"""
        
        # Pattern 1: Consistently low health score
        if metrics.health_score < self.fault_tolerance_threshold:
            metrics.consecutive_failures += 1
            if metrics.consecutive_failures >= self.max_consecutive_failures:
                return True
        else:
            metrics.consecutive_failures = 0
        
        # Pattern 2: Rapid anomaly accumulation
        if metrics.last_anomaly_time:
            time_since_anomaly = datetime.now() - metrics.last_anomaly_time
            if time_since_anomaly < timedelta(minutes=5) and metrics.anomaly_count > 5:
                return True
        
        # Pattern 3: Inconsistent response times (potential manipulation)
        agent = self.agents[agent_id]
        current_response_time = agent.performance_metrics.get("average_response_time", 0)
        metrics.response_time_history.append(current_response_time)
        
        # Keep only recent history
        if len(metrics.response_time_history) > 10:
            metrics.response_time_history = metrics.response_time_history[-10:]
        
        # Check for erratic response patterns
        if len(metrics.response_time_history) >= 5:
            avg_response = sum(metrics.response_time_history) / len(metrics.response_time_history)
            variance = sum((x - avg_response) ** 2 for x in metrics.response_time_history) / len(metrics.response_time_history)
            if variance > avg_response:  # High variance indicates erratic behavior
                return True
        
        return False
    
    async def handle_byzantine_agent(self, agent_id: str):
        """Handle agent exhibiting Byzantine behavior"""
        try:
            if agent_id not in self.agents:
                return
            
            agent = self.agents[agent_id]
            logger.warning(f"Handling Byzantine agent {agent_id}")
            
            # Quarantine the agent
            agent.status = AgentStatus.ERROR
            self.quarantine_agents[agent_id] = datetime.now()
            
            # Log Byzantine detection event
            log_swarm_event(
                session_id=self.session_id,
                agent_id=agent_id,
                event_type="byzantine_detected",
                event_data={
                    "health_score": self.byzantine_metrics.get(agent_id, {}).health_score if agent_id in self.byzantine_metrics else 0,
                    "action": "quarantine",
                    "task_queue_size": len(agent.task_queue),
                    "anomaly_count": self.byzantine_metrics.get(agent_id, {}).anomaly_count if agent_id in self.byzantine_metrics else 0
                }
            )
            
            # Redistribute tasks from Byzantine agent
            await self.redistribute_agent_tasks(agent_id)
            
            # Spawn replacement agent if needed
            await self.spawn_replacement_agent(agent.config.agent_type)
            
        except Exception as e:
            logger.error(f"Error handling Byzantine agent {agent_id}: {e}")
    
    async def redistribute_agent_tasks(self, failed_agent_id: str):
        """Redistribute tasks from a failed/Byzantine agent"""
        try:
            if failed_agent_id not in self.agents:
                return
            
            failed_agent = self.agents[failed_agent_id]
            tasks_to_redistribute = failed_agent.task_queue.copy()
            failed_agent.task_queue.clear()
            
            if not tasks_to_redistribute:
                return
            
            # Find healthy agents to redistribute tasks to
            healthy_agents = [
                agent_id for agent_id, agent in self.agents.items()
                if agent.status == AgentStatus.ACTIVE and agent_id != failed_agent_id
                and agent_id not in self.quarantine_agents
            ]
            
            if not healthy_agents:
                logger.warning("No healthy agents available for task redistribution")
                return
            
            # Distribute tasks evenly among healthy agents
            for i, task in enumerate(tasks_to_redistribute):
                target_agent_id = healthy_agents[i % len(healthy_agents)]
                self.agents[target_agent_id].task_queue.append(task)
            
            logger.info(f"Redistributed {len(tasks_to_redistribute)} tasks from {failed_agent_id} to {len(healthy_agents)} agents")
            
        except Exception as e:
            logger.error(f"Error redistributing tasks from {failed_agent_id}: {e}")
    
    async def spawn_replacement_agent(self, agent_type: AgentType):
        """Spawn a replacement agent for a failed one"""
        try:
            replacement_id = await self.spawn_agent(agent_type)
            if replacement_id:
                logger.info(f"Spawned replacement agent {replacement_id} for type {agent_type.value}")
            else:
                logger.error(f"Failed to spawn replacement agent for type {agent_type.value}")
        except Exception as e:
            logger.error(f"Error spawning replacement agent: {e}")
    
    async def byzantine_consensus_check(self, decision_data: Dict[str, Any]) -> bool:
        """Perform Byzantine fault tolerant consensus on critical decisions"""
        try:
            healthy_agents = [
                agent_id for agent_id, agent in self.agents.items()
                if agent.status == AgentStatus.ACTIVE and agent_id not in self.quarantine_agents
            ]
            
            if len(healthy_agents) < self.consensus_quorum:
                logger.warning(f"Insufficient healthy agents ({len(healthy_agents)}) for consensus (need {self.consensus_quorum})")
                return False
            
            # Simulate consensus mechanism (in real implementation, this would query agents)
            consensus_votes = 0
            for agent_id in healthy_agents[:self.consensus_quorum]:
                # In practice, this would send the decision to the agent and await response
                vote = await self._simulate_agent_vote(agent_id, decision_data)
                if vote:
                    consensus_votes += 1
            
            consensus_reached = consensus_votes >= (self.consensus_quorum // 2) + 1
            
            log_swarm_event(
                session_id=self.session_id,
                agent_id="coordinator",
                event_type="consensus_check",
                event_data={
                    "decision": decision_data,
                    "votes": consensus_votes,
                    "quorum": self.consensus_quorum,
                    "consensus_reached": consensus_reached
                }
            )
            
            return consensus_reached
            
        except Exception as e:
            logger.error(f"Error in Byzantine consensus check: {e}")
            return False
    
    async def _simulate_agent_vote(self, agent_id: str, decision_data: Dict[str, Any]) -> bool:
        """Simulate agent voting on a decision (placeholder for actual implementation)"""
        # In real implementation, this would communicate with the agent
        # For now, simulate based on agent health
        if agent_id in self.byzantine_metrics:
            health = self.byzantine_metrics[agent_id].health_score
            return health > 0.7
        return True
    
    async def cleanup_quarantined_agents(self):
        """Clean up agents that have been quarantined for too long"""
        try:
            current_time = datetime.now()
            quarantine_timeout = timedelta(hours=1)  # 1 hour quarantine
            
            agents_to_remove = []
            for agent_id, quarantine_time in self.quarantine_agents.items():
                if current_time - quarantine_time > quarantine_timeout:
                    agents_to_remove.append(agent_id)
            
            for agent_id in agents_to_remove:
                # Remove from quarantine and agent registry
                del self.quarantine_agents[agent_id]
                if agent_id in self.agents:
                    del self.agents[agent_id]
                if agent_id in self.byzantine_metrics:
                    del self.byzantine_metrics[agent_id]
                
                logger.info(f"Removed quarantined agent {agent_id} after timeout")
                
        except Exception as e:
            logger.error(f"Error cleaning up quarantined agents: {e}")