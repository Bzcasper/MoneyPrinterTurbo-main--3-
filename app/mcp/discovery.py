"""
MCP Service Discovery and Load Balancing

Provides service registry, discovery, and load balancing capabilities for MCP services,
enabling distributed coordination and high availability.
"""

import asyncio
import json
import random
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import aiohttp
import redis

from app.config import config


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


@dataclass
class ServiceInstance:
    """Represents a service instance in the registry"""
    id: str
    name: str
    address: str
    port: int
    metadata: Dict
    status: ServiceStatus = ServiceStatus.HEALTHY
    last_heartbeat: float = 0
    health_check_url: Optional[str] = None
    load_factor: float = 1.0  # Higher = more load capacity
    
    @property 
    def endpoint(self) -> str:
        return f"{self.address}:{self.port}"
        
    @property
    def is_healthy(self) -> bool:
        return self.status == ServiceStatus.HEALTHY
        
    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_heartbeat


class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    LEAST_RESPONSE_TIME = "least_response_time"


class MCPServiceRegistry:
    """Service registry for MCP services with health checking and load balancing"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.connection_counts: Dict[str, int] = {}
        self.response_times: Dict[str, List[float]] = {}
        self.round_robin_indexes: Dict[str, int] = {}
        
        # Configuration
        self.heartbeat_interval = config.app.get("mcp_heartbeat_interval", 30)
        self.health_check_interval = config.app.get("mcp_health_check_interval", 10)
        self.service_timeout = config.app.get("mcp_service_timeout", 90)
        
        # Redis for distributed registry (optional)
        self.redis_client = None
        if config.app.get("enable_redis", False):
            try:
                redis_url = f"redis://:{config.app.get('redis_password', '')}@{config.app.get('redis_host', 'redis')}:{config.app.get('redis_port', 6379)}/{config.app.get('redis_db', 0)}"
                self.redis_client = redis.Redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for service registry: {e}")
                
        # Start background tasks
        self._health_check_task = None
        self._cleanup_task = None
        self._start_background_tasks()
        
    def _start_background_tasks(self):
        """Start background tasks for health checking and cleanup"""
        loop = asyncio.get_event_loop()
        self._health_check_task = loop.create_task(self._health_check_loop())
        self._cleanup_task = loop.create_task(self._cleanup_loop())
        
    async def register_service(self, service_name: str, endpoint: str, 
                             metadata: Optional[Dict] = None) -> str:
        """Register a service instance"""
        try:
            # Parse endpoint
            if "://" in endpoint:
                protocol, address_port = endpoint.split("://", 1)
                address, port = address_port.split(":", 1)
                port = int(port)
            else:
                address, port = endpoint.split(":", 1)
                port = int(port)
                
            service_id = f"{service_name}_{address}_{port}_{int(time.time())}"
            
            instance = ServiceInstance(
                id=service_id,
                name=service_name,
                address=address,
                port=port,
                metadata=metadata or {},
                last_heartbeat=time.time(),
                health_check_url=f"http://{address}:{port}/health" if metadata and metadata.get("http_health_check") else None
            )
            
            if service_name not in self.services:
                self.services[service_name] = []
                self.round_robin_indexes[service_name] = 0
                
            self.services[service_name].append(instance)
            self.connection_counts[service_id] = 0
            self.response_times[service_id] = []
            
            # Store in Redis if available
            if self.redis_client:
                key = f"mcp_services:{service_name}:{service_id}"
                self.redis_client.setex(key, self.service_timeout, json.dumps({
                    "id": service_id,
                    "name": service_name,
                    "address": address,
                    "port": port,
                    "metadata": metadata or {},
                    "registered_at": time.time()
                }))
                
            logger.info(f"Registered MCP service: {service_name} at {endpoint} (ID: {service_id})")
            return service_id
            
        except Exception as e:
            logger.error(f"Error registering service {service_name}: {str(e)}")
            raise
            
    async def unregister_service(self, service_id: str):
        """Unregister a service instance"""
        for service_name, instances in self.services.items():
            self.services[service_name] = [
                instance for instance in instances
                if instance.id != service_id
            ]
            
        if service_id in self.connection_counts:
            del self.connection_counts[service_id]
        if service_id in self.response_times:
            del self.response_times[service_id]
            
        # Remove from Redis
        if self.redis_client:
            pattern = f"mcp_services:*:{service_id}"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                
        logger.info(f"Unregistered MCP service: {service_id}")
        
    async def heartbeat(self, service_id: str):
        """Record heartbeat from service instance"""
        for instances in self.services.values():
            for instance in instances:
                if instance.id == service_id:
                    instance.last_heartbeat = time.time()
                    instance.status = ServiceStatus.HEALTHY
                    return
                    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover all instances of a service"""
        # Load from Redis if available
        if self.redis_client:
            await self._load_from_redis(service_name)
            
        return self.services.get(service_name, [])
        
    async def get_service_instance(self, service_name: str, 
                                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
                                 healthy_only: bool = True) -> Optional[ServiceInstance]:
        """Get a service instance using the specified load balancing strategy"""
        instances = await self.discover_services(service_name)
        
        if healthy_only:
            instances = [i for i in instances if i.is_healthy]
            
        if not instances:
            return None
            
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(service_name, instances)
        elif strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(instances)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(instances)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(service_name, instances)
        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(instances)
        else:
            return instances[0]
            
    def _round_robin_select(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin load balancing"""
        if service_name not in self.round_robin_indexes:
            self.round_robin_indexes[service_name] = 0
            
        index = self.round_robin_indexes[service_name]
        instance = instances[index % len(instances)]
        self.round_robin_indexes[service_name] = (index + 1) % len(instances)
        return instance
        
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections load balancing"""
        min_connections = float('inf')
        selected_instance = instances[0]
        
        for instance in instances:
            connections = self.connection_counts.get(instance.id, 0)
            if connections < min_connections:
                min_connections = connections
                selected_instance = instance
                
        return selected_instance
        
    def _weighted_round_robin_select(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round-robin load balancing"""
        # Create weighted list based on load_factor
        weighted_instances = []
        for instance in instances:
            weight = int(instance.load_factor * 10)  # Scale for integer weights
            weighted_instances.extend([instance] * weight)
            
        if not weighted_instances:
            return instances[0]
            
        return self._round_robin_select(f"{service_name}_weighted", weighted_instances)
        
    def _least_response_time_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least average response time load balancing"""
        min_response_time = float('inf')
        selected_instance = instances[0]
        
        for instance in instances:
            response_times = self.response_times.get(instance.id, [])
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                if avg_response_time < min_response_time:
                    min_response_time = avg_response_time
                    selected_instance = instance
            else:
                # No data, give it a chance
                selected_instance = instance
                break
                
        return selected_instance
        
    async def record_connection(self, service_id: str):
        """Record new connection to service"""
        if service_id in self.connection_counts:
            self.connection_counts[service_id] += 1
            
    async def record_disconnection(self, service_id: str):
        """Record disconnection from service"""
        if service_id in self.connection_counts:
            self.connection_counts[service_id] = max(0, self.connection_counts[service_id] - 1)
            
    async def record_response_time(self, service_id: str, response_time: float):
        """Record response time for service"""
        if service_id not in self.response_times:
            self.response_times[service_id] = []
            
        self.response_times[service_id].append(response_time)
        
        # Keep only recent response times (last 100)
        if len(self.response_times[service_id]) > 100:
            self.response_times[service_id] = self.response_times[service_id][-100:]
            
    async def update_service_status(self, service_id: str, status: ServiceStatus):
        """Update service status"""
        for instances in self.services.values():
            for instance in instances:
                if instance.id == service_id:
                    instance.status = status
                    return
                    
    async def get_service_stats(self, service_name: str) -> Dict:
        """Get statistics for a service"""
        instances = await self.discover_services(service_name)
        
        total_instances = len(instances)
        healthy_instances = len([i for i in instances if i.is_healthy])
        total_connections = sum(self.connection_counts.get(i.id, 0) for i in instances)
        
        avg_response_times = {}
        for instance in instances:
            response_times = self.response_times.get(instance.id, [])
            if response_times:
                avg_response_times[instance.id] = sum(response_times) / len(response_times)
                
        return {
            "service_name": service_name,
            "total_instances": total_instances,
            "healthy_instances": healthy_instances,
            "total_connections": total_connections,
            "average_response_times": avg_response_times,
            "instances": [
                {
                    "id": i.id,
                    "endpoint": i.endpoint,
                    "status": i.status,
                    "connections": self.connection_counts.get(i.id, 0),
                    "last_heartbeat": i.last_heartbeat,
                    "age_seconds": i.age_seconds
                }
                for i in instances
            ]
        }
        
    async def _load_from_redis(self, service_name: str):
        """Load service instances from Redis"""
        if not self.redis_client:
            return
            
        try:
            pattern = f"mcp_services:{service_name}:*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    service_data = json.loads(data)
                    service_id = service_data["id"]
                    
                    # Check if already loaded
                    existing = any(
                        i.id == service_id 
                        for i in self.services.get(service_name, [])
                    )
                    
                    if not existing:
                        instance = ServiceInstance(
                            id=service_id,
                            name=service_data["name"],
                            address=service_data["address"],
                            port=service_data["port"],
                            metadata=service_data["metadata"],
                            last_heartbeat=service_data["registered_at"]
                        )
                        
                        if service_name not in self.services:
                            self.services[service_name] = []
                            
                        self.services[service_name].append(instance)
                        
        except Exception as e:
            logger.error(f"Error loading services from Redis: {str(e)}")
            
    async def _health_check_loop(self):
        """Background health checking loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                
    async def _perform_health_checks(self):
        """Perform health checks on all services"""
        for service_name, instances in self.services.items():
            for instance in instances:
                if instance.health_check_url:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                instance.health_check_url, 
                                timeout=aiohttp.ClientTimeout(total=5)
                            ) as response:
                                if response.status == 200:
                                    instance.status = ServiceStatus.HEALTHY
                                else:
                                    instance.status = ServiceStatus.UNHEALTHY
                                    
                    except Exception as e:
                        logger.warning(f"Health check failed for {instance.id}: {str(e)}")
                        instance.status = ServiceStatus.UNHEALTHY
                        
    async def _cleanup_loop(self):
        """Background cleanup loop for stale services"""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                await self._cleanup_stale_services()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                
    async def _cleanup_stale_services(self):
        """Remove stale service instances"""
        current_time = time.time()
        
        for service_name, instances in list(self.services.items()):
            active_instances = []
            
            for instance in instances:
                if current_time - instance.last_heartbeat < self.service_timeout:
                    active_instances.append(instance)
                else:
                    logger.info(f"Removing stale service instance: {instance.id}")
                    if instance.id in self.connection_counts:
                        del self.connection_counts[instance.id]
                    if instance.id in self.response_times:
                        del self.response_times[instance.id]
                        
            self.services[service_name] = active_instances
            
            # Remove empty service entries
            if not active_instances:
                del self.services[service_name]
                if service_name in self.round_robin_indexes:
                    del self.round_robin_indexes[service_name]
                    
    async def shutdown(self):
        """Shutdown the service registry"""
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
            
        logger.info("Service registry shutdown complete")