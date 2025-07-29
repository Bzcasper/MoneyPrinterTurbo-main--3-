# MoneyPrinterTurbo Service Communication Patterns

## Overview

This document defines the communication patterns, protocols, and message formats for all services in the MoneyPrinterTurbo microservices architecture. It serves as the contract specification for inter-service communication.

## Communication Architecture

### Service Communication Matrix

```
┌─────────────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│     SERVICE     │   GW    │  USER   │ CONTENT │   TTS   │  VIDEO  │ MATERIAL│  ORCH   │  NOTIF  │
├─────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ API Gateway     │    -    │  HTTP   │  HTTP   │  HTTP   │  HTTP   │  HTTP   │  HTTP   │   WS    │
│ User Mgmt       │  HTTP   │    -    │  gRPC   │    -    │    -    │    -    │  gRPC   │  Event  │
│ Content Gen     │  HTTP   │  gRPC   │    -    │  Event  │  Event  │  gRPC   │  Event  │  Event  │
│ TTS Service     │  HTTP   │    -    │  Event  │    -    │  Event  │    -    │  Event  │  Event  │
│ Video Process   │  HTTP   │    -    │  Event  │  Event  │    -    │  gRPC   │  Event  │  Event  │
│ Material Mgmt   │  HTTP   │    -    │  gRPC   │    -    │  gRPC   │    -    │  gRPC   │    -    │
│ Orchestration   │  HTTP   │  gRPC   │  Event  │  Event  │  Event  │  gRPC   │    -    │  Event  │
│ Notification    │   WS    │  Event  │  Event  │  Event  │  Event  │    -    │  Event  │    -    │
└─────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

Legend:
HTTP  - Synchronous HTTP REST API
gRPC  - High-performance RPC calls  
Event - Asynchronous event-driven messaging
WS    - WebSocket connections
```

## Communication Protocols

### 1. HTTP REST API (Synchronous)

**Use Cases:**
- User-facing operations
- Real-time data queries
- Service health checks
- Administrative operations

**Standard HTTP Status Codes:**
```python
# http_responses.py
class HTTPStatus:
    # Success
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    
    # Client Errors
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    
    # Server Errors
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504
```

**Request/Response Format:**
```python
# Standard API Response Format
{
    "success": boolean,
    "data": object | array | null,
    "message": string,
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "uuid",
    "errors": [
        {
            "code": "VALIDATION_ERROR",
            "field": "email",
            "message": "Invalid email format"
        }
    ]
}
```

**Example API Client:**
```python
# api_client.py
class ServiceClient:
    def __init__(self, base_url: str, service_key: str):
        self.base_url = base_url
        self.service_key = service_key
        self.session = httpx.AsyncClient()
        
    async def post(self, endpoint: str, data: dict):
        headers = {
            "Content-Type": "application/json",
            "X-Service-Key": self.service_key,
            "X-Request-ID": str(uuid.uuid4())
        }
        
        response = await self.session.post(
            f"{self.base_url}{endpoint}",
            json=data,
            headers=headers,
            timeout=30.0
        )
        
        if response.status_code >= 400:
            raise ServiceError(response.status_code, response.json())
            
        return response.json()
```

### 2. gRPC (High-Performance RPC)

**Use Cases:**
- Service-to-service communication
- High-frequency operations
- Binary data transfer
- Type-safe interfaces

**Service Definitions:**
```protobuf
// user_service.proto
syntax = "proto3";

package moneyprinter.user;

service UserService {
    rpc GetUser(GetUserRequest) returns (UserResponse);
    rpc ValidateToken(ValidateTokenRequest) returns (ValidateTokenResponse);
    rpc UpdateUsage(UpdateUsageRequest) returns (UpdateUsageResponse);
}

message GetUserRequest {
    string user_id = 1;
}

message UserResponse {
    string user_id = 1;
    string email = 2;
    string full_name = 3;
    string subscription_tier = 4;
    bool is_active = 5;
    int64 created_at = 6;
}

message ValidateTokenRequest {
    string token = 1;
}

message ValidateTokenResponse {
    bool valid = 1;
    UserResponse user = 2;
    int64 expires_at = 3;
}
```

**gRPC Client Implementation:**
```python
# grpc_client.py
import grpc
from .protos import user_service_pb2, user_service_pb2_grpc

class UserServiceClient:
    def __init__(self, host: str, port: int):
        self.channel = grpc.aio.insecure_channel(f"{host}:{port}")
        self.stub = user_service_pb2_grpc.UserServiceStub(self.channel)
        
    async def get_user(self, user_id: str):
        request = user_service_pb2.GetUserRequest(user_id=user_id)
        response = await self.stub.GetUser(request)
        
        return {
            "user_id": response.user_id,
            "email": response.email,
            "full_name": response.full_name,
            "subscription_tier": response.subscription_tier,
            "is_active": response.is_active
        }
        
    async def validate_token(self, token: str):
        request = user_service_pb2.ValidateTokenRequest(token=token)
        response = await self.stub.ValidateToken(request)
        
        return {
            "valid": response.valid,
            "user": response.user if response.valid else None,
            "expires_at": response.expires_at
        }
```

### 3. Event-Driven Messaging (Asynchronous)

**Use Cases:**
- Workflow coordination
- Long-running processes
- Decoupled operations
- Event notifications

**Message Bus Architecture:**
```yaml
# rabbitmq-topology.yaml
exchanges:
  - name: moneyprinter.events
    type: topic
    durable: true
    
  - name: moneyprinter.dlx
    type: direct
    durable: true

queues:
  # Video Generation Events
  - name: video.content.generate
    exchange: moneyprinter.events
    routing_key: video.content.requested
    consumer: content-service
    
  - name: video.speech.synthesize
    exchange: moneyprinter.events
    routing_key: video.speech.requested
    consumer: tts-service
    
  - name: video.processing.create
    exchange: moneyprinter.events
    routing_key: video.processing.requested
    consumer: video-service
    
  # Notification Events
  - name: notifications.email
    exchange: moneyprinter.events
    routing_key: user.#
    consumer: notification-service
    
  - name: notifications.websocket
    exchange: moneyprinter.events
    routing_key: video.#
    consumer: notification-service
    
  # Dead Letter Queue
  - name: failed.messages
    exchange: moneyprinter.dlx
    routing_key: failed
```

**Event Message Format:**
```python
# events.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any
import uuid

@dataclass
class BaseEvent:
    event_id: str
    event_type: str
    source_service: str
    timestamp: datetime
    version: str = "1.0"
    correlation_id: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_service": self.source_service,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "correlation_id": self.correlation_id,
            "data": self.__dict__
        }

@dataclass
class VideoGenerationRequested(BaseEvent):
    user_id: str
    video_id: str
    topic: str
    style: str
    duration_target: int
    voice_settings: Dict[str, Any]
    video_settings: Dict[str, Any]
    
    def __post_init__(self):
        self.event_type = "video.generation.requested"

@dataclass
class ContentGenerationCompleted(BaseEvent):
    content_id: str
    user_id: str
    video_id: str
    script: str
    estimated_duration: int
    sections: list
    
    def __post_init__(self):
        self.event_type = "content.generation.completed"

@dataclass
class SpeechSynthesisCompleted(BaseEvent):
    audio_id: str
    content_id: str
    video_id: str
    audio_url: str
    duration: float
    provider: str
    
    def __post_init__(self):
        self.event_type = "speech.synthesis.completed"
```

**Event Publisher:**
```python
# event_publisher.py
import aio_pika
import json
from typing import Any

class EventPublisher:
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.connection = None
        self.channel = None
        
    async def connect(self):
        self.connection = await aio_pika.connect_robust(self.connection_url)
        self.channel = await self.connection.channel()
        
    async def publish_event(self, event: BaseEvent, routing_key: str):
        exchange = await self.channel.get_exchange("moneyprinter.events")
        
        message = aio_pika.Message(
            json.dumps(event.to_dict()).encode(),
            headers={
                "event_type": event.event_type,
                "source_service": event.source_service,
                "correlation_id": event.correlation_id
            },
            timestamp=event.timestamp
        )
        
        await exchange.publish(message, routing_key=routing_key)
        
    async def close(self):
        if self.connection:
            await self.connection.close()
```

**Event Consumer:**
```python
# event_consumer.py
import aio_pika
import json
from typing import Callable, Dict

class EventConsumer:
    def __init__(self, connection_url: str, queue_name: str):
        self.connection_url = connection_url
        self.queue_name = queue_name
        self.handlers: Dict[str, Callable] = {}
        
    def register_handler(self, event_type: str, handler: Callable):
        self.handlers[event_type] = handler
        
    async def start_consuming(self):
        connection = await aio_pika.connect_robust(self.connection_url)
        channel = await connection.channel()
        queue = await channel.get_queue(self.queue_name)
        
        async def process_message(message: aio_pika.Message):
            async with message.process():
                try:
                    event_data = json.loads(message.body.decode())
                    event_type = event_data.get("event_type")
                    
                    if event_type in self.handlers:
                        await self.handlers[event_type](event_data)
                    else:
                        print(f"No handler for event type: {event_type}")
                        
                except Exception as e:
                    print(f"Error processing message: {e}")
                    # Message will be nacked and sent to DLX
                    raise
                    
        await queue.consume(process_message)
```

### 4. WebSocket Connections (Real-time)

**Use Cases:**
- Real-time notifications
- Live progress updates
- Bidirectional communication
- Server-sent events

**WebSocket Manager:**
```python
# websocket_manager.py
import asyncio
import json
from typing import Dict, Set
from fastapi import WebSocket
import redis.asyncio as redis

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.redis_client = None
        
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
            
        self.active_connections[user_id].add(websocket)
        
        # Subscribe to user-specific Redis channel
        if not self.redis_client:
            self.redis_client = await redis.from_url("redis://redis:6379")
            
        await self.subscribe_to_notifications(user_id)
        
    async def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                
    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            disconnected = []
            
            for websocket in self.active_connections[user_id]:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.append(websocket)
                    
            # Clean up disconnected sockets
            for websocket in disconnected:
                self.active_connections[user_id].discard(websocket)
                
    async def broadcast_to_all(self, message: dict):
        for user_id in self.active_connections:
            await self.send_to_user(user_id, message)
            
    async def subscribe_to_notifications(self, user_id: str):
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(f"notifications:{user_id}")
        
        async def listen_for_messages():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    notification = json.loads(message["data"])
                    await self.send_to_user(user_id, notification)
                    
        asyncio.create_task(listen_for_messages())
```

## Service-Specific Communication Patterns

### 1. Video Generation Workflow

**Sequence Diagram:**
```
User → API Gateway → Orchestration Service
  ↓
Orchestration → Content Service (HTTP)
  ↓
Content Service → Event Bus (content.generated)
  ↓
TTS Service ← Event Bus
  ↓
TTS Service → Event Bus (speech.synthesized)
  ↓
Video Service ← Event Bus
  ↓
Video Service → Material Service (gRPC - assets)
  ↓
Video Service → Event Bus (video.created)
  ↓
Notification Service ← Event Bus
  ↓
WebSocket → User (real-time update)
```

**Implementation:**
```python
# video_generation_workflow.py
class VideoGenerationWorkflow:
    def __init__(self, event_publisher: EventPublisher):
        self.event_publisher = event_publisher
        
    async def start_generation(self, request: VideoGenerationRequest):
        # Step 1: Initiate content generation
        event = VideoGenerationRequested(
            event_id=str(uuid.uuid4()),
            source_service="orchestration-service",
            timestamp=datetime.utcnow(),
            user_id=request.user_id,
            video_id=request.video_id,
            topic=request.topic,
            style=request.style,
            duration_target=request.duration_target,
            voice_settings=request.voice_settings,
            video_settings=request.video_settings
        )
        
        await self.event_publisher.publish_event(
            event, 
            routing_key="video.content.requested"
        )
        
        return {"workflow_id": event.event_id, "status": "initiated"}

# Content Service Event Handler
class ContentServiceHandler:
    async def handle_content_request(self, event_data: dict):
        # Generate content
        content = await self.generate_content(
            topic=event_data["topic"],
            style=event_data["style"],
            duration=event_data["duration_target"]
        )
        
        # Publish completion event
        completion_event = ContentGenerationCompleted(
            event_id=str(uuid.uuid4()),
            source_service="content-service",
            timestamp=datetime.utcnow(),
            correlation_id=event_data["event_id"],
            content_id=content.id,
            user_id=event_data["user_id"],
            video_id=event_data["video_id"],
            script=content.script,
            estimated_duration=content.estimated_duration,
            sections=content.sections
        )
        
        await self.event_publisher.publish_event(
            completion_event,
            routing_key="content.generation.completed"
        )
```

### 2. User Authentication Flow

**gRPC Communication:**
```python
# API Gateway → User Service
class AuthenticationFlow:
    def __init__(self, user_service_client: UserServiceClient):
        self.user_service = user_service_client
        
    async def authenticate_request(self, token: str):
        # Validate token with User Service
        validation_result = await self.user_service.validate_token(token)
        
        if not validation_result["valid"]:
            raise AuthenticationError("Invalid token")
            
        return validation_result["user"]
        
    async def get_user_permissions(self, user_id: str):
        user = await self.user_service.get_user(user_id)
        
        permissions = {
            "can_create_video": user["subscription_tier"] in ["pro", "enterprise"],
            "max_duration": self.get_max_duration(user["subscription_tier"]),
            "api_rate_limit": self.get_rate_limit(user["subscription_tier"])
        }
        
        return permissions
```

### 3. Asset Management Communication

**Mixed Protocol Usage:**
```python
# Video Service → Material Service (gRPC)
class AssetManager:
    def __init__(self, material_service_client: MaterialServiceClient):
        self.material_service = material_service_client
        
    async def get_visual_assets(self, video_requirements: dict):
        # Use gRPC for high-performance asset queries
        assets = await self.material_service.search_assets(
            category="image",
            tags=video_requirements["keywords"],
            style=video_requirements["visual_style"],
            limit=10
        )
        
        return assets
        
    async def upload_generated_thumbnail(self, video_id: str, thumbnail_data: bytes):
        # Use HTTP for file uploads
        upload_url = f"http://material-service:8005/v1/assets"
        
        files = {"file": ("thumbnail.jpg", thumbnail_data, "image/jpeg")}
        data = {
            "category": "image",
            "tags": ["thumbnail", "generated"],
            "video_id": video_id
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(upload_url, files=files, data=data)
            
        return response.json()
```

## Error Handling and Resilience

### 1. Circuit Breaker Pattern

```python
# circuit_breaker.py
import asyncio
from enum import Enum
from typing import Callable, Any
import time

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Service is unavailable")
                
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        return (
            time.time() - self.last_failure_time > self.recovery_timeout
        )
        
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

### 2. Retry Mechanism with Exponential Backoff

```python
# retry_handler.py
import asyncio
import random
from typing import Callable, Any

class RetryHandler:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                    
                delay = self._calculate_delay(attempt)
                await asyncio.sleep(delay)
                
        raise last_exception
        
    def _calculate_delay(self, attempt: int) -> float:
        # Exponential backoff with jitter
        delay = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter
```

### 3. Timeout Management

```python
# timeout_manager.py
import asyncio
from typing import Callable, Any

class TimeoutManager:
    @staticmethod
    async def execute_with_timeout(
        func: Callable, 
        timeout_seconds: float, 
        *args, 
        **kwargs
    ) -> Any:
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs), 
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            raise ServiceTimeoutError(
                f"Operation timed out after {timeout_seconds} seconds"
            )

# Service-specific timeout configurations
SERVICE_TIMEOUTS = {
    "user-service": {
        "validate_token": 5.0,
        "get_user": 3.0,
        "update_usage": 10.0
    },
    "content-service": {
        "generate_content": 60.0,
        "enhance_content": 30.0
    },
    "tts-service": {
        "synthesize": 120.0,
        "batch_synthesize": 300.0
    },
    "video-service": {
        "create_video": 600.0,
        "optimize_video": 300.0
    }
}
```

## Service Discovery and Load Balancing

### 1. Service Registry

```python
# service_registry.py
import consul.aio
from typing import List, Dict, Optional

class ServiceRegistry:
    def __init__(self, consul_host: str = "consul", consul_port: int = 8500):
        self.consul = consul.aio.Consul(host=consul_host, port=consul_port)
        
    async def register_service(
        self, 
        service_name: str, 
        service_id: str, 
        host: str, 
        port: int,
        health_check_url: str = None
    ):
        check = None
        if health_check_url:
            check = consul.Check.http(health_check_url, interval="10s")
            
        await self.consul.agent.service.register(
            name=service_name,
            service_id=service_id,
            address=host,
            port=port,
            check=check,
            tags=[f"version-{os.getenv('SERVICE_VERSION', '1.0.0')}"]
        )
        
    async def discover_service(self, service_name: str) -> List[Dict]:
        _, services = await self.consul.health.service(service_name, passing=True)
        
        return [
            {
                "id": service["Service"]["ID"],
                "host": service["Service"]["Address"],
                "port": service["Service"]["Port"],
                "tags": service["Service"]["Tags"]
            }
            for service in services
        ]
        
    async def deregister_service(self, service_id: str):
        await self.consul.agent.service.deregister(service_id)
```

### 2. Load Balancer

```python
# load_balancer.py
import random
from typing import List, Dict, Optional
from collections import defaultdict
import time

class LoadBalancer:
    def __init__(self):
        self.service_instances: Dict[str, List[Dict]] = defaultdict(list)
        self.last_updated: Dict[str, float] = {}
        self.round_robin_index: Dict[str, int] = defaultdict(int)
        
    async def get_service_instance(
        self, 
        service_name: str, 
        strategy: str = "round_robin"
    ) -> Optional[Dict]:
        # Refresh service instances if needed
        if self._should_refresh(service_name):
            await self._refresh_instances(service_name)
            
        instances = self.service_instances.get(service_name, [])
        if not instances:
            return None
            
        return self._select_instance(service_name, instances, strategy)
        
    def _select_instance(
        self, 
        service_name: str, 
        instances: List[Dict], 
        strategy: str
    ) -> Dict:
        if strategy == "round_robin":
            index = self.round_robin_index[service_name] % len(instances)
            self.round_robin_index[service_name] += 1
            return instances[index]
            
        elif strategy == "random":
            return random.choice(instances)
            
        elif strategy == "least_connections":
            # For simplicity, using random here
            # In production, track active connections
            return min(instances, key=lambda x: x.get("active_connections", 0))
            
        else:
            return instances[0]  # Default to first instance
            
    def _should_refresh(self, service_name: str) -> bool:
        last_update = self.last_updated.get(service_name, 0)
        return time.time() - last_update > 30  # Refresh every 30 seconds
        
    async def _refresh_instances(self, service_name: str):
        # This would integrate with service discovery
        # For now, using placeholder
        self.last_updated[service_name] = time.time()
```

## Message Serialization and Validation

### 1. Pydantic Models for Message Validation

```python
# message_models.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    VIDEO_GENERATION_REQUESTED = "video.generation.requested"
    CONTENT_GENERATED = "content.generated"
    SPEECH_SYNTHESIZED = "speech.synthesized"
    VIDEO_CREATED = "video.created"
    
class BaseMessage(BaseModel):
    event_id: str = Field(..., description="Unique event identifier")
    event_type: MessageType
    source_service: str
    timestamp: datetime
    correlation_id: Optional[str] = None
    version: str = "1.0"
    
    class Config:
        use_enum_values = True

class VideoGenerationRequestMessage(BaseMessage):
    event_type: MessageType = MessageType.VIDEO_GENERATION_REQUESTED
    user_id: str
    video_id: str
    topic: str = Field(..., min_length=1, max_length=1000)
    style: str = Field(..., regex="^(engaging|professional|casual|educational)$")
    duration_target: int = Field(..., ge=15, le=600)
    voice_settings: Dict[str, Any]
    video_settings: Dict[str, Any]
    
    @validator('voice_settings')
    def validate_voice_settings(cls, v):
        required_fields = ['provider', 'voice_id']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        return v

class ContentGeneratedMessage(BaseMessage):
    event_type: MessageType = MessageType.CONTENT_GENERATED
    content_id: str
    user_id: str
    video_id: str
    script: str
    estimated_duration: int
    sections: List[Dict[str, Any]]
    
class SpeechSynthesizedMessage(BaseMessage):
    event_type: MessageType = MessageType.SPEECH_SYNTHESIZED
    audio_id: str
    content_id: str
    video_id: str
    audio_url: str
    duration: float
    provider: str
```

### 2. Message Serializer

```python
# message_serializer.py
import json
from typing import Type, TypeVar, Union
from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)

class MessageSerializer:
    @staticmethod
    def serialize(message: BaseModel) -> bytes:
        """Serialize a Pydantic model to JSON bytes"""
        return json.dumps(message.dict()).encode('utf-8')
        
    @staticmethod
    def deserialize(data: bytes, model_class: Type[T]) -> T:
        """Deserialize JSON bytes to a Pydantic model"""
        try:
            json_data = json.loads(data.decode('utf-8'))
            return model_class(**json_data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise MessageSerializationError(f"Failed to deserialize message: {e}")
            
    @staticmethod
    def get_message_type(data: bytes) -> str:
        """Extract message type without full deserialization"""
        try:
            json_data = json.loads(data.decode('utf-8'))
            return json_data.get('event_type', 'unknown')
        except json.JSONDecodeError:
            return 'invalid'
```

## Performance Optimization

### 1. Connection Pooling

```python
# connection_pool.py
import asyncio
import aiohttp
import asyncpg
from typing import Dict, Any

class ConnectionPoolManager:
    def __init__(self):
        self.http_sessions: Dict[str, aiohttp.ClientSession] = {}
        self.db_pools: Dict[str, asyncpg.Pool] = {}
        
    async def get_http_session(self, service_name: str) -> aiohttp.ClientSession:
        if service_name not in self.http_sessions:
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Per host limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
            )
            
            timeout = aiohttp.ClientTimeout(total=30, connect=5)
            
            self.http_sessions[service_name] = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            
        return self.http_sessions[service_name]
        
    async def get_db_pool(self, service_name: str) -> asyncpg.Pool:
        if service_name not in self.db_pools:
            db_config = await self.load_db_config(service_name)
            
            self.db_pools[service_name] = await asyncpg.create_pool(
                **db_config,
                min_size=5,
                max_size=20,
                max_queries=50000,
                max_inactive_connection_lifetime=300
            )
            
        return self.db_pools[service_name]
        
    async def close_all(self):
        # Close HTTP sessions
        for session in self.http_sessions.values():
            await session.close()
            
        # Close database pools
        for pool in self.db_pools.values():
            await pool.close()
```

### 2. Caching Strategy

```python
# cache_manager.py
import redis.asyncio as redis
import json
from typing import Any, Optional
import hashlib

class CacheManager:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        
    async def get(self, key: str) -> Optional[Any]:
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None
        
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = 3600
    ):
        await self.redis.setex(
            key, 
            ttl, 
            json.dumps(value, default=str)
        )
        
    async def delete(self, key: str):
        await self.redis.delete(key)
        
    async def cache_function_result(
        self, 
        func_name: str, 
        args: tuple, 
        kwargs: dict, 
        result: Any, 
        ttl: int = 3600
    ):
        # Create cache key from function signature
        key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
        cache_key = f"func_cache:{hashlib.md5(key_data.encode()).hexdigest()}"
        
        await self.set(cache_key, result, ttl)
        
    async def get_cached_function_result(
        self, 
        func_name: str, 
        args: tuple, 
        kwargs: dict
    ) -> Optional[Any]:
        key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
        cache_key = f"func_cache:{hashlib.md5(key_data.encode()).hexdigest()}"
        
        return await self.get(cache_key)
```

## Monitoring and Observability

### 1. Distributed Tracing Integration

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

def setup_tracing(service_name: str):
    # Set up tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(service_name)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=6831,
    )
    
    # Set up span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrument frameworks
    FastAPIInstrumentor.instrument()
    HTTPXClientInstrumentor.instrument()
    AsyncPGInstrumentor.instrument()
    
    return tracer

# Usage in service
tracer = setup_tracing("content-service")

@tracer.start_as_current_span("generate_content")
async def generate_content(topic: str, style: str):
    span = trace.get_current_span()
    span.set_attribute("content.topic", topic)
    span.set_attribute("content.style", style)
    
    # Business logic here
    result = await perform_content_generation(topic, style)
    
    span.set_attribute("content.length", len(result))
    return result
```

### 2. Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
from functools import wraps

# Service-level metrics
REQUEST_COUNT = Counter(
    'service_requests_total',
    'Total service requests',
    ['service', 'method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'service_request_duration_seconds',
    'Request duration in seconds',
    ['service', 'method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'service_active_connections',
    'Active connections',
    ['service', 'connection_type']
)

# Business metrics
VIDEOS_CREATED = Counter(
    'videos_created_total',
    'Total videos created',
    ['user_tier', 'style']
)

CONTENT_GENERATION_DURATION = Histogram(
    'content_generation_duration_seconds',
    'Content generation duration',
    ['style', 'length_category']
)

TTS_REQUESTS = Counter(
    'tts_requests_total',
    'TTS requests',
    ['provider', 'language', 'status']
)

def monitor_request(service_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_COUNT.labels(
                    service=service_name,
                    method=func.__name__,
                    endpoint=getattr(func, '__endpoint__', 'unknown'),
                    status=status
                ).inc()
                
                REQUEST_DURATION.labels(
                    service=service_name,
                    method=func.__name__,
                    endpoint=getattr(func, '__endpoint__', 'unknown')
                ).observe(duration)
                
        return wrapper
    return decorator
```

This comprehensive service communication specification provides the foundation for reliable, scalable inter-service communication in the MoneyPrinterTurbo microservices architecture. The patterns and implementations shown here ensure type safety, error resilience, and observability across all service interactions.