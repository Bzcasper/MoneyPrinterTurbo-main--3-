# Phase 2: MoneyPrinterTurbo Domain Model & Modular Architecture

## Executive Summary

This document defines the domain model for MoneyPrinterTurbo's modular architecture, establishing core entities, relationships, and data structures that support the functional requirements defined in Phase 1. The model focuses on clear separation of concerns, dependency injection patterns, and testable component boundaries.

## Domain Overview

### Core Business Domains

```
MoneyPrinterTurbo Domain Architecture:

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Video Domain   │  │  Audio Domain   │  │ Content Domain  │
│                 │  │                 │  │                 │
│ • Video Clips   │  │ • TTS Services  │  │ • Scripts       │
│ • Transitions   │  │ • Voice Synth   │  │ • Subtitles     │ 
│ • Effects       │  │ • BGM Manager   │  │ • Materials     │
│ • Codecs        │  │ • Audio Mixing  │  │ • Fonts         │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                 Processing Domain                            │
│                                                             │
│ • Video Pipeline     • Memory Management                    │
│ • Parallel Processor • Resource Pool                       │
│ • Concatenation      • Performance Monitor                 │
│ • Validation Engine  • Cache Manager                       │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│              Infrastructure Domain                          │
│                                                             │
│ • Configuration   • Security      • Storage                │
│ • Database       • Authentication • File System            │
│ • Redis Cache    • Authorization  • Logging                │
│ • External APIs  • Rate Limiting  • Monitoring             │
└─────────────────────────────────────────────────────────────┘
```

## Core Domain Entities

### 1. Video Domain Entities

#### VideoClip
```python
// Core video clip entity with validation and metadata
Entity VideoClip:
    Properties:
        clip_id: UUID                    // Unique identifier
        file_path: FilePath             // Validated file location
        start_time: Timestamp           // Clip start position
        end_time: Timestamp             // Clip end position
        duration: Duration              // Calculated duration
        dimensions: VideoDimensions     // Width x Height
        format: VideoFormat             // Codec and container info
        metadata: VideoMetadata         // Additional properties
        validation_status: ValidationStatus
        created_at: Timestamp
        
    Invariants:
        - end_time > start_time
        - duration = end_time - start_time
        - dimensions.width >= 64 AND dimensions.height >= 64
        - file_path must exist and be readable
        - format must be supported
        
    Business Rules:
        - Maximum clip duration: 30 seconds
        - Minimum dimensions: 64x64 pixels
        - Supported formats: MP4, AVI, MOV, WebM
        - Maximum file size: 2GB per clip

// TEST: VideoClip creation with valid parameters succeeds
// TEST: VideoClip creation with invalid dimensions fails
// TEST: VideoClip validation detects corrupted files
// TEST: VideoClip invariants prevent invalid state
```

#### VideoTransition
```python
// Video transition effects and timing
Entity VideoTransition:
    Properties:
        transition_id: UUID
        type: TransitionType            // fade_in, fade_out, slide, etc.
        duration: Duration              // Transition length
        parameters: TransitionParams    // Type-specific config
        easing_function: EasingType     // Animation curve
        
    Invariants:
        - duration > 0 AND duration <= 5 seconds
        - parameters must be valid for transition type
        
    Business Rules:
        - Fade transitions: 0.5-3 seconds
        - Slide transitions: 1-5 seconds
        - Custom transitions: validated parameters required

// TEST: Transition creation with valid types succeeds
// TEST: Transition duration constraints enforced
// TEST: Parameter validation for each transition type
```

#### CodecProfile
```python
// Hardware-optimized codec configuration
Entity CodecProfile:
    Properties:
        profile_id: UUID
        name: String                    // Human-readable name
        codec_type: CodecType          // H264, H265, VP9, etc.
        hardware_acceleration: HardwareType  // NVENC, QSV, VAAPI
        quality_settings: QualityProfile
        performance_tier: PerformanceTier
        compatibility_matrix: CompatibilityInfo
        
    Invariants:
        - name must be unique
        - quality_settings valid for codec_type
        - hardware_acceleration compatible with codec_type
        
    Business Rules:
        - Quality profiles: speed, balanced, quality
        - Hardware detection automatic on startup
        - Fallback to software encoding if hardware unavailable

// TEST: Codec profile validation with hardware detection
// TEST: Quality settings compatibility checks
// TEST: Hardware fallback behavior
```

### 2. Audio Domain Entities

#### TTSRequest
```python
// Text-to-speech service request
Entity TTSRequest:
    Properties:
        request_id: UUID
        text_content: String            // Text to synthesize
        voice_config: VoiceConfig      // Voice selection and settings
        output_format: AudioFormat     // MP3, WAV, etc.
        quality_level: AudioQuality    // Low, medium, high
        provider_preference: TTSProvider // Primary provider choice
        fallback_providers: List[TTSProvider]
        priority: RequestPriority
        created_at: Timestamp
        
    Invariants:
        - text_content length > 0 AND length <= 10000 chars
        - voice_config must be valid for selected provider
        - at least one fallback provider required
        
    Business Rules:
        - Maximum text length: 10,000 characters
        - Supported providers: OpenAI, Google, Azure, Character.ai
        - Fallback chain: minimum 2 providers
        - Rate limiting: 100 requests/minute per user

// TEST: TTS request validation with different providers
// TEST: Fallback chain execution on provider failure
// TEST: Rate limiting enforcement
// TEST: Text length validation
```

#### AudioMixer
```python
// Audio mixing and composition service
Entity AudioMixer:
    Properties:
        mixer_id: UUID
        voice_track: AudioTrack        // Primary voice audio
        bgm_track: AudioTrack          // Background music
        effects_tracks: List[AudioTrack] // Sound effects
        mix_profile: MixingProfile     // Volume levels and EQ
        output_format: AudioFormat
        target_duration: Duration
        
    Invariants:
        - voice_track is required
        - all tracks must have compatible sample rates
        - target_duration > 0
        
    Business Rules:
        - Voice track: -3dB to 0dB range
        - BGM track: -20dB to -10dB range
        - Auto-ducking when voice active
        - Fade in/out: 1-3 second transitions

// TEST: Audio mixing with multiple tracks
// TEST: Volume level validation and adjustment
// TEST: Sample rate compatibility checks
// TEST: Auto-ducking behavior
```

### 3. Content Domain Entities

#### VideoScript
```python
// Script generation and management
Entity VideoScript:
    Properties:
        script_id: UUID
        title: String
        content_blocks: List[ContentBlock] // Segmented script content
        metadata: ScriptMetadata          // Topic, style, keywords
        language: LanguageCode
        estimated_duration: Duration     // Based on reading speed
        generation_params: GenerationParams
        validation_status: ValidationStatus
        created_at: Timestamp
        
    Invariants:
        - content_blocks not empty
        - estimated_duration calculated from content
        - language code valid ISO 639-1
        
    Business Rules:
        - Maximum script length: 5000 words
        - Content blocks: 50-200 words each
        - Reading speed: 150 words per minute
        - Profanity filtering required

// TEST: Script generation with valid parameters
// TEST: Content block segmentation logic
// TEST: Duration estimation accuracy
// TEST: Profanity filtering effectiveness
```

#### SubtitleTrack
```python
// Subtitle timing and formatting
Entity SubtitleTrack:
    Properties:
        subtitle_id: UUID
        segments: List[SubtitleSegment]   // Timed text segments
        styling: SubtitleStyle           // Font, color, position
        language: LanguageCode
        format: SubtitleFormat          // SRT, VTT, etc.
        sync_offset: Duration           // Timing adjustment
        
    Invariants:
        - segments must be chronologically ordered
        - no overlapping segments allowed
        - sync_offset between -10s and +10s
        
    Business Rules:
        - Maximum segment duration: 6 seconds
        - Minimum gap between segments: 0.5 seconds
        - Character limit per segment: 100 characters
        - Reading speed consideration: 21 characters/second

// TEST: Subtitle segment timing validation
// TEST: Overlap detection and prevention
// TEST: Character limit enforcement
// TEST: Reading speed calculation
```

## Aggregate Boundaries and Consistency Rules

### Video Processing Aggregate
```python
// Main video processing workflow orchestrator
Aggregate VideoProcessingWorkflow:
    Root Entity: VideoProject
    
    Entities:
        - VideoProject (root)
        - VideoClip (multiple)
        - AudioTrack (multiple)
        - SubtitleTrack (optional)
        - ProcessingTask (multiple)
        
    Value Objects:
        - VideoParameters
        - QualitySettings
        - OutputConfiguration
        
    Invariants:
        - All clips must have valid dimensions
        - Total duration must match audio duration
        - Output format must be supported
        - Processing tasks must be in valid sequence
        
    Consistency Rules:
        - Video and audio synchronization maintained
        - Clip order preserved during processing
        - Quality settings applied consistently
        - Error recovery maintains partial progress

// TEST: Aggregate consistency during clip addition
// TEST: Audio-video synchronization validation
// TEST: Processing task sequencing
// TEST: Error recovery behavior
```

### Configuration Aggregate
```python
// Application configuration and security
Aggregate ConfigurationContext:
    Root Entity: ApplicationConfig
    
    Entities:
        - ApplicationConfig (root)
        - EnvironmentConfig
        - ServiceConfig (multiple)
        - SecurityConfig
        
    Value Objects:
        - DatabaseSettings
        - RedisSettings
        - APICredentials
        - PerformanceSettings
        
    Invariants:
        - No secrets in configuration files
        - All required settings present
        - Settings valid for environment
        - Security constraints enforced
        
    Consistency Rules:
        - Environment-specific overrides
        - Credential rotation without service interruption
        - Configuration validation on startup
        - Secure default values

// TEST: Configuration loading and validation
// TEST: Environment-specific overrides
// TEST: Security constraint enforcement
// TEST: Hot reloading capabilities
```

## Domain Services

### VideoProcessingOrchestrator
```python
// Main orchestration service for video generation
DomainService VideoProcessingOrchestrator:
    Dependencies:
        - clip_processor: ClipProcessor
        - audio_mixer: AudioMixer
        - concatenation_service: ConcatenationService
        - codec_optimizer: CodecOptimizer
        - validation_engine: ValidationEngine
        - performance_monitor: PerformanceMonitor
        
    Operations:
        process_video_request(params: VideoParams) -> VideoResult:
            // Orchestrate complete video processing pipeline
            
        validate_inputs(clips: List[VideoClip]) -> ValidationResult:
            // Comprehensive input validation
            
        optimize_processing_plan(clips: List[VideoClip]) -> ProcessingPlan:
            // Create optimal processing strategy
            
        monitor_progress(workflow_id: UUID) -> ProgressStatus:
            // Track processing progress and performance

// TEST: End-to-end video processing workflow
// TEST: Input validation across multiple clips
// TEST: Processing plan optimization
// TEST: Progress monitoring accuracy
```

### TTSServiceCoordinator
```python
// Coordinate multiple TTS providers with fallback
DomainService TTSServiceCoordinator:
    Dependencies:
        - provider_registry: TTSProviderRegistry
        - circuit_breaker: CircuitBreaker
        - cache_manager: CacheManager
        - rate_limiter: RateLimiter
        
    Operations:
        synthesize_speech(request: TTSRequest) -> AudioResult:
            // Handle TTS with provider fallback
            
        check_provider_health() -> ProviderHealthStatus:
            // Monitor provider availability
            
        cache_audio_result(request: TTSRequest, audio: AudioData):
            // Cache successful TTS results
            
        get_optimal_provider(request: TTSRequest) -> TTSProvider:
            // Select best provider based on current conditions

// TEST: Provider fallback on failure
// TEST: Circuit breaker activation
// TEST: Cache hit/miss behavior
// TEST: Rate limiting enforcement
```

## Value Objects and Data Structures

### VideoParameters
```python
// Immutable video generation parameters
ValueObject VideoParameters:
    Properties:
        aspect_ratio: AspectRatio       // 16:9, 9:16, 1:1, etc.
        resolution: Resolution          // 1080p, 720p, etc.
        frame_rate: FrameRate          // 24, 30, 60 fps
        quality_target: QualityTarget   // Speed, balanced, quality
        codec_preference: CodecType
        hardware_acceleration: Boolean
        
    Validation Rules:
        - aspect_ratio must be supported
        - resolution compatible with aspect_ratio
        - frame_rate in valid range (1-120 fps)
        - quality_target maps to valid settings

// TEST: Parameter validation with different combinations
// TEST: Compatibility checks between settings
// TEST: Default value assignment
```

### ProcessingMetrics
```python
// Performance and resource usage tracking
ValueObject ProcessingMetrics:
    Properties:
        processing_time: Duration
        memory_peak: MemorySize
        memory_average: MemorySize
        cpu_utilization: Percentage
        gpu_utilization: Percentage
        cache_hit_rate: Percentage
        error_count: Integer
        throughput: ClipsPerSecond
        
    Calculated Properties:
        efficiency_score: Float         // Overall performance metric
        resource_utilization: Float    // Combined CPU/GPU/Memory
        
    Business Rules:
        - Efficiency score: 0.0-1.0 range
        - Memory tracking: peak and average
        - Error rate threshold: <1% target

// TEST: Metrics calculation accuracy
// TEST: Efficiency score computation
// TEST: Resource utilization tracking
```

## Integration Interfaces

### ExternalServiceInterface
```python
// Standard interface for external service integration
Interface ExternalServiceInterface:
    Methods:
        authenticate(credentials: Credentials) -> AuthResult
        make_request(request: ServiceRequest) -> ServiceResponse
        handle_error(error: ServiceError) -> ErrorResponse
        check_health() -> HealthStatus
        
    Error Handling:
        - Network timeout: retry with exponential backoff
        - Authentication failure: refresh credentials
        - Rate limiting: respect retry-after headers
        - Service unavailable: circuit breaker activation

// TEST: Interface compliance for all external services
// TEST: Error handling behavior
// TEST: Authentication flow
// TEST: Health check implementation
```

### StorageInterface
```python
// Abstraction for file and data storage
Interface StorageInterface:
    Methods:
        store_file(file_data: FileData, path: StoragePath) -> StorageResult
        retrieve_file(path: StoragePath) -> FileData
        delete_file(path: StoragePath) -> DeletionResult
        list_files(prefix: PathPrefix) -> List[StoragePath]
        get_file_metadata(path: StoragePath) -> FileMetadata
        
    Properties:
        storage_type: StorageType       // Local, S3, GCS, etc.
        encryption_enabled: Boolean
        backup_configured: Boolean
        
    Performance Requirements:
        - File upload: <30 seconds for 1GB
        - File retrieval: <10 seconds for 1GB
        - Metadata operations: <1 second

// TEST: Storage operations with different backends
// TEST: Performance requirements validation
// TEST: Error handling for storage failures
// TEST: Encryption and security features
```

## Event Flow and Domain Events

### VideoProcessingEvents
```python
// Domain events for video processing workflow
DomainEvent VideoProcessingStarted:
    Properties:
        workflow_id: UUID
        user_id: UUID
        parameters: VideoParameters
        estimated_duration: Duration
        timestamp: Timestamp

DomainEvent ClipProcessingCompleted:
    Properties:
        workflow_id: UUID
        clip_id: UUID
        processing_time: Duration
        output_size: FileSize
        quality_metrics: QualityMetrics
        timestamp: Timestamp

DomainEvent VideoProcessingCompleted:
    Properties:
        workflow_id: UUID
        output_file_path: FilePath
        final_duration: Duration
        total_processing_time: Duration
        performance_metrics: ProcessingMetrics
        timestamp: Timestamp

DomainEvent ProcessingErrorOccurred:
    Properties:
        workflow_id: UUID
        error_type: ErrorType
        error_message: String
        context: ErrorContext
        retry_count: Integer
        timestamp: Timestamp

// TEST: Event generation during processing
// TEST: Event handler execution
// TEST: Event ordering and sequencing
// TEST: Error event handling
```

## Repository Patterns

### VideoProjectRepository
```python
// Repository for video project persistence
Repository VideoProjectRepository:
    Methods:
        create(project: VideoProject) -> RepositoryResult
        find_by_id(project_id: UUID) -> Optional[VideoProject]
        find_by_user(user_id: UUID) -> List[VideoProject]
        update(project: VideoProject) -> RepositoryResult
        delete(project_id: UUID) -> RepositoryResult
        
    Query Methods:
        find_by_status(status: ProcessingStatus) -> List[VideoProject]
        find_recent(user_id: UUID, limit: Integer) -> List[VideoProject]
        find_by_date_range(start: Date, end: Date) -> List[VideoProject]
        
    Performance Requirements:
        - Create operation: <1 second
        - Find by ID: <0.5 seconds
        - Query operations: <2 seconds
        - Pagination support for large result sets

// TEST: CRUD operations correctness
// TEST: Query performance benchmarks
// TEST: Concurrent access handling
// TEST: Data consistency validation
```

### ConfigurationRepository
```python
// Repository for configuration management
Repository ConfigurationRepository:
    Methods:
        load_configuration(environment: Environment) -> ApplicationConfig
        save_configuration(config: ApplicationConfig) -> RepositoryResult
        get_setting(key: SettingKey) -> Optional[SettingValue]
        update_setting(key: SettingKey, value: SettingValue) -> RepositoryResult
        
    Security Features:
        - Credential encryption at rest
        - Access audit logging
        - Role-based access control
        - Configuration versioning
        
    Validation Rules:
        - Schema validation on load
        - Environment-specific constraints
        - Security policy enforcement
        - Default value fallbacks

// TEST: Configuration loading and validation
// TEST: Security feature enforcement
// TEST: Schema validation behavior
// TEST: Environment-specific handling
```

## Glossary of Domain Terms

| Term | Definition | Context |
|------|------------|---------|
| **Clip** | A segment of video with defined start/end times | Video processing |
| **Concatenation** | Joining multiple video clips into a single file | Video processing |
| **Codec Profile** | Configuration for video encoding optimization | Video optimization |
| **TTS Provider** | External service for text-to-speech synthesis | Audio generation |
| **Circuit Breaker** | Pattern to prevent cascading failures | Service reliability |
| **Aggregate** | Consistency boundary for related entities | Domain modeling |
| **Value Object** | Immutable data structure with no identity | Domain modeling |
| **Domain Event** | Significant occurrence within the domain | Event-driven design |
| **Repository** | Data access abstraction layer | Data persistence |
| **Workflow** | Complete video generation process | Business process |

---

## Model Validation Checklist

- [ ] All core entities identified with clear responsibilities
- [ ] Aggregate boundaries defined with consistency rules
- [ ] Value objects immutable and self-validating
- [ ] Domain services address complex business logic
- [ ] Repository interfaces abstract data access
- [ ] Events capture significant domain occurrences
- [ ] Integration interfaces standardize external communication
- [ ] Business rules explicitly documented
- [ ] Validation rules comprehensive and testable
- [ ] Performance requirements specified

---

*Document Version*: 1.0
*Last Updated*: 2025-01-29
*Next Phase*: Core Module Pseudocode Design
*Dependencies*: Phase 1 Requirements Analysis
*Validation*: Domain Expert Review Required