"""
Supabase Database Models and Schemas

This module defines the data models, schemas, and table structures for the
MoneyPrinterTurbo application using Supabase as the backend database.
"""

import enum
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import json


class ProjectStatus(str, Enum):
    """Project status enumeration."""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoStatus(str, Enum):
    """Video generation status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


@dataclass
class DatabaseBaseModel:
    """Base model for database entities."""
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model instance from dictionary."""
        # Convert ISO format strings back to datetime objects
        for key, value in data.items():
            if isinstance(value, str) and key.endswith('_at'):
                try:
                    data[key] = datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    pass
        return cls(**data)


@dataclass
class User(DatabaseBaseModel):
    """User model for authentication and user management."""
    email: Optional[str] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True
    last_login: Optional[datetime] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    avatar_url: Optional[str] = None
    
    # Supabase Auth fields
    auth_id: Optional[str] = None  # Links to Supabase auth.users.id


@dataclass
class Project(DatabaseBaseModel):
    """Project model for video generation projects."""
    name: str = ""
    description: Optional[str] = None
    user_id: Optional[str] = None
    status: ProjectStatus = ProjectStatus.DRAFT
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Content
    script: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    # Media paths
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    
    # Metadata
    duration: Optional[float] = None  # in seconds
    resolution: Optional[str] = None  # e.g., "1920x1080"
    fps: Optional[int] = None
    file_size: Optional[int] = None  # in bytes
    
    # Processing info
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Publication
    published_url: Optional[str] = None
    published_at: Optional[datetime] = None


@dataclass
class Video(DatabaseBaseModel):
    """Video model for individual video files and metadata."""
    project_id: Optional[str] = None
    title: str = ""
    description: Optional[str] = None
    status: VideoStatus = VideoStatus.PENDING
    
    # File information
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_format: Optional[str] = None  # mp4, avi, etc.
    
    # Video properties
    duration: Optional[float] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    bitrate: Optional[int] = None
    codec: Optional[str] = None
    
    # Generation metadata
    generation_config: Dict[str, Any] = field(default_factory=dict)
    generation_log: List[str] = field(default_factory=list)
    
    # Processing timestamps
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Analytics
    view_count: int = 0
    download_count: int = 0


@dataclass
class Task(DatabaseBaseModel):
    """Task model for background job tracking."""
    name: str = ""
    description: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    
    # Task configuration
    task_type: str = ""  # video_generation, audio_processing, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Progress tracking
    progress: float = 0.0  # 0.0 to 100.0
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    
    # Execution info
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Results
    result: Optional[Dict[str, Any]] = None
    output_files: List[str] = field(default_factory=list)


@dataclass
class Analytics(DatabaseBaseModel):
    """Analytics model for tracking usage and performance."""
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    video_id: Optional[str] = None
    
    # Event information
    event_type: str = ""  # view, download, generation, etc.
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    # Session info
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Performance metrics
    processing_time: Optional[float] = None  # in seconds
    file_size: Optional[int] = None
    quality_score: Optional[float] = None


@dataclass
class SystemConfig(DatabaseBaseModel):
    """System configuration model for application settings."""
    key: str = ""
    value: Any = None
    description: Optional[str] = None
    category: str = "general"
    is_encrypted: bool = False
    is_active: bool = True
    
    def set_value(self, value: Any) -> None:
        """Set configuration value with proper serialization."""
        if isinstance(value, (dict, list)):
            self.value = json.dumps(value)
        else:
            self.value = str(value)
    
    def get_value(self) -> Any:
        """Get configuration value with proper deserialization."""
        if self.value is None:
            return None
        
        try:
            # Try to parse as JSON first
            return json.loads(self.value)
        except (json.JSONDecodeError, TypeError):
            # Return as string if not JSON
            return self.value


# Table schemas for Supabase SQL creation
SUPABASE_SCHEMAS = {
    'users': """
        CREATE TABLE IF NOT EXISTS users (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            auth_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
            email varchar(255) UNIQUE NOT NULL,
            username varchar(100) UNIQUE,
            full_name varchar(255),
            role varchar(50) DEFAULT 'user',
            is_active boolean DEFAULT true,
            last_login timestamptz,
            preferences jsonb DEFAULT '{}',
            avatar_url text,
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        );
        
        -- Enable RLS
        ALTER TABLE users ENABLE ROW LEVEL SECURITY;
        
        -- Create policies
        CREATE POLICY "Users can view own data" ON users FOR SELECT USING (auth.uid() = auth_id);
        CREATE POLICY "Users can update own data" ON users FOR UPDATE USING (auth.uid() = auth_id);
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_users_auth_id ON users(auth_id);
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    """,
    
    'projects': """
        CREATE TABLE IF NOT EXISTS projects (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            name varchar(255) NOT NULL,
            description text,
            user_id uuid REFERENCES users(id) ON DELETE CASCADE,
            status varchar(50) DEFAULT 'draft',
            config jsonb DEFAULT '{}',
            script text,
            keywords text[],
            audio_path text,
            video_path text,
            thumbnail_path text,
            duration numeric,
            resolution varchar(20),
            fps integer,
            file_size bigint,
            processing_started_at timestamptz,
            processing_completed_at timestamptz,
            error_message text,
            published_url text,
            published_at timestamptz,
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        );
        
        -- Enable RLS
        ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
        
        -- Create policies
        CREATE POLICY "Users can view own projects" ON projects FOR SELECT USING (user_id IN (SELECT id FROM users WHERE auth_id = auth.uid()));
        CREATE POLICY "Users can manage own projects" ON projects FOR ALL USING (user_id IN (SELECT id FROM users WHERE auth_id = auth.uid()));
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
        CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
        CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at);
    """,
    
    'videos': """
        CREATE TABLE IF NOT EXISTS videos (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id uuid REFERENCES projects(id) ON DELETE CASCADE,
            title varchar(255) NOT NULL,
            description text,
            status varchar(50) DEFAULT 'pending',
            file_path text,
            file_size bigint,
            file_format varchar(10),
            duration numeric,
            resolution varchar(20),
            fps integer,
            bitrate integer,
            codec varchar(50),
            generation_config jsonb DEFAULT '{}',
            generation_log text[],
            processing_started_at timestamptz,
            processing_completed_at timestamptz,
            error_message text,
            view_count integer DEFAULT 0,
            download_count integer DEFAULT 0,
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        );
        
        -- Enable RLS
        ALTER TABLE videos ENABLE ROW LEVEL SECURITY;
        
        -- Create policies
        CREATE POLICY "Users can view own videos" ON videos FOR SELECT USING (project_id IN (SELECT id FROM projects WHERE user_id IN (SELECT id FROM users WHERE auth_id = auth.uid())));
        CREATE POLICY "Users can manage own videos" ON videos FOR ALL USING (project_id IN (SELECT id FROM projects WHERE user_id IN (SELECT id FROM users WHERE auth_id = auth.uid())));
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_videos_project_id ON videos(project_id);
        CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
        CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos(created_at);
    """,
    
    'tasks': """
        CREATE TABLE IF NOT EXISTS tasks (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            name varchar(255) NOT NULL,
            description text,
            project_id uuid REFERENCES projects(id) ON DELETE CASCADE,
            user_id uuid REFERENCES users(id) ON DELETE CASCADE,
            status varchar(50) DEFAULT 'pending',
            task_type varchar(100) NOT NULL,
            parameters jsonb DEFAULT '{}',
            progress numeric DEFAULT 0.0,
            current_step text,
            total_steps integer,
            started_at timestamptz,
            completed_at timestamptz,
            error_message text,
            retry_count integer DEFAULT 0,
            max_retries integer DEFAULT 3,
            result jsonb,
            output_files text[],
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        );
        
        -- Enable RLS
        ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;
        
        -- Create policies
        CREATE POLICY "Users can view own tasks" ON tasks FOR SELECT USING (user_id IN (SELECT id FROM users WHERE auth_id = auth.uid()));
        CREATE POLICY "Users can manage own tasks" ON tasks FOR ALL USING (user_id IN (SELECT id FROM users WHERE auth_id = auth.uid()));
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks(project_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_task_type ON tasks(task_type);
        CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
    """,
    
    'analytics': """
        CREATE TABLE IF NOT EXISTS analytics (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id uuid REFERENCES users(id) ON DELETE SET NULL,
            project_id uuid REFERENCES projects(id) ON DELETE SET NULL,
            video_id uuid REFERENCES videos(id) ON DELETE SET NULL,
            event_type varchar(100) NOT NULL,
            event_data jsonb DEFAULT '{}',
            session_id varchar(255),
            ip_address inet,
            user_agent text,
            processing_time numeric,
            file_size bigint,
            quality_score numeric,
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        );
        
        -- Enable RLS
        ALTER TABLE analytics ENABLE ROW LEVEL SECURITY;
        
        -- Create policies
        CREATE POLICY "Users can view own analytics" ON analytics FOR SELECT USING (user_id IN (SELECT id FROM users WHERE auth_id = auth.uid()));
        CREATE POLICY "Anonymous analytics allowed" ON analytics FOR INSERT WITH CHECK (true);
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics(user_id);
        CREATE INDEX IF NOT EXISTS idx_analytics_project_id ON analytics(project_id);
        CREATE INDEX IF NOT EXISTS idx_analytics_video_id ON analytics(video_id);
        CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics(event_type);
        CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics(created_at);
    """,
    
    'system_config': """
        CREATE TABLE IF NOT EXISTS system_config (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            key varchar(255) UNIQUE NOT NULL,
            value text,
            description text,
            category varchar(100) DEFAULT 'general',
            is_encrypted boolean DEFAULT false,
            is_active boolean DEFAULT true,
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        );
        
        -- Enable RLS
        ALTER TABLE system_config ENABLE ROW LEVEL SECURITY;
        
        -- Create policies (admin only)
        CREATE POLICY "Admin can manage config" ON system_config FOR ALL USING (
            EXISTS (SELECT 1 FROM users WHERE auth_id = auth.uid() AND role = 'admin')
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config(key);
        CREATE INDEX IF NOT EXISTS idx_system_config_category ON system_config(category);
        CREATE INDEX IF NOT EXISTS idx_system_config_is_active ON system_config(is_active);
    """
}


# Functions for database operations
def get_create_table_sql(table_name: str) -> str:
    """
    Get SQL for creating a specific table.
    
    Args:
        table_name: Name of the table
        
    Returns:
        SQL string for table creation
        
    Raises:
        ValueError: If table name is not found
    """
    if table_name not in SUPABASE_SCHEMAS:
        raise ValueError(f"Table '{table_name}' not found in schemas")
    
    return SUPABASE_SCHEMAS[table_name]


def get_all_tables_sql() -> str:
    """
    Get SQL for creating all tables.
    
    Returns:
        Combined SQL string for all table creation
    """
    return '\n\n'.join(SUPABASE_SCHEMAS.values())


# Model registry for dynamic model access
MODEL_REGISTRY = {
    'User': User,
    'Project': Project,
    'Video': Video,
    'Task': Task,
    'Analytics': Analytics,
    'SystemConfig': SystemConfig
}


def get_model_class(model_name: str):
    """
    Get model class by name.
    
    Args:
        model_name: Name of the model class
        
    Returns:
        Model class
        
    Raises:
        ValueError: If model name is not found
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry")
    
    return MODEL_REGISTRY[model_name]


# Validation functions
def validate_model_data(model_class, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate data against model schema.
    
    Args:
        model_class: Model class to validate against
        data: Data to validate
        
    Returns:
        Validated data dictionary
        
    Raises:
        ValueError: If validation fails
    """
    try:
        # Create instance to validate
        instance = model_class.from_dict(data)
        return instance.to_dict()
    except Exception as e:
        raise ValueError(f"Model validation failed: {str(e)}")


class TTSProvider(str, Enum):
    """TTS provider enumeration."""
    GOOGLE_TTS = "google_tts"
    CHARACTERBOX = "characterbox"
    AZURE_TTS = "azure_tts"
    GPT_SOVITS = "gpt_sovits"
    SILICONFLOW = "siliconflow"


class TTSJobStatus(str, Enum):
    """TTS job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TTSJob(DatabaseBaseModel):
    """TTS job model for tracking text-to-speech operations."""
    text: str = ""
    provider: TTSProvider = TTSProvider.GOOGLE_TTS
    voice_name: str = ""
    language_code: str = "en-US"
    
    # Job metadata
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    status: TTSJobStatus = TTSJobStatus.PENDING
    
    # Configuration
    voice_settings: Dict[str, Any] = field(default_factory=dict)
    audio_format: str = "mp3"
    quality: str = "high"
    
    # Processing info
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None  # in seconds
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Output
    audio_file_path: Optional[str] = None
    file_size: Optional[int] = None  # in bytes
    duration: Optional[float] = None  # in seconds
    
    # Provider-specific data
    provider_job_id: Optional[str] = None
    provider_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Character(DatabaseBaseModel):
    """CharacterBox character model for voice personalities."""
    character_id: str = ""
    name: str = ""
    description: Optional[str] = None
    provider: str = "characterbox"
    
    # Character properties
    personality_traits: Dict[str, Any] = field(default_factory=dict)
    voice_characteristics: Dict[str, Any] = field(default_factory=dict)
    language_codes: List[str] = field(default_factory=list)
    
    # Availability and quality
    is_active: bool = True
    quality_rating: Optional[float] = None  # 0.0 to 5.0
    usage_count: int = 0
    
    # Custom character data
    is_custom: bool = False
    created_by_user_id: Optional[str] = None
    voice_sample_path: Optional[str] = None
    
    # Provider metadata
    provider_character_id: Optional[str] = None
    provider_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    last_used_at: Optional[datetime] = None


@dataclass
class TTSVoice(DatabaseBaseModel):
    """TTS voice model for managing available voices across providers."""
    voice_id: str = ""
    name: str = ""
    provider: TTSProvider = TTSProvider.GOOGLE_TTS
    language_code: str = "en-US"
    
    # Voice properties
    gender: Optional[str] = None  # Male, Female, Neutral
    age_group: Optional[str] = None  # Child, Adult, Senior
    accent: Optional[str] = None
    style: Optional[str] = None  # Conversational, News, etc.
    
    # Technical details
    sample_rate: Optional[int] = None
    supported_formats: List[str] = field(default_factory=lambda: ["mp3", "wav"])
    
    # Quality and availability
    is_available: bool = True
    quality_score: Optional[float] = None
    usage_count: int = 0
    
    # Provider-specific
    provider_voice_id: str = ""
    provider_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    last_updated: Optional[datetime] = None


# Add new schemas to SUPABASE_SCHEMAS
SUPABASE_SCHEMAS.update({
    'tts_jobs': """
        CREATE TABLE IF NOT EXISTS tts_jobs (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            text text NOT NULL,
            provider varchar(50) NOT NULL DEFAULT 'google_tts',
            voice_name varchar(255) NOT NULL,
            language_code varchar(20) DEFAULT 'en-US',
            user_id uuid REFERENCES users(id) ON DELETE SET NULL,
            project_id uuid REFERENCES projects(id) ON DELETE SET NULL,
            status varchar(50) DEFAULT 'pending',
            voice_settings jsonb DEFAULT '{}',
            audio_format varchar(20) DEFAULT 'mp3',
            quality varchar(20) DEFAULT 'high',
            started_at timestamptz,
            completed_at timestamptz,
            processing_time numeric,
            error_message text,
            retry_count integer DEFAULT 0,
            audio_file_path text,
            file_size bigint,
            duration numeric,
            provider_job_id varchar(255),
            provider_response jsonb DEFAULT '{}',
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        );
        
        -- Enable RLS
        ALTER TABLE tts_jobs ENABLE ROW LEVEL SECURITY;
        
        -- Create policies
        CREATE POLICY "Users can view own TTS jobs" ON tts_jobs FOR SELECT USING (user_id IN (SELECT id FROM users WHERE auth_id = auth.uid()));
        CREATE POLICY "Users can manage own TTS jobs" ON tts_jobs FOR ALL USING (user_id IN (SELECT id FROM users WHERE auth_id = auth.uid()));
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_tts_jobs_user_id ON tts_jobs(user_id);
        CREATE INDEX IF NOT EXISTS idx_tts_jobs_project_id ON tts_jobs(project_id);
        CREATE INDEX IF NOT EXISTS idx_tts_jobs_status ON tts_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_tts_jobs_provider ON tts_jobs(provider);
        CREATE INDEX IF NOT EXISTS idx_tts_jobs_created_at ON tts_jobs(created_at);
    """,
    
    'characters': """
        CREATE TABLE IF NOT EXISTS characters (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            character_id varchar(255) UNIQUE NOT NULL,
            name varchar(255) NOT NULL,
            description text,
            provider varchar(50) DEFAULT 'characterbox',
            personality_traits jsonb DEFAULT '{}',
            voice_characteristics jsonb DEFAULT '{}',
            language_codes text[],
            is_active boolean DEFAULT true,
            quality_rating numeric CHECK (quality_rating >= 0 AND quality_rating <= 5),
            usage_count integer DEFAULT 0,
            is_custom boolean DEFAULT false,
            created_by_user_id uuid REFERENCES users(id) ON DELETE SET NULL,
            voice_sample_path text,
            provider_character_id varchar(255),
            provider_metadata jsonb DEFAULT '{}',
            last_used_at timestamptz,
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        );
        
        -- Enable RLS
        ALTER TABLE characters ENABLE ROW LEVEL SECURITY;
        
        -- Create policies
        CREATE POLICY "All users can view active characters" ON characters FOR SELECT USING (is_active = true);
        CREATE POLICY "Users can manage own custom characters" ON characters FOR ALL USING (
            is_custom = true AND created_by_user_id IN (SELECT id FROM users WHERE auth_id = auth.uid())
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_characters_character_id ON characters(character_id);
        CREATE INDEX IF NOT EXISTS idx_characters_provider ON characters(provider);
        CREATE INDEX IF NOT EXISTS idx_characters_is_active ON characters(is_active);
        CREATE INDEX IF NOT EXISTS idx_characters_is_custom ON characters(is_custom);
        CREATE INDEX IF NOT EXISTS idx_characters_created_by_user_id ON characters(created_by_user_id);
    """,
    
    'tts_voices': """
        CREATE TABLE IF NOT EXISTS tts_voices (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            voice_id varchar(255) UNIQUE NOT NULL,
            name varchar(255) NOT NULL,
            provider varchar(50) NOT NULL,
            language_code varchar(20) NOT NULL,
            gender varchar(20),
            age_group varchar(20),
            accent varchar(100),
            style varchar(100),
            sample_rate integer,
            supported_formats text[],
            is_available boolean DEFAULT true,
            quality_score numeric,
            usage_count integer DEFAULT 0,
            provider_voice_id varchar(255) NOT NULL,
            provider_metadata jsonb DEFAULT '{}',
            last_updated timestamptz,
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        );
        
        -- Enable RLS
        ALTER TABLE tts_voices ENABLE ROW LEVEL SECURITY;
        
        -- Create policies
        CREATE POLICY "All users can view available voices" ON tts_voices FOR SELECT USING (is_available = true);
        CREATE POLICY "Admin can manage voices" ON tts_voices FOR ALL USING (
            EXISTS (SELECT 1 FROM users WHERE auth_id = auth.uid() AND role = 'admin')
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_tts_voices_voice_id ON tts_voices(voice_id);
        CREATE INDEX IF NOT EXISTS idx_tts_voices_provider ON tts_voices(provider);
        CREATE INDEX IF NOT EXISTS idx_tts_voices_language_code ON tts_voices(language_code);
        CREATE INDEX IF NOT EXISTS idx_tts_voices_is_available ON tts_voices(is_available);
        CREATE INDEX IF NOT EXISTS idx_tts_voices_provider_voice_id ON tts_voices(provider_voice_id);
    """
})

# Update MODEL_REGISTRY
MODEL_REGISTRY.update({
    'TTSJob': TTSJob,
    'Character': Character,
    'TTSVoice': TTSVoice
})

# Export all models and utilities
__all__ = [
    'DatabaseBaseModel',
    'User',
    'Project', 
    'Video',
    'Task',
    'Analytics',
    'SystemConfig',
    'TTSJob',
    'Character',
    'TTSVoice',
    'ProjectStatus',
    'VideoStatus', 
    'TaskStatus',
    'UserRole',
    'TTSProvider',
    'TTSJobStatus',
    'SUPABASE_SCHEMAS',
    'MODEL_REGISTRY',
    'get_create_table_sql',
    'get_all_tables_sql',
    'get_model_class',
    'validate_model_data'
]