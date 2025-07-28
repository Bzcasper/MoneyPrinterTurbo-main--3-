"""
Comprehensive SQLite Database Models for MoneyPrinterTurbo
Handles video projects, tasks, sessions, and processing history
"""

import enum
import uuid
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text, 
    ForeignKey, Index, CheckConstraint, UniqueConstraint,
    create_engine, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.sql import func
from sqlalchemy.engine import Engine
import sqlite3

# Enable WAL mode and foreign keys for SQLite
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys=ON")
        # Optimize SQLite settings
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

Base = declarative_base()


class TaskStatus(enum.Enum):
    """Task status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(enum.Enum):
    """Task type enumeration"""
    VIDEO_GENERATION = "video_generation"
    SCRIPT_GENERATION = "script_generation"
    AUDIO_GENERATION = "audio_generation"
    SUBTITLE_GENERATION = "subtitle_generation"
    MATERIAL_DOWNLOAD = "material_download"
    VIDEO_PROCESSING = "video_processing"


class ProcessingStage(enum.Enum):
    """Processing stage enumeration"""
    SCRIPT = "script"
    TERMS = "terms"
    AUDIO = "audio"
    SUBTITLE = "subtitle"
    MATERIALS = "materials"
    VIDEO = "video"


class User(Base):
    """User accounts table"""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    preferences = Column(JSON, default=dict)
    
    # Relationships
    projects = relationship("VideoProject", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    tasks = relationship("Task", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_users_username', 'username'),
        Index('idx_users_email', 'email'),
        Index('idx_users_active', 'is_active'),
    )


class VideoProject(Base):
    """Video projects table for organizing related video generation tasks"""
    __tablename__ = "video_projects"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    subject = Column(String(500), nullable=False)
    language = Column(String(10), nullable=True)
    
    # Project configuration
    default_params = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    
    # Status and metrics
    status = Column(String(20), default="active")
    total_videos = Column(Integer, default=0)
    completed_videos = Column(Integer, default=0)
    total_duration = Column(Float, default=0.0)  # in seconds
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="projects")
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")
    videos = relationship("VideoMetadata", back_populates="project", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_projects_user_id', 'user_id'),
        Index('idx_projects_status', 'status'), 
        Index('idx_projects_created_at', 'created_at'),
        Index('idx_projects_subject', 'subject'),
    )


class Task(Base):
    """Tasks table for tracking video generation and processing tasks"""
    __tablename__ = "tasks"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    project_id = Column(String(36), ForeignKey("video_projects.id"), nullable=True)
    
    # Task details
    task_type = Column(String(50), nullable=False)
    status = Column(String(20), default=TaskStatus.PENDING.value)
    priority = Column(Integer, default=5)  # 1-10, higher is more priority
    
    # Progress tracking
    progress = Column(Integer, default=0)  # 0-100
    current_stage = Column(String(20), nullable=True)
    stages_completed = Column(JSON, default=list)
    
    # Task configuration and data
    parameters = Column(JSON, default=dict)
    result_data = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)
    
    # File paths and resources
    task_directory = Column(String(500), nullable=True)
    input_files = Column(JSON, default=list)
    output_files = Column(JSON, default=list)
    
    # Timing and performance
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    estimated_duration = Column(Integer, nullable=True)  # seconds
    actual_duration = Column(Integer, nullable=True)  # seconds
    
    # Resource usage
    cpu_time = Column(Float, nullable=True)
    memory_peak = Column(Integer, nullable=True)  # MB
    gpu_utilized = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="tasks")
    project = relationship("VideoProject", back_populates="tasks")
    processing_logs = relationship("ProcessingLog", back_populates="task", cascade="all, delete-orphan")
    videos = relationship("VideoMetadata", back_populates="task", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_tasks_status', 'status'),
        Index('idx_tasks_type', 'task_type'),
        Index('idx_tasks_user_id', 'user_id'),
        Index('idx_tasks_project_id', 'project_id'),
        Index('idx_tasks_created_at', 'created_at'),
        Index('idx_tasks_priority', 'priority'),
        Index('idx_tasks_progress', 'progress'),
        CheckConstraint('progress >= 0 AND progress <= 100', name='check_progress_range'),
        CheckConstraint('priority >= 1 AND priority <= 10', name='check_priority_range'),
    )


class ProcessingLog(Base):
    """Processing logs for detailed task execution tracking"""
    __tablename__ = "processing_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False)
    
    # Log details
    stage = Column(String(50), nullable=False)
    level = Column(String(10), default="INFO")  # DEBUG, INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    details = Column(JSON, default=dict)
    
    # Performance metrics
    stage_duration = Column(Float, nullable=True)  # seconds
    memory_usage = Column(Integer, nullable=True)  # MB
    cpu_usage = Column(Float, nullable=True)  # percentage
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    task = relationship("Task", back_populates="processing_logs")
    
    __table_args__ = (
        Index('idx_logs_task_id', 'task_id'),
        Index('idx_logs_stage', 'stage'),
        Index('idx_logs_level', 'level'),
        Index('idx_logs_timestamp', 'timestamp'),
    )


class VideoMetadata(Base):
    """Video metadata and file information"""
    __tablename__ = "video_metadata"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=True)
    project_id = Column(String(36), ForeignKey("video_projects.id"), nullable=True)
    
    # Video identification
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=True)  # SHA-256
    file_size = Column(Integer, nullable=False)  # bytes
    
    # Video properties
    duration = Column(Float, nullable=False)  # seconds
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    fps = Column(Float, nullable=False)
    aspect_ratio = Column(String(10), nullable=False)  # e.g., "16:9"
    codec = Column(String(20), nullable=True)
    bitrate = Column(Integer, nullable=True)  # kbps
    
    # Content information
    subject = Column(String(500), nullable=True)
    script_text = Column(Text, nullable=True)
    search_terms = Column(JSON, default=list)
    language = Column(String(10), nullable=True)
    
    # Processing details
    voice_name = Column(String(100), nullable=True)
    bgm_type = Column(String(50), nullable=True)
    subtitle_enabled = Column(Boolean, default=True)
    video_source = Column(String(50), nullable=True)  # pexels, local, etc.
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)  # 0-100
    processing_time = Column(Float, nullable=True)  # seconds
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    task = relationship("Task", back_populates="videos")
    project = relationship("VideoProject", back_populates="videos")
    materials = relationship("MaterialMetadata", back_populates="video", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_videos_task_id', 'task_id'),
        Index('idx_videos_project_id', 'project_id'),
        Index('idx_videos_filename', 'filename'),
        Index('idx_videos_created_at', 'created_at'),
        Index('idx_videos_duration', 'duration'),
        Index('idx_videos_aspect_ratio', 'aspect_ratio'),
        Index('idx_videos_subject', 'subject'),
        Index('idx_videos_language', 'language'),
        UniqueConstraint('file_path', name='unique_file_path'),
    )


class MaterialMetadata(Base):
    """Metadata for video materials used in generation"""
    __tablename__ = "material_metadata"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String(36), ForeignKey("video_metadata.id"), nullable=False)
    
    # Material identification
    material_id = Column(String(100), nullable=True)  # external ID from source
    source = Column(String(50), nullable=False)  # pexels, local, etc.
    source_url = Column(String(500), nullable=True)
    file_path = Column(String(500), nullable=False)
    
    # Material properties
    duration = Column(Float, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # Usage in video
    used_start_time = Column(Float, nullable=True)  # start time in final video
    used_duration = Column(Float, nullable=True)  # duration used in final video
    search_term = Column(String(200), nullable=True)  # term used to find this material
    
    # Timestamps
    downloaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    video = relationship("VideoMetadata", back_populates="materials")
    
    __table_args__ = (
        Index('idx_materials_video_id', 'video_id'),
        Index('idx_materials_source', 'source'),
        Index('idx_materials_search_term', 'search_term'),
        Index('idx_materials_downloaded_at', 'downloaded_at'),
    )


class UserSession(Base):
    """User sessions for web interface and API access"""
    __tablename__ = "user_sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    
    # Session details
    session_token = Column(String(255), unique=True, nullable=False)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    # Session state
    is_active = Column(Boolean, default=True)
    last_activity = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    
    # Session data
    session_data = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    __table_args__ = (
        Index('idx_sessions_token', 'session_token'),
        Index('idx_sessions_user_id', 'user_id'),
        Index('idx_sessions_active', 'is_active'),
        Index('idx_sessions_expires_at', 'expires_at'),
        Index('idx_sessions_last_activity', 'last_activity'),
    )


class SystemMetrics(Base):
    """System performance and usage metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Metric identification
    metric_type = Column(String(50), nullable=False)  # cpu, memory, disk, gpu, task_throughput
    metric_name = Column(String(100), nullable=False)
    
    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)  # %, MB, seconds, etc.
    
    # Context
    hostname = Column(String(100), nullable=True)
    process_id = Column(Integer, nullable=True)
    context_data = Column(JSON, default=dict)
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_metrics_type', 'metric_type'),
        Index('idx_metrics_name', 'metric_name'),
        Index('idx_metrics_timestamp', 'timestamp'),
        Index('idx_metrics_hostname', 'hostname'),
    )


class CacheEntry(Base):
    """Cache for storing frequently accessed data"""
    __tablename__ = "cache_entries"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Cache key and data
    cache_key = Column(String(255), unique=True, nullable=False)
    cache_data = Column(Text, nullable=False)  # JSON encoded
    cache_type = Column(String(50), nullable=False)  # video_materials, scripts, etc.
    
    # Cache metadata
    size_bytes = Column(Integer, default=0)
    hit_count = Column(Integer, default=0)
    
    # Expiration
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_cache_key', 'cache_key'),
        Index('idx_cache_type', 'cache_type'),
        Index('idx_cache_expires_at', 'expires_at'),
        Index('idx_cache_last_accessed', 'last_accessed'),
    )


class DatabaseMigration(Base):
    """Database migration tracking"""
    __tablename__ = "database_migrations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), unique=True, nullable=False)
    description = Column(String(255), nullable=False)
    applied_at = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    __table_args__ = (
        Index('idx_migrations_version', 'version'),
        Index('idx_migrations_applied_at', 'applied_at'),
    )


# Database initialization and utility functions
def get_database_url(db_path: str = None) -> str:
    """Get database URL for SQLite connection"""
    if db_path is None:
        db_path = "./storage/moneyprinterturbo.db"
    return f"sqlite:///{db_path}"


def create_database_engine(db_path: str = None, echo: bool = False):
    """Create database engine with optimized settings"""
    database_url = get_database_url(db_path)
    engine = create_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,
        pool_recycle=3600,  # Recycle connections every hour
        connect_args={
            'check_same_thread': False,
            'timeout': 30
        }
    )
    return engine


def create_all_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(engine)


def get_session_maker(engine):
    """Get session maker for database operations"""
    return sessionmaker(bind=engine)


# Performance optimization views (created via raw SQL)
PERFORMANCE_VIEWS = [
    """
    CREATE VIEW IF NOT EXISTS task_performance_summary AS
    SELECT 
        t.task_type,
        COUNT(*) as total_tasks,
        AVG(t.actual_duration) as avg_duration,
        AVG(t.memory_peak) as avg_memory,
        SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
        SUM(CASE WHEN t.status = 'failed' THEN 1 ELSE 0 END) as failed_tasks,
        (SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as success_rate
    FROM tasks t
    WHERE t.created_at >= datetime('now', '-30 days')
    GROUP BY t.task_type;
    """,
    
    """
    CREATE VIEW IF NOT EXISTS video_production_stats AS
    SELECT 
        DATE(v.created_at) as production_date,
        COUNT(*) as videos_created,
        SUM(v.duration) as total_duration,
        AVG(v.duration) as avg_duration,
        AVG(v.quality_score) as avg_quality,
        COUNT(DISTINCT v.project_id) as projects_active
    FROM video_metadata v
    WHERE v.created_at >= datetime('now', '-90 days')
    GROUP BY DATE(v.created_at)
    ORDER BY production_date DESC;
    """,
    
    """
    CREATE VIEW IF NOT EXISTS system_health_overview AS
    SELECT 
        sm.metric_type,
        AVG(sm.value) as avg_value,
        MIN(sm.value) as min_value,
        MAX(sm.value) as max_value,
        COUNT(*) as sample_count,
        sm.unit
    FROM system_metrics sm
    WHERE sm.timestamp >= datetime('now', '-1 hour')
    GROUP BY sm.metric_type, sm.unit;
    """
]