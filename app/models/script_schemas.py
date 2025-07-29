"""
Enhanced Script Generation Data Models and Schemas
Provides comprehensive data models for multi-provider LLM routing and domain-specific script generation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from datetime import datetime
import uuid


class ContentCategory(str, Enum):
    """Domain-specific content categories for targeted script generation."""
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    TECHNOLOGY = "technology"
    LIFESTYLE = "lifestyle"
    BUSINESS = "business"
    HEALTH_FITNESS = "health_fitness"
    GAMING = "gaming"
    COOKING = "cooking"
    TRAVEL = "travel"
    NEWS = "news"
    COMEDY = "comedy"
    DOCUMENTARY = "documentary"
    TUTORIAL = "tutorial"
    REVIEW = "review"
    VLOG = "vlog"


class LLMProvider(str, Enum):
    """Supported LLM providers for multi-provider routing."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    COHERE = "cohere"
    HUGGING_FACE = "hugging_face"
    OLLAMA = "ollama"
    REPLICATE = "replicate"


class ScriptTone(str, Enum):
    """Script tone options for content personalization."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ENERGETIC = "energetic"
    EDUCATIONAL = "educational"
    HUMOROUS = "humorous"
    DRAMATIC = "dramatic"
    CONVERSATIONAL = "conversational"
    AUTHORITATIVE = "authoritative"


class ScriptLength(str, Enum):
    """Video length categories with appropriate script lengths."""
    SHORT = "short"  # 15-60 seconds
    MEDIUM = "medium"  # 1-5 minutes
    LONG = "long"  # 5-15 minutes
    EXTENDED = "extended"  # 15+ minutes


class ProviderConfig(BaseModel):
    """Configuration for individual LLM providers."""
    provider: LLMProvider
    api_key: str
    model_name: str
    max_tokens: int = 2000
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    retry_attempts: int = 3
    priority: int = 1  # Lower number = higher priority
    enabled: bool = True
    cost_per_token: float = 0.0
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 90000


class ScriptGenerationRequest(BaseModel):
    """Request model for enhanced script generation."""
    topic: str = Field(..., min_length=5, max_length=500, description="Main topic or theme for the script")
    category: ContentCategory = Field(ContentCategory.EDUCATIONAL, description="Content category for domain-specific optimization")
    tone: ScriptTone = Field(ScriptTone.CONVERSATIONAL, description="Desired tone and style of the script")
    length: ScriptLength = Field(ScriptLength.MEDIUM, description="Target video length category")
    target_audience: str = Field("general", description="Target audience demographic")
    keywords: List[str] = Field(default_factory=list, description="Keywords to include for SEO optimization")
    style_references: List[str] = Field(default_factory=list, description="Reference styles or creators to emulate")
    custom_instructions: Optional[str] = Field(None, description="Additional custom instructions")
    language: str = Field("en", description="Script language (ISO code)")
    include_hooks: bool = Field(True, description="Include attention-grabbing hooks and CTAs")
    optimize_engagement: bool = Field(True, description="Apply engagement optimization techniques")
    preferred_providers: List[LLMProvider] = Field(default_factory=list, description="Preferred LLM providers in order")
    max_retries: int = Field(3, description="Maximum retry attempts across providers")
    
    @validator('keywords')
    def validate_keywords(cls, v):
        if len(v) > 20:
            raise ValueError("Maximum 20 keywords allowed")
        return v


class ScriptSection(BaseModel):
    """Individual section of a generated script."""
    section_type: str = Field(..., description="Type of section (intro, main, conclusion, etc.)")
    content: str = Field(..., description="Section content")
    duration_seconds: Optional[int] = Field(None, description="Estimated duration for this section")
    visual_cues: List[str] = Field(default_factory=list, description="Suggested visual elements")
    audio_cues: List[str] = Field(default_factory=list, description="Suggested audio elements")
    engagement_score: Optional[float] = Field(None, description="Predicted engagement score for this section")


class ScriptQualityMetrics(BaseModel):
    """Quality assessment metrics for generated scripts."""
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score (0-100)")
    engagement_score: float = Field(..., ge=0, le=100, description="Predicted engagement potential")
    clarity_score: float = Field(..., ge=0, le=100, description="Content clarity and coherence")
    seo_score: float = Field(..., ge=0, le=100, description="SEO optimization score")
    originality_score: float = Field(..., ge=0, le=100, description="Content originality assessment")
    viral_potential: float = Field(..., ge=0, le=100, description="Predicted viral potential")
    readability_score: float = Field(..., ge=0, le=100, description="Script readability score")
    emotion_score: float = Field(..., ge=0, le=100, description="Emotional impact score")
    
    # Detailed breakdowns
    strengths: List[str] = Field(default_factory=list, description="Identified script strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")
    suggestions: List[str] = Field(default_factory=list, description="Specific improvement suggestions")


class LLMProviderResponse(BaseModel):
    """Response from individual LLM provider."""
    provider: LLMProvider
    model_used: str
    response_time_ms: int
    token_count: int
    cost_estimate: float
    success: bool
    error_message: Optional[str] = None
    content: Optional[str] = None
    confidence_score: Optional[float] = None


class EnhancedScriptResponse(BaseModel):
    """Complete response for enhanced script generation."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    script_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Generated content
    title: str = Field(..., description="Generated or optimized title")
    script: str = Field(..., description="Complete generated script")
    sections: List[ScriptSection] = Field(default_factory=list, description="Structured script sections")
    
    # Metadata
    category: ContentCategory
    tone: ScriptTone
    length: ScriptLength
    estimated_duration: int = Field(..., description="Estimated video duration in seconds")
    word_count: int = Field(..., description="Script word count")
    
    # Quality assessment
    quality_metrics: ScriptQualityMetrics
    
    # Generation details
    provider_used: LLMProvider
    model_used: str
    generation_time_ms: int
    total_cost: float
    provider_responses: List[LLMProviderResponse] = Field(default_factory=list)
    
    # Optimization features
    hooks: List[str] = Field(default_factory=list, description="Attention-grabbing hooks identified")
    cta_suggestions: List[str] = Field(default_factory=list, description="Call-to-action suggestions")
    seo_keywords: List[str] = Field(default_factory=list, description="SEO keywords incorporated")
    hashtag_suggestions: List[str] = Field(default_factory=list, description="Relevant hashtag suggestions")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DomainTemplate(BaseModel):
    """Domain-specific template for script generation."""
    category: ContentCategory
    name: str
    description: str
    structure: List[str] = Field(..., description="Ordered list of section types")
    prompts: Dict[str, str] = Field(..., description="Section-specific prompts")
    default_tone: ScriptTone
    optimal_length: ScriptLength
    key_elements: List[str] = Field(default_factory=list, description="Essential elements for this domain")
    engagement_tactics: List[str] = Field(default_factory=list, description="Category-specific engagement techniques")
    cta_templates: List[str] = Field(default_factory=list, description="Call-to-action templates")
    visual_suggestions: List[str] = Field(default_factory=list, description="Category-appropriate visual elements")


class ProviderHealthCheck(BaseModel):
    """Health check status for LLM providers."""
    provider: LLMProvider
    status: str = Field(..., description="online, offline, degraded, maintenance")
    response_time_ms: int
    success_rate: float = Field(..., ge=0, le=1, description="Success rate in last 100 requests")
    last_check: datetime
    error_count: int = 0
    consecutive_failures: int = 0
    next_retry: Optional[datetime] = None


class ScriptOptimizationRequest(BaseModel):
    """Request for script optimization and enhancement."""
    script_id: str
    original_script: str
    optimization_goals: List[str] = Field(..., description="engagement, seo, clarity, virality, etc.")
    target_metrics: Dict[str, float] = Field(default_factory=dict, description="Target quality scores")
    preserve_sections: List[str] = Field(default_factory=list, description="Sections to preserve during optimization")


class BatchScriptRequest(BaseModel):
    """Request for batch script generation."""
    requests: List[ScriptGenerationRequest] = Field(..., max_items=10)
    parallel_processing: bool = Field(True, description="Process requests in parallel")
    priority_order: bool = Field(False, description="Process in order of priority")


class ScriptAnalytics(BaseModel):
    """Analytics data for script performance tracking."""
    script_id: str
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    engagement_rate: float = 0.0
    retention_rate: float = 0.0
    click_through_rate: float = 0.0
    conversion_rate: float = 0.0
    actual_duration: Optional[int] = None
    platform_performance: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    audience_feedback: List[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Export all models for easy importing
__all__ = [
    'ContentCategory', 'LLMProvider', 'ScriptTone', 'ScriptLength',
    'ProviderConfig', 'ScriptGenerationRequest', 'ScriptSection',
    'ScriptQualityMetrics', 'LLMProviderResponse', 'EnhancedScriptResponse',
    'DomainTemplate', 'ProviderHealthCheck', 'ScriptOptimizationRequest',
    'BatchScriptRequest', 'ScriptAnalytics'
]