import warnings
from enum import Enum
from typing import Any, List, Optional, Union, Dict
from datetime import datetime

import pydantic
from pydantic import BaseModel, Field

# 忽略 Pydantic 的特定警告
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Field name.*shadows an attribute in parent.*",
)


class VideoConcatMode(str, Enum):
    random = "random"
    sequential = "sequential"


class VideoTransitionMode(str, Enum):
    none = "none"
    shuffle = "shuffle"
    fade_in = "fade_in"
    fade_out = "fade_out"
    slide_in = "slide_in"
    slide_out = "slide_out"
    
    @classmethod
    def _missing_(cls, value):
        if value is None or value == "":
            return cls.none
        return super()._missing_(value)


class VideoAspect(str, Enum):
    landscape = "16:9"
    portrait = "9:16"
    square = "1:1"

    def to_resolution(self):
        if self == VideoAspect.landscape.value:
            return 1920, 1080
        elif self == VideoAspect.portrait.value:
            return 1080, 1920
        elif self == VideoAspect.square.value:
            return 1080, 1080
        return 1080, 1920


class _Config:
    arbitrary_types_allowed = True


@pydantic.dataclasses.dataclass(config=_Config)
class MaterialInfo:
    provider: str = "pexels"
    url: str = ""
    duration: int = 0


class VideoParams(BaseModel):
    """
    {
      "video_subject": "",
      "video_aspect": "横屏 16:9（西瓜视频）",
      "voice_name": "女生-晓晓",
      "bgm_name": "random",
      "font_name": "STHeitiMedium 黑体-中",
      "text_color": "#FFFFFF",
      "font_size": 60,
      "stroke_color": "#000000",
      "stroke_width": 1.5
    }
    """

    video_subject: str
    video_script: str = ""  # Script used to generate the video
    video_terms: Optional[Union[str, List[str]]] = None  # Keywords used to generate the video
    video_aspect: Optional[VideoAspect] = VideoAspect.portrait.value
    video_concat_mode: Optional[VideoConcatMode] = VideoConcatMode.random.value
    video_transition_mode: Optional[VideoTransitionMode] = None
    video_clip_duration: Optional[int] = 5
    video_count: Optional[int] = 1

    video_source: Optional[str] = "pexels"
    video_materials: Optional[List[MaterialInfo]] = (
        None  # Materials used to generate the video
    )

    video_language: Optional[str] = ""  # auto detect

    voice_name: Optional[str] = ""
    voice_volume: Optional[float] = 1.0
    voice_rate: Optional[float] = 1.0
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = 0.2

    subtitle_enabled: Optional[bool] = True
    subtitle_position: Optional[str] = "bottom"  # top, bottom, center
    custom_position: float = 70.0
    font_name: Optional[str] = "STHeitiMedium.ttc"
    text_fore_color: Optional[str] = "#FFFFFF"
    text_background_color: Union[bool, str] = True

    font_size: int = 60
    stroke_color: Optional[str] = "#000000"
    stroke_width: float = 1.5
    n_threads: Optional[int] = 2
    paragraph_number: Optional[int] = 1


class SubtitleRequest(BaseModel):
    video_script: str
    video_language: Optional[str] = ""
    voice_name: Optional[str] = "zh-CN-XiaoxiaoNeural-Female"
    voice_volume: Optional[float] = 1.0
    voice_rate: Optional[float] = 1.2
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = 0.2
    subtitle_position: Optional[str] = "bottom"
    font_name: Optional[str] = "STHeitiMedium.ttc"
    text_fore_color: Optional[str] = "#FFFFFF"
    text_background_color: Union[bool, str] = True
    font_size: int = 60
    stroke_color: Optional[str] = "#000000"
    stroke_width: float = 1.5
    video_source: Optional[str] = "local"
    subtitle_enabled: Optional[str] = "true"


class AudioRequest(BaseModel):
    video_script: str
    video_language: Optional[str] = ""
    voice_name: Optional[str] = "zh-CN-XiaoxiaoNeural-Female"
    voice_volume: Optional[float] = 1.0
    voice_rate: Optional[float] = 1.2
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = 0.2
    video_source: Optional[str] = "local"


class VideoScriptParams:
    """
    {
      "video_subject": "春天的花海",
      "video_language": "",
      "paragraph_number": 1
    }
    """

    video_subject: Optional[str] = "春天的花海"
    video_language: Optional[str] = ""
    paragraph_number: Optional[int] = 1


class VideoTermsParams:
    """
    {
      "video_subject": "",
      "video_script": "",
      "amount": 5
    }
    """

    video_subject: Optional[str] = "春天的花海"
    video_script: Optional[str] = (
        "春天的花海，如诗如画般展现在眼前。万物复苏的季节里，大地披上了一袭绚丽多彩的盛装。金黄的迎春、粉嫩的樱花、洁白的梨花、艳丽的郁金香……"
    )
    amount: Optional[int] = 5


class BaseResponse(BaseModel):
    status: int = 200
    message: Optional[str] = "success"
    data: Any = None


class TaskVideoRequest(VideoParams, BaseModel):
    pass


class TaskQueryRequest(BaseModel):
    pass


class VideoScriptRequest(VideoScriptParams, BaseModel):
    pass


class VideoTermsRequest(VideoTermsParams, BaseModel):
    pass


######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
class TaskResponse(BaseResponse):
    class TaskResponseData(BaseModel):
        task_id: str

    data: TaskResponseData

    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {"task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558"},
            },
        }


class TaskQueryResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "state": 1,
                    "progress": 100,
                    "videos": [
                        "http://127.0.0.1:8080/tasks/6c85c8cc-a77a-42b9-bc30-947815aa0558/final-1.mp4"
                    ],
                    "combined_videos": [
                        "http://127.0.0.1:8080/tasks/6c85c8cc-a77a-42b9-bc30-947815aa0558/combined-1.mp4"
                    ],
                },
            },
        }


class TaskDeletionResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "state": 1,
                    "progress": 100,
                    "videos": [
                        "http://127.0.0.1:8080/tasks/6c85c8cc-a77a-42b9-bc30-947815aa0558/final-1.mp4"
                    ],
                    "combined_videos": [
                        "http://127.0.0.1:8080/tasks/6c85c8cc-a77a-42b9-bc30-947815aa0558/combined-1.mp4"
                    ],
                },
            },
        }


class VideoScriptResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "video_script": "春天的花海，是大自然的一幅美丽画卷。在这个季节里，大地复苏，万物生长，花朵争相绽放，形成了一片五彩斑斓的花海..."
                },
            },
        }


class VideoTermsResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {"video_terms": ["sky", "tree"]},
            },
        }


class BgmRetrieveResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "files": [
                        {
                            "name": "output013.mp3",
                            "size": 1891269,
                            "file": "/MoneyPrinterTurbo/resource/songs/output013.mp3",
                        }
                    ]
                },
            },
        }


class BgmUploadResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {"file": "/MoneyPrinterTurbo/resource/songs/example.mp3"},
            },
        }


# TTS-related schema models

class TTSProvider(BaseModel):
    """TTS provider information"""
    name: str = Field(..., description="Provider identifier")
    display_name: str = Field(..., description="Human-readable provider name")
    is_active: bool = Field(True, description="Whether provider is enabled")
    capabilities: List[str] = Field(default_factory=list, description="Provider capabilities")
    priority: int = Field(0, description="Provider priority (higher = preferred)")


class VoiceInfo(BaseModel):
    """Voice information for TTS"""
    name: str = Field(..., description="Voice identifier")
    display_name: Optional[str] = Field(None, description="Human-readable voice name")
    language: str = Field(..., description="Language code (e.g., en-US)")
    gender: Optional[str] = Field(None, description="Voice gender")
    age_group: Optional[str] = Field(None, description="Age group (adult, child, elderly)")
    style: Optional[str] = Field(None, description="Voice style (casual, professional, etc.)")
    natural_sample_rate: Optional[int] = Field(None, description="Natural sample rate in Hz")
    is_neural: bool = Field(False, description="Whether this is a neural voice")
    is_premium: bool = Field(False, description="Whether voice requires premium access")
    character_info: Optional[Dict[str, Any]] = Field(None, description="Character information for character voices")


class TTSRequest(BaseModel):
    """TTS synthesis request"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    provider: str = Field("google", description="TTS provider to use")
    voice_name: str = Field(..., description="Voice identifier")
    language_code: str = Field("en-US", description="Language code")
    speaking_rate: float = Field(1.0, ge=0.25, le=4.0, description="Speaking rate multiplier")
    pitch: float = Field(0.0, ge=-20.0, le=20.0, description="Pitch adjustment in semitones")
    volume_gain: float = Field(0.0, ge=-96.0, le=16.0, description="Volume gain in dB")
    gender: Optional[str] = Field(None, description="Preferred voice gender")
    
    # Character-specific fields
    character_id: Optional[str] = Field(None, description="Character ID for character voices")
    emotion: str = Field("neutral", description="Emotion to express")
    voice_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional voice settings")
    
    # Output options
    output_format: str = Field("mp3", description="Audio output format")
    sample_rate: Optional[int] = Field(None, description="Audio sample rate")


class TTSResponse(BaseModel):
    """TTS synthesis response"""
    audio_file_path: Optional[str] = Field(None, description="Path to generated audio file")
    audio_url: Optional[str] = Field(None, description="URL to audio file if hosted")
    audio_format: str = Field("mp3", description="Audio format")
    duration: float = Field(..., description="Audio duration in seconds")
    voice_info: VoiceInfo = Field(..., description="Information about the voice used")
    
    # Quality and metadata
    quality_score: Optional[float] = Field(None, description="Quality score (0.0-1.0)")
    emotion_score: Optional[float] = Field(None, description="Emotion confidence score")
    synthesis_time: Optional[float] = Field(None, description="Time taken to synthesize")
    
    # Subtitle and timing data
    subtitle_data: List[Dict[str, Any]] = Field(default_factory=list, description="Word-level timing data")
    
    # Caching
    cache_hit: bool = Field(False, description="Whether response was served from cache")
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_file_path": "/path/to/audio.mp3",
                "audio_format": "mp3",
                "duration": 5.2,
                "voice_info": {
                    "name": "en-US-Wavenet-D",
                    "language": "en-US",
                    "gender": "male",
                    "is_neural": True
                },
                "quality_score": 0.95,
                "synthesis_time": 1.2,
                "cache_hit": False
            }
        }


class VoiceListResponse(BaseModel):
    """Response for voice listing"""
    provider: str = Field(..., description="TTS provider")
    voices: List[VoiceInfo] = Field(..., description="Available voices")
    total_count: int = Field(..., description="Total number of voices")
    
    class Config:
        json_schema_extra = {
            "example": {
                "provider": "google",
                "voices": [
                    {
                        "name": "en-US-Wavenet-D",
                        "display_name": "US English (Male, Wavenet)",
                        "language": "en-US",
                        "gender": "male",
                        "is_neural": True
                    }
                ],
                "total_count": 1
            }
        }


class BatchTTSRequest(BaseModel):
    """Batch TTS synthesis request"""
    requests: List[TTSRequest] = Field(..., min_items=1, max_items=50, description="TTS requests to process")
    parallel_processing: bool = Field(True, description="Whether to process requests in parallel")
    max_concurrent: int = Field(5, ge=1, le=20, description="Maximum concurrent requests")


class BatchTTSResponse(BaseModel):
    """Batch TTS synthesis response"""
    results: List[Union[TTSResponse, Dict[str, str]]] = Field(..., description="Results or errors for each request")
    successful_count: int = Field(..., description="Number of successful syntheses")
    failed_count: int = Field(..., description="Number of failed syntheses")
    total_time: float = Field(..., description="Total processing time")


# Character-specific models (re-exported from characterbox service)

class CharacterInfo(BaseModel):
    """Character information for character voices"""
    character_id: str = Field(..., description="Character identifier")
    name: str = Field(..., description="Character name")
    description: Optional[str] = Field(None, description="Character description")
    personality_traits: List[str] = Field(default_factory=list, description="Personality traits")
    supported_emotions: List[str] = Field(default_factory=list, description="Supported emotions")
    language_codes: List[str] = Field(default_factory=list, description="Supported languages")
    avatar_url: Optional[str] = Field(None, description="Character avatar URL")
    is_premium: bool = Field(False, description="Whether character requires premium access")


class CharacterSpeechRequest(BaseModel):
    """Request for character-based speech synthesis"""
    character_id: str = Field(..., description="Character to use")
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    emotion: str = Field("neutral", description="Emotion to express")
    language_code: str = Field("en-US", description="Language code")
    voice_settings: Dict[str, Any] = Field(default_factory=dict, description="Voice customization")


class ConversationRequest(BaseModel):
    """Request for multi-character conversation"""
    characters: List[str] = Field(..., min_items=2, description="Character IDs")
    script: str = Field(..., min_length=1, description="Conversation script")
    conversation_type: str = Field("dialogue", description="Type of conversation")
    scene_setting: Optional[str] = Field(None, description="Scene context")


class ConversationResponse(BaseModel):
    """Response from conversation generation"""
    conversation_id: str = Field(..., description="Conversation identifier")
    audio_segments: List[Dict[str, Any]] = Field(..., description="Audio segments")
    total_duration: float = Field(..., description="Total duration")
    combined_audio_url: Optional[str] = Field(None, description="Combined audio URL")


# Analytics and monitoring models

class TTSUsageStats(BaseModel):
    """TTS usage statistics"""
    provider: str = Field(..., description="TTS provider")
    total_requests: int = Field(0, description="Total requests")
    successful_requests: int = Field(0, description="Successful requests") 
    average_duration: float = Field(0.0, description="Average audio duration")
    average_synthesis_time: float = Field(0.0, description="Average synthesis time")
    cache_hit_rate: float = Field(0.0, description="Cache hit rate percentage")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


class ProviderHealthStatus(BaseModel):
    """Health status for TTS providers"""
    provider: str = Field(..., description="Provider name")
    is_healthy: bool = Field(..., description="Whether provider is healthy")
    last_check: datetime = Field(..., description="Last health check timestamp")
    response_time: Optional[float] = Field(None, description="Average response time")
    error_rate: float = Field(0.0, description="Recent error rate percentage")
    message: Optional[str] = Field(None, description="Status message")


class TTSAnalyticsResponse(BaseModel):
    """TTS analytics dashboard response"""
    usage_stats: List[TTSUsageStats] = Field(..., description="Usage statistics by provider")
    provider_health: List[ProviderHealthStatus] = Field(..., description="Provider health status")
    popular_voices: List[Dict[str, Any]] = Field(..., description="Most used voices")
    cost_analysis: Dict[str, Any] = Field(..., description="Cost breakdown")
    recommendations: List[str] = Field(..., description="Optimization recommendations")
