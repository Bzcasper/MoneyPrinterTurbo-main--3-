"""
SiliconFlow TTS Service Implementation

Wrapper for the existing SiliconFlow TTS implementation in voice.py,
integrating it into the new TTS service architecture.
"""

import asyncio
import logging
from typing import List, Dict, Any
import os

from .base_tts_service import BaseTTSService, TTSRequest, TTSResponse, VoiceInfo, TTSServiceError
from app.services.voice import siliconflow_tts, get_siliconflow_voices, is_siliconflow_voice

logger = logging.getLogger(__name__)


class SiliconFlowTTSService(BaseTTSService):
    """SiliconFlow TTS service implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Service capabilities
        self.supports_ssml = False
        self.supports_emotions = False
        self.supports_characters = False
        self.max_text_length = 5000
        
        # Voice caching
        self._voices_cache = None
        self._voices_cache_valid = False
        
        # Configuration validation
        self.api_key = config.get("api_key", "")
        if not self.api_key:
            logger.warning("SiliconFlow API key not provided")
        
        logger.info("SiliconFlow TTS service initialized")
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech using SiliconFlow TTS
        
        Args:
            request: TTSRequest containing text and voice parameters
            
        Returns:
            TTSResponse with synthesized audio
        """
        try:
            logger.info(f"Starting SiliconFlow TTS synthesis with voice: {request.voice_name}")
            
            # Validate request
            self._validate_request(request)
            
            # Parse voice name to extract model and voice
            model, voice = self._parse_voice_name(request.voice_name)
            
            # Create temporary audio file
            import tempfile
            import uuid
            temp_dir = tempfile.gettempdir()
            audio_file = os.path.join(temp_dir, f"siliconflow_tts_{uuid.uuid4().hex}.mp3")
            
            # Call existing SiliconFlow TTS implementation
            sub_maker = siliconflow_tts(
                request.text,
                model,
                voice,
                request.speaking_rate,
                audio_file,
                request.volume_gain + 1.0  # Convert from dB to multiplier
            )
            
            if sub_maker is None:
                raise TTSServiceError("SiliconFlow TTS synthesis failed", provider="siliconflow")
            
            # Read generated audio file
            audio_content = b""
            if os.path.exists(audio_file):
                with open(audio_file, 'rb') as f:
                    audio_content = f.read()
                
                # Clean up temporary file
                try:
                    os.remove(audio_file)
                except:
                    pass
            
            # Calculate duration from audio content
            duration = self._calculate_duration(audio_content)
            
            # Convert SubMaker to subtitle data
            subtitle_data = self._convert_submaker_to_subtitles(sub_maker)
            
            # Create voice info
            voice_info = VoiceInfo(
                name=request.voice_name,
                language=request.language_code,
                gender=self._extract_gender_from_voice(request.voice_name),
                natural_sample_rate=32000  # SiliconFlow default
            )
            
            logger.info(f"SiliconFlow TTS synthesis completed successfully")
            
            return TTSResponse(
                audio_content=audio_content,
                audio_format="mp3",
                duration=duration,
                voice_info=voice_info,
                subtitle_data=subtitle_data,
                quality_score=0.85  # High quality
            )
            
        except Exception as e:
            logger.error(f"SiliconFlow TTS synthesis failed: {e}")
            raise TTSServiceError(f"SiliconFlow TTS synthesis failed: {e}", provider="siliconflow")
    
    def get_voices(self) -> List[VoiceInfo]:
        """
        Get available SiliconFlow TTS voices
        
        Returns:
            List of VoiceInfo objects
        """
        if self._voices_cache and self._voices_cache_valid:
            return self._voices_cache
        
        try:
            # Get voices from existing implementation
            siliconflow_voices = get_siliconflow_voices()
            
            voices = []
            for voice_string in siliconflow_voices:
                # Parse voice string: "siliconflow:model:voice-Gender"
                parts = voice_string.split(":")
                if len(parts) >= 3:
                    model = parts[1]
                    voice_info_part = parts[2]  # e.g., "alex-Male"
                    voice_parts = voice_info_part.split("-")
                    voice_name = voice_parts[0]
                    gender = voice_parts[1].lower() if len(voice_parts) > 1 else "neutral"
                    
                    voice_info = VoiceInfo(
                        name=voice_string,
                        language="en-US",  # SiliconFlow primarily supports English
                        gender=gender,
                        natural_sample_rate=32000,
                        is_neural=True
                    )
                    voices.append(voice_info)
            
            self._voices_cache = voices
            self._voices_cache_valid = True
            
            logger.info(f"Retrieved {len(voices)} SiliconFlow TTS voices")
            return voices
            
        except Exception as e:
            logger.error(f"Failed to get SiliconFlow TTS voices: {e}")
            return []
    
    def validate_config(self) -> bool:
        """
        Validate SiliconFlow TTS configuration
        
        Returns:
            True if API key is available
        """
        return bool(self.api_key)
    
    def _validate_request(self, request: TTSRequest):
        """Validate TTS request parameters"""
        if not self.api_key:
            raise TTSServiceError("SiliconFlow API key not configured", provider="siliconflow")
        
        if not request.text or not request.text.strip():
            raise TTSServiceError("Text cannot be empty", provider="siliconflow")
        
        if len(request.text) > self.max_text_length:
            raise TTSServiceError(
                f"Text length {len(request.text)} exceeds maximum {self.max_text_length}",
                provider="siliconflow"
            )
        
        if not request.voice_name:
            raise TTSServiceError("Voice name is required", provider="siliconflow")
        
        if not is_siliconflow_voice(request.voice_name):
            raise TTSServiceError(f"Invalid SiliconFlow voice: {request.voice_name}", provider="siliconflow")
    
    def _parse_voice_name(self, voice_name: str) -> tuple[str, str]:
        """
        Parse SiliconFlow voice name to extract model and voice
        
        Args:
            voice_name: Voice name in format "siliconflow:model:voice-Gender"
            
        Returns:
            Tuple of (model, full_voice_path)
        """
        parts = voice_name.split(":")
        if len(parts) < 3:
            raise TTSServiceError(f"Invalid SiliconFlow voice format: {voice_name}", provider="siliconflow")
        
        model = parts[1]
        voice_info = parts[2]
        voice = voice_info.split("-")[0]  # Remove gender suffix
        
        # Construct full voice path as expected by SiliconFlow API
        full_voice = f"{model}:{voice}"
        
        return model, full_voice
    
    def _convert_submaker_to_subtitles(self, sub_maker) -> List[Dict]:
        """Convert SiliconFlow SubMaker to subtitle data"""
        if not sub_maker or not hasattr(sub_maker, 'subs') or not sub_maker.subs:
            return []
        
        subtitle_data = []
        
        # Handle both timing and non-timing formats
        if hasattr(sub_maker, 'offset') and sub_maker.offset:
            # Format with timing offsets
            for i, (sub_text, offset) in enumerate(zip(sub_maker.subs, sub_maker.offset)):
                start_time = offset[0] / 10000000  # Convert from 100ns to seconds
                end_time = offset[1] / 10000000
                
                subtitle_data.append({
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "text": sub_text.strip(),
                    "word_index": i
                })
        else:
            # Simple format - estimate timing
            total_duration = self._estimate_duration_from_text(sub_maker.subs)
            segment_duration = total_duration / len(sub_maker.subs) if sub_maker.subs else 1.0
            
            for i, sub_text in enumerate(sub_maker.subs):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                
                subtitle_data.append({
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "text": sub_text.strip(),
                    "word_index": i
                })
        
        return subtitle_data
    
    def _extract_gender_from_voice(self, voice_name: str) -> str:
        """Extract gender from SiliconFlow voice name"""
        voice_lower = voice_name.lower()
        
        if "-male" in voice_lower:
            return "male"
        elif "-female" in voice_lower:
            return "female"
        
        # Check common voice name patterns
        male_names = ["alex", "benjamin", "charles", "david"]
        female_names = ["anna", "bella", "claire", "diana"]
        
        for name in male_names:
            if name in voice_lower:
                return "male"
        
        for name in female_names:
            if name in voice_lower:
                return "female"
        
        return "neutral"
    
    def _estimate_duration_from_text(self, text_segments: List[str]) -> float:
        """Estimate audio duration from text segments"""
        total_chars = sum(len(seg) for seg in text_segments)
        # Rough estimate: 150 words per minute, average 5 characters per word
        words = total_chars / 5
        duration = (words / 150) * 60  # Convert to seconds
        return max(1.0, duration)  # Minimum 1 second
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get SiliconFlow TTS provider information"""
        info = super().get_provider_info()
        info.update({
            "display_name": "SiliconFlow TTS",
            "neural_voices": True,
            "premium_voices": True,
            "languages_supported": 10,  # Primarily English and Chinese
            "pricing_model": "pay_per_character",
            "api_docs": "https://docs.siliconflow.cn/"
        })
        return info
