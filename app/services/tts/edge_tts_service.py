"""
Edge TTS Service Implementation

Wrapper for the existing Edge TTS implementation in voice.py,
integrating it into the new TTS service architecture.
"""

import asyncio
import logging
from typing import List, Dict, Any
import os

from .base_tts_service import BaseTTSService, TTSRequest, TTSResponse, VoiceInfo, TTSServiceError
from app.services.voice import azure_tts_v1, get_all_azure_voices, is_azure_v2_voice, azure_tts_v2

logger = logging.getLogger(__name__)


class EdgeTTSService(BaseTTSService):
    """Microsoft Edge TTS service implementation (free)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Service capabilities
        self.supports_ssml = True
        self.supports_emotions = False
        self.supports_characters = False
        self.max_text_length = 32767  # Edge TTS limit
        
        # Voice caching
        self._voices_cache = None
        self._voices_cache_valid = False
        
        logger.info("Edge TTS service initialized")
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech using Edge TTS
        
        Args:
            request: TTSRequest containing text and voice parameters
            
        Returns:
            TTSResponse with synthesized audio
        """
        try:
            logger.info(f"Starting Edge TTS synthesis with voice: {request.voice_name}")
            
            # Validate request
            self._validate_request(request)
            
            # Create temporary audio file
            import tempfile
            import uuid
            temp_dir = tempfile.gettempdir()
            audio_file = os.path.join(temp_dir, f"edge_tts_{uuid.uuid4().hex}.mp3")
            
            # Call existing Edge TTS implementation
            if is_azure_v2_voice(request.voice_name):
                sub_maker = azure_tts_v2(request.text, request.voice_name, audio_file)
            else:
                sub_maker = azure_tts_v1(
                    request.text,
                    request.voice_name,
                    request.speaking_rate,
                    audio_file
                )
            
            if sub_maker is None:
                raise TTSServiceError("Edge TTS synthesis failed", provider="edge")
            
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
                natural_sample_rate=22050
            )
            
            logger.info(f"Edge TTS synthesis completed successfully")
            
            return TTSResponse(
                audio_content=audio_content,
                audio_format="mp3",
                duration=duration,
                voice_info=voice_info,
                subtitle_data=subtitle_data,
                quality_score=0.75  # Good quality for free service
            )
            
        except Exception as e:
            logger.error(f"Edge TTS synthesis failed: {e}")
            raise TTSServiceError(f"Edge TTS synthesis failed: {e}", provider="edge")
    
    def get_voices(self) -> List[VoiceInfo]:
        """
        Get available Edge TTS voices
        
        Returns:
            List of VoiceInfo objects
        """
        if self._voices_cache and self._voices_cache_valid:
            return self._voices_cache
        
        try:
            # Get voices from existing implementation
            azure_voices = get_all_azure_voices()
            
            voices = []
            for voice_line in azure_voices:
                if voice_line.startswith("Name: "):
                    voice_name = voice_line.replace("Name: ", "").strip()
                    
                    # Extract language from voice name
                    language = self._extract_language_from_voice(voice_name)
                    gender = self._extract_gender_from_voice(voice_name)
                    
                    voice_info = VoiceInfo(
                        name=voice_name,
                        language=language,
                        gender=gender,
                        natural_sample_rate=22050,
                        is_neural=True  # Most Edge voices are neural
                    )
                    voices.append(voice_info)
            
            self._voices_cache = voices
            self._voices_cache_valid = True
            
            logger.info(f"Retrieved {len(voices)} Edge TTS voices")
            return voices
            
        except Exception as e:
            logger.error(f"Failed to get Edge TTS voices: {e}")
            return []
    
    def validate_config(self) -> bool:
        """
        Validate Edge TTS configuration (always valid as it's free)
        
        Returns:
            True (Edge TTS is always available)
        """
        return True
    
    def _validate_request(self, request: TTSRequest):
        """Validate TTS request parameters"""
        if not request.text or not request.text.strip():
            raise TTSServiceError("Text cannot be empty", provider="edge")
        
        if len(request.text) > self.max_text_length:
            raise TTSServiceError(
                f"Text length {len(request.text)} exceeds maximum {self.max_text_length}",
                provider="edge"
            )
        
        if not request.voice_name:
            raise TTSServiceError("Voice name is required", provider="edge")
    
    def _convert_submaker_to_subtitles(self, sub_maker) -> List[Dict]:
        """Convert Edge TTS SubMaker to subtitle data"""
        if not sub_maker or not hasattr(sub_maker, 'subs') or not sub_maker.subs:
            return []
        
        subtitle_data = []
        
        # Handle both old and new SubMaker formats
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
            # Simple format without timing
            total_duration = self._calculate_duration_from_text(sub_maker.subs)
            words_per_second = len(sub_maker.subs) / total_duration if total_duration > 0 else 1
            
            current_time = 0.0
            for i, sub_text in enumerate(sub_maker.subs):
                word_duration = 1.0 / words_per_second
                
                subtitle_data.append({
                    "start": round(current_time, 2),
                    "end": round(current_time + word_duration, 2),
                    "text": sub_text.strip(),
                    "word_index": i
                })
                
                current_time += word_duration
        
        return subtitle_data
    
    def _extract_language_from_voice(self, voice_name: str) -> str:
        """Extract language code from voice name"""
        if voice_name.startswith("en-"):
            return "en-US"
        elif voice_name.startswith("zh-"):
            return "zh-CN"
        elif voice_name.startswith("es-"):
            return "es-ES"
        elif voice_name.startswith("fr-"):
            return "fr-FR"
        elif voice_name.startswith("de-"):
            return "de-DE"
        elif voice_name.startswith("ja-"):
            return "ja-JP"
        elif voice_name.startswith("ko-"):
            return "ko-KR"
        else:
            # Default fallback
            parts = voice_name.split("-")
            if len(parts) >= 2:
                return f"{parts[0]}-{parts[1]}"
            return "en-US"
    
    def _extract_gender_from_voice(self, voice_name: str) -> str:
        """Extract gender from voice name"""
        voice_lower = voice_name.lower()
        if "female" in voice_lower or voice_lower.endswith("neural"):
            # Many neural voices with specific names are female
            female_patterns = ["aria", "jenny", "nancy", "sara", "jane", "emma"]
            if any(pattern in voice_lower for pattern in female_patterns):
                return "female"
        
        if "male" in voice_lower:
            return "male"
        
        # Try to determine from common name patterns
        male_patterns = ["guy", "davis", "tony", "brian", "ryan", "adam"]
        female_patterns = ["aria", "jenny", "nancy", "sara", "jane", "emma", "michelle"]
        
        for pattern in female_patterns:
            if pattern in voice_lower:
                return "female"
        
        for pattern in male_patterns:
            if pattern in voice_lower:
                return "male"
        
        return "neutral"
    
    def _calculate_duration_from_text(self, text_segments: List[str]) -> float:
        """Estimate duration from text segments"""
        total_chars = sum(len(seg) for seg in text_segments)
        # Rough estimate: 150 words per minute, average 5 characters per word
        words = total_chars / 5
        duration = (words / 150) * 60  # Convert to seconds
        return max(1.0, duration)  # Minimum 1 second
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get Edge TTS provider information"""
        info = super().get_provider_info()
        info.update({
            "display_name": "Microsoft Edge TTS (Free)",
            "neural_voices": True,
            "premium_voices": False,
            "languages_supported": 100+,
            "pricing_model": "free",
            "api_docs": "https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/"
        })
        return info
