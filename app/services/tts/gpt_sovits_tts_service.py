"""
GPT-SoVITS TTS Service Implementation

Wrapper for the existing GPT-SoVITS TTS implementation in voice.py,
integrating it into the new TTS service architecture.
"""

import asyncio
import logging
from typing import List, Dict, Any
import os

from .base_tts_service import BaseTTSService, TTSRequest, TTSResponse, VoiceInfo, TTSServiceError
from app.services.voice import gpt_sovits_tts, get_gpt_sovits_voices, is_gpt_sovits_voice

logger = logging.getLogger(__name__)


class GPTSoVITSTTSService(BaseTTSService):
    """GPT-SoVITS TTS service implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Service capabilities
        self.supports_ssml = False
        self.supports_emotions = True  # GPT-SoVITS supports emotional speech
        self.supports_characters = True  # Voice cloning capabilities
        self.max_text_length = 5000
        
        # Voice caching
        self._voices_cache = None
        self._voices_cache_valid = False
        
        # Configuration
        self.api_url = config.get("api_url", "http://127.0.0.1:9880")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 120)  # GPT-SoVITS can be slow
        
        logger.info("GPT-SoVITS TTS service initialized")
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech using GPT-SoVITS TTS
        
        Args:
            request: TTSRequest containing text and voice parameters
            
        Returns:
            TTSResponse with synthesized audio
        """
        try:
            logger.info(f"Starting GPT-SoVITS TTS synthesis with voice: {request.voice_name}")
            
            # Validate request
            self._validate_request(request)
            
            # Extract voice parameters
            voice_params = self._parse_voice_name(request.voice_name)
            
            # Create temporary audio file
            import tempfile
            import uuid
            temp_dir = tempfile.gettempdir()
            audio_file = os.path.join(temp_dir, f"gpt_sovits_tts_{uuid.uuid4().hex}.wav")
            
            # Call existing GPT-SoVITS TTS implementation
            sub_maker = gpt_sovits_tts(
                request.text,
                voice_params["voice"],
                audio_file,
                request.speaking_rate  # GPT-SoVITS handles speed internally
            )
            
            if sub_maker is None:
                raise TTSServiceError("GPT-SoVITS TTS synthesis failed", provider="gpt_sovits")
            
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
                gender=voice_params["gender"],
                natural_sample_rate=32000,  # GPT-SoVITS typical output
                is_neural=True  # GPT-SoVITS uses neural synthesis
            )
            
            logger.info(f"GPT-SoVITS TTS synthesis completed successfully")
            
            return TTSResponse(
                audio_content=audio_content,
                audio_format="wav",
                duration=duration,
                voice_info=voice_info,
                subtitle_data=subtitle_data,
                quality_score=0.90  # High quality neural synthesis
            )
            
        except Exception as e:
            logger.error(f"GPT-SoVITS TTS synthesis failed: {e}")
            raise TTSServiceError(f"GPT-SoVITS TTS synthesis failed: {e}", provider="gpt_sovits")
    
    def get_voices(self) -> List[VoiceInfo]:
        """
        Get available GPT-SoVITS TTS voices
        
        Returns:
            List of VoiceInfo objects
        """
        if self._voices_cache and self._voices_cache_valid:
            return self._voices_cache
        
        try:
            # Get voices from existing implementation
            gpt_sovits_voices = get_gpt_sovits_voices()
            
            voices = []
            for voice_string in gpt_sovits_voices:
                # Parse voice string: "gpt_sovits:character_name:language"
                voice_params = self._parse_voice_name(voice_string)
                
                voice_info = VoiceInfo(
                    name=voice_string,
                    language=voice_params.get("language", "en-US"),
                    gender=voice_params.get("gender", "neutral"),
                    natural_sample_rate=32000,
                    is_neural=True,
                    supports_emotions=True
                )
                voices.append(voice_info)
            
            self._voices_cache = voices
            self._voices_cache_valid = True
            
            logger.info(f"Retrieved {len(voices)} GPT-SoVITS TTS voices")
            return voices
            
        except Exception as e:
            logger.error(f"Failed to get GPT-SoVITS TTS voices: {e}")
            return []
    
    def validate_config(self) -> bool:
        """
        Validate GPT-SoVITS TTS configuration
        
        Returns:
            True if API URL is accessible
        """
        try:
            import requests
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"GPT-SoVITS health check failed: {e}")
            return False
    
    def _validate_request(self, request: TTSRequest):
        """Validate TTS request parameters"""
        if not request.text or not request.text.strip():
            raise TTSServiceError("Text cannot be empty", provider="gpt_sovits")
        
        if len(request.text) > self.max_text_length:
            raise TTSServiceError(
                f"Text length {len(request.text)} exceeds maximum {self.max_text_length}",
                provider="gpt_sovits"
            )
        
        if not request.voice_name:
            raise TTSServiceError("Voice name is required", provider="gpt_sovits")
        
        if not is_gpt_sovits_voice(request.voice_name):
            raise TTSServiceError(f"Invalid GPT-SoVITS voice: {request.voice_name}", provider="gpt_sovits")
    
    def _parse_voice_name(self, voice_name: str) -> Dict[str, str]:
        """
        Parse GPT-SoVITS voice name to extract parameters
        
        Args:
            voice_name: Voice name in format "gpt_sovits:character:language"
            
        Returns:
            Dictionary with voice parameters
        """
        parts = voice_name.split(":")
        if len(parts) < 2:
            raise TTSServiceError(f"Invalid GPT-SoVITS voice format: {voice_name}", provider="gpt_sovits")
        
        character = parts[1] if len(parts) > 1 else "default"
        language = parts[2] if len(parts) > 2 else "en"
        
        # Extract character info
        character_lower = character.lower()
        
        # Determine gender from character name patterns
        gender = "neutral"
        if any(name in character_lower for name in ["male", "man", "boy", "father", "brother"]):
            gender = "male"
        elif any(name in character_lower for name in ["female", "woman", "girl", "mother", "sister"]):
            gender = "female"
        
        return {
            "voice": character,
            "language": language,
            "gender": gender,
            "character": character
        }
    
    def _convert_submaker_to_subtitles(self, sub_maker) -> List[Dict]:
        """Convert GPT-SoVITS SubMaker to subtitle data"""
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
            # Simple format - estimate timing based on text length
            total_duration = self._estimate_duration_from_text(sub_maker.subs)
            char_weights = [len(seg) for seg in sub_maker.subs]
            total_chars = sum(char_weights)
            
            current_time = 0.0
            for i, (sub_text, char_count) in enumerate(zip(sub_maker.subs, char_weights)):
                segment_duration = (char_count / total_chars) * total_duration if total_chars > 0 else 1.0
                start_time = current_time
                end_time = current_time + segment_duration
                
                subtitle_data.append({
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "text": sub_text.strip(),
                    "word_index": i
                })
                
                current_time = end_time
        
        return subtitle_data
    
    def _estimate_duration_from_text(self, text_segments: List[str]) -> float:
        """Estimate audio duration from text segments"""
        total_chars = sum(len(seg) for seg in text_segments)
        # GPT-SoVITS is typically slower than normal TTS (more expressive)
        # Estimate: 120 words per minute, average 5 characters per word
        words = total_chars / 5
        duration = (words / 120) * 60  # Convert to seconds
        return max(2.0, duration)  # Minimum 2 seconds for GPT-SoVITS
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get GPT-SoVITS TTS provider information"""
        info = super().get_provider_info()
        info.update({
            "display_name": "GPT-SoVITS",
            "neural_voices": True,
            "voice_cloning": True,
            "emotional_speech": True,
            "languages_supported": 50,  # Multilingual support
            "pricing_model": "free_self_hosted",
            "api_docs": "https://github.com/RVC-Boss/GPT-SoVITS"
        })
        return info
    
    def supports_voice_cloning(self) -> bool:
        """Check if the service supports voice cloning"""
        return True
    
    def supports_emotional_synthesis(self) -> bool:
        """Check if the service supports emotional speech synthesis"""
        return True
