#!/usr/bin/env python3
"""
GPT-SoVITS Configuration for High-Quality Chinese Voice Synthesis
AudioSpecialist Agent Configuration for MoneyPrinterTurbo
Optimized for YouTube Shorts Motivational Content
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class VoiceModel:
    """Voice model configuration for GPT-SoVITS"""
    model: str
    voice: str
    gender: str
    description: str
    use_case: str
    emotional_range: str
    sample_rate: int
    settings: Dict[str, float]

class GPTSoVITSConfig:
    """GPT-SoVITS Configuration Manager"""
    
    def __init__(self, base_path: str = "/home/bobby/Documents/MoneyPrinterTurbo"):
        self.base_path = Path(base_path)
        self.config_path = self.base_path / "config.toml"
        
        # YouTube Shorts Audio Specifications
        self.youtube_specs = {
            "format": "AAC",
            "bitrate": 384000,  # 384 kbps in bits per second
            "sample_rate": 48000,  # 48kHz
            "channels": 2,  # Stereo
            "max_duration": 180,  # 3 minutes max
            "target_sample_rate": 32000  # GPT-SoVITS optimal rate
        }
        
        # Chinese Voice Models for Motivational Content
        self.voice_models = {
            "motivational_male_deep": VoiceModel(
                model="gpt-sovits-v2",
                voice="narrator",
                gender="Male",
                description="深沉叙述者",
                use_case="Authoritative motivational content",
                emotional_range="Deep, commanding, inspiring",
                sample_rate=32000,
                settings={
                    "speed_factor": 1.0,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 5,
                    "batch_size": 1,
                    "fragment_interval": 0.3,
                    "repetition_penalty": 1.35
                }
            ),
            "motivational_female_warm": VoiceModel(
                model="gpt-sovits-v2",
                voice="emma",
                gender="Female", 
                description="温和女声",
                use_case="Gentle, encouraging motivational content",
                emotional_range="Warm, supportive, uplifting",
                sample_rate=32000,
                settings={
                    "speed_factor": 0.95,
                    "temperature": 1.1,
                    "top_p": 0.9,
                    "top_k": 5,
                    "batch_size": 1,
                    "fragment_interval": 0.3,
                    "repetition_penalty": 1.25
                }
            ),
            "motivational_male_energetic": VoiceModel(
                model="gpt-sovits-v2",
                voice="energetic",
                gender="Male",
                description="活力青春",
                use_case="High-energy motivational content",
                emotional_range="Dynamic, enthusiastic, powerful",
                sample_rate=32000,
                settings={
                    "speed_factor": 1.1,
                    "temperature": 1.2,
                    "top_p": 0.85,
                    "top_k": 8,
                    "batch_size": 1,
                    "fragment_interval": 0.2,
                    "repetition_penalty": 1.4
                }
            ),
            "motivational_female_professional": VoiceModel(
                model="gpt-sovits-v2",
                voice="professional",
                gender="Female",
                description="专业播报",
                use_case="Professional motivational presentations",
                emotional_range="Clear, confident, articulate",
                sample_rate=32000,
                settings={
                    "speed_factor": 1.0,
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 5,
                    "batch_size": 1,
                    "fragment_interval": 0.3,
                    "repetition_penalty": 1.3
                }
            )
        }
        
        # API Configuration
        self.api_config = {
            "base_url": "http://localhost:9880",
            "endpoints": {
                "set_weights": "/set_gpt_weights",
                "tts": "/tts",
                "status": "/status"
            },
            "timeout": 60,
            "retry_attempts": 3
        }
        
        # BGM Configuration for YouTube Shorts
        self.bgm_config = {
            "mixing_ratios": {
                "voice_primary": 0.7,  # 70% voice volume
                "bgm_secondary": 0.3,  # 30% BGM volume
                "voice_fade": 0.5,     # 0.5s fade
                "bgm_fade": 1.0        # 1.0s fade
            },
            "recommended_bgm_types": [
                "chinese_instrumental",
                "motivational_upbeat", 
                "corporate_inspiring",
                "energetic_percussion"
            ],
            "audio_processing": {
                "normalize_audio": True,
                "apply_compression": True,
                "eq_settings": {
                    "voice_boost_freq": [1000, 3000],  # Hz range for voice clarity
                    "bgm_reduce_freq": [1000, 3000]    # Reduce BGM in voice range
                }
            }
        }

    def get_gpt_sovits_payload(self, text: str, voice_model_key: str) -> Dict:
        """Generate GPT-SoVITS API payload for given text and voice model"""
        if voice_model_key not in self.voice_models:
            raise ValueError(f"Voice model '{voice_model_key}' not found")
        
        model = self.voice_models[voice_model_key]
        
        payload = {
            "text": text,
            "text_lang": "zh",  # Chinese language
            "ref_audio_path": None,  # Will be set based on model
            "aux_ref_audio_paths": [],
            "prompt_lang": "zh",
            "prompt_text": "",
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
            **model.settings
        }
        
        return payload

    def get_optimal_voice_for_content(self, content_type: str, gender_preference: Optional[str] = None) -> str:
        """Recommend optimal voice model based on content type and gender preference"""
        content_mapping = {
            "success_stories": "motivational_male_deep",
            "daily_motivation": "motivational_female_warm", 
            "workout_energy": "motivational_male_energetic",
            "business_tips": "motivational_female_professional",
            "life_advice": "motivational_male_deep",
            "personal_growth": "motivational_female_warm"
        }
        
        default_voice = content_mapping.get(content_type, "motivational_male_deep")
        
        # Apply gender preference if specified
        if gender_preference:
            gender_filtered = {k: v for k, v in self.voice_models.items() 
                             if v.gender.lower() == gender_preference.lower()}
            if gender_filtered:
                # Find best match within preferred gender
                for key in gender_filtered:
                    if content_type in self.voice_models[key].use_case.lower():
                        return key
                # Return first available in preferred gender
                return list(gender_filtered.keys())[0]
        
        return default_voice

    def generate_audio_pipeline_config(self) -> Dict:
        """Generate complete audio processing pipeline configuration"""
        return {
            "input_processing": {
                "text_preprocessing": {
                    "remove_special_chars": True,
                    "normalize_punctuation": True,
                    "split_long_sentences": True,
                    "max_sentence_length": 100
                }
            },
            "voice_synthesis": {
                "provider": "gpt-sovits",
                "api_config": self.api_config,
                "quality_settings": {
                    "sample_rate": self.youtube_specs["target_sample_rate"],
                    "bitrate": 256000,  # High quality for processing
                    "channels": 1  # Mono for voice, will be converted to stereo
                }
            },
            "bgm_integration": {
                "bgm_sources": [
                    f"{self.base_path}/resource/songs/",
                    f"{self.base_path}/storage/bgm/"
                ],
                "mixing_config": self.bgm_config["mixing_ratios"],
                "audio_processing": self.bgm_config["audio_processing"]
            },
            "output_optimization": {
                "youtube_specs": self.youtube_specs,
                "format_conversion": {
                    "from_wav_to_aac": True,
                    "apply_loudness_normalization": True,
                    "target_lufs": -16  # YouTube recommended loudness
                }
            },
            "quality_testing": {
                "pronunciation_check": True,
                "emotional_tone_analysis": True,
                "cultural_appropriateness_check": True,
                "audio_sync_validation": True
            }
        }

    def save_config_to_toml(self) -> bool:
        """Save GPT-SoVITS configuration to config.toml"""
        try:
            # Read existing config.toml
            config_content = ""
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_content = f.read()
            
            # Add GPT-SoVITS section if not exists
            gpt_sovits_section = f"""
[gpt_sovits]
# GPT-SoVITS API Configuration for Chinese TTS
api_base_url = "{self.api_config['base_url']}"
api_key = ""  # Optional API key
timeout = {self.api_config['timeout']}
retry_attempts = {self.api_config['retry_attempts']}

# Recommended voice models for motivational content
default_male_voice = "motivational_male_deep"
default_female_voice = "motivational_female_warm"
energetic_voice = "motivational_male_energetic"
professional_voice = "motivational_female_professional"

# Audio quality settings for YouTube Shorts
target_sample_rate = {self.youtube_specs['target_sample_rate']}
output_sample_rate = {self.youtube_specs['sample_rate']}
output_bitrate = {self.youtube_specs['bitrate']}
"""
            
            if "[gpt_sovits]" not in config_content:
                config_content += gpt_sovits_section
                
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    f.write(config_content)
                    
                return True
            return False  # Section already exists
            
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

# Test phrases for Chinese pronunciation and emotional tone
TEST_PHRASES = {
    "motivational": [
        "每一天都是新的开始，相信自己的力量！",
        "成功的路上没有捷径，但有决心就有希望。",
        "不要害怕失败，害怕的应该是从未尝试。",
        "你的潜力远比你想象的要大，勇敢地追求梦想吧！"
    ],
    "business": [
        "机会总是青睐有准备的人。",
        "创新是企业发展的核心动力。",
        "团队合作能够创造无限可能。"
    ],
    "personal_growth": [
        "成长就是不断突破自己的舒适圈。",
        "学习是一辈子的事业。",
        "今天的努力是明天成功的基础。"
    ]
}

if __name__ == "__main__":
    # Initialize configuration
    config = GPTSoVITSConfig()
    
    # Save configuration to TOML
    success = config.save_config_to_toml()
    print(f"Configuration saved: {success}")
    
    # Test voice model selection
    test_voice = config.get_optimal_voice_for_content("success_stories", "Male")
    print(f"Recommended voice for success stories (Male): {test_voice}")
    
    # Generate pipeline config
    pipeline_config = config.generate_audio_pipeline_config()
    print("Audio pipeline configuration generated")
    
    # Test payload generation
    test_text = "每一天都是新的开始，相信自己的力量！"
    payload = config.get_gpt_sovits_payload(test_text, test_voice)
    print(f"Generated payload for test text: {json.dumps(payload, indent=2, ensure_ascii=False)}")