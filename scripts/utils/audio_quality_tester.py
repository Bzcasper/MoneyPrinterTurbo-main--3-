#!/usr/bin/env python3
"""
Audio Quality Testing Suite for GPT-SoVITS Chinese Voice Synthesis
Validates pronunciation accuracy, emotional tone, and cultural appropriateness
"""

import asyncio
import os
import json
import wave
import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import requests
from datetime import datetime

@dataclass
class AudioQualityMetrics:
    """Audio quality assessment metrics"""
    pronunciation_score: float  # 0-100
    emotional_tone_score: float  # 0-100
    cultural_appropriateness: float  # 0-100
    audio_clarity: float  # 0-100
    youtube_compliance: bool
    overall_score: float  # 0-100

class ChineseAudioTester:
    """Chinese voice synthesis quality tester"""
    
    def __init__(self, base_path: str = "/home/bobby/Documents/MoneyPrinterTurbo"):
        self.base_path = Path(base_path)
        self.test_output_dir = self.base_path / "storage" / "audio_tests"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chinese pronunciation patterns
        self.chinese_phonetic_patterns = {
            "tones": ["阴平", "阳平", "上声", "去声", "轻声"],
            "initials": ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s"],
            "finals": ["a", "o", "e", "i", "u", "ü", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong"]
        }
        
        # Motivational content evaluation criteria
        self.motivational_criteria = {
            "energy_level": {"low": 0-30, "medium": 31-70, "high": 71-100},
            "emotional_warmth": {"cold": 0-25, "neutral": 26-60, "warm": 61-100},
            "authority": {"weak": 0-40, "moderate": 41-75, "strong": 76-100},
            "clarity": {"poor": 0-30, "good": 31-80, "excellent": 81-100}
        }
        
        # Test phrases with expected emotional characteristics
        self.test_phrases = {
            "high_energy": {
                "text": "突破自己，超越极限！成功就在前方等着你！",
                "expected_energy": "high",
                "expected_warmth": "warm",
                "expected_authority": "strong"
            },
            "gentle_encouragement": {
                "text": "每一步都算数，慢慢来，你一定可以做到的。",
                "expected_energy": "medium",
                "expected_warmth": "warm", 
                "expected_authority": "moderate"
            },
            "business_authority": {
                "text": "专注目标，保持执行力，这是成功的关键要素。",
                "expected_energy": "medium",
                "expected_warmth": "neutral",
                "expected_authority": "strong"
            },
            "inspirational_story": {
                "text": "从零开始并不可怕，可怕的是永远不开始行动。",
                "expected_energy": "high",
                "expected_warmth": "warm",
                "expected_authority": "strong"
            }
        }

    async def test_gpt_sovits_voice(self, 
                                    text: str, 
                                    voice_model: str,
                                    api_config: Dict) -> Tuple[str, AudioQualityMetrics]:
        """Test GPT-SoVITS voice synthesis and return audio file path with quality metrics"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.test_output_dir / f"test_{voice_model}_{timestamp}.wav"
        
        # Prepare GPT-SoVITS API request
        payload = {
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": None,
            "aux_ref_audio_paths": [],
            "prompt_lang": "zh", 
            "prompt_text": "",
            "top_k": 5,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut5",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": 1.0,
            "fragment_interval": 0.3,
            "seed": -1,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
            "repetition_penalty": 1.35
        }
        
        try:
            # Call GPT-SoVITS API
            response = requests.post(
                f"{api_config['base_url']}/tts",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=api_config.get('timeout', 60)
            )
            
            if response.status_code == 200:
                # Save audio file
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                # Analyze audio quality
                metrics = await self.analyze_audio_quality(output_file, text, voice_model)
                return str(output_file), metrics
            else:
                raise Exception(f"API request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error testing voice {voice_model}: {e}")
            # Return default failed metrics
            return "", AudioQualityMetrics(0, 0, 0, 0, False, 0)

    async def analyze_audio_quality(self, 
                                   audio_file_path: str, 
                                   original_text: str,
                                   voice_model: str) -> AudioQualityMetrics:
        """Analyze audio file for quality metrics"""
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Basic audio analysis
            audio_clarity = self.analyze_audio_clarity(y, sr)
            youtube_compliance = self.check_youtube_compliance(audio_file_path, duration)
            
            # Chinese pronunciation analysis (simplified)
            pronunciation_score = self.analyze_pronunciation(y, sr, original_text)
            
            # Emotional tone analysis
            emotional_tone_score = self.analyze_emotional_tone(y, sr, voice_model)
            
            # Cultural appropriateness (basic check)
            cultural_appropriateness = self.check_cultural_appropriateness(original_text, voice_model)
            
            # Calculate overall score
            overall_score = (
                pronunciation_score * 0.3 +
                emotional_tone_score * 0.25 +
                cultural_appropriateness * 0.2 +
                audio_clarity * 0.25
            )
            
            return AudioQualityMetrics(
                pronunciation_score=pronunciation_score,
                emotional_tone_score=emotional_tone_score,
                cultural_appropriateness=cultural_appropriateness,
                audio_clarity=audio_clarity,
                youtube_compliance=youtube_compliance,
                overall_score=overall_score
            )
            
        except Exception as e:
            print(f"Error analyzing audio quality: {e}")
            return AudioQualityMetrics(0, 0, 0, 0, False, 0)

    def analyze_audio_clarity(self, y: np.ndarray, sr: int) -> float:
        """Analyze audio clarity based on spectral features"""
        
        # Calculate spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Calculate spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Calculate zero crossing rate (voice activity)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Calculate MFCC for voice characteristics
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Simple clarity scoring based on spectral features
        # Higher spectral centroid generally indicates clearer speech
        clarity_score = min(100, (np.mean(spectral_centroids) / 1000) * 20)
        
        # Adjust based on voice activity (higher ZCR in speech ranges)
        voice_activity = np.mean(zcr)
        if 0.05 < voice_activity < 0.2:  # Typical speech ZCR range
            clarity_score *= 1.2
        
        return min(100, max(0, clarity_score))

    def analyze_pronunciation(self, y: np.ndarray, sr: int, text: str) -> float:
        """Analyze pronunciation accuracy for Chinese text"""
        
        # Basic pronunciation analysis using audio features
        # In a real implementation, this would use Chinese phoneme recognition
        
        # Calculate formant frequencies (approximate)
        formants = self.estimate_formants(y, sr)
        
        # Check if formants are within expected ranges for Chinese speech
        f1_score = 80 if 200 <= formants['f1'] <= 1000 else 60
        f2_score = 80 if 800 <= formants['f2'] <= 3000 else 60
        
        # Check for tonal patterns (simplified)
        pitch = librosa.feature.chroma_stft(y=y, sr=sr)
        pitch_variation = np.std(pitch) * 100
        
        # Chinese is tonal, so we expect moderate pitch variation
        tonal_score = 90 if 0.5 <= pitch_variation <= 2.0 else 70
        
        # Combine scores
        pronunciation_score = (f1_score * 0.3 + f2_score * 0.3 + tonal_score * 0.4)
        
        return min(100, max(0, pronunciation_score))

    def estimate_formants(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Estimate formant frequencies"""
        
        # Simple formant estimation using LPC
        # This is a simplified approach
        
        # Calculate spectral centroid as proxy for F2
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Estimate F1 and F2 based on spectral features
        f1_estimate = spectral_centroid * 0.3  # Very rough approximation
        f2_estimate = spectral_centroid * 1.2
        
        return {
            "f1": f1_estimate,
            "f2": f2_estimate
        }

    def analyze_emotional_tone(self, y: np.ndarray, sr: int, voice_model: str) -> float:
        """Analyze emotional appropriateness for motivational content"""
        
        # Calculate energy/intensity
        rms = librosa.feature.rms(y=y)[0]
        energy_level = np.mean(rms) * 1000
        
        # Calculate pitch variation for emotional expression
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_range = np.ptp(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
        
        # Calculate spectral contrast for voice quality
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast)
        
        # Score based on expected characteristics for motivational content
        energy_score = min(100, energy_level * 50)  # Higher energy is better
        pitch_score = min(100, pitch_range / 10)    # Moderate pitch variation
        contrast_score = min(100, contrast_mean * 30)  # Good spectral contrast
        
        # Weight scores based on voice model type
        if "energetic" in voice_model.lower():
            emotional_score = energy_score * 0.5 + pitch_score * 0.3 + contrast_score * 0.2
        elif "warm" in voice_model.lower() or "gentle" in voice_model.lower():
            emotional_score = energy_score * 0.2 + pitch_score * 0.4 + contrast_score * 0.4
        elif "professional" in voice_model.lower():
            emotional_score = energy_score * 0.3 + pitch_score * 0.2 + contrast_score * 0.5
        else:
            emotional_score = energy_score * 0.33 + pitch_score * 0.33 + contrast_score * 0.34
        
        return min(100, max(0, emotional_score))

    def check_cultural_appropriateness(self, text: str, voice_model: str) -> float:
        """Check cultural appropriateness for Chinese motivational content"""
        
        score = 100  # Start with perfect score
        
        # Check for appropriate tone/formality
        formal_indicators = ["您", "请", "谢谢", "专业", "成功"]
        casual_indicators = ["你", "咋", "啥", "搞"]
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in text)
        casual_count = sum(1 for indicator in casual_indicators if indicator in text)
        
        # Motivational content should be more formal/respectful
        if casual_count > formal_count:
            score -= 20
        
        # Check for positive language
        positive_words = ["成功", "努力", "坚持", "梦想", "希望", "相信", "突破", "成长"]
        positive_count = sum(1 for word in positive_words if word in text)
        
        if positive_count >= 2:
            score += 10
        elif positive_count == 0:
            score -= 30
        
        # Check voice model appropriateness
        if "professional" in voice_model and formal_count > 0:
            score += 10
        elif "energetic" in voice_model and "突破" in text:
            score += 10
        elif "warm" in voice_model and any(word in text for word in ["相信", "希望", "可以"]):
            score += 10
        
        return min(100, max(0, score))

    def check_youtube_compliance(self, audio_file_path: str, duration: float) -> bool:
        """Check if audio meets YouTube Shorts specifications"""
        
        try:
            # Check duration (max 180 seconds for Shorts)
            if duration > 180:
                return False
            
            # Check audio file properties
            with wave.open(audio_file_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
            
            # YouTube prefers 48kHz, but 32kHz is acceptable for TTS
            sample_rate_ok = sample_rate in [32000, 44100, 48000]
            
            # Check if audio can be converted to stereo AAC
            format_ok = sample_width >= 2  # At least 16-bit
            
            return sample_rate_ok and format_ok
            
        except Exception:
            return False

    async def run_comprehensive_test(self, api_config: Dict) -> Dict:
        """Run comprehensive audio quality tests for all voice models"""
        
        print("🎙️ Starting Comprehensive Chinese Voice Quality Tests...")
        
        # Import voice models from config
        from audio_specialist_config import GPTSoVITSConfig
        config = GPTSoVITSConfig()
        
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "api_config": api_config,
            "voice_model_results": {},
            "summary": {}
        }
        
        for model_key, voice_model in config.voice_models.items():
            print(f"\n🧪 Testing voice model: {model_key} ({voice_model.description})")
            
            model_results = {
                "model_info": {
                    "key": model_key,
                    "description": voice_model.description,
                    "gender": voice_model.gender,
                    "use_case": voice_model.use_case
                },
                "phrase_tests": {}
            }
            
            # Test each phrase type
            for phrase_key, phrase_data in self.test_phrases.items():
                print(f"  📝 Testing phrase: {phrase_key}")
                
                audio_file, metrics = await self.test_gpt_sovits_voice(
                    phrase_data["text"], 
                    model_key,
                    api_config
                )
                
                model_results["phrase_tests"][phrase_key] = {
                    "text": phrase_data["text"],
                    "audio_file": audio_file,
                    "metrics": {
                        "pronunciation_score": metrics.pronunciation_score,
                        "emotional_tone_score": metrics.emotional_tone_score,
                        "cultural_appropriateness": metrics.cultural_appropriateness,
                        "audio_clarity": metrics.audio_clarity,
                        "youtube_compliance": metrics.youtube_compliance,
                        "overall_score": metrics.overall_score
                    },
                    "expected_characteristics": {
                        "energy": phrase_data["expected_energy"],
                        "warmth": phrase_data["expected_warmth"],
                        "authority": phrase_data["expected_authority"]
                    }
                }
                
                print(f"    ✅ Overall Score: {metrics.overall_score:.1f}/100")
            
            # Calculate average scores for this voice model
            all_scores = [test["metrics"]["overall_score"] 
                         for test in model_results["phrase_tests"].values()]
            model_results["average_score"] = np.mean(all_scores) if all_scores else 0
            
            results["voice_model_results"][model_key] = model_results
        
        # Generate summary
        results["summary"] = self.generate_test_summary(results["voice_model_results"])
        
        # Save results
        results_file = self.test_output_dir / f"audio_quality_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Test results saved to: {results_file}")
        
        return results

    def generate_test_summary(self, voice_model_results: Dict) -> Dict:
        """Generate summary of test results"""
        
        summary = {
            "best_overall_voice": None,
            "best_by_category": {},
            "recommendations": []
        }
        
        # Find best overall voice
        best_score = 0
        best_voice = None
        
        scores_by_category = {
            "pronunciation": {},
            "emotional_tone": {},
            "cultural_appropriateness": {},
            "audio_clarity": {}
        }
        
        for voice_key, results in voice_model_results.items():
            avg_score = results["average_score"]
            if avg_score > best_score:
                best_score = avg_score
                best_voice = voice_key
            
            # Collect category scores
            for phrase_key, phrase_results in results["phrase_tests"].items():
                metrics = phrase_results["metrics"]
                for category in scores_by_category:
                    if voice_key not in scores_by_category[category]:
                        scores_by_category[category][voice_key] = []
                    scores_by_category[category][voice_key].append(
                        metrics[f"{category.replace('_', '_')}_score"]
                    )
        
        summary["best_overall_voice"] = {
            "voice": best_voice,
            "score": best_score
        }
        
        # Find best voice for each category
        for category, voice_scores in scores_by_category.items():
            avg_scores = {voice: np.mean(scores) for voice, scores in voice_scores.items()}
            best_voice_category = max(avg_scores, key=avg_scores.get)
            summary["best_by_category"][category] = {
                "voice": best_voice_category,
                "score": avg_scores[best_voice_category]
            }
        
        # Generate recommendations
        summary["recommendations"] = [
            f"Best overall voice model: {summary['best_overall_voice']['voice']} (Score: {summary['best_overall_voice']['score']:.1f})",
            f"Best for pronunciation: {summary['best_by_category']['pronunciation']['voice']}",
            f"Best for emotional tone: {summary['best_by_category']['emotional_tone']['voice']}",
            f"Best for cultural appropriateness: {summary['best_by_category']['cultural_appropriateness']['voice']}"
        ]
        
        return summary

async def main():
    """Main testing function"""
    tester = ChineseAudioTester()
    
    # API configuration (adjust as needed)
    api_config = {
        "base_url": "http://localhost:9880",
        "timeout": 60
    }
    
    # Run comprehensive tests
    results = await tester.run_comprehensive_test(api_config)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 AUDIO QUALITY TEST SUMMARY")
    print("="*60)
    
    summary = results["summary"]
    for recommendation in summary["recommendations"]:
        print(f"✨ {recommendation}")
    
    print("\n📈 Detailed results saved in storage/audio_tests/ directory")

if __name__ == "__main__":
    asyncio.run(main())