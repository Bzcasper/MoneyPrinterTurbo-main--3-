"""
CharacterBox character generation service

This service provides AI-generated character avatars and animations
for video content using various character generation APIs.
"""

import os
import base64
import requests
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
from loguru import logger

from app.config import config


@dataclass
class CharacterStyle:
    """Character style configuration"""
    style_id: str
    name: str
    description: str
    category: str
    thumbnail_url: Optional[str] = None


@dataclass
class CharacterConfig:
    """Character generation configuration"""
    style: str = "realistic"
    gender: str = "female"
    age_group: str = "adult"
    ethnicity: str = "mixed"
    clothing: str = "casual"
    background: str = "neutral"
    pose: str = "standing"
    expression: str = "neutral"
    quality: str = "high"


class CharacterBoxService:
    """Service for generating AI characters and avatars"""
    
    def __init__(self):
        self.api_key = config.characterbox.get("api_key", "")
        self.base_url = config.characterbox.get("base_url", "https://api.characterbox.ai/v1")
        self.timeout = config.characterbox.get("timeout", 30)
        
    def get_available_styles(self) -> List[CharacterStyle]:
        """Get list of available character styles"""
        
        # Default styles when API is not available
        default_styles = [
            CharacterStyle(
                style_id="realistic",
                name="Realistic",
                description="Photorealistic human characters",
                category="realistic"
            ),
            CharacterStyle(
                style_id="anime", 
                name="Anime",
                description="Japanese anime-style characters",
                category="animated"
            ),
            CharacterStyle(
                style_id="cartoon",
                name="Cartoon",
                description="Western cartoon-style characters", 
                category="animated"
            ),
            CharacterStyle(
                style_id="3d_render",
                name="3D Render",
                description="High-quality 3D rendered characters",
                category="3d"
            ),
            CharacterStyle(
                style_id="pixel_art",
                name="Pixel Art", 
                description="Retro pixel art characters",
                category="pixel"
            ),
            CharacterStyle(
                style_id="sketch",
                name="Sketch",
                description="Hand-drawn sketch style",
                category="artistic"
            ),
            CharacterStyle(
                style_id="watercolor",
                name="Watercolor",
                description="Watercolor painting style",
                category="artistic"
            ),
            CharacterStyle(
                style_id="cyberpunk",
                name="Cyberpunk",
                description="Futuristic cyberpunk aesthetic",
                category="themed"
            ),
            CharacterStyle(
                style_id="fantasy",
                name="Fantasy",
                description="Fantasy RPG character style",
                category="themed"
            ),
            CharacterStyle(
                style_id="corporate",
                name="Corporate",
                description="Professional business style",
                category="themed"
            )
        ]
        
        if not self.api_key:
            logger.warning("CharacterBox API key not configured, using default styles")
            return default_styles
            
        try:
            response = requests.get(
                f"{self.base_url}/styles",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                styles = []
                for style_data in data.get("styles", []):
                    style = CharacterStyle(
                        style_id=style_data.get("id"),
                        name=style_data.get("name"),
                        description=style_data.get("description", ""),
                        category=style_data.get("category", "other"),
                        thumbnail_url=style_data.get("thumbnail_url")
                    )
                    styles.append(style)
                return styles
            else:
                logger.warning(f"Failed to fetch styles from API: {response.status_code}")
                return default_styles
                
        except Exception as e:
            logger.error(f"Error fetching character styles: {str(e)}")
            return default_styles
    
    def generate_character(self, 
                         character_config: CharacterConfig,
                         prompt: Optional[str] = None,
                         negative_prompt: Optional[str] = None) -> Optional[Dict]:
        """
        Generate a character image based on configuration
        
        Args:
            character_config: Character configuration object
            prompt: Additional text prompt for character generation
            negative_prompt: Negative prompt to avoid certain features
            
        Returns:
            Dictionary with character data or None if failed
        """
        
        if not self.api_key:
            logger.error("CharacterBox API key not configured")
            return self._generate_placeholder_character(character_config)
        
        try:
            # Build prompt from config
            config_prompt = self._build_prompt_from_config(character_config)
            
            # Combine with custom prompt if provided
            final_prompt = config_prompt
            if prompt:
                final_prompt += f", {prompt}"
            
            payload = {
                "style": character_config.style,
                "prompt": final_prompt,
                "negative_prompt": negative_prompt or "blurry, low quality, distorted, ugly",
                "width": 512,
                "height": 768,
                "quality": character_config.quality,
                "seed": -1  # Random seed
            }
            
            logger.info(f"Generating character with style: {character_config.style}")
            
            response = requests.post(
                f"{self.base_url}/generate", 
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                character_data = {
                    "id": result.get("id"),
                    "image_url": result.get("image_url"),
                    "image_data": result.get("image_data"),  # base64 encoded
                    "prompt": final_prompt,
                    "style": character_config.style,
                    "config": character_config.__dict__,
                    "metadata": result.get("metadata", {})
                }
                
                logger.success(f"Character generated successfully: {character_data.get('id')}")
                return character_data
            else:
                logger.error(f"Character generation failed: {response.status_code} - {response.text}")
                return self._generate_placeholder_character(character_config)
                
        except Exception as e:
            logger.error(f"Error generating character: {str(e)}")
            return self._generate_placeholder_character(character_config)
    
    def save_character_image(self, character_data: Dict, output_path: str) -> bool:
        """
        Save character image to file
        
        Args:
            character_data: Character data from generate_character
            output_path: Path to save the image file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if character_data.get("image_data"):
                # Save from base64 data
                image_data = base64.b64decode(character_data["image_data"])
                with open(output_path, "wb") as f:
                    f.write(image_data)
                logger.info(f"Character image saved to: {output_path}")
                return True
                
            elif character_data.get("image_url"):
                # Download from URL
                response = requests.get(character_data["image_url"], timeout=30)
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"Character image downloaded and saved to: {output_path}")
                    return True
                else:
                    logger.error(f"Failed to download character image: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error saving character image: {str(e)}")
            
        return False
    
    def _build_prompt_from_config(self, config: CharacterConfig) -> str:
        """Build text prompt from character configuration"""
        prompt_parts = []
        
        # Style and quality
        if config.style == "realistic":
            prompt_parts.append("photorealistic portrait")
        elif config.style == "anime":
            prompt_parts.append("anime style character")
        elif config.style == "cartoon":
            prompt_parts.append("cartoon character")
        elif config.style == "3d_render":
            prompt_parts.append("high quality 3D rendered character")
        else:
            prompt_parts.append(f"{config.style} style character")
        
        # Demographics
        if config.gender != "mixed":
            prompt_parts.append(f"{config.gender}")
        
        if config.age_group != "mixed":
            if config.age_group == "young":
                prompt_parts.append("young adult")
            elif config.age_group == "adult":
                prompt_parts.append("adult")
            elif config.age_group == "elderly":
                prompt_parts.append("elderly person")
        
        if config.ethnicity != "mixed":
            prompt_parts.append(f"{config.ethnicity}")
        
        # Appearance
        if config.clothing != "none":
            prompt_parts.append(f"wearing {config.clothing} clothing")
        
        if config.pose != "neutral":
            prompt_parts.append(f"{config.pose} pose")
        
        if config.expression != "neutral":
            prompt_parts.append(f"{config.expression} expression")
        
        # Background
        if config.background != "none":
            if config.background == "neutral":
                prompt_parts.append("neutral background")
            else:
                prompt_parts.append(f"{config.background} background")
        
        # Quality modifiers
        if config.quality == "high":
            prompt_parts.append("high quality, detailed, sharp focus")
        elif config.quality == "ultra":
            prompt_parts.append("ultra high quality, extremely detailed, professional")
        
        return ", ".join(prompt_parts)
    
    def _generate_placeholder_character(self, config: CharacterConfig) -> Dict:
        """Generate placeholder character data when API is unavailable"""
        
        # Create a simple colored placeholder
        placeholder_data = {
            "id": f"placeholder_{hash(str(config.__dict__)) % 10000}",
            "image_url": None,
            "image_data": None,
            "prompt": self._build_prompt_from_config(config),
            "style": config.style, 
            "config": config.__dict__,
            "metadata": {
                "is_placeholder": True,
                "message": "CharacterBox API not available - using placeholder"
            }
        }
        
        logger.warning("Generated placeholder character data")
        return placeholder_data


def get_character_styles() -> List[CharacterStyle]:
    """Get available character styles"""
    service = CharacterBoxService()
    return service.get_available_styles()


def generate_character(config: CharacterConfig, 
                      prompt: Optional[str] = None,
                      negative_prompt: Optional[str] = None) -> Optional[Dict]:
    """Generate character with given configuration"""
    service = CharacterBoxService()
    return service.generate_character(config, prompt, negative_prompt)


def save_character_image(character_data: Dict, output_path: str) -> bool:
    """Save character image to file"""
    service = CharacterBoxService()
    return service.save_character_image(character_data, output_path)