"""
CharacterBox Service Package

Provides AI character voice synthesis and interaction capabilities.
Supports character personalities, emotions, and conversation generation.
"""

from .characterbox_service import CharacterBoxService, CharacterBoxError
from .character_models import CharacterInfo, CharacterRequest, CharacterResponse, ConversationRequest, ConversationResponse

__all__ = [
    "CharacterBoxService",
    "CharacterBoxError",
    "CharacterInfo",
    "CharacterRequest", 
    "CharacterResponse",
    "ConversationRequest",
    "ConversationResponse"
]
