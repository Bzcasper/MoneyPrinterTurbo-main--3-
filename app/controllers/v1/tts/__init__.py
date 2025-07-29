"""
TTS API Controllers Package

RESTful API endpoints for Text-to-Speech services including:
- Provider management
- Voice listing and selection
- Speech synthesis
- Batch processing
- Character-based voices
"""

from .providers import router as providers_router
from .synthesis import router as synthesis_router
from .characters import router as characters_router

__all__ = [
    "providers_router",
    "synthesis_router", 
    "characters_router"
]
