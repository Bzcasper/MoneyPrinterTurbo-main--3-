"""Application configuration - root APIRouter.

Defines all FastAPI application endpoints.

Resources:
    1. https://fastapi.tiangolo.com/tutorial/bigger-applications

"""

# No replacement needed
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, APIRouter

from app.controllers.v1 import llm, video
from app.controllers import ping

root_api_router = APIRouter()

# Include health check endpoint
root_api_router.include_router(ping.router)

# v1 endpoints
root_api_router.include_router(video.router)
root_api_router.include_router(llm.router)

# Enhanced router configuration
def configure_cors(app: FastAPI):
    """Configure CORS middleware for the application."""
    # Get allowed origins from environment or config
    import os
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,  # Restrict to specific origins
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],  # Specific methods only
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],  # Specific headers only
        max_age=3600,  # Cache preflight responses
    )

# Export configured router
__all__ = ["root_api_router", "configure_cors"]
