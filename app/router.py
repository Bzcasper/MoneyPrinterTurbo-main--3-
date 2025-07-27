"""Application configuration - root APIRouter.

Defines all FastAPI application endpoints.

Resources:
    1. https://fastapi.tiangolo.com/tutorial/bigger-applications

"""

from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Export configured router
__all__ = ["root_api_router", "configure_cors"]
