import uvicorn
from loguru import logger
import os
import sys
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.router import root_api_router, configure_cors
from app.models.exception import HttpException
from app.middleware import RateLimitMiddleware
# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="MoneyPrinterTurbo API",
    description="Advanced video generation API with GPU acceleration",
    version="2.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
)

# Add security middleware
trusted_hosts = os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1,0.0.0.0").split(",")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

# Add rate limiting middleware
try:
    from app.config import config
    if config.app.get("enable_redis", False):
        import redis
        redis_client = redis.Redis.from_url(
            f"redis://:{config.app.get('redis_password', '')}@{config.app.get('redis_host', 'redis')}:{config.app.get('redis_port', 6379)}/{config.app.get('redis_db', 0)}"
        )
        app.add_middleware(RateLimitMiddleware, redis_client=redis_client)
    else:
        app.add_middleware(RateLimitMiddleware)
except Exception as e:
    logger.warning(f"Rate limiting setup failed: {e}, using memory-based rate limiting")
    app.add_middleware(RateLimitMiddleware)

# Configure CORS and include all API routers
configure_cors(app)

# Global exception handler
@app.exception_handler(HttpException)
async def http_exception_handler(request: Request, exc: HttpException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": exc.status_code,
            "message": exc.message,
            "data": exc.data,
            "path": str(request.url)
        }
    )

@app.exception_handler(HTTPException)
async def fastapi_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": exc.status_code,
            "message": exc.detail,
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": 500,
            "message": "Internal server error",
            "path": str(request.url)
        }
    )

app.include_router(root_api_router)

@app.get("/")
async def root():
    return {"message": "MoneyPrinterTurbo API is running"}

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    import psutil
    import time
    
    health_status = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        },
        "services": {}
    }
    
    # Check Redis connection if enabled
    try:
        from app.config import config
        if config.app.get("enable_redis", False):
            import redis
            redis_client = redis.Redis.from_url(
                f"redis://:{config.app.get('redis_password', '')}@{config.app.get('redis_host', 'redis')}:{config.app.get('redis_port', 6379)}/{config.app.get('redis_db', 0)}"
            )
            redis_client.ping()
            health_status["services"]["redis"] = "healthy"
        else:
            health_status["services"]["redis"] = "disabled"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check GPU availability
    try:
        from app.services.gpu_manager import get_gpu_manager
        gpu_manager = get_gpu_manager()
        gpu_count = len(gpu_manager.available_gpus)
        health_status["services"]["gpu"] = f"available: {gpu_count} GPUs"
    except Exception as e:
        health_status["services"]["gpu"] = f"unavailable: {str(e)}"
    
    return health_status

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    app.state.start_time = time.time()
    logger.info("MoneyPrinterTurbo API started successfully")
    
    # Initialize GPU resources
    try:
        from app.services.gpu_manager import initialize_gpu_resources
        gpu_manager = initialize_gpu_resources()
        logger.info(f"GPU resources initialized: {len(gpu_manager.available_gpus)} GPUs available")
    except Exception as e:
        logger.warning(f"GPU initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("MoneyPrinterTurbo API shutting down")
    
    # Stop GPU monitoring
    try:
        from app.services.gpu_manager import get_gpu_manager
        gpu_manager = get_gpu_manager()
        gpu_manager.stop_monitoring()
    except Exception as e:
        logger.warning(f"GPU cleanup failed: {e}")


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    
    logger.info(f"Starting MoneyPrinterTurbo API server on {host}:{port}")
    
    try:
        uvicorn.run(
            app="app.main:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
