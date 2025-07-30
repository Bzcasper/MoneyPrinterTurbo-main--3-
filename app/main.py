import uvicorn
from loguru import logger
import os
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.router import root_api_router, configure_cors
from app.models.exception import HttpException
from app.middleware import RateLimitMiddleware
from app.middleware.supabase_middleware import SupabaseMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    app.state.start_time = time.time()
    logger.info("MoneyPrinterTurbo API starting up...")

    # Run database migrations
    try:
        from app.database.migrations import check_database_status, migrate_database
        from app.database.connection import get_supabase_connection

        # Patch httpx.Client to handle 'proxy' keyword argument
        import httpx
        def patched_init(self, *args, **kwargs):
            if 'proxy' in kwargs:
                proxy = kwargs.pop('proxy')
                if proxy is not None:
                    kwargs['proxies'] = proxy
            self.__original_init__(*args, **kwargs)
        if not hasattr(httpx.Client, '__original_init__'):
            httpx.Client.__original_init__ = httpx.Client.__init__
            httpx.Client.__init__ = patched_init

        # Ensure connection is established before checking status
        db_connection = get_supabase_connection()
        if not db_connection.is_connected:
            await db_connection.connect(use_service_key=True)

        status = await check_database_status()
        if status.get('migration_needed', False):
            logger.info("Database migration needed. Running migrations...")
            migration_results = await migrate_database(force=False)
            if migration_results.get('success', False):
                logger.info("Database migration completed successfully.")
            else:
                logger.error(f"Database migration failed: {migration_results.get('errors')}")
        else:
            logger.info("Database schema is up to date.")
    except Exception as e:
        logger.error(f"Database migration check failed: {e}")

    # Initialize GPU resources
    try:
        from app.services.gpu_manager import initialize_gpu_resources
        gpu_manager = initialize_gpu_resources()
        logger.info(f"GPU resources initialized: {len(gpu_manager.available_gpus)} GPUs available")
    except Exception as e:
        logger.warning(f"GPU initialization failed: {e}")
    
    # Initialize MCP server
    try:
        from app.controllers.v1.mcp import initialize_mcp_server
        await initialize_mcp_server()
        logger.info("MCP server initialized successfully")
    except Exception as e:
        logger.warning(f"MCP server initialization failed: {e}")

    logger.info("MoneyPrinterTurbo API started successfully")
    
    yield  # Application is running
    
    # Shutdown
    logger.info("MoneyPrinterTurbo API shutting down")
    
    # Shutdown MCP server
    try:
        from app.controllers.v1.mcp import shutdown_mcp_server
        await shutdown_mcp_server()
        logger.info("MCP server shutdown complete")
    except Exception as e:
        logger.warning(f"MCP server shutdown failed: {e}")
    
    # Stop GPU monitoring
    try:
        from app.services.gpu_manager import get_gpu_manager
        gpu_manager = get_gpu_manager()
        gpu_manager.stop_monitoring()
    except Exception as e:
        logger.warning(f"GPU cleanup failed: {e}")


# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="MoneyPrinterTurbo API",
    description="Advanced video generation API with GPU acceleration",
    version="2.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
    lifespan=lifespan,
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

# Add Supabase middleware for database connection and authentication
try:
    # Enable authentication and logging for Supabase middleware
    enable_auth = os.getenv("SUPABASE_ENABLE_AUTH", "true").lower() == "true"
    enable_logging = os.getenv("SUPABASE_ENABLE_LOGGING", "true").lower() == "true"
    
    app.add_middleware(
        SupabaseMiddleware,
        enable_auth=enable_auth,
        enable_logging=enable_logging
    )
    logger.info("Supabase middleware added successfully")
except Exception as e:
    logger.warning(f"Supabase middleware setup failed: {e}")

# Configure CORS and include all API routers
configure_cors(app)

# Mount static files
try:
    from fastapi.staticfiles import StaticFiles
    import os
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info("Static files mounted successfully")
except Exception as e:
    logger.warning(f"Static files mounting failed: {e}")

# Add favicon route
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import FileResponse
    import os
    favicon_path = os.path.join(os.path.dirname(__file__), "static", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    else:
        raise HTTPException(status_code=404, detail="Favicon not found")

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