import uvicorn
from loguru import logger
import os
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Create FastAPI app
app = FastAPI(title="MoneyPrinterTurbo API")

@app.get("/")
async def root():
    return {"message": "MoneyPrinterTurbo API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/docs")
async def docs():
    return {"message": "API documentation available at /docs"}

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
