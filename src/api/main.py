"""
Main FastAPI application for GMF Time Series Forecasting system.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
import time
import logging
from typing import Dict, Any
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GMF Time Series Forecasting API",
    description="Advanced time series forecasting and portfolio optimization API",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Request timing middleware


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    inprogress_gauge.inc()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        request_latency_histogram.observe(process_time)
        request_counter.labels(
            method=request.method,
            path=request.url.path,
            status_code=str(response.status_code)
        ).inc()
        return response
    finally:
        inprogress_gauge.dec()

# Global exception handler


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "timestamp": time.time()
        }
    )

# Prometheus metrics
request_counter = Counter(
    "gmf_request_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"]
)

request_latency_histogram = Histogram(
    "gmf_request_latency_seconds",
    "Latency of HTTP requests in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)

inprogress_gauge = Gauge(
    "gmf_requests_in_progress",
    "Number of HTTP requests in progress"
)

# Health check endpoint


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

# Root endpoint


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "message": "GMF Time Series Forecasting API",
        "version": "3.0.0",
        "status": "running",
        "documentation": "/docs",
        "health_check": "/health"
    }


# Metrics endpoint
@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# API info endpoint


@app.get("/api/info")
async def api_info() -> Dict[str, Any]:
    """Get detailed API information."""
    return {
        "name": "GMF Time Series Forecasting API",
        "version": "3.0.0",
        "description": "Advanced financial forecasting and portfolio optimization",
        "features": [
            "Time series forecasting (LSTM, ARIMA)",
            "Portfolio optimization",
            "Risk analysis",
            "SHAP model explainability",
            "Real-time data streaming",
            "Advanced analytics",
            "Automated reporting"
        ],
        "endpoints": {
            "forecasting": "/api/v1/forecasting",
            "portfolio": "/api/v1/portfolio",
            "analytics": "/api/v1/analytics",
            "reports": "/api/v1/reports"
        }
    }

if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")

    logger.info(f"Starting GMF API server on {host}:{port}")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=True
    )
