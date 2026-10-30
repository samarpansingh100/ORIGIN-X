"""
FastAPI Web Application for ORIGIN X.
Provides RESTful inference endpoints, health checks, error sanitization,
and serves the Apple Design-grounded web user interface.
"""

from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from originx.config import (
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_MIME_TYPES,
    API_PREFIX,
)
from originx.data.validator import ImageValidationError
from originx.web.service import InferenceService

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="ORIGIN X — Image Origin Estimator",
    description="Probabilistic estimation of human-created vs AI-generated imagery.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.get(f"{API_PREFIX}/health")
async def health_check() -> Dict[str, Any]:
    """Health status and model readiness check."""
    try:
        service = InferenceService.get_instance()
        return {
            "status": "healthy",
            "model_ready": True,
            "model_metadata": service.model_metadata,
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "model_ready": False,
            "error": str(exc),
        }


@app.get(f"{API_PREFIX}/model-info")
async def model_info() -> Dict[str, Any]:
    """Returns model architecture, temperature calibration, and operational limits."""
    service = InferenceService.get_instance()
    return {
        "model_name": service.model_metadata.get("model_name", "convnet_multiscale"),
        "temperature": service.model_metadata.get("temperature", 1.5),
        "calibration_method": "Post-Hoc Temperature Scaling",
        "supported_mime_types": list(SUPPORTED_MIME_TYPES),
        "max_file_size_mb": MAX_FILE_SIZE_BYTES / (1024 * 1024),
        "confidence_thresholds": {
            "ai_high_confidence": 0.75,
            "inconclusive_low": 0.42,
            "inconclusive_high": 0.58,
            "human_high_confidence": 0.25,
        },
    }


@app.post(f"{API_PREFIX}/analyze")
async def analyze_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Ingests and analyzes an uploaded image in volatile memory.
    Returns calibrated probability, confidence intervals, diagnostic signals, and plain-language explanation.
    """
    # Check filename extension or content type if provided
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    # Read binary payload with size limit guard
    data = await file.read(MAX_FILE_SIZE_BYTES + 1024)
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB."
        )

    try:
        service = InferenceService.get_instance()
        result = service.analyze_image_bytes(data)
        return result
    except ImageValidationError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(val_err)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while analyzing the image: {str(exc)}"
        )


# Serve single page web application
@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"message": "ORIGIN X API running. Frontend static files pending."})
    return FileResponse(index_file)

# Mount remaining static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
