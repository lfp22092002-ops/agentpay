"""
Health and root/landing endpoints.
"""
import os

from fastapi import APIRouter, Request
from starlette.responses import FileResponse

from api.middleware import limiter
from api.models import HealthResponse

router = APIRouter(tags=["health"])

_project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@router.get("/v1/health", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse, include_in_schema=False)
@limiter.limit("30/minute")
async def health(request: Request):
    """Check API health status. Returns service name and version."""
    return HealthResponse(status="ok", service="agentpay", version="0.1.0")


@router.get("/")
async def root():
    """Serve landing page."""
    return FileResponse(os.path.join(_project_dir, "landing", "index.html"))
