"""
Health and root/landing endpoints.
"""
import os
import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import FileResponse

from api.middleware import limiter
from api.models import HealthResponse
from models.database import get_db

router = APIRouter(tags=["health"])

_project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_start_time = time.time()


@router.get("/v1/health", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse, include_in_schema=False)
@limiter.limit("30/minute")
async def health(request: Request):
    """Check API health status. Returns service name and version."""
    return HealthResponse(status="ok", service="agentpay", version="0.1.0")


@router.get("/v1/health/detailed")
@limiter.limit("10/minute")
async def health_detailed(request: Request, db: AsyncSession = Depends(get_db)):
    """Detailed health check including database connectivity and uptime."""
    db_status = "ok"
    db_latency_ms = None
    try:
        t0 = time.monotonic()
        await db.execute(text("SELECT 1"))
        db_latency_ms = round((time.monotonic() - t0) * 1000, 2)
    except Exception as e:
        db_status = f"error: {str(e)[:100]}"

    uptime_seconds = int(time.time() - _start_time)

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "service": "agentpay",
        "version": "0.1.0",
        "uptime_seconds": uptime_seconds,
        "checks": {
            "database": {
                "status": db_status,
                "latency_ms": db_latency_ms,
            }
        },
    }


@router.get("/")
async def root():
    """Serve landing page."""
    return FileResponse(os.path.join(_project_dir, "landing", "index.html"))
