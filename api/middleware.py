"""
Middleware and rate limiting for the AgentPay API.
"""
import collections
import hashlib
import threading
import time
import time as _time
import logging

from fastapi import Request
from starlette.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger("agentpay.api")

# ═══════════════════════════════════════
# IP-BASED RATE LIMITER (slowapi)
# ═══════════════════════════════════════

limiter = Limiter(key_func=get_remote_address)


# ═══════════════════════════════════════
# PER-API-KEY RATE LIMITER (in-memory)
# ═══════════════════════════════════════

_api_key_requests: dict[str, collections.deque] = {}
_rate_lock = threading.Lock()
API_KEY_RATE_LIMIT = 60  # requests per minute
API_KEY_RATE_WINDOW = 60  # seconds


_last_gc = 0.0
_GC_INTERVAL = 300  # purge stale keys every 5 minutes


def check_api_key_rate_limit(api_key: str) -> bool:
    """Return True if allowed, False if rate-limited. Uses hash to avoid storing raw keys in memory."""
    global _last_gc
    now = time.time()
    cutoff = now - API_KEY_RATE_WINDOW
    key_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    with _rate_lock:
        # Periodic GC: remove keys with no recent requests
        if now - _last_gc > _GC_INTERVAL:
            stale = [k for k, dq in _api_key_requests.items() if not dq or dq[-1] < cutoff]
            for k in stale:
                del _api_key_requests[k]
            _last_gc = now

        dq = _api_key_requests.get(key_id)
        if dq is None:
            dq = collections.deque()
            _api_key_requests[key_id] = dq
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= API_KEY_RATE_LIMIT:
            return False
        dq.append(now)
        return True


# Keep old name for backward compatibility (tests reference it)
_check_api_key_rate_limit = check_api_key_rate_limit


async def log_requests_middleware(request: Request, call_next):
    """Log requests and add security headers."""
    start = _time.time()
    response = await call_next(request)
    duration = _time.time() - start
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    if "server" in response.headers:
        del response.headers["server"]
    # Skip noisy health check logs
    if request.url.path != "/v1/health":
        logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration:.3f}s)")
    return response


async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Slow down."},
    )


async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions — never leak stack traces to clients."""
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
