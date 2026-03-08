"""
AgentPay API — Payment infrastructure for AI agents.

This is the main entry point. All endpoint logic lives in api/routes/.
"""
import os
import logging

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from models.database import init_db
from config.settings import API_SECRET
from api.middleware import (
    limiter, log_requests_middleware,
    rate_limit_handler, global_exception_handler,
)

# Re-export for backward compatibility (tests import from api.main)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("agentpay.api")

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════
# BACKWARD COMPAT — tests import these from api.main
# ═══════════════════════════════════════

# Trust score calculation (tests reference it)

# Miniapp auth (tests/routes reference it)
MINIAPP_JWT_SECRET = API_SECRET
MINIAPP_JWT_ALGO = "HS256"
MINIAPP_JWT_EXPIRY_HOURS = 24
ADMIN_TELEGRAM_ID = 5360481016


def _validate_telegram_init_data(init_data, bot_token):
    from api.routes.miniapp import _validate_telegram_init_data as _impl
    return _impl(init_data, bot_token)


def _create_miniapp_jwt(user_id, telegram_user):
    from api.routes.miniapp import _create_miniapp_jwt as _impl
    return _impl(user_id, telegram_user)


def _decode_miniapp_jwt(token):
    from api.routes.miniapp import _decode_miniapp_jwt as _impl
    return _impl(token)


async def get_miniapp_user(authorization: str = Header(..., alias="Authorization")):
    from api.routes.miniapp import get_miniapp_user as _impl
    return await _impl(authorization)


# ═══════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("✅ API database ready")
    from core.webhooks import load_webhooks_from_db
    await load_webhooks_from_db()
    yield

app = FastAPI(
    title="AgentPay API",
    description="Payment API for AI agents. Let your bot spend money.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
ALLOWED_ORIGINS = [
    "https://leofundmybot.dev",
    "https://web.telegram.org",
    "https://t.me",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.telegram\.org",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["X-API-Key", "Authorization", "Content-Type", "Accept"],
)

app.state.limiter = limiter

# Register middleware and exception handlers
app.middleware("http")(log_requests_middleware)

from slowapi.errors import RateLimitExceeded
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.add_exception_handler(Exception, global_exception_handler)


# ═══════════════════════════════════════
# REGISTER ROUTERS
# ═══════════════════════════════════════

from api.routes.health import router as health_router
from api.routes.agents import router as agents_router
from api.routes.wallets import router as wallets_router
from api.routes.identity import router as identity_router
from api.routes.admin import router as admin_router
from api.routes.miniapp import router as miniapp_router

app.include_router(health_router)
app.include_router(agents_router)
app.include_router(wallets_router)
app.include_router(identity_router)
app.include_router(admin_router)
app.include_router(miniapp_router)


# ═══════════════════════════════════════
# STATIC FILE SERVING
# ═══════════════════════════════════════

app.mount("/app", StaticFiles(directory=os.path.join(_project_dir, "miniapp"), html=True), name="miniapp")
app.mount("/docs-site", StaticFiles(directory=os.path.join(_project_dir, "landing", "docs"), html=True), name="docs-site")
app.mount("/landing", StaticFiles(directory=os.path.join(_project_dir, "landing"), html=True), name="landing")
