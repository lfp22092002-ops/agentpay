"""
Shared dependencies for the AgentPay API.
"""
import hashlib
import time
import logging

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db
from models.schema import Agent
from core.wallet import get_agent_by_api_key
from api.middleware import check_api_key_rate_limit

logger = logging.getLogger("agentpay.api")


async def get_agent_auth(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> tuple[Agent, AsyncSession]:
    """Authenticate an agent by API key. Returns (agent, db_session)."""
    if not check_api_key_rate_limit(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded: 60 requests/minute per API key")
    agent = await get_agent_by_api_key(db, x_api_key)
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not agent.is_active:
        raise HTTPException(status_code=403, detail="Agent is deactivated")
    return agent, db


def get_miniapp_user_dep():
    """Return the get_miniapp_user dependency (imported at call time to avoid circular imports)."""
    from api.routes.miniapp import get_miniapp_user
    return get_miniapp_user
