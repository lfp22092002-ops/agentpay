"""
Agent CRUD endpoints: balance, key rotation, export.
"""
import csv
import io
from decimal import Decimal

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from api.dependencies import get_agent_auth
from api.middleware import limiter
from api.models import (
    BalanceResponse, RotateKeyResponse, TransactionOut,
)
from core.wallet import get_daily_spent, get_agent_transactions, rotate_api_key
from models.schema import Agent

router = APIRouter(prefix="/v1", tags=["agents"])


@router.get("/balance", response_model=BalanceResponse)
@limiter.limit("60/minute")
async def get_balance(request: Request, auth: tuple = Depends(get_agent_auth)):
    agent, db = auth
    daily_spent = await get_daily_spent(db, agent)
    return BalanceResponse(
        agent_id=agent.id,
        agent_name=agent.name,
        balance_usd=float(agent.balance_usd),
        daily_limit_usd=float(agent.daily_limit_usd),
        daily_spent_usd=float(daily_spent),
        daily_remaining_usd=float(agent.daily_limit_usd - daily_spent),
        tx_limit_usd=float(agent.tx_limit_usd),
        is_active=agent.is_active,
    )


@router.get("/transactions", response_model=list[TransactionOut])
async def list_transactions(limit: int = 20, auth: tuple = Depends(get_agent_auth)):
    agent, db = auth
    txs = await get_agent_transactions(db, agent, limit=min(limit, 100))
    return [
        TransactionOut(
            id=tx.id,
            type=tx.tx_type.value,
            amount=float(tx.amount_usd),
            fee=float(tx.fee_usd),
            description=tx.description,
            status=tx.status.value,
            created_at=tx.created_at.isoformat(),
        )
        for tx in txs
    ]


@router.post("/agent/rotate-key", response_model=RotateKeyResponse)
@limiter.limit("5/minute")
async def api_rotate_key(request: Request, auth: tuple = Depends(get_agent_auth)):
    """Rotate the API key for this agent. Returns the new key ONCE. Old key stops working immediately."""
    agent, db = auth
    new_key = await rotate_api_key(db, agent)
    return RotateKeyResponse(
        success=True,
        new_api_key=new_key,
        key_prefix=new_key[:8] + "...",
    )


@router.get("/export")
@limiter.limit("5/minute")
async def export_csv(request: Request, auth: tuple = Depends(get_agent_auth)):
    agent, db = auth
    txs = await get_agent_transactions(db, agent, limit=1000)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "type", "amount", "fee", "description", "status", "id"])
    for tx in txs:
        writer.writerow([
            tx.created_at.isoformat(),
            tx.tx_type.value,
            float(tx.amount_usd),
            float(tx.fee_usd),
            tx.description or "",
            tx.status.value,
            tx.id,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=agentpay-{agent.name}-transactions.csv"},
    )
