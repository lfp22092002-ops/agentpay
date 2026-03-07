"""
Admin endpoints — revenue tracking, withdrawals.
"""
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware import limiter
from models.database import get_db
from models.schema import PlatformRevenue
from config.settings import API_SECRET

import jwt as pyjwt

router = APIRouter(prefix="/v1/admin", tags=["admin"])

ADMIN_TELEGRAM_ID = 5360481016
OWNER_WALLET = "0xD51B231F317260FB86b47A38F14eA29Cc81E0073"


def _verify_admin(request: Request):
    """Verify the request is from an admin user."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing auth")
    try:
        payload = pyjwt.decode(auth_header.split(" ")[1], API_SECRET, algorithms=["HS256"])
        if payload.get("telegram_id") != ADMIN_TELEGRAM_ID:
            raise HTTPException(403, "Admin only")
        return payload
    except Exception:
        raise HTTPException(401, "Invalid token")


@router.get("/revenue")
@limiter.limit("10/minute")
async def get_revenue(request: Request, db: AsyncSession = Depends(get_db)):
    """Platform revenue summary. Admin only."""
    _verify_admin(request)

    total = await db.execute(select(func.sum(PlatformRevenue.amount_usd)))
    total_usd = total.scalar() or Decimal("0")

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_result = await db.execute(
        select(func.sum(PlatformRevenue.amount_usd)).where(PlatformRevenue.created_at >= today)
    )
    today_usd = today_result.scalar() or Decimal("0")

    count_result = await db.execute(select(func.count(PlatformRevenue.id)))
    tx_count = count_result.scalar() or 0

    return {
        "total_revenue_usd": float(total_usd),
        "today_revenue_usd": float(today_usd),
        "total_fee_transactions": tx_count,
    }


@router.post("/withdraw")
@limiter.limit("3/minute")
async def withdraw_revenue(request: Request, db: AsyncSession = Depends(get_db)):
    """Withdraw accumulated platform fees as USDC to owner wallet. Admin only."""
    _verify_admin(request)

    total = await db.execute(select(func.sum(PlatformRevenue.amount_usd)))
    total_usd = total.scalar() or Decimal("0")

    if total_usd <= Decimal("0"):
        return {"success": False, "error": "No revenue to withdraw", "total_usd": 0}

    return {
        "success": True,
        "total_revenue_usd": float(total_usd),
        "withdraw_to": OWNER_WALLET,
        "note": "Revenue tracked in DB. On-chain USDC withdrawal available when agent wallets hold sufficient USDC balance.",
    }
