"""
Telegram Mini App endpoints — auth, agent management, dashboard, analytics.
"""
import hashlib
import hmac
import json
import time
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from urllib.parse import parse_qs

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from sqlalchemy import select, func, cast, Date
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import JSONResponse

from api.middleware import limiter
from api.models import (
    TelegramAuthRequest, TelegramAuthResponse,
    AgentSettingsUpdate,
    AgentIdentityOut, TrustScoreBreakdown,
    DashboardResponse, AgentAnalyticsResponse,
)
from models.database import get_db
from models.schema import (
    Agent, User, Transaction, TransactionType, TransactionStatus,
    AgentIdentity, PlatformRevenue,
)
from core.wallet import get_daily_spent
from providers.local_wallet import (
    get_wallet_address, get_wallet_balance,
    CHAIN_CONFIGS, SUPPORTED_EVM_CHAINS,
)
from providers.solana_wallet import (
    create_solana_wallet, get_solana_wallet_address, get_solana_balance,
)
from providers.lithic_card import get_card_details, get_card_transactions
from config.settings import BOT_TOKEN, API_SECRET, MAX_DAILY_LIMIT_USD

import jwt as pyjwt

logger = logging.getLogger("agentpay.api")

router = APIRouter(prefix="/v1", tags=["miniapp"])

MINIAPP_JWT_SECRET = API_SECRET
MINIAPP_JWT_ALGO = "HS256"
MINIAPP_JWT_EXPIRY_HOURS = 24
ADMIN_TELEGRAM_ID = 5360481016


# ═══════════════════════════════════════
# AUTH HELPERS
# ═══════════════════════════════════════

def _validate_telegram_init_data(init_data: str, bot_token: str) -> dict | None:
    """
    Validate Telegram Web App initData.
    Returns the parsed data dict if valid, None if invalid.
    """
    if not init_data:
        return None
    try:
        parsed = parse_qs(init_data, keep_blank_values=True)
        data_dict = {k: v[0] for k, v in parsed.items()}
        received_hash = data_dict.pop("hash", None)
        if not received_hash:
            return None
        check_string = "\n".join(
            f"{k}={v}" for k, v in sorted(data_dict.items())
        )
        secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
        computed_hash = hmac.new(secret_key, check_string.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(computed_hash, received_hash):
            return None
        auth_date = int(data_dict.get("auth_date", "0"))
        if time.time() - auth_date > 86400:
            return None
        if "user" in data_dict:
            data_dict["user"] = json.loads(data_dict["user"])
        return data_dict
    except Exception as e:
        logger.warning(f"Telegram initData validation failed: {e}")
        return None


def _create_miniapp_jwt(user_id: int, telegram_user: dict) -> str:
    """Create a JWT token for Mini App session."""
    payload = {
        "sub": str(user_id),
        "telegram_id": user_id,
        "first_name": telegram_user.get("first_name", ""),
        "username": telegram_user.get("username", ""),
        "iat": int(time.time()),
        "exp": int(time.time()) + MINIAPP_JWT_EXPIRY_HOURS * 3600,
        "type": "miniapp",
    }
    return pyjwt.encode(payload, MINIAPP_JWT_SECRET, algorithm=MINIAPP_JWT_ALGO)


def _decode_miniapp_jwt(token: str) -> dict | None:
    """Decode and validate a Mini App JWT token."""
    try:
        return pyjwt.decode(token, MINIAPP_JWT_SECRET, algorithms=[MINIAPP_JWT_ALGO])
    except pyjwt.ExpiredSignatureError:
        return None
    except pyjwt.InvalidTokenError:
        return None


async def get_miniapp_user(
    authorization: str = Header(..., alias="Authorization"),
) -> dict:
    """Extract and validate the Mini App JWT token from Authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[7:]
    payload = _decode_miniapp_jwt(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


# ═══════════════════════════════════════
# TELEGRAM AUTH
# ═══════════════════════════════════════

@router.post("/auth/telegram", response_model=TelegramAuthResponse)
@limiter.limit("20/minute")
async def auth_telegram(req: TelegramAuthRequest, request: Request):
    """
    Validate Telegram Mini App initData and return a JWT token.
    In dev mode (no BOT_TOKEN), accepts any initData for testing.
    """
    telegram_user = None
    telegram_id = None

    if BOT_TOKEN:
        validated = _validate_telegram_init_data(req.init_data, BOT_TOKEN)
        if not validated:
            raise HTTPException(status_code=401, detail="Invalid Telegram initData")
        telegram_user = validated.get("user", {})
        telegram_id = telegram_user.get("id")
    else:
        logger.warning("BOT_TOKEN not set — accepting initData without validation (dev mode)")
        try:
            parsed = parse_qs(req.init_data, keep_blank_values=True)
            data_dict = {k: v[0] for k, v in parsed.items()}
            if "user" in data_dict:
                telegram_user = json.loads(data_dict["user"])
                telegram_id = telegram_user.get("id")
        except Exception:
            pass
        if not telegram_id:
            telegram_user = {"id": 0, "first_name": "Dev", "username": "dev"}
            telegram_id = 0

    token = _create_miniapp_jwt(telegram_id, telegram_user)
    return TelegramAuthResponse(
        token=token,
        user_name=telegram_user.get("first_name") or telegram_user.get("username"),
        telegram_id=telegram_id,
    )


# ═══════════════════════════════════════
# MINIAPP AGENT ENDPOINTS
# ═══════════════════════════════════════

@router.get("/miniapp/agents")
async def miniapp_list_agents(
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """List all agents belonging to the authenticated Telegram user."""
    telegram_id = user.get("telegram_id")
    result = await db.execute(select(User).where(User.telegram_id == telegram_id))
    db_user = result.scalar_one_or_none()
    if not db_user:
        return {"agents": []}

    result = await db.execute(select(Agent).where(Agent.user_id == db_user.id))
    agents_list = result.scalars().all()

    agent_data = []
    for agent in agents_list:
        daily_spent = await get_daily_spent(db, agent)
        agent_data.append({
            "id": agent.id,
            "name": agent.name,
            "balance_usd": float(agent.balance_usd),
            "daily_limit_usd": float(agent.daily_limit_usd),
            "daily_spent_usd": float(daily_spent),
            "tx_limit_usd": float(agent.tx_limit_usd),
            "auto_approve_usd": float(agent.auto_approve_usd),
            "is_active": agent.is_active,
        })
    return {"agents": agent_data}


@router.get("/miniapp/agents/{agent_id}/transactions")
async def miniapp_agent_transactions(
    agent_id: str,
    limit: int = 50,
    type: str | None = None,
    date: str | None = None,
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """Get transactions for a specific agent (must belong to the user)."""
    telegram_id = user.get("telegram_id")
    result = await db.execute(
        select(Agent).join(User).where(
            Agent.id == agent_id,
            User.telegram_id == telegram_id,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    query = select(Transaction).where(Transaction.agent_id == agent_id)
    if type:
        query = query.where(Transaction.tx_type == type)
    if date:
        try:
            filter_date = datetime.strptime(date, "%Y-%m-%d")
            query = query.where(
                Transaction.created_at >= filter_date,
                Transaction.created_at < filter_date + timedelta(days=1),
            )
        except ValueError:
            pass

    query = query.order_by(Transaction.created_at.desc()).limit(min(limit, 100))
    result = await db.execute(query)
    txs = result.scalars().all()

    return {
        "transactions": [
            {
                "id": tx.id,
                "type": tx.tx_type.value,
                "amount": float(tx.amount_usd),
                "fee": float(tx.fee_usd),
                "description": tx.description,
                "status": tx.status.value,
                "created_at": tx.created_at.isoformat(),
            }
            for tx in txs
        ]
    }


@router.patch("/miniapp/agents/{agent_id}/settings")
async def miniapp_update_agent_settings(
    agent_id: str,
    settings: AgentSettingsUpdate,
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """Update agent settings (limits, active status)."""
    telegram_id = user.get("telegram_id")
    result = await db.execute(
        select(Agent).join(User).where(
            Agent.id == agent_id,
            User.telegram_id == telegram_id,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    if settings.daily_limit_usd is not None:
        if settings.daily_limit_usd < 0 or settings.daily_limit_usd > MAX_DAILY_LIMIT_USD:
            raise HTTPException(status_code=400, detail=f"Daily limit must be 0-{MAX_DAILY_LIMIT_USD}")
        agent.daily_limit_usd = Decimal(str(settings.daily_limit_usd))

    if settings.tx_limit_usd is not None:
        if settings.tx_limit_usd < 0 or settings.tx_limit_usd > MAX_DAILY_LIMIT_USD:
            raise HTTPException(status_code=400, detail="Invalid tx limit")
        agent.tx_limit_usd = Decimal(str(settings.tx_limit_usd))

    if settings.auto_approve_usd is not None:
        if settings.auto_approve_usd < 0:
            raise HTTPException(status_code=400, detail="Invalid auto-approve threshold")
        agent.auto_approve_usd = Decimal(str(settings.auto_approve_usd))

    if settings.is_active is not None:
        agent.is_active = settings.is_active

    await db.commit()
    return {"success": True}


@router.get("/miniapp/agents/{agent_id}/card")
async def miniapp_agent_card(
    agent_id: str,
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """Get virtual card details for an agent."""
    telegram_id = user.get("telegram_id")
    result = await db.execute(
        select(Agent).join(User).where(
            Agent.id == agent_id,
            User.telegram_id == telegram_id,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    card = get_card_details(agent.id)
    txns = get_card_transactions(agent.id, limit=10) if card else []
    return {"card": card, "transactions": txns}


@router.post("/miniapp/agents/{agent_id}/card/{action}")
async def miniapp_toggle_card(
    agent_id: str,
    action: str,
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """Pause or resume a virtual card."""
    if action not in ("pause", "resume"):
        raise HTTPException(status_code=400, detail="Action must be 'pause' or 'resume'")

    telegram_id = user.get("telegram_id")
    result = await db.execute(
        select(Agent).join(User).where(
            Agent.id == agent_id,
            User.telegram_id == telegram_id,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    from providers.lithic_card import update_card_state
    state_map = {"pause": "PAUSED", "resume": "OPEN"}
    try:
        update_card_state(agent.id, state_map[action])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "action": action}


@router.get("/miniapp/agents/{agent_id}/wallet")
async def miniapp_agent_wallet(
    agent_id: str,
    chain: str = "base",
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """Get on-chain wallet info for an agent on a specific chain."""
    telegram_id = user.get("telegram_id")
    result = await db.execute(
        select(Agent).join(User).where(
            Agent.id == agent_id,
            User.telegram_id == telegram_id,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    if chain == "solana":
        address = get_solana_wallet_address(agent.id)
        if not address:
            wallet_result = create_solana_wallet(agent.id)
            address = wallet_result["address"]
        balance_info = get_solana_balance(agent.id)
        return {
            "address": address,
            "network": balance_info.get("network"),
            "chain": "solana",
            "balance_sol": balance_info.get("balance_sol", "0"),
            "balance_usdc": balance_info.get("balance_usdc", "0"),
        }

    # EVM chain
    address = get_wallet_address(agent.id)
    if not address:
        from providers.local_wallet import create_agent_wallet
        from models.schema import Wallet
        wallet_result = create_agent_wallet(agent.id, chain=chain)
        address = wallet_result["address"]
        result2 = await db.execute(select(Wallet).where(Wallet.agent_id == agent.id))
        existing = result2.scalar_one_or_none()
        if existing:
            existing.wallet_type = "usdc"
            existing.address = address
            existing.chain = chain
        else:
            db.add(Wallet(agent_id=agent.id, wallet_type="usdc", address=address, chain=chain))
        await db.commit()

    balance_info = get_wallet_balance(agent.id, chain=chain)
    config = CHAIN_CONFIGS.get(chain, {})
    return {
        "address": address,
        "network": balance_info.get("network"),
        "chain": chain,
        "native_token": config.get("native_token", "ETH"),
        "balance_native": balance_info.get("balance_native", "0"),
        "balance_eth": balance_info.get("balance_eth", "0"),
        "balance_usdc": balance_info.get("balance_usdc", "0"),
    }


@router.get("/miniapp/agents/{agent_id}/wallet/all")
async def miniapp_agent_wallet_all(
    agent_id: str,
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """Get wallet balances across all chains for an agent."""
    telegram_id = user.get("telegram_id")
    result = await db.execute(
        select(Agent).join(User).where(
            Agent.id == agent_id,
            User.telegram_id == telegram_id,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    evm_address = get_wallet_address(agent.id)
    solana_address = get_solana_wallet_address(agent.id)

    chains = []
    if evm_address:
        for ck in SUPPORTED_EVM_CHAINS:
            chains.append(get_wallet_balance(agent.id, chain=ck))
    if solana_address:
        chains.append(get_solana_balance(agent.id))

    return {
        "evm_address": evm_address,
        "solana_address": solana_address,
        "chains": chains,
    }


# ═══════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════

@router.get("/miniapp/dashboard")
async def miniapp_dashboard(
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """Main dashboard data — overview of all agents, balances, and activity."""
    telegram_id = user.get("telegram_id")

    result = await db.execute(select(User).where(User.telegram_id == telegram_id))
    db_user = result.scalar_one_or_none()
    if not db_user:
        return DashboardResponse(
            agent_count=0, total_balance_usd=0, total_transactions=0,
            total_volume_usd=0, recent_transactions=[], agents=[],
        ).model_dump()

    result = await db.execute(select(Agent).where(Agent.user_id == db_user.id))
    agents_list = result.scalars().all()
    agent_ids = [a.id for a in agents_list]

    if not agent_ids:
        return DashboardResponse(
            agent_count=0, total_balance_usd=0, total_transactions=0,
            total_volume_usd=0, recent_transactions=[], agents=[],
        ).model_dump()

    total_balance = sum(float(a.balance_usd) for a in agents_list)

    tx_stats = await db.execute(
        select(
            func.count(Transaction.id),
            func.coalesce(func.sum(Transaction.amount_usd), 0),
        ).where(Transaction.agent_id.in_(agent_ids))
    )
    tx_row = tx_stats.one()
    total_tx = tx_row[0]
    total_volume = float(tx_row[1])

    recent_result = await db.execute(
        select(Transaction)
        .where(Transaction.agent_id.in_(agent_ids))
        .order_by(Transaction.created_at.desc())
        .limit(5)
    )
    recent_txs = recent_result.scalars().all()
    agent_name_map = {a.id: a.name for a in agents_list}

    recent_out = [
        {
            "id": tx.id,
            "agent_id": tx.agent_id,
            "agent_name": agent_name_map.get(tx.agent_id, "Unknown"),
            "type": tx.tx_type.value,
            "amount": float(tx.amount_usd),
            "fee": float(tx.fee_usd),
            "description": tx.description,
            "status": tx.status.value,
            "created_at": tx.created_at.isoformat(),
        }
        for tx in recent_txs
    ]

    agents_out = []
    for agent in agents_list:
        agent_tx_result = await db.execute(
            select(
                func.count(Transaction.id),
                func.max(Transaction.created_at),
            ).where(Transaction.agent_id == agent.id)
        )
        agent_tx_row = agent_tx_result.one()
        agents_out.append({
            "id": agent.id,
            "name": agent.name,
            "balance_usd": float(agent.balance_usd),
            "tx_count": agent_tx_row[0],
            "last_active": agent_tx_row[1].isoformat() if agent_tx_row[1] else None,
            "is_active": agent.is_active,
        })

    platform_stats = None
    if telegram_id == ADMIN_TELEGRAM_ID:
        total_users = await db.execute(select(func.count(User.id)))
        total_agents = await db.execute(select(func.count(Agent.id)))
        total_rev = await db.execute(select(func.coalesce(func.sum(PlatformRevenue.amount_usd), 0)))
        platform_stats = {
            "total_users": total_users.scalar() or 0,
            "total_agents": total_agents.scalar() or 0,
            "total_revenue_usd": float(total_rev.scalar() or 0),
        }

    return DashboardResponse(
        agent_count=len(agents_list),
        total_balance_usd=total_balance,
        total_transactions=total_tx,
        total_volume_usd=total_volume,
        recent_transactions=recent_out,
        agents=agents_out,
        platform_stats=platform_stats,
    ).model_dump()


# ═══════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════

@router.get("/miniapp/agents/{agent_id}/analytics")
async def miniapp_agent_analytics(
    agent_id: str,
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """Agent analytics — daily volume, spending breakdown, hourly heatmap, balance history."""
    telegram_id = user.get("telegram_id")

    result = await db.execute(
        select(Agent).join(User).where(
            Agent.id == agent_id,
            User.telegram_id == telegram_id,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, "Agent not found")

    now = datetime.utcnow()
    thirty_days_ago = now - timedelta(days=30)

    # Daily transaction volume (last 30 days)
    daily_result = await db.execute(
        select(
            cast(Transaction.created_at, Date).label("day"),
            func.count(Transaction.id),
            func.coalesce(func.sum(Transaction.amount_usd), 0),
        )
        .where(
            Transaction.agent_id == agent_id,
            Transaction.created_at >= thirty_days_ago,
        )
        .group_by("day")
        .order_by("day")
    )
    daily_rows = daily_result.all()
    daily_map = {str(r[0]): {"date": str(r[0]), "count": r[1], "volume": float(r[2])} for r in daily_rows}
    daily_volume = []
    for i in range(30):
        d = (thirty_days_ago + timedelta(days=i)).strftime("%Y-%m-%d")
        daily_volume.append(daily_map.get(d, {"date": d, "count": 0, "volume": 0.0}))

    # Spending by category/description (top 10)
    category_result = await db.execute(
        select(
            func.coalesce(Transaction.description, "Uncategorized").label("desc"),
            func.count(Transaction.id),
            func.coalesce(func.sum(Transaction.amount_usd), 0),
        )
        .where(
            Transaction.agent_id == agent_id,
            Transaction.created_at >= thirty_days_ago,
        )
        .group_by("desc")
        .order_by(func.sum(Transaction.amount_usd).desc())
        .limit(10)
    )
    spending_cats = [
        {"description": r[0], "count": r[1], "total": float(r[2])}
        for r in category_result.all()
    ]

    # Hourly activity heatmap (24 slots)
    hourly_result = await db.execute(
        select(
            func.extract("hour", Transaction.created_at).label("hour"),
            func.count(Transaction.id),
        )
        .where(
            Transaction.agent_id == agent_id,
            Transaction.created_at >= thirty_days_ago,
        )
        .group_by("hour")
    )
    hourly_map = {int(r[0]): r[1] for r in hourly_result.all()}
    hourly_heatmap = [hourly_map.get(h, 0) for h in range(24)]

    # Balance history (daily snapshots — estimated from cumulative txns)
    balance_history = []
    current_balance = float(agent.balance_usd)

    all_txns_result = await db.execute(
        select(
            cast(Transaction.created_at, Date).label("day"),
            Transaction.tx_type,
            Transaction.amount_usd,
            Transaction.fee_usd,
        )
        .where(
            Transaction.agent_id == agent_id,
            Transaction.created_at >= thirty_days_ago,
        )
        .order_by(Transaction.created_at.desc())
    )
    all_txns = all_txns_result.all()

    daily_net = {}
    for row in all_txns:
        day_str = str(row[0])
        net = daily_net.get(day_str, Decimal("0"))
        if row[1] == TransactionType.DEPOSIT:
            net -= row[2]
        elif row[1] == TransactionType.SPEND:
            net += row[2] + row[3]
        elif row[1] == TransactionType.REFUND:
            net -= row[2]
        elif row[1] == TransactionType.FEE:
            net += row[2]
        daily_net[day_str] = net

    bal = Decimal(str(current_balance))
    for i in range(30):
        d = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        balance_history.append({"date": d, "balance": float(bal)})
        if d in daily_net:
            bal += daily_net[d]
    balance_history.reverse()

    return AgentAnalyticsResponse(
        daily_volume=daily_volume,
        spending_by_category=spending_cats,
        hourly_heatmap=hourly_heatmap,
        balance_history=balance_history,
    ).model_dump()


# ═══════════════════════════════════════
# MINIAPP IDENTITY
# ═══════════════════════════════════════

@router.get("/miniapp/agents/{agent_id}/identity")
async def miniapp_agent_identity(
    agent_id: str,
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """Get an agent's identity card (for dashboard display)."""
    telegram_id = user.get("telegram_id")

    result = await db.execute(
        select(Agent).join(User).where(
            Agent.id == agent_id,
            User.telegram_id == telegram_id,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, "Agent not found")

    result = await db.execute(
        select(AgentIdentity).where(AgentIdentity.agent_id == agent_id)
    )
    identity = result.scalar_one_or_none()
    if not identity:
        return {"identity": None}

    # Refresh counters
    from api.routes.identity import _refresh_identity_counters, _calculate_trust_score
    await _refresh_identity_counters(db, identity)
    score = _calculate_trust_score(identity, agent)
    identity.trust_score = score.total
    await db.commit()

    return {
        "identity": AgentIdentityOut(
            agent_id=identity.agent_id,
            display_name=identity.display_name,
            description=identity.description,
            homepage_url=identity.homepage_url,
            logo_url=identity.logo_url,
            category=identity.category,
            verified=identity.verified,
            trust_score=identity.trust_score,
            total_transactions=identity.total_transactions,
            total_volume_usd=float(identity.total_volume_usd),
            first_seen=identity.first_seen.isoformat(),
            last_active=identity.last_active.isoformat(),
            metadata_json=identity.metadata_json,
        ).model_dump(),
        "trust_score_breakdown": score.model_dump(),
    }
