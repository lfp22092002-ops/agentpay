"""
Agent Identity (KYA) endpoints — CRUD and public directory.
"""
from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_agent_auth
from api.middleware import limiter
from api.models import (
    AgentIdentityUpdate, AgentIdentityOut,
    TrustScoreBreakdown, DirectoryEntry, DirectoryResponse,
    VALID_CATEGORIES,
)
from models.database import get_db
from models.schema import Agent, AgentIdentity, Transaction, TransactionStatus

router = APIRouter(prefix="/v1", tags=["identity"])


# ═══════════════════════════════════════
# TRUST SCORE CALCULATION
# ═══════════════════════════════════════

def _calculate_trust_score(identity, agent) -> TrustScoreBreakdown:
    """Calculate trust score (0-100) based on activity and profile."""
    now = datetime.utcnow()

    age_weeks = (now - agent.created_at).days / 7
    account_age_pts = min(int(age_weeks), 15)

    tx_count_pts = min(identity.total_transactions // 10, 25)
    volume_pts = min(int(float(identity.total_volume_usd) / 100), 25)

    profile_pts = 0
    detail = {}
    if identity.display_name:
        profile_pts += 5
        detail["name"] = 5
    if identity.description:
        profile_pts += 5
        detail["description"] = 5
    if identity.homepage_url:
        profile_pts += 3
        detail["homepage_url"] = 3
    if identity.logo_url:
        profile_pts += 2
        detail["logo_url"] = 2
    profile_pts = min(profile_pts, 15)

    verified_pts = 20 if identity.verified else 0
    total = account_age_pts + tx_count_pts + volume_pts + profile_pts + verified_pts

    return TrustScoreBreakdown(
        total=min(total, 100),
        account_age_pts=account_age_pts,
        transaction_count_pts=tx_count_pts,
        volume_pts=volume_pts,
        profile_completeness_pts=profile_pts,
        verified_pts=verified_pts,
        details={
            "account_age_weeks": round(age_weeks, 1),
            "total_transactions": identity.total_transactions,
            "total_volume_usd": float(identity.total_volume_usd),
            "profile_fields": detail,
            "verified": identity.verified,
        },
    )


async def _refresh_identity_counters(db: AsyncSession, identity: AgentIdentity):
    """Refresh total_transactions and total_volume_usd from actual transaction data."""
    result = await db.execute(
        select(
            func.count(Transaction.id),
            func.coalesce(func.sum(Transaction.amount_usd), 0),
        ).where(
            Transaction.agent_id == identity.agent_id,
            Transaction.status == TransactionStatus.COMPLETED,
        )
    )
    row = result.one()
    identity.total_transactions = row[0]
    identity.total_volume_usd = row[1]
    identity.last_active = datetime.utcnow()


# ═══════════════════════════════════════
# IDENTITY ENDPOINTS
# ═══════════════════════════════════════

@router.get("/agent/identity")
@limiter.limit("30/minute")
async def get_own_identity(request: Request, auth: tuple = Depends(get_agent_auth)):
    """Get the authenticated agent's identity profile."""
    agent, db = auth
    result = await db.execute(select(AgentIdentity).where(AgentIdentity.agent_id == agent.id))
    identity = result.scalar_one_or_none()
    if not identity:
        return {"identity": None, "message": "No identity profile set. Use PUT /v1/agent/identity to create one."}

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
        ).model_dump()
    }


@router.put("/agent/identity")
@limiter.limit("10/minute")
async def upsert_identity(req: AgentIdentityUpdate, request: Request, auth: tuple = Depends(get_agent_auth)):
    """Create or update the authenticated agent's identity profile."""
    agent, db = auth

    if req.category and req.category not in VALID_CATEGORIES:
        raise HTTPException(400, f"Invalid category. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}")

    result = await db.execute(select(AgentIdentity).where(AgentIdentity.agent_id == agent.id))
    identity = result.scalar_one_or_none()

    now = datetime.utcnow()
    if identity:
        identity.display_name = req.display_name
        identity.description = req.description
        identity.homepage_url = req.homepage_url
        identity.logo_url = req.logo_url
        identity.category = req.category
        identity.metadata_json = req.metadata_json
        identity.updated_at = now
    else:
        identity = AgentIdentity(
            agent_id=agent.id,
            display_name=req.display_name,
            description=req.description,
            homepage_url=req.homepage_url,
            logo_url=req.logo_url,
            category=req.category,
            metadata_json=req.metadata_json,
            first_seen=agent.created_at,
            last_active=now,
        )
        db.add(identity)

    await _refresh_identity_counters(db, identity)
    score = _calculate_trust_score(identity, agent)
    identity.trust_score = score.total
    await db.commit()

    return {
        "success": True,
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
        ).model_dump()
    }


@router.get("/agent/identity/score")
@limiter.limit("30/minute")
async def get_trust_score(request: Request, auth: tuple = Depends(get_agent_auth)):
    """Get detailed trust score breakdown for the authenticated agent."""
    agent, db = auth
    result = await db.execute(select(AgentIdentity).where(AgentIdentity.agent_id == agent.id))
    identity = result.scalar_one_or_none()
    if not identity:
        raise HTTPException(404, "No identity profile. Create one with PUT /v1/agent/identity first.")

    await _refresh_identity_counters(db, identity)
    score = _calculate_trust_score(identity, agent)
    identity.trust_score = score.total
    await db.commit()

    return score.model_dump()


# ═══════════════════════════════════════
# PUBLIC DIRECTORY
# ═══════════════════════════════════════

@router.get("/directory")
@limiter.limit("60/minute")
async def agent_directory(
    request: Request,
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Public agent directory — browse registered agents with optional category filter."""
    page_size = min(max(page_size, 1), 50)
    page = max(page, 1)
    offset = (page - 1) * page_size

    if category:
        category = category.replace("\x00", "").strip()
        if len(category) > 50 or not category:
            category = None

    query = select(AgentIdentity)
    count_query = select(func.count(AgentIdentity.id))

    if category:
        query = query.where(AgentIdentity.category == category)
        count_query = count_query.where(AgentIdentity.category == category)

    query = query.order_by(AgentIdentity.trust_score.desc()).offset(offset).limit(page_size)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    result = await db.execute(query)
    identities = result.scalars().all()

    return DirectoryResponse(
        agents=[
            DirectoryEntry(
                agent_id=i.agent_id,
                display_name=i.display_name,
                description=i.description,
                category=i.category,
                verified=i.verified,
                trust_score=i.trust_score,
                total_transactions=i.total_transactions,
                total_volume_usd=float(i.total_volume_usd),
                logo_url=i.logo_url,
            )
            for i in identities
        ],
        total=total,
        page=page,
        page_size=page_size,
    ).model_dump()


@router.get("/directory/{agent_id}")
@limiter.limit("60/minute")
async def agent_public_profile(
    agent_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Get a public agent profile by agent ID."""
    result = await db.execute(select(AgentIdentity).where(AgentIdentity.agent_id == agent_id))
    identity = result.scalar_one_or_none()
    if not identity:
        raise HTTPException(404, "Agent not found in directory")

    return AgentIdentityOut(
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
    ).model_dump()
