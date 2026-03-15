"""
Payee Rules endpoints — manage per-agent whitelist/blocklist of allowed payees.
"""
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_agent_auth
from api.middleware import limiter
from models.database import get_db
from models.schema import PayeeRule

router = APIRouter(prefix="/v1", tags=["payee-rules"])


# ═══════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════

class PayeeRuleCreate(BaseModel):
    rule_type: str = Field("allow", pattern="^(allow|deny)$", description="'allow' or 'deny'")
    payee_type: str = Field(..., pattern="^(agent_id|domain|category|address)$",
                           description="Type of payee: agent_id, domain, category, or address")
    payee_value: str = Field(..., min_length=1, max_length=512, description="The payee identifier")
    max_amount_usd: float | None = Field(None, ge=0, description="Per-payee transaction cap (USD)")
    note: str | None = Field(None, max_length=255, description="Optional note")


class PayeeRuleOut(BaseModel):
    id: str
    rule_type: str
    payee_type: str
    payee_value: str
    max_amount_usd: float | None
    note: str | None
    is_active: bool
    created_at: str


# ═══════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════

@router.get("/agent/payee-rules")
@limiter.limit("30/minute")
async def list_payee_rules(request: Request, auth: tuple = Depends(get_agent_auth)):
    """List all payee rules for the authenticated agent."""
    agent, db = auth
    result = await db.execute(
        select(PayeeRule)
        .where(PayeeRule.agent_id == agent.id)
        .order_by(PayeeRule.created_at.desc())
    )
    rules = result.scalars().all()

    return {
        "rules": [
            PayeeRuleOut(
                id=r.id,
                rule_type=r.rule_type,
                payee_type=r.payee_type,
                payee_value=r.payee_value,
                max_amount_usd=float(r.max_amount_usd) if r.max_amount_usd is not None else None,
                note=r.note,
                is_active=r.is_active,
                created_at=r.created_at.isoformat(),
            ).model_dump()
            for r in rules
        ],
        "total": len(rules),
    }


@router.post("/agent/payee-rules", status_code=201)
@limiter.limit("20/minute")
async def create_payee_rule(
    req: PayeeRuleCreate,
    request: Request,
    auth: tuple = Depends(get_agent_auth),
):
    """Create a new payee rule (allow or deny)."""
    agent, db = auth

    # Check for duplicate
    result = await db.execute(
        select(PayeeRule).where(
            PayeeRule.agent_id == agent.id,
            PayeeRule.payee_type == req.payee_type,
            PayeeRule.payee_value == req.payee_value,
            PayeeRule.rule_type == req.rule_type,
            PayeeRule.is_active == True,
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(409, "A matching active rule already exists")

    rule = PayeeRule(
        agent_id=agent.id,
        rule_type=req.rule_type,
        payee_type=req.payee_type,
        payee_value=req.payee_value,
        max_amount_usd=Decimal(str(req.max_amount_usd)) if req.max_amount_usd is not None else None,
        note=req.note,
    )
    db.add(rule)
    await db.commit()
    await db.refresh(rule)

    return {
        "success": True,
        "rule": PayeeRuleOut(
            id=rule.id,
            rule_type=rule.rule_type,
            payee_type=rule.payee_type,
            payee_value=rule.payee_value,
            max_amount_usd=float(rule.max_amount_usd) if rule.max_amount_usd is not None else None,
            note=rule.note,
            is_active=rule.is_active,
            created_at=rule.created_at.isoformat(),
        ).model_dump(),
    }


@router.delete("/agent/payee-rules/{rule_id}")
@limiter.limit("20/minute")
async def delete_payee_rule(
    rule_id: str,
    request: Request,
    auth: tuple = Depends(get_agent_auth),
):
    """Delete (deactivate) a payee rule."""
    agent, db = auth

    result = await db.execute(
        select(PayeeRule).where(
            PayeeRule.id == rule_id,
            PayeeRule.agent_id == agent.id,
        )
    )
    rule = result.scalar_one_or_none()
    if not rule:
        raise HTTPException(404, "Rule not found")

    rule.is_active = False
    await db.commit()

    return {"success": True, "message": f"Rule {rule_id} deactivated"}


# ═══════════════════════════════════════
# EVALUATION HELPER (used by spend logic)
# ═══════════════════════════════════════

async def check_payee_allowed(
    db: AsyncSession,
    agent_id: str,
    payee_type: str,
    payee_value: str,
    amount_usd: Decimal,
) -> tuple[bool, str | None]:
    """Check if a payment to a specific payee is allowed.

    Returns (allowed: bool, reason: str | None).
    Logic: if any active deny rule matches → blocked.
    If any active allow rules exist but none match → blocked.
    If no rules exist → allowed (open by default).
    """
    result = await db.execute(
        select(PayeeRule).where(
            PayeeRule.agent_id == agent_id,
            PayeeRule.is_active == True,
        )
    )
    rules = result.scalars().all()

    if not rules:
        return True, None  # No rules = open

    # Check deny rules first
    for r in rules:
        if r.rule_type == "deny" and r.payee_type == payee_type and r.payee_value == payee_value:
            return False, f"Payment to {payee_type}:{payee_value} denied by rule"

    # Check allow rules
    allow_rules = [r for r in rules if r.rule_type == "allow"]
    if not allow_rules:
        return True, None  # Only deny rules exist, no match = allowed

    # If allow rules exist, must match one
    for r in allow_rules:
        if r.payee_type == payee_type and r.payee_value == payee_value:
            if r.max_amount_usd is not None and amount_usd > r.max_amount_usd:
                return False, f"Amount ${amount_usd} exceeds payee cap ${r.max_amount_usd}"
            return True, None

    return False, f"Payment to {payee_type}:{payee_value} not in allowlist"
