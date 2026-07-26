"""
Batch/multi-agent payment endpoints — parallel funding, sectioned workloads.

Supports the orchestrator→sub-agent pattern from Anthropic's building-effective-agents research:
split a budget across N agents in parallel, aggregate results back atomically.
"""
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_agent_auth
from api.middleware import limiter
from models.database import get_db
from models.schema import Agent, Transaction, TransactionStatus, TransactionType


router = APIRouter(prefix="/v1", tags=["batch"])


class BatchTransferRequest(BaseModel):
    """Split a total budget across N sub-agents."""
    agent_ids: list[str]
    amounts_usd: list[float]  # same length as agent_ids
    description: str | None = None


class BatchPaymentItem(BaseModel):
    agent_id: str
    amount_usd: float


class BatchResultItem(BaseModel):
    agent_id: str
    success: bool
    amount_usd: float
    tx_id: str | None = None
    error: str | None = None


class BatchTransferResponse(BaseModel):
    total_requested_usd: float
    total_successful_usd: float
    total_failed_usd: float
    items: list[BatchResultItem]


@router.post("/batch/transfer", response_model=BatchTransferResponse, status_code=201)
@limiter.limit("10/minute")
async def batch_transfer_agents(request: Request, auth: tuple = Depends(get_agent_auth)):
    """
    Transfer funds from one agent to multiple sub-agents in parallel.
    
    Orchestrator splits budget → sub-agents execute independently.
    Partial failures are ok — only affected transfers fail, others succeed.
    """
    # Parse request
    body = await request.json()
    if not body.get("payments") or len(body["payments"]) == 0:
        raise HTTPException(400, "Need at least one payment")
    
    payments = body["payments"]
    agent, db = auth
    
    # Validate inputs
    for i, p in enumerate(payments):
        if not isinstance(p, dict):
            raise HTTPException(400, f"Payment {i} must be an object with agent_id and amount_usd")
        if "agent_id" not in p or "amount_usd" not in p:
            raise HTTPException(400, f"Payment {i} requires agent_id and amount_usd")
        try:
            p["amount_usd"] = Decimal(str(p["amount_usd"]))
        except Exception:
            raise HTTPException(400, f"Payment {i} has invalid amount")
    
    total = sum(p["amount_usd"] for p in payments)
    if agent.balance_usd < total:
        raise HTTPException(409, f"Insufficient balance. Need ${total}, have ${agent.balance_usd}")
    
    # Find all target agents (parallel query)
    from sqlalchemy import select
    result = await db.execute(select(Agent).where(Agent.id.in_(p["agent_id"] for p in payments)))
    targets = {a.id: a for a in result.scalars().all()}
    
    results: list[BatchResultItem] = []
    total_success = Decimal("0")
    total_failed = Decimal("0")
    pending_txs = []
    pending_updates = {}  # agent_id -> amount
    
    # Process each transfer independently (partial failure is acceptable)
    for p in payments:
        target_id = p["agent_id"]
        amount = p["amount_usd"]
        
        if target_id not in targets:
            results.append(BatchResultItem(
                agent_id=target_id, success=False, amount_usd=float(amount),
                error="Target agent not found"
            ))
            total_failed += amount
            continue
        
        to_agent = targets[target_id]
        
        # Same-owner check
        if agent.user_id != to_agent.user_id:
            results.append(BatchResultItem(
                agent_id=target_id, success=False, amount_usd=float(amount),
                error="Can only transfer between your own agents"
            ))
            total_failed += amount
            continue
        
        # Record in pending state (will commit all-or-nothing)
        from_tx = Transaction(
            agent_id=agent.id, tx_type=TransactionType.SPEND,
            status=TransactionStatus.COMPLETED,
            amount_usd=amount, fee_usd=Decimal("0"),
            description=p.get("description") or f"Batch transfer to {to_agent.name}",
        )
        to_tx = Transaction(
            agent_id=target_id, tx_type=TransactionType.DEPOSIT,
            status=TransactionStatus.COMPLETED,
            amount_usd=amount, fee_usd=Decimal("0"),
            description=p.get("description") or f"Batch transfer from {agent.name}",
        )
        pending_txs.extend([from_tx, to_tx])
        pending_updates[to_agent.id] = pending_updates.get(to_agent.id, Decimal("0")) + amount
    
    # Commit everything atomically
    try:
        for tx in pending_txs:
            db.add(tx)
        agent.balance_usd -= total
        for tid, amt in pending_updates.items():
            if tid in targets:
                targets[tid].balance_usd += amt
        await db.commit()
        
        for i, p in enumerate(payments):
            target_id = p["agent_id"]
            if target_id not in targets:
                continue  # already failed above
            
            tx_idx = i * 2 + 1  # credit tx index in pending_txs
            results.append(BatchResultItem(
                agent_id=target_id, success=True, amount_usd=float(p["amount_usd"]),
                tx_id=pending_txs[tx_idx].id if pending_txs else None
            ))
            total_success += p["amount_usd"]
    except Exception as e:
        await db.rollback()
        raise HTTPException(500, f"Transfer failed: {str(e)}")
    
    return BatchTransferResponse(
        total_requested_usd=float(total),
        total_successful_usd=float(total_success),
        total_failed_usd=float(total_failed),
        items=results,
    )
