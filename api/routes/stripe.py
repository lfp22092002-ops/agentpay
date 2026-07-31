"""
Stripe payment integration for AgentPay funding.

Enables credit/debit card deposits via Stripe Checkout.
Agents can be funded from anywhere in the world, not just Telegram.
"""
from decimal import Decimal
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from starlette.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select as sa_select

from api.middleware import limiter
from models.database import get_db
from models.schema import Agent, PlatformRevenue, Transaction, TransactionStatus, TransactionType
from core.wallet import hash_api_key
from config.settings import get_stripe_secret_key, get_stripe_webhook_secret

import stripe

router = APIRouter(prefix="/v1/funding", tags=["funding"])


@router.post("/stripe/create-checkout")
@limiter.limit("5/minute")
async def create_checkout(
    request: Request,
    agent_id: str = Query(...),
    amount_usd: float = Query(..., gt=0, le=10000),
    db: AsyncSession = Depends(get_db),
):
    """Create a Stripe Checkout session to fund an agent via credit/debit card."""
    # Validate auth
    auth_header = request.headers.get("X-API-Key", "")
    if not auth_header or not auth_header.startswith("ap_"):
        raise HTTPException(401, "Invalid or missing API key")
    
    result = await db.execute(sa_select(Agent).where(Agent.api_key_hash == hash_api_key(auth_header)))
    agent = result.scalar_one_or_none()
    if not agent or not agent.is_active:
        raise HTTPException(401, "Agent not found or inactive")
    
    # Verify agent ownership
    result = await db.execute(sa_select(Agent).where(Agent.id == agent_id))
    target_agent = result.scalar_one_or_none()
    if not target_agent or target_agent.user_id != agent.user_id:
        raise HTTPException(403, "Invalid agent ID")
    
    # Ensure Stripe key is set
    stripe_key = get_stripe_secret_key()
    if not stripe_key:
        raise HTTPException(503, "Stripe not configured. Set STRIPE_SECRET_KEY in environment.")
    
    stripe.api_key = stripe_key
    
    host = request.url.hostname or "leofundmybot.dev"
    
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "unit_amount": int(amount_usd * 100),
                "product_data": {
                    "name": f"AgentPay Funding: {target_agent.name}",
                    "description": f"${amount_usd:.2f} credit for your agent wallet",
                },
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=f"https://{host}/app/fund/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"https://{host}/app/fund/cancel",
        metadata={
            "agentpay_agent_id": target_agent.id,
            "agentpay_user_id": agent.user_id,
            "amount_usd": str(amount_usd),
        },
    )
    
    return {
        "checkout_url": session.url,
        "session_id": session.id,
        "amount_usd": amount_usd,
    }


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """Handle Stripe webhook events for funding confirmation."""
    # Check webhook secret at runtime (lazy load)
    wh_secret = get_stripe_webhook_secret()
    if not wh_secret:
        return JSONResponse(status_code=503, content={"error": "Stripe webhook not configured"})
    
    stripe.api_key = wh_secret
    
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, wh_secret)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid payload"})
    except stripe.error.SignatureVerificationError:
        return JSONResponse(status_code=400, content={"error": "Invalid signature"})
    
    if event.type == "checkout.session.completed":
        session = event.data.object
        
        agent_id = session.metadata.get("agentpay_agent_id")
        amount_usd = float(session.metadata.get("amount_usd", 0))
        
        if not agent_id:
            return JSONResponse(status_code=200, content={"status": "ignored"})
        
        result = await db.execute(sa_select(Agent).where(Agent.id == agent_id))
        target_agent = result.scalar_one_or_none()
        if not target_agent:
            return JSONResponse(status_code=404, content={"error": "Agent not found"})
        
        # Credit agent balance (platform fee is 1% for card processing)
        fee = min(Decimal(str(amount_usd * 0.01)), Decimal("0.50"))
        net_amount = Decimal(str(amount_usd)) - fee
        
        target_agent.balance_usd += net_amount
        
        # Record platform revenue
        revenue = PlatformRevenue(
            id=str(uuid.uuid4()),
            transaction_id=session.id,
            agent_id=target_agent.id,
            amount_usd=float(fee),
        )
        
        tx = Transaction(
            agent_id=target_agent.id,
            tx_type=TransactionType.DEPOSIT,
            status=TransactionStatus.COMPLETED,
            amount_usd=float(net_amount),
            fee_usd=float(fee),
            description=f"Stripe funding: ${amount_usd:.2f} (after 1% processing fee)",
        )
        
        db.add(revenue)
        db.add(tx)
        await db.commit()
    
    return JSONResponse(status_code=200, content={"status": "ok"})
