"""
Wallet, balance, send, spend, refund, transfer endpoints.
"""
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select

from api.dependencies import get_agent_auth
from api.middleware import limiter
from api.models import (
    SpendRequest, SpendResponse,
    RefundRequest, RefundResponse,
    TransferRequest, TransferResponse,
    WalletResponse, MultiChainWalletResponse,
    SendUsdcRequest, SendUsdcResponse,
    SendNativeRequest, SendNativeResponse,
    CardResponse, CardTransactionOut,
    WebhookSetRequest, WebhookResponse,
    ApprovalStatusResponse,
    X402PayRequest, X402PayResponse, X402ProbeResponse,
)
from core.wallet import spend, refund, transfer_between_agents
from models.schema import Wallet
from providers.local_wallet import (
    get_wallet_address, get_wallet_balance, send_usdc, send_native,
    CHAIN_CONFIGS, SUPPORTED_EVM_CHAINS,
)
from providers.solana_wallet import (
    create_solana_wallet, get_solana_wallet_address, get_solana_balance,
    send_sol, send_solana_usdc,
)
from providers.lithic_card import get_card_details, get_card_transactions

router = APIRouter(prefix="/v1", tags=["wallets"])


# ═══════════════════════════════════════
# SPEND / REFUND / TRANSFER
# ═══════════════════════════════════════

@router.post("/spend", response_model=SpendResponse)
@limiter.limit("30/minute")
async def do_spend(req: SpendRequest, request: Request, auth: tuple = Depends(get_agent_auth)):
    agent, db = auth
    amount = Decimal(str(req.amount)).quantize(Decimal("0.0001"))

    tx, error = await spend(
        db, agent, amount,
        description=req.description,
        idempotency_key=req.idempotency_key,
        require_approval=not req.skip_approval,
    )

    if error:
        if error.startswith("approval_pending:"):
            approval_id = error.split(":")[1]
            return SpendResponse(
                success=True,
                amount=float(amount),
                fee=0,
                remaining_balance=float(agent.balance_usd),
                approval_id=approval_id,
                status="pending_approval",
            )
        return SpendResponse(
            success=False,
            amount=float(amount),
            fee=0,
            remaining_balance=float(agent.balance_usd),
            error=error,
        )

    return SpendResponse(
        success=True,
        transaction_id=tx.id,
        amount=float(tx.amount_usd),
        fee=float(tx.fee_usd),
        remaining_balance=float(agent.balance_usd),
    )


@router.post("/refund", response_model=RefundResponse)
@limiter.limit("10/minute")
async def do_refund(req: RefundRequest, request: Request, auth: tuple = Depends(get_agent_auth)):
    """Refund a completed spend transaction. Credits amount + fee back to agent."""
    agent, db = auth
    tx, error = await refund(db, agent, req.transaction_id)
    if error:
        return RefundResponse(success=False, error=error)
    return RefundResponse(
        success=True,
        refund_transaction_id=tx.id,
        amount_refunded=float(tx.amount_usd),
        new_balance=float(agent.balance_usd),
    )


@router.post("/transfer", response_model=TransferResponse)
@limiter.limit("20/minute")
async def do_transfer(req: TransferRequest, request: Request, auth: tuple = Depends(get_agent_auth)):
    """Transfer funds from this agent to another agent (same owner only). No fees."""
    agent, db = auth
    amount = Decimal(str(req.amount)).quantize(Decimal("0.0001"))
    tx, error = await transfer_between_agents(db, agent, req.to_agent_id, amount, req.description)
    if error:
        return TransferResponse(success=False, error=error)
    return TransferResponse(
        success=True,
        transaction_id=tx.id,
        amount=float(amount),
        from_balance=float(agent.balance_usd),
    )


# ═══════════════════════════════════════
# CHAINS
# ═══════════════════════════════════════

@router.get("/chains")
@limiter.limit("30/minute")
async def list_chains(request: Request):
    """List all supported blockchain networks."""
    chains = []
    for chain_key, config in CHAIN_CONFIGS.items():
        chains.append({
            "id": chain_key,
            "name": config["name"],
            "type": "evm",
            "native_token": config["native_token"],
            "usdc_supported": True,
            "explorer": config["explorer"],
        })
    chains.append({
        "id": "solana",
        "name": "Solana",
        "type": "solana",
        "native_token": "SOL",
        "usdc_supported": True,
        "explorer": "https://solscan.io",
    })
    return {"chains": chains}


# ═══════════════════════════════════════
# ON-CHAIN WALLET
# ═══════════════════════════════════════

@router.get("/wallet")
async def wallet_info(chain: str = "base", auth: tuple = Depends(get_agent_auth)):
    """Get wallet info for a specific chain."""
    agent, db = auth

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

    address = get_wallet_address(agent.id)
    if not address:
        from providers.local_wallet import create_agent_wallet
        wallet_result = create_agent_wallet(agent.id, chain=chain)
        address = wallet_result["address"]
        result = await db.execute(select(Wallet).where(Wallet.agent_id == agent.id))
        existing = result.scalar_one_or_none()
        if existing:
            existing.wallet_type = "usdc"
            existing.address = address
            existing.chain = chain
        else:
            db.add(Wallet(agent_id=agent.id, wallet_type="usdc", address=address, chain=chain))
        await db.commit()

    balance_info = get_wallet_balance(agent.id, chain=chain)
    return WalletResponse(
        address=address,
        network=balance_info.get("network"),
        chain=chain,
        balance=balance_info.get("balance_eth"),
    )


@router.get("/wallet/all")
async def wallet_all_chains(auth: tuple = Depends(get_agent_auth)):
    """Get wallet info across all supported chains."""
    agent, db = auth
    evm_address = get_wallet_address(agent.id)
    solana_address = get_solana_wallet_address(agent.id)
    chain_balances = []
    if evm_address:
        for chain_key in SUPPORTED_EVM_CHAINS:
            balance = get_wallet_balance(agent.id, chain=chain_key)
            chain_balances.append(balance)
    if solana_address:
        balance = get_solana_balance(agent.id)
        chain_balances.append(balance)
    return MultiChainWalletResponse(
        evm_address=evm_address,
        solana_address=solana_address,
        chains=chain_balances,
    )


@router.post("/wallet/send-usdc", response_model=SendUsdcResponse)
@limiter.limit("10/minute")
async def api_send_usdc(req: SendUsdcRequest, request: Request, auth: tuple = Depends(get_agent_auth)):
    """Send USDC on the specified chain."""
    agent, db = auth
    if req.chain == "solana":
        result = await send_solana_usdc(agent.id, req.to_address, req.amount)
    elif req.chain in SUPPORTED_EVM_CHAINS:
        result = await send_usdc(agent.id, req.to_address, req.amount, chain=req.chain)
    else:
        return SendUsdcResponse(success=False, error=f"Unsupported chain: {req.chain}")
    return SendUsdcResponse(**result)


@router.post("/wallet/send-native", response_model=SendNativeResponse)
@limiter.limit("10/minute")
async def api_send_native(req: SendNativeRequest, request: Request, auth: tuple = Depends(get_agent_auth)):
    """Send native token (ETH/POL/BNB/SOL) on the specified chain."""
    agent, db = auth
    if req.chain == "solana":
        result = await send_sol(agent.id, req.to_address, req.amount)
    elif req.chain in SUPPORTED_EVM_CHAINS:
        result = await send_native(agent.id, req.to_address, req.amount, chain=req.chain)
    else:
        return SendNativeResponse(success=False, error=f"Unsupported chain: {req.chain}")
    return SendNativeResponse(**result)


# ═══════════════════════════════════════
# VIRTUAL CARD
# ═══════════════════════════════════════

@router.get("/card", response_model=CardResponse)
async def card_info(auth: tuple = Depends(get_agent_auth)):
    agent, db = auth
    details = get_card_details(agent.id)
    if not details:
        return CardResponse(last4=None, exp_month=None, exp_year=None, state=None, spend_limit_cents=None)
    return CardResponse(**details)


@router.get("/card/transactions", response_model=list[CardTransactionOut])
async def card_transactions(limit: int = 10, auth: tuple = Depends(get_agent_auth)):
    agent, db = auth
    txns = get_card_transactions(agent.id, limit=min(limit, 50))
    return [CardTransactionOut(**t) for t in txns]


# ═══════════════════════════════════════
# WEBHOOKS
# ═══════════════════════════════════════

@router.post("/webhook", response_model=WebhookResponse)
@limiter.limit("10/minute")
async def set_webhook(req: WebhookSetRequest, request: Request, auth: tuple = Depends(get_agent_auth)):
    from core.webhooks import register_webhook, generate_webhook_secret
    agent, db = auth
    secret = generate_webhook_secret()
    await register_webhook(agent.id, req.url, secret, req.events)
    return WebhookResponse(url=req.url, secret=secret, events=req.events or ["all"])


@router.get("/webhook", response_model=WebhookResponse)
async def get_webhook(auth: tuple = Depends(get_agent_auth)):
    from core.webhooks import get_webhook_config
    agent, db = auth
    config = get_webhook_config(agent.id)
    if not config:
        return WebhookResponse(url=None, secret=None, events=None)
    return WebhookResponse(
        url=config["url"],
        secret="whsec_****" + config["secret"][-8:],
        events=config["events"],
    )


@router.delete("/webhook")
async def delete_webhook(auth: tuple = Depends(get_agent_auth)):
    from core.webhooks import unregister_webhook
    agent, db = auth
    await unregister_webhook(agent.id)
    return {"success": True}


# ═══════════════════════════════════════
# APPROVALS
# ═══════════════════════════════════════

@router.get("/approvals/{approval_id}", response_model=ApprovalStatusResponse)
async def check_approval(approval_id: str, auth: tuple = Depends(get_agent_auth)):
    from core.approvals import get_pending
    approval = get_pending(approval_id)
    if not approval:
        raise HTTPException(status_code=404, detail="Approval not found or expired")
    if approval.agent_id != auth[0].id:
        raise HTTPException(status_code=403, detail="Not your approval")
    result_str = None
    if approval.resolved and approval.result and approval.result.done():
        r = approval.result.result()
        result_str = "approved" if r["approved"] else "denied"
    return ApprovalStatusResponse(
        approval_id=approval.id,
        agent_id=approval.agent_id,
        amount_usd=float(approval.amount_usd),
        description=approval.description,
        resolved=approval.resolved,
        result=result_str,
    )


# ═══════════════════════════════════════
# x402
# ═══════════════════════════════════════

@router.post("/x402/pay", response_model=X402PayResponse)
@limiter.limit("20/minute")
async def x402_pay(req: X402PayRequest, request: Request, auth: tuple = Depends(get_agent_auth)):
    """Agent pays for an x402-gated resource using their on-chain wallet."""
    from providers.x402_protocol import pay_x402_resource
    agent, db = auth
    result = await pay_x402_resource(agent.id, req.url, req.method, req.body, req.max_price_usd)
    return X402PayResponse(**result)


@router.get("/x402/probe")
@limiter.limit("30/minute")
async def x402_probe(url: str, request: Request):
    """Probe a URL to check if it's x402-gated and see pricing. No auth needed."""
    url = url.replace("\x00", "").strip()
    if not url.startswith(("http://", "https://")) or len(url) > 2048:
        raise HTTPException(400, "Invalid URL")
    from providers.x402_protocol import estimate_x402_cost
    result = estimate_x402_cost(url)
    if result.get("error"):
        raise HTTPException(502, f"Probe failed: {result['error']}")
    return X402ProbeResponse(**result)
