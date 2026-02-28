import logging
import hashlib
import hmac
import json
import time
import time as _time
import csv
import io
from datetime import datetime, timedelta
from decimal import Decimal
from urllib.parse import parse_qs, unquote
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select, func, cast, Date
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse, FileResponse, StreamingResponse
from starlette.staticfiles import StaticFiles
from models.database import get_db, init_db
from models.schema import Agent, User, Transaction, PlatformRevenue, AgentIdentity
from core.wallet import get_agent_by_api_key, spend, get_agent_transactions, get_daily_spent, refund, transfer_between_agents, hash_api_key, rotate_api_key
from providers.local_wallet import get_wallet_address, get_wallet_balance, send_usdc, send_eth, send_native, CHAIN_CONFIGS, SUPPORTED_EVM_CHAINS, get_all_chain_balances
from providers.solana_wallet import (
    create_solana_wallet, get_solana_wallet_address, get_solana_balance,
    send_sol, send_solana_usdc,
)
from providers.lithic_card import get_card_details, get_card_transactions
from contextlib import asynccontextmanager
import jwt as pyjwt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("agentpay.api")

limiter = Limiter(key_func=get_remote_address)


# ═══════════════════════════════════════
# PER-API-KEY RATE LIMITER (in-memory)
# ═══════════════════════════════════════

import collections
import threading

_api_key_requests: dict[str, collections.deque] = {}
_rate_lock = threading.Lock()
API_KEY_RATE_LIMIT = 60  # requests per minute
API_KEY_RATE_WINDOW = 60  # seconds


def _check_api_key_rate_limit(api_key: str) -> bool:
    """Return True if allowed, False if rate-limited. Uses hash to avoid storing raw keys in memory."""
    now = time.time()
    cutoff = now - API_KEY_RATE_WINDOW
    # Use hash so raw API keys never sit in memory dict
    key_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    with _rate_lock:
        dq = _api_key_requests.get(key_id)
        if dq is None:
            dq = collections.deque()
            _api_key_requests[key_id] = dq
        # Evict old entries
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= API_KEY_RATE_LIMIT:
            return False
        dq.append(now)
        return True


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

# CORS — production only (no localhost)
ALLOWED_ORIGINS = [
    "https://leofundmybot.dev",
    "https://web.telegram.org",
    "https://t.me",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.telegram\.org",  # Telegram Mini App webviews
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["X-API-Key", "Authorization", "Content-Type", "Accept"],
)

app.state.limiter = limiter

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = _time.time()
    response = await call_next(request)
    duration = _time.time() - start
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    # Hide server version
    if "server" in response.headers:
        del response.headers["server"]
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration:.3f}s)")
    return response

# JWT secret for Mini App sessions (reuse API_SECRET)
from config.settings import BOT_TOKEN, API_SECRET
MINIAPP_JWT_SECRET = API_SECRET
MINIAPP_JWT_ALGO = "HS256"
MINIAPP_JWT_EXPIRY_HOURS = 24

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Slow down."},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions — never leak stack traces to clients."""
    import logging
    logging.getLogger("agentpay.api").error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ═══════════════════════════════════════
# AUTH
# ═══════════════════════════════════════

async def get_agent_auth(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> tuple[Agent, AsyncSession]:
    if not _check_api_key_rate_limit(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded: 60 requests/minute per API key")
    agent = await get_agent_by_api_key(db, x_api_key)
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not agent.is_active:
        raise HTTPException(status_code=403, detail="Agent is deactivated")
    return agent, db


# ═══════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════

class SpendRequest(BaseModel):
    amount: float = Field(gt=0, le=10000, description="Amount in USD")
    description: str | None = Field(None, max_length=500)
    idempotency_key: str | None = Field(None, max_length=64, description="Prevent duplicate charges")
    skip_approval: bool = Field(False, description="Skip approval workflow (only works within auto-approve limit)")


class SpendResponse(BaseModel):
    success: bool
    transaction_id: str | None = None
    amount: float
    fee: float
    remaining_balance: float
    approval_id: str | None = None
    status: str = "completed"  # completed, pending_approval
    error: str | None = None


class BalanceResponse(BaseModel):
    agent_id: str
    agent_name: str
    balance_usd: float
    daily_limit_usd: float
    daily_spent_usd: float
    daily_remaining_usd: float
    tx_limit_usd: float
    is_active: bool


class TransactionOut(BaseModel):
    id: str
    type: str
    amount: float
    fee: float
    description: str | None
    status: str
    created_at: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ═══════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════

@app.get("/v1/health", response_model=HealthResponse)
@limiter.limit("30/minute")
async def health(request: Request):
    return HealthResponse(status="ok", service="agentpay", version="0.1.0")


@app.get("/v1/balance", response_model=BalanceResponse)
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


@app.post("/v1/spend", response_model=SpendResponse)
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
        # Check if it's a pending approval
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


@app.get("/v1/transactions", response_model=list[TransactionOut])
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


import os
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.get("/")
async def root():
    """Serve landing page."""
    return FileResponse(os.path.join(_project_dir, "landing", "index.html"))


# ═══════════════════════════════════════
# REFUND ENDPOINT
# ═══════════════════════════════════════

class RefundRequest(BaseModel):
    transaction_id: str = Field(..., description="ID of the spend transaction to refund")


class RefundResponse(BaseModel):
    success: bool
    refund_transaction_id: str | None = None
    amount_refunded: float = 0
    new_balance: float = 0
    error: str | None = None


@app.post("/v1/refund", response_model=RefundResponse)
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


# ═══════════════════════════════════════
# TRANSFER ENDPOINT
# ═══════════════════════════════════════

class TransferRequest(BaseModel):
    to_agent_id: str = Field(..., description="Target agent ID to transfer to")
    amount: float = Field(..., gt=0, le=10000, description="Amount to transfer")
    description: str | None = None


class TransferResponse(BaseModel):
    success: bool
    transaction_id: str | None = None
    amount: float = 0
    from_balance: float = 0
    error: str | None = None


@app.post("/v1/transfer", response_model=TransferResponse)
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
# KEY ROTATION ENDPOINT
# ═══════════════════════════════════════

class RotateKeyResponse(BaseModel):
    success: bool
    new_api_key: str | None = None
    key_prefix: str | None = None
    error: str | None = None


@app.post("/v1/agent/rotate-key", response_model=RotateKeyResponse)
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


# ═══════════════════════════════════════
# CHAINS ENDPOINT
# ═══════════════════════════════════════

@app.get("/v1/chains")
@limiter.limit("30/minute")
async def list_chains(request: Request):
    """List all supported blockchain networks."""
    chains = []
    # EVM chains
    for chain_key, config in CHAIN_CONFIGS.items():
        chains.append({
            "id": chain_key,
            "name": config["name"],
            "type": "evm",
            "native_token": config["native_token"],
            "usdc_supported": True,
            "explorer": config["explorer"],
        })
    # Solana
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
# ON-CHAIN WALLET ENDPOINTS (Multi-Chain)
# ═══════════════════════════════════════

class WalletResponse(BaseModel):
    address: str | None
    network: str | None
    chain: str | None = None
    balance: str | None = None


class MultiChainWalletResponse(BaseModel):
    evm_address: str | None = None
    solana_address: str | None = None
    chains: list[dict] = []


class SendUsdcRequest(BaseModel):
    to_address: str
    amount: float = Field(gt=0, le=10000)
    chain: str = Field("base", description="Chain to send on: base, polygon, bnb, solana")
    description: str | None = None


class SendNativeRequest(BaseModel):
    to_address: str
    amount: float = Field(gt=0, le=100)
    chain: str = Field("base", description="Chain to send on: base, polygon, bnb, solana")


class SendUsdcResponse(BaseModel):
    success: bool
    tx_hash: str | None = None
    amount: float | None = None
    to: str | None = None
    chain: str | None = None
    error: str | None = None


class SendNativeResponse(BaseModel):
    success: bool
    tx_hash: str | None = None
    amount: float | None = None
    to: str | None = None
    chain: str | None = None
    native_token: str | None = None
    error: str | None = None


@app.get("/v1/wallet")
async def wallet_info(chain: str = "base", auth: tuple = Depends(get_agent_auth)):
    """Get wallet info for a specific chain. Default: base (backward compatible)."""
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

    # EVM chains (base, polygon, bnb)
    address = get_wallet_address(agent.id)

    # Auto-create EVM wallet if none exists
    if not address:
        from providers.local_wallet import create_agent_wallet
        from models.schema import Wallet
        wallet_result = create_agent_wallet(agent.id, chain=chain)
        address = wallet_result["address"]
        # Check if DB wallet row exists (may already exist from bot)
        result = await db.execute(
            select(Wallet).where(Wallet.agent_id == agent.id)
        )
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


@app.get("/v1/wallet/all")
async def wallet_all_chains(auth: tuple = Depends(get_agent_auth)):
    """Get wallet info across all supported chains."""
    agent, db = auth

    evm_address = get_wallet_address(agent.id)
    solana_address = get_solana_wallet_address(agent.id)

    chain_balances = []

    # EVM chains (same address across all)
    if evm_address:
        for chain_key in SUPPORTED_EVM_CHAINS:
            balance = get_wallet_balance(agent.id, chain=chain_key)
            chain_balances.append(balance)

    # Solana
    if solana_address:
        balance = get_solana_balance(agent.id)
        chain_balances.append(balance)

    return MultiChainWalletResponse(
        evm_address=evm_address,
        solana_address=solana_address,
        chains=chain_balances,
    )


@app.post("/v1/wallet/send-usdc", response_model=SendUsdcResponse)
@limiter.limit("10/minute")
async def api_send_usdc(req: SendUsdcRequest, request: Request, auth: tuple = Depends(get_agent_auth)):
    """Send USDC on the specified chain. Default: base."""
    agent, db = auth

    if req.chain == "solana":
        result = await send_solana_usdc(agent.id, req.to_address, req.amount)
    elif req.chain in SUPPORTED_EVM_CHAINS:
        result = await send_usdc(agent.id, req.to_address, req.amount, chain=req.chain)
    else:
        return SendUsdcResponse(success=False, error=f"Unsupported chain: {req.chain}")

    return SendUsdcResponse(**result)


@app.post("/v1/wallet/send-native", response_model=SendNativeResponse)
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
# VIRTUAL CARD ENDPOINTS
# ═══════════════════════════════════════

class CardResponse(BaseModel):
    last4: str | None
    exp_month: str | None
    exp_year: str | None
    state: str | None
    spend_limit_cents: int | None


class CardTransactionOut(BaseModel):
    amount_cents: int
    merchant: str
    status: str
    created: str


@app.get("/v1/card", response_model=CardResponse)
async def card_info(auth: tuple = Depends(get_agent_auth)):
    agent, db = auth
    details = get_card_details(agent.id)
    if not details:
        return CardResponse(last4=None, exp_month=None, exp_year=None, state=None, spend_limit_cents=None)
    return CardResponse(**details)


@app.get("/v1/card/transactions", response_model=list[CardTransactionOut])
async def card_transactions(limit: int = 10, auth: tuple = Depends(get_agent_auth)):
    agent, db = auth
    txns = get_card_transactions(agent.id, limit=min(limit, 50))
    return [CardTransactionOut(**t) for t in txns]


# ═══════════════════════════════════════
# WEBHOOK ENDPOINTS
# ═══════════════════════════════════════

class WebhookSetRequest(BaseModel):
    url: str = Field(..., max_length=512, description="HTTPS URL for webhook delivery")
    events: list[str] | None = Field(None, description="Event types to subscribe to (default: all)")


class WebhookResponse(BaseModel):
    url: str | None
    secret: str | None
    events: list[str] | None


@app.post("/v1/webhook", response_model=WebhookResponse)
@limiter.limit("10/minute")
async def set_webhook(req: WebhookSetRequest, request: Request, auth: tuple = Depends(get_agent_auth)):
    from core.webhooks import register_webhook, generate_webhook_secret
    agent, db = auth

    secret = generate_webhook_secret()
    await register_webhook(agent.id, req.url, secret, req.events)

    # DB already persisted by register_webhook

    return WebhookResponse(url=req.url, secret=secret, events=req.events or ["all"])


@app.get("/v1/webhook", response_model=WebhookResponse)
async def get_webhook(auth: tuple = Depends(get_agent_auth)):
    from core.webhooks import get_webhook_config
    agent, db = auth
    config = get_webhook_config(agent.id)
    if not config:
        return WebhookResponse(url=None, secret=None, events=None)
    return WebhookResponse(
        url=config["url"],
        secret="whsec_****" + config["secret"][-8:],  # masked
        events=config["events"],
    )


@app.delete("/v1/webhook")
async def delete_webhook(auth: tuple = Depends(get_agent_auth)):
    from core.webhooks import unregister_webhook
    agent, db = auth
    await unregister_webhook(agent.id)
    return {"success": True}


# ═══════════════════════════════════════
# APPROVAL ENDPOINTS
# ═══════════════════════════════════════

class ApprovalStatusResponse(BaseModel):
    approval_id: str
    agent_id: str
    amount_usd: float
    description: str | None
    resolved: bool
    result: str | None = None


@app.get("/v1/approvals/{approval_id}", response_model=ApprovalStatusResponse)
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
# x402 ENDPOINTS — Pay for resources via agent wallet
# ═══════════════════════════════════════

class X402PayRequest(BaseModel):
    url: str = Field(..., description="x402-gated resource URL")
    method: str = Field("GET", description="HTTP method")
    body: dict | None = Field(None, description="Request body for POST")
    max_price_usd: float = Field(1.0, gt=0, le=100, description="Max price willing to pay")


class X402PayResponse(BaseModel):
    success: bool
    status: int | None = None
    data: str | None = None
    paid_usd: float = 0
    error: str | None = None


class X402ProbeResponse(BaseModel):
    gated: bool | None = None
    status: int | None = None
    costs: list[dict] | None = None
    error: str | None = None


@app.post("/v1/x402/pay", response_model=X402PayResponse)
@limiter.limit("20/minute")
async def x402_pay(req: X402PayRequest, request: Request, auth: tuple = Depends(get_agent_auth)):
    """Agent pays for an x402-gated resource using their on-chain wallet."""
    from providers.x402_protocol import pay_x402_resource
    agent, db = auth
    result = await pay_x402_resource(
        agent.id, req.url, req.method, req.body, req.max_price_usd
    )
    return X402PayResponse(**result)


@app.get("/v1/x402/probe")
@limiter.limit("30/minute")
async def x402_probe(url: str, request: Request):
    """Probe a URL to check if it's x402-gated and see pricing. No auth needed."""
    # Sanitize URL
    url = url.replace("\x00", "").strip()
    if not url.startswith(("http://", "https://")) or len(url) > 2048:
        raise HTTPException(400, "Invalid URL")
    from providers.x402_protocol import estimate_x402_cost
    result = estimate_x402_cost(url)
    if result.get("error"):
        raise HTTPException(502, f"Probe failed: {result['error']}")
    return X402ProbeResponse(**result)


# ═══════════════════════════════════════
# TELEGRAM MINI APP ENDPOINTS
# ═══════════════════════════════════════

def _validate_telegram_init_data(init_data: str, bot_token: str) -> dict | None:
    """
    Validate Telegram Web App initData.
    Returns the parsed data dict if valid, None if invalid.
    See: https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app
    """
    if not init_data:
        return None

    try:
        parsed = parse_qs(init_data, keep_blank_values=True)
        # parse_qs returns lists; flatten
        data_dict = {k: v[0] for k, v in parsed.items()}

        received_hash = data_dict.pop("hash", None)
        if not received_hash:
            return None

        # Build check string: sort keys alphabetically, join with \n
        check_string = "\n".join(
            f"{k}={v}" for k, v in sorted(data_dict.items())
        )

        # HMAC-SHA256(secret_key, check_string)
        secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
        computed_hash = hmac.new(secret_key, check_string.encode(), hashlib.sha256).hexdigest()

        if not hmac.compare_digest(computed_hash, received_hash):
            return None

        # Check auth_date is not too old (allow 24h)
        auth_date = int(data_dict.get("auth_date", "0"))
        if time.time() - auth_date > 86400:
            return None

        # Parse the user JSON
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


class TelegramAuthRequest(BaseModel):
    init_data: str = Field(..., description="Telegram Web App initData string")


class TelegramAuthResponse(BaseModel):
    token: str
    user_name: str | None = None
    telegram_id: int


@app.post("/v1/auth/telegram", response_model=TelegramAuthResponse)
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
        # Dev mode: try to parse initData loosely, or use dummy
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


@app.get("/v1/miniapp/agents")
async def miniapp_list_agents(
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """List all agents belonging to the authenticated Telegram user."""
    telegram_id = user.get("telegram_id")

    # Find user by telegram_id
    result = await db.execute(
        select(User).where(User.telegram_id == telegram_id)
    )
    db_user = result.scalar_one_or_none()
    if not db_user:
        return {"agents": []}

    # Get all agents for this user
    result = await db.execute(
        select(Agent).where(Agent.user_id == db_user.id)
    )
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


@app.get("/v1/miniapp/agents/{agent_id}/transactions")
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

    # Verify ownership
    result = await db.execute(
        select(Agent).join(User).where(
            Agent.id == agent_id,
            User.telegram_id == telegram_id,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Query transactions
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


class AgentSettingsUpdate(BaseModel):
    daily_limit_usd: float | None = None
    tx_limit_usd: float | None = None
    auto_approve_usd: float | None = None
    is_active: bool | None = None


@app.patch("/v1/miniapp/agents/{agent_id}/settings")
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

    from config.settings import MAX_DAILY_LIMIT_USD

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


@app.get("/v1/miniapp/agents/{agent_id}/card")
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

    return {
        "card": card,
        "transactions": txns,
    }


@app.post("/v1/miniapp/agents/{agent_id}/card/{action}")
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

    # Delegate to lithic provider
    from providers.lithic_card import update_card_state
    state_map = {"pause": "PAUSED", "resume": "OPEN"}
    try:
        update_card_state(agent.id, state_map[action])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "action": action}


@app.get("/v1/miniapp/agents/{agent_id}/wallet")
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
        "balance_eth": balance_info.get("balance_eth", "0"),  # backward compat
        "balance_usdc": balance_info.get("balance_usdc", "0"),
    }


@app.get("/v1/miniapp/agents/{agent_id}/wallet/all")
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
# EXPORT ENDPOINT
# ═══════════════════════════════════════

@app.get("/v1/export")
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


# ═══════════════════════════════════════
# ADMIN — Revenue Tracking
# ═══════════════════════════════════════

ADMIN_TELEGRAM_ID = 5360481016  # G only

@app.get("/v1/admin/revenue")
@limiter.limit("10/minute")
async def get_revenue(request: Request, db: AsyncSession = Depends(get_db)):
    """Platform revenue summary. Admin only (requires JWT from Mini App auth)."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing auth")
    try:
        payload = pyjwt.decode(auth_header.split(" ")[1], API_SECRET, algorithms=["HS256"])
        if payload.get("telegram_id") != ADMIN_TELEGRAM_ID:
            raise HTTPException(403, "Admin only")
    except Exception:
        raise HTTPException(401, "Invalid token")

    # Total revenue
    total = await db.execute(select(func.sum(PlatformRevenue.amount_usd)))
    total_usd = total.scalar() or Decimal("0")

    # Today's revenue
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_result = await db.execute(
        select(func.sum(PlatformRevenue.amount_usd)).where(PlatformRevenue.created_at >= today)
    )
    today_usd = today_result.scalar() or Decimal("0")

    # Count
    count_result = await db.execute(select(func.count(PlatformRevenue.id)))
    tx_count = count_result.scalar() or 0

    return {
        "total_revenue_usd": float(total_usd),
        "today_revenue_usd": float(today_usd),
        "total_fee_transactions": tx_count,
    }


OWNER_WALLET = "0xD51B231F317260FB86b47A38F14eA29Cc81E0073"

@app.post("/v1/admin/withdraw")
@limiter.limit("3/minute")
async def withdraw_revenue(request: Request, db: AsyncSession = Depends(get_db)):
    """Withdraw accumulated platform fees as USDC to owner wallet. Admin only."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing auth")
    try:
        payload = pyjwt.decode(auth_header.split(" ")[1], API_SECRET, algorithms=["HS256"])
        if payload.get("telegram_id") != ADMIN_TELEGRAM_ID:
            raise HTTPException(403, "Admin only")
    except Exception:
        raise HTTPException(401, "Invalid token")

    # Calculate total unwithdrawn revenue
    total = await db.execute(select(func.sum(PlatformRevenue.amount_usd)))
    total_usd = total.scalar() or Decimal("0")

    if total_usd <= Decimal("0"):
        return {"success": False, "error": "No revenue to withdraw", "total_usd": 0}

    return {
        "success": True,
        "total_revenue_usd": float(total_usd),
        "withdraw_to": OWNER_WALLET,
        "note": "Revenue tracked in DB. On-chain USDC withdrawal available when agent wallets hold sufficient USDC balance. Use /v1/wallet/send-usdc on agent wallets to move funds to your wallet.",
    }


# ═══════════════════════════════════════
# AGENT IDENTITY (KYA) — SCHEMAS
# ═══════════════════════════════════════

VALID_CATEGORIES = {"trading", "research", "content", "automation", "defi", "analytics", "infrastructure", "social", "gaming", "other"}


class AgentIdentityUpdate(BaseModel):
    display_name: str = Field(..., max_length=255)
    description: str | None = Field(None, max_length=2000)
    homepage_url: str | None = Field(None, max_length=512)
    logo_url: str | None = Field(None, max_length=512)
    category: str | None = Field(None, max_length=50)
    metadata_json: str | None = Field(None, max_length=10000)


class AgentIdentityOut(BaseModel):
    agent_id: str
    display_name: str
    description: str | None = None
    homepage_url: str | None = None
    logo_url: str | None = None
    category: str | None = None
    verified: bool = False
    trust_score: int = 0
    total_transactions: int = 0
    total_volume_usd: float = 0
    first_seen: str
    last_active: str
    metadata_json: str | None = None


class TrustScoreBreakdown(BaseModel):
    total: int
    account_age_pts: int
    account_age_max: int = 15
    transaction_count_pts: int
    transaction_count_max: int = 25
    volume_pts: int
    volume_max: int = 25
    profile_completeness_pts: int
    profile_completeness_max: int = 15
    verified_pts: int
    verified_max: int = 20
    details: dict


class DirectoryEntry(BaseModel):
    agent_id: str
    display_name: str
    description: str | None = None
    category: str | None = None
    verified: bool = False
    trust_score: int = 0
    total_transactions: int = 0
    total_volume_usd: float = 0
    logo_url: str | None = None


class DirectoryResponse(BaseModel):
    agents: list[DirectoryEntry]
    total: int
    page: int
    page_size: int


# ═══════════════════════════════════════
# AGENT IDENTITY (KYA) — TRUST SCORE CALC
# ═══════════════════════════════════════

def _calculate_trust_score(identity: AgentIdentity, agent: Agent) -> TrustScoreBreakdown:
    """Calculate trust score (0-100) based on activity and profile."""
    now = datetime.utcnow()

    # Account age: 1pt/week, max 15
    age_weeks = (now - agent.created_at).days / 7
    account_age_pts = min(int(age_weeks), 15)

    # Transaction count: 1pt/10 txns, max 25
    tx_count_pts = min(identity.total_transactions // 10, 25)

    # Volume: 1pt/$100, max 25
    volume_pts = min(int(float(identity.total_volume_usd) / 100), 25)

    # Profile completeness: name=5, desc=5, url=3, logo=2
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

    # Verified badge: 20 pts
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
    from models.schema import TransactionStatus
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
# AGENT IDENTITY (KYA) — ENDPOINTS
# ═══════════════════════════════════════

@app.get("/v1/agent/identity")
@limiter.limit("30/minute")
async def get_own_identity(request: Request, auth: tuple = Depends(get_agent_auth)):
    """Get the authenticated agent's identity profile."""
    agent, db = auth
    result = await db.execute(
        select(AgentIdentity).where(AgentIdentity.agent_id == agent.id)
    )
    identity = result.scalar_one_or_none()
    if not identity:
        return {"identity": None, "message": "No identity profile set. Use PUT /v1/agent/identity to create one."}

    # Refresh counters
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


@app.put("/v1/agent/identity")
@limiter.limit("10/minute")
async def upsert_identity(req: AgentIdentityUpdate, request: Request, auth: tuple = Depends(get_agent_auth)):
    """Create or update the authenticated agent's identity profile."""
    agent, db = auth

    # Validate category
    if req.category and req.category not in VALID_CATEGORIES:
        raise HTTPException(400, f"Invalid category. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}")

    result = await db.execute(
        select(AgentIdentity).where(AgentIdentity.agent_id == agent.id)
    )
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

    # Refresh counters & trust score
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


@app.get("/v1/agent/identity/score")
@limiter.limit("30/minute")
async def get_trust_score(request: Request, auth: tuple = Depends(get_agent_auth)):
    """Get detailed trust score breakdown for the authenticated agent."""
    agent, db = auth
    result = await db.execute(
        select(AgentIdentity).where(AgentIdentity.agent_id == agent.id)
    )
    identity = result.scalar_one_or_none()
    if not identity:
        raise HTTPException(404, "No identity profile. Create one with PUT /v1/agent/identity first.")

    await _refresh_identity_counters(db, identity)
    score = _calculate_trust_score(identity, agent)
    identity.trust_score = score.total
    await db.commit()

    return score.model_dump()


@app.get("/v1/directory")
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

    # Sanitize category — strip null bytes, limit length
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


@app.get("/v1/directory/{agent_id}")
@limiter.limit("60/minute")
async def agent_public_profile(
    agent_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Get a public agent profile by agent ID."""
    result = await db.execute(
        select(AgentIdentity).where(AgentIdentity.agent_id == agent_id)
    )
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


# ═══════════════════════════════════════
# DASHBOARD API — SCHEMAS
# ═══════════════════════════════════════

class DashboardAgentSummary(BaseModel):
    id: str
    name: str
    balance_usd: float
    tx_count: int
    last_active: str | None = None


class DashboardResponse(BaseModel):
    agent_count: int
    total_balance_usd: float
    total_transactions: int
    total_volume_usd: float
    recent_transactions: list[dict]
    agents: list[dict]
    platform_stats: dict | None = None


class DailyVolume(BaseModel):
    date: str
    count: int
    volume: float


class SpendingCategory(BaseModel):
    description: str
    count: int
    total: float


class AgentAnalyticsResponse(BaseModel):
    daily_volume: list[dict]
    spending_by_category: list[dict]
    hourly_heatmap: list[int]
    balance_history: list[dict]


# ═══════════════════════════════════════
# DASHBOARD API — ENDPOINTS
# ═══════════════════════════════════════

@app.get("/v1/miniapp/dashboard")
async def miniapp_dashboard(
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """Main dashboard data — overview of all agents, balances, and activity."""
    telegram_id = user.get("telegram_id")

    result = await db.execute(
        select(User).where(User.telegram_id == telegram_id)
    )
    db_user = result.scalar_one_or_none()
    if not db_user:
        return DashboardResponse(
            agent_count=0, total_balance_usd=0, total_transactions=0,
            total_volume_usd=0, recent_transactions=[], agents=[],
        ).model_dump()

    # Get all agents
    result = await db.execute(
        select(Agent).where(Agent.user_id == db_user.id)
    )
    agents_list = result.scalars().all()
    agent_ids = [a.id for a in agents_list]

    if not agent_ids:
        return DashboardResponse(
            agent_count=0, total_balance_usd=0, total_transactions=0,
            total_volume_usd=0, recent_transactions=[], agents=[],
        ).model_dump()

    # Total balance
    total_balance = sum(float(a.balance_usd) for a in agents_list)

    # Total transactions & volume
    tx_stats = await db.execute(
        select(
            func.count(Transaction.id),
            func.coalesce(func.sum(Transaction.amount_usd), 0),
        ).where(Transaction.agent_id.in_(agent_ids))
    )
    tx_row = tx_stats.one()
    total_tx = tx_row[0]
    total_volume = float(tx_row[1])

    # Recent 5 transactions
    recent_result = await db.execute(
        select(Transaction)
        .where(Transaction.agent_id.in_(agent_ids))
        .order_by(Transaction.created_at.desc())
        .limit(5)
    )
    recent_txs = recent_result.scalars().all()

    # Map agent_id → name
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

    # Per-agent summary
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

    # Platform stats (admin only)
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


@app.get("/v1/miniapp/agents/{agent_id}/analytics")
async def miniapp_agent_analytics(
    agent_id: str,
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """Agent analytics — daily volume, spending breakdown, hourly heatmap, balance history."""
    telegram_id = user.get("telegram_id")

    # Verify ownership
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

    # ── Daily transaction volume (last 30 days) ──
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

    # Fill in missing days with zeros
    daily_map = {str(r[0]): {"date": str(r[0]), "count": r[1], "volume": float(r[2])} for r in daily_rows}
    daily_volume = []
    for i in range(30):
        d = (thirty_days_ago + timedelta(days=i)).strftime("%Y-%m-%d")
        daily_volume.append(daily_map.get(d, {"date": d, "count": 0, "volume": 0.0}))

    # ── Spending by category/description (top 10) ──
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

    # ── Hourly activity heatmap (24 slots) ──
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

    # ── Balance history (daily snapshots — estimated from cumulative txns) ──
    # We reconstruct by running a cumulative sum backwards from current balance
    balance_history = []
    current_balance = float(agent.balance_usd)
    # Get daily net changes (deposits - spends)
    daily_net_result = await db.execute(
        select(
            cast(Transaction.created_at, Date).label("day"),
            func.sum(
                # deposits add, spends/fees subtract
                Transaction.amount_usd
            ),
            func.array_agg(Transaction.tx_type),
        )
        .where(
            Transaction.agent_id == agent_id,
            Transaction.created_at >= thirty_days_ago,
        )
        .group_by("day")
        .order_by("day")
    )

    # Simpler approach: just track balance per day going backwards
    # Get all txns in the period ordered by date
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

    # Build daily net-change map
    from models.schema import TransactionType
    daily_net = {}
    for row in all_txns:
        day_str = str(row[0])
        net = daily_net.get(day_str, Decimal("0"))
        if row[1] == TransactionType.DEPOSIT:
            net -= row[2]  # reverse: removing deposit decreases balance
        elif row[1] == TransactionType.SPEND:
            net += row[2] + row[3]  # reverse: removing spend increases balance
        elif row[1] == TransactionType.REFUND:
            net -= row[2]  # reverse
        elif row[1] == TransactionType.FEE:
            net += row[2]  # reverse
        daily_net[day_str] = net

    # Walk backwards from today's balance
    bal = Decimal(str(current_balance))
    for i in range(30):
        d = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        balance_history.append({"date": d, "balance": float(bal)})
        # Reverse the day's transactions to get previous day's balance
        if d in daily_net:
            bal += daily_net[d]

    balance_history.reverse()

    return AgentAnalyticsResponse(
        daily_volume=daily_volume,
        spending_by_category=spending_cats,
        hourly_heatmap=hourly_heatmap,
        balance_history=balance_history,
    ).model_dump()


@app.get("/v1/miniapp/agents/{agent_id}/identity")
async def miniapp_agent_identity(
    agent_id: str,
    user: dict = Depends(get_miniapp_user),
    db: AsyncSession = Depends(get_db),
):
    """Get an agent's identity card (for dashboard display)."""
    telegram_id = user.get("telegram_id")

    # Verify ownership
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


# ═══════════════════════════════════════
# STATIC FILE SERVING — Mini App + Landing + Docs
# ═══════════════════════════════════════

# Mount static files (order matters — mount after all API routes)
app.mount("/app", StaticFiles(directory=os.path.join(_project_dir, "miniapp"), html=True), name="miniapp")
app.mount("/docs-site", StaticFiles(directory=os.path.join(_project_dir, "landing", "docs"), html=True), name="docs-site")
app.mount("/landing", StaticFiles(directory=os.path.join(_project_dir, "landing"), html=True), name="landing")
