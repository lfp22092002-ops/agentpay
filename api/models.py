"""
Pydantic request/response models for the AgentPay API.
"""
from pydantic import BaseModel, Field


# ═══════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ═══════════════════════════════════════
# BALANCE
# ═══════════════════════════════════════

class BalanceResponse(BaseModel):
    agent_id: str
    agent_name: str
    balance_usd: float
    daily_limit_usd: float
    daily_spent_usd: float
    daily_remaining_usd: float
    tx_limit_usd: float
    is_active: bool


# ═══════════════════════════════════════
# SPEND / REFUND / TRANSFER
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
    status: str = "completed"
    error: str | None = None


class RefundRequest(BaseModel):
    transaction_id: str = Field(..., description="ID of the spend transaction to refund")


class RefundResponse(BaseModel):
    success: bool
    refund_transaction_id: str | None = None
    amount_refunded: float = 0
    new_balance: float = 0
    error: str | None = None


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


class TransactionOut(BaseModel):
    id: str
    type: str
    amount: float
    fee: float
    description: str | None
    status: str
    created_at: str


# ═══════════════════════════════════════
# KEY ROTATION
# ═══════════════════════════════════════

class RotateKeyResponse(BaseModel):
    success: bool
    new_api_key: str | None = None
    key_prefix: str | None = None
    error: str | None = None


# ═══════════════════════════════════════
# WALLET
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


# ═══════════════════════════════════════
# VIRTUAL CARD
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


# ═══════════════════════════════════════
# WEBHOOKS
# ═══════════════════════════════════════

class WebhookSetRequest(BaseModel):
    url: str = Field(..., max_length=512, description="HTTPS URL for webhook delivery")
    events: list[str] | None = Field(None, description="Event types to subscribe to (default: all)")


class WebhookResponse(BaseModel):
    url: str | None
    secret: str | None
    events: list[str] | None


# ═══════════════════════════════════════
# APPROVALS
# ═══════════════════════════════════════

class ApprovalStatusResponse(BaseModel):
    approval_id: str
    agent_id: str
    amount_usd: float
    description: str | None
    resolved: bool
    result: str | None = None


# ═══════════════════════════════════════
# x402
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


# ═══════════════════════════════════════
# TELEGRAM AUTH
# ═══════════════════════════════════════

class TelegramAuthRequest(BaseModel):
    init_data: str = Field(..., description="Telegram Web App initData string")


class TelegramAuthResponse(BaseModel):
    token: str
    user_name: str | None = None
    telegram_id: int


# ═══════════════════════════════════════
# MINI APP SETTINGS
# ═══════════════════════════════════════

class AgentSettingsUpdate(BaseModel):
    daily_limit_usd: float | None = None
    tx_limit_usd: float | None = None
    auto_approve_usd: float | None = None
    is_active: bool | None = None


# ═══════════════════════════════════════
# AGENT IDENTITY (KYA)
# ═══════════════════════════════════════

VALID_CATEGORIES = {
    "trading", "research", "content", "automation", "defi",
    "analytics", "infrastructure", "social", "gaming", "other",
}


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
# DASHBOARD
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
