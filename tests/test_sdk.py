"""
Tests for the AgentPay Python SDK (sync + async clients).

Uses httpx mock transport — no network calls needed.
"""
import pytest
import httpx

# Add SDK directory so we import the *SDK* agentpay package, not the project root
import sys
import os
SDK_DIR = os.path.join(os.path.dirname(__file__), "..", "sdk")
sys.path.insert(0, SDK_DIR)

# Remove project root from path to avoid the empty __init__.py shadowing the SDK
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
while PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)

# Clear cached 'agentpay' module if it was loaded from project root
if "agentpay" in sys.modules:
    mod = sys.modules["agentpay"]
    mod_file = getattr(mod, "__file__", "") or ""
    if "sdk" not in mod_file:
        del sys.modules["agentpay"]

from agentpay import (
    AgentPayClient,
    AgentPayAsyncClient,
    AgentPayError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
    __version__,
)


# ---------------------------------------------------------------------------
# Mock transport
# ---------------------------------------------------------------------------

def _make_response(data: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data)


class MockTransport(httpx.BaseTransport):
    """Deterministic transport that returns canned responses by path."""

    def __init__(self, routes: dict | None = None):
        self._routes = routes or {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode()
        method = request.method

        # Check for exact method+path match first
        key = f"{method} {path}"
        if key in self._routes:
            return self._routes[key]

        # Fallback to path-only
        if path in self._routes:
            return self._routes[path]

        return httpx.Response(404, json={"detail": "Not found"})


class AsyncMockTransport(httpx.AsyncBaseTransport):
    """Async version of MockTransport."""

    def __init__(self, routes: dict | None = None):
        self._routes = routes or {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode()
        method = request.method

        key = f"{method} {path}"
        if key in self._routes:
            return self._routes[key]
        if path in self._routes:
            return self._routes[path]

        return httpx.Response(404, json={"detail": "Not found"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BALANCE_RESPONSE = {
    "agent_id": "agent_123",
    "agent_name": "test-agent",
    "balance_usd": 100.0,
    "daily_limit_usd": 50.0,
    "daily_spent_usd": 5.0,
    "daily_remaining_usd": 45.0,
    "tx_limit_usd": 25.0,
    "is_active": True,
}

SPEND_RESPONSE = {
    "success": True,
    "transaction_id": "tx_abc123",
    "amount": 0.50,
    "fee": 0.0,
    "remaining_balance": 99.50,
    "status": "completed",
}

TRANSACTIONS_RESPONSE = [
    {
        "id": "tx_001",
        "type": "spend",
        "amount": 5.00,
        "fee": 0.0,
        "description": "API call",
        "status": "completed",
        "created_at": "2026-03-07T12:00:00Z",
    },
    {
        "id": "tx_002",
        "type": "deposit",
        "amount": 50.00,
        "fee": 0.0,
        "description": "Stars deposit",
        "status": "completed",
        "created_at": "2026-03-07T11:00:00Z",
    },
]

WALLET_RESPONSE = {
    "address": "0xabc123def456",
    "chain": "base",
    "balance": "42.50",
}

WEBHOOK_RESPONSE = {
    "url": "https://example.com/webhook",
    "secret": "whsec_test123",
    "events": ["spend", "refund"],
}

X402_RESPONSE = {
    "success": True,
    "status": 200,
    "data": "premium content here",
    "paid_usd": 0.10,
}

CHAINS_RESPONSE = {
    "chains": [
        {"id": "base", "name": "Base", "type": "evm", "native_token": "ETH", "usdc_supported": True, "explorer": "https://basescan.org"},
        {"id": "polygon", "name": "Polygon PoS", "type": "evm", "native_token": "MATIC", "usdc_supported": True, "explorer": "https://polygonscan.com"},
    ]
}


def _sync_client(routes: dict) -> AgentPayClient:
    """Create a sync client with mocked transport."""
    client = AgentPayClient("ap_test_key", base_url="https://test.local")
    # Swap out the internal httpx client with our mock
    client._client = httpx.Client(
        base_url="https://test.local",
        headers={"X-API-Key": "ap_test_key"},
        transport=MockTransport(routes),
    )
    return client


def _async_client(routes: dict) -> AgentPayAsyncClient:
    """Create an async client with mocked transport."""
    client = AgentPayAsyncClient.__new__(AgentPayAsyncClient)
    client._api_key = "ap_test_key"
    client._base_url = "https://test.local"
    client._max_retries = 3
    client._client = httpx.AsyncClient(
        base_url="https://test.local",
        headers={"X-API-Key": "ap_test_key"},
        transport=AsyncMockTransport(routes),
    )
    return client


# ===========================================================================
# Sync client tests
# ===========================================================================


class TestSyncClientInit:
    def test_version_exists(self):
        assert __version__ == "0.1.0"

    def test_default_base_url(self):
        client = AgentPayClient("ap_key")
        assert client._base_url == "https://leofundmybot.dev"
        client.close()

    def test_strips_trailing_slash(self):
        client = AgentPayClient("ap_key", base_url="https://example.com/")
        assert client._base_url == "https://example.com"
        client.close()

    def test_context_manager(self):
        routes = {"GET /v1/balance": _make_response(BALANCE_RESPONSE)}
        with _sync_client(routes) as c:
            balance = c.get_balance()
            assert balance.balance_usd == 100.0


class TestSyncBalance:
    def test_get_balance(self):
        routes = {"GET /v1/balance": _make_response(BALANCE_RESPONSE)}
        client = _sync_client(routes)
        balance = client.get_balance()
        assert balance.agent_id == "agent_123"
        assert balance.balance_usd == 100.0
        assert balance.daily_remaining_usd == 45.0
        assert balance.is_active is True
        client.close()


class TestSyncSpend:
    def test_spend(self):
        routes = {"POST /v1/spend": _make_response(SPEND_RESPONSE)}
        client = _sync_client(routes)
        result = client.spend(0.50, "GPT-4 API call")
        assert result.transaction_id == "tx_abc123"
        assert result.status == "completed"
        client.close()

    def test_spend_with_idempotency_key(self):
        routes = {"POST /v1/spend": _make_response(SPEND_RESPONSE)}
        client = _sync_client(routes)
        result = client.spend(0.50, "Test", idempotency_key="idem_123")
        assert result.transaction_id == "tx_abc123"
        client.close()


class TestSyncTransactions:
    def test_get_transactions(self):
        routes = {"GET /v1/transactions?limit=20": _make_response(TRANSACTIONS_RESPONSE)}
        client = _sync_client(routes)
        txs = client.get_transactions()
        assert len(txs) == 2
        assert txs[0].id == "tx_001"
        assert txs[1].type == "deposit"
        client.close()

    def test_get_transactions_custom_limit(self):
        routes = {"GET /v1/transactions?limit=5": _make_response(TRANSACTIONS_RESPONSE[:1])}
        client = _sync_client(routes)
        txs = client.get_transactions(limit=5)
        assert len(txs) == 1
        client.close()


class TestSyncWallet:
    def test_get_wallet(self):
        routes = {"GET /v1/wallet?chain=base": _make_response(WALLET_RESPONSE)}
        client = _sync_client(routes)
        wallet = client.get_wallet("base")
        assert wallet.address == "0xabc123def456"
        assert wallet.chain == "base"
        client.close()


class TestSyncChains:
    def test_list_chains(self):
        routes = {"GET /v1/chains": _make_response(CHAINS_RESPONSE)}
        client = _sync_client(routes)
        chains = client.list_chains()
        assert len(chains) == 2
        assert chains[0].id == "base"
        assert chains[1].name == "Polygon PoS"
        client.close()


class TestSyncWebhook:
    def test_register_webhook(self):
        routes = {"POST /v1/webhook": _make_response(WEBHOOK_RESPONSE)}
        client = _sync_client(routes)
        wh = client.register_webhook("https://example.com/webhook", events=["spend", "refund"])
        assert wh.url == "https://example.com/webhook"
        assert wh.secret == "whsec_test123"
        client.close()


class TestSyncX402:
    def test_x402_pay(self):
        routes = {"POST /v1/x402/pay": _make_response(X402_RESPONSE)}
        client = _sync_client(routes)
        result = client.x402_pay("https://api.example.com/premium", max_amount=1.0)
        assert result.status == 200
        assert result.data == "premium content here"
        client.close()


class TestSyncRefund:
    def test_refund(self):
        refund_resp = {"success": True, "refund_transaction_id": "tx_ref_123", "amount_refunded": 0.50, "new_balance": 100.50}
        routes = {"POST /v1/refund": _make_response(refund_resp)}
        client = _sync_client(routes)
        result = client.refund("tx_abc123")
        assert result.amount_refunded == 0.50
        client.close()


class TestSyncTransfer:
    def test_transfer(self):
        transfer_resp = {"success": True, "transaction_id": "tx_tr_123", "amount": 5.00, "from_balance": 95.00}
        routes = {"POST /v1/transfer": _make_response(transfer_resp)}
        client = _sync_client(routes)
        result = client.transfer("agent_456", 5.0, description="Payment")
        assert result.amount == 5.0
        client.close()


# ===========================================================================
# Error handling tests
# ===========================================================================


class TestSyncErrors:
    def test_auth_error_401(self):
        routes = {"GET /v1/balance": httpx.Response(401, json={"detail": "Invalid API key"})}
        client = _sync_client(routes)
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            client.get_balance()
        client.close()

    def test_rate_limit_429(self):
        routes = {"GET /v1/balance": httpx.Response(429, json={"detail": "Too many requests"})}
        client = _sync_client(routes)
        with pytest.raises(RateLimitError, match="Too many requests"):
            client.get_balance()
        client.close()

    def test_insufficient_balance_402(self):
        routes = {"POST /v1/spend": httpx.Response(402, json={"detail": "Insufficient balance"})}
        client = _sync_client(routes)
        with pytest.raises(InsufficientBalanceError, match="Insufficient balance"):
            client.spend(999.99, "Too much")
        client.close()

    def test_insufficient_balance_400_with_balance_hint(self):
        routes = {"POST /v1/spend": httpx.Response(400, json={"detail": "Not enough balance"})}
        client = _sync_client(routes)
        with pytest.raises(InsufficientBalanceError):
            client.spend(999.99, "Too much")
        client.close()

    def test_generic_error_500(self):
        routes = {"GET /v1/balance": httpx.Response(500, json={"detail": "Internal error"})}
        client = _sync_client(routes)
        with pytest.raises(AgentPayError):
            client.get_balance()
        client.close()

    def test_not_found_404(self):
        client = _sync_client({})  # No routes = 404 for everything
        with pytest.raises(AgentPayError):
            client.get_balance()
        client.close()


# ===========================================================================
# Async client tests
# ===========================================================================


class TestAsyncBalance:
    @pytest.mark.asyncio
    async def test_get_balance(self):
        routes = {"GET /v1/balance": _make_response(BALANCE_RESPONSE)}
        client = _async_client(routes)
        balance = await client.get_balance()
        assert balance.agent_id == "agent_123"
        assert balance.balance_usd == 100.0
        assert balance.is_active is True
        await client.close()


class TestAsyncSpend:
    @pytest.mark.asyncio
    async def test_spend(self):
        routes = {"POST /v1/spend": _make_response(SPEND_RESPONSE)}
        client = _async_client(routes)
        result = await client.spend(0.50, "Test call")
        assert result.transaction_id == "tx_abc123"
        await client.close()


class TestAsyncErrors:
    @pytest.mark.asyncio
    async def test_auth_error(self):
        routes = {"GET /v1/balance": httpx.Response(401, json={"detail": "Bad key"})}
        client = _async_client(routes)
        with pytest.raises(AuthenticationError):
            await client.get_balance()
        await client.close()

    @pytest.mark.asyncio
    async def test_rate_limit(self):
        routes = {"GET /v1/balance": httpx.Response(429, json={"detail": "Rate limited"})}
        client = _async_client(routes)
        with pytest.raises(RateLimitError):
            await client.get_balance()
        await client.close()


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        routes = {"GET /v1/balance": _make_response(BALANCE_RESPONSE)}
        client = _async_client(routes)
        async with client as c:
            balance = await c.get_balance()
            assert balance.balance_usd == 100.0
