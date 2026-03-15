"""
Tests for sdk/agentpay/async_client.py — async SDK mirror of sync client tests.

Uses httpx MockTransport to test all async methods without network calls.
"""
import os
import sys

import httpx
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sdk.agentpay.async_client import AgentPayAsyncClient
from sdk.agentpay.exceptions import (
    AgentPayError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
)
from sdk.agentpay.models import Balance, SpendResponse, RefundResponse, TransferResponse, Wallet, Chain, Transaction, Webhook, X402Response


def make_transport(status: int = 200, body: dict | list | None = None):
    """Create an httpx MockTransport that returns a fixed response."""
    async def handler(request: httpx.Request):
        return httpx.Response(status, json=body or {})
    return httpx.MockTransport(handler)


def make_client(status: int = 200, body: dict | list | None = None) -> AgentPayAsyncClient:
    """Create an AgentPayAsyncClient with a mocked transport."""
    client = AgentPayAsyncClient("ap_test_key", base_url="https://test.local")
    client._client = httpx.AsyncClient(
        transport=make_transport(status, body),
        base_url="https://test.local",
        headers={"X-API-Key": "ap_test_key"},
    )
    return client


# ═══════════════════════════════════════
# BALANCE
# ═══════════════════════════════════════

class TestAsyncBalance:
    @pytest.mark.asyncio
    async def test_get_balance(self):
        client = make_client(200, {
            "agent_id": "1",
            "agent_name": "test-agent",
            "balance_usd": 42.50,
            "daily_spent_usd": 7.50,
            "daily_limit_usd": 50.0,
            "daily_remaining_usd": 42.50,
            "tx_limit_usd": 25.0,
            "is_active": True,
        })
        async with client:
            b = await client.get_balance()
        assert isinstance(b, Balance)
        assert b.balance_usd == 42.50
        assert b.daily_spent_usd == 7.50


# ═══════════════════════════════════════
# WALLET
# ═══════════════════════════════════════

class TestAsyncWallet:
    @pytest.mark.asyncio
    async def test_get_wallet(self):
        client = make_client(200, {
            "address": "0xabc123",
            "network": "Base Mainnet",
            "chain": "base",
            "balance": "0.05",
        })
        async with client:
            w = await client.get_wallet("base")
        assert isinstance(w, Wallet)
        assert w.address == "0xabc123"

    @pytest.mark.asyncio
    async def test_list_chains(self):
        client = make_client(200, {"chains": [
            {"id": "base", "name": "Base", "type": "evm", "native_token": "ETH", "usdc_supported": True, "explorer": "https://basescan.org"},
            {"id": "solana", "name": "Solana", "type": "solana", "native_token": "SOL", "usdc_supported": True, "explorer": "https://solscan.io"},
        ]})
        async with client:
            chains = await client.list_chains()
        assert len(chains) == 2
        assert isinstance(chains[0], Chain)
        assert chains[1].id == "solana"


# ═══════════════════════════════════════
# SPEND
# ═══════════════════════════════════════

class TestAsyncSpend:
    @pytest.mark.asyncio
    async def test_spend_success(self):
        client = make_client(200, {
            "success": True,
            "transaction_id": "tx_001",
            "amount": 5.00,
            "fee": 0.0,
            "remaining_balance": 95.0,
        })
        async with client:
            r = await client.spend(5.00, "Test charge")
        assert isinstance(r, SpendResponse)
        assert r.success is True
        assert r.transaction_id == "tx_001"

    @pytest.mark.asyncio
    async def test_spend_with_idempotency(self):
        client = make_client(200, {
            "success": True,
            "transaction_id": "tx_002",
            "amount": 3.00,
            "fee": 0.0,
            "remaining_balance": 97.0,
        })
        async with client:
            r = await client.spend(3.00, "Idempotent", idempotency_key="key-123")
        assert r.success is True

    @pytest.mark.asyncio
    async def test_spend_insufficient_balance(self):
        client = make_client(400, {"detail": "Insufficient balance"})
        async with client:
            with pytest.raises(InsufficientBalanceError):
                await client.spend(999.00, "Too much")


# ═══════════════════════════════════════
# REFUND
# ═══════════════════════════════════════

class TestAsyncRefund:
    @pytest.mark.asyncio
    async def test_refund_success(self):
        client = make_client(200, {
            "success": True,
            "refund_transaction_id": "tx_ref_001",
            "amount_refunded": 5.00,
            "new_balance": 100.0,
        })
        async with client:
            r = await client.refund("tx_001")
        assert isinstance(r, RefundResponse)
        assert r.success is True
        assert r.amount_refunded == 5.00


# ═══════════════════════════════════════
# TRANSFER
# ═══════════════════════════════════════

class TestAsyncTransfer:
    @pytest.mark.asyncio
    async def test_transfer_success(self):
        client = make_client(200, {
            "success": True,
            "transaction_id": "tx_xfer_001",
            "amount": 10.0,
            "from_balance": 90.0,
        })
        async with client:
            r = await client.transfer("agent_other", 10.0, "Payment")
        assert isinstance(r, TransferResponse)
        assert r.success is True

    @pytest.mark.asyncio
    async def test_transfer_no_description(self):
        client = make_client(200, {
            "success": True,
            "transaction_id": "tx_xfer_002",
            "amount": 5.0,
            "from_balance": 95.0,
        })
        async with client:
            r = await client.transfer("agent_other", 5.0)
        assert r.success is True


# ═══════════════════════════════════════
# TRANSACTIONS
# ═══════════════════════════════════════

class TestAsyncTransactions:
    @pytest.mark.asyncio
    async def test_get_transactions(self):
        client = make_client(200, [
            {"id": "tx_1", "type": "spend", "amount": 5.0, "fee": 0.0, "status": "completed", "created_at": "2026-01-01T00:00:00Z"},
            {"id": "tx_2", "type": "deposit", "amount": 50.0, "fee": 0.0, "status": "completed", "created_at": "2026-01-01T00:00:00Z"},
        ])
        async with client:
            txs = await client.get_transactions(limit=10)
        assert len(txs) == 2
        assert isinstance(txs[0], Transaction)
        assert txs[0].id == "tx_1"


# ═══════════════════════════════════════
# WEBHOOKS
# ═══════════════════════════════════════

class TestAsyncWebhooks:
    @pytest.mark.asyncio
    async def test_register_webhook(self):
        client = make_client(200, {
            "url": "https://example.com/hook",
            "secret": "whsec_abc123",
            "events": ["spend", "refund"],
        })
        async with client:
            w = await client.register_webhook("https://example.com/hook", events=["spend", "refund"])
        assert isinstance(w, Webhook)
        assert w.url == "https://example.com/hook"

    @pytest.mark.asyncio
    async def test_register_webhook_all_events(self):
        client = make_client(200, {
            "url": "https://example.com/hook",
            "secret": "whsec_xyz",
            "events": ["all"],
        })
        async with client:
            w = await client.register_webhook("https://example.com/hook")
        assert w.events == ["all"]


# ═══════════════════════════════════════
# x402
# ═══════════════════════════════════════

class TestAsyncX402:
    @pytest.mark.asyncio
    async def test_x402_pay(self):
        client = make_client(200, {
            "success": True,
            "data": "protected content here",
            "price_usd": 0.01,
        })
        async with client:
            r = await client.x402_pay("https://paywall.example/data", max_amount=0.05)
        assert isinstance(r, X402Response)
        assert r.success is True

    @pytest.mark.asyncio
    async def test_x402_pay_no_max(self):
        client = make_client(200, {
            "success": True,
            "data": "content",
        })
        async with client:
            r = await client.x402_pay("https://paywall.example/data")
        assert r.success is True


# ═══════════════════════════════════════
# ERROR HANDLING
# ═══════════════════════════════════════

class TestAsyncErrors:
    @pytest.mark.asyncio
    async def test_auth_error(self):
        client = make_client(401, {"detail": "Invalid API key"})
        async with client:
            with pytest.raises(AuthenticationError):
                await client.get_balance()

    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        client = make_client(429, {"detail": "Rate limited"})
        async with client:
            with pytest.raises(RateLimitError):
                await client.get_balance()

    @pytest.mark.asyncio
    async def test_generic_error(self):
        client = make_client(500, {"detail": "Internal error"})
        async with client:
            with pytest.raises(AgentPayError):
                await client.get_balance()

    @pytest.mark.asyncio
    async def test_error_without_json(self):
        """Non-JSON error body still raises."""
        async def handler(request: httpx.Request):
            return httpx.Response(503, text="Service Unavailable")
        client = AgentPayAsyncClient("ap_test", base_url="https://test.local")
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://test.local",
        )
        async with client:
            with pytest.raises(AgentPayError):
                await client.get_balance()


# ═══════════════════════════════════════
# CONTEXT MANAGER
# ═══════════════════════════════════════

class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Client works as async context manager."""
        client = make_client(200, {
            "agent_id": "1", "agent_name": "t", "balance_usd": 10.0,
            "daily_limit_usd": 50.0, "daily_spent_usd": 0.0,
            "daily_remaining_usd": 50.0, "tx_limit_usd": 25.0, "is_active": True,
        })
        async with client as c:
            b = await c.get_balance()
            assert b.balance_usd == 10.0

    @pytest.mark.asyncio
    async def test_close(self):
        """Explicit close works."""
        client = make_client(200, {
            "agent_id": "1", "agent_name": "t", "balance_usd": 0,
            "daily_limit_usd": 50.0, "daily_spent_usd": 0.0,
            "daily_remaining_usd": 50.0, "tx_limit_usd": 25.0, "is_active": True,
        })
        await client.get_balance()
        await client.close()
