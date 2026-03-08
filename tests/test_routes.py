"""
Tests for wallet, admin, and chain API endpoints.
Covers: /v1/spend, /v1/refund, /v1/transfer, /v1/chains,
        /v1/wallet, /v1/card, /v1/webhook, /v1/admin/*
"""
import os
import sys
from decimal import Decimal
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import Base
from models.schema import User, Agent
from core.wallet import hash_api_key


@pytest_asyncio.fixture
async def wallet_app():
    """Create a test app with agent and API key for wallet tests."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    from api.main import app
    from models.database import get_db
    app.dependency_overrides[get_db] = override_get_db

    # Create user + agent
    api_key = "ap_wallet_test_key_abcdef1234567890abcdef1234"
    async with session_factory() as db:
        user = User(telegram_id=12345678, username="walletuser", first_name="Wallet")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        agent = Agent(
            user_id=user.id,
            name="wallet-agent",
            api_key_hash=hash_api_key(api_key),
            api_key_prefix="ap_wallet...",
            balance_usd=Decimal("100.0000"),
            daily_limit_usd=Decimal("50.0000"),
            tx_limit_usd=Decimal("25.0000"),
            auto_approve_usd=Decimal("10.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

    yield app, api_key, agent

    app.dependency_overrides.clear()
    await engine.dispose()


# ═══════════════════════════════════════
# CHAINS ENDPOINT
# ═══════════════════════════════════════

class TestChainsEndpoint:
    @pytest.mark.asyncio
    async def test_list_chains(self, wallet_app):
        """GET /v1/chains returns supported chains."""
        app, api_key, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/chains",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "chains" in data
            chain_ids = [c["id"] for c in data["chains"]]
            assert "solana" in chain_ids
            # Should have at least base + solana
            assert len(data["chains"]) >= 2
            # Each chain has required fields
            for chain in data["chains"]:
                assert "id" in chain
                assert "name" in chain
                assert "native_token" in chain


# ═══════════════════════════════════════
# SPEND ENDPOINT
# ═══════════════════════════════════════

class TestSpendEndpoint:
    @pytest.mark.asyncio
    async def test_spend_success(self, wallet_app):
        """POST /v1/spend with valid amount succeeds."""
        app, api_key, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/spend",
                json={"amount": 5.0, "description": "Test spend", "skip_approval": True},
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["amount"] == 5.0
            assert data["remaining_balance"] == 95.0

    @pytest.mark.asyncio
    async def test_spend_exceeds_tx_limit(self, wallet_app):
        """POST /v1/spend exceeding tx limit fails."""
        app, api_key, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/spend",
                json={"amount": 30.0, "description": "Over tx limit", "skip_approval": True},
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False
            assert "limit" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_spend_no_auth(self, wallet_app):
        """POST /v1/spend without API key returns 401/422."""
        app, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/spend",
                json={"amount": 5.0},
            )
            assert resp.status_code in (401, 422)


# ═══════════════════════════════════════
# REFUND ENDPOINT
# ═══════════════════════════════════════

class TestRefundEndpoint:
    @pytest.mark.asyncio
    async def test_refund_nonexistent_tx(self, wallet_app):
        """POST /v1/refund with invalid tx ID returns error."""
        app, api_key, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/refund",
                json={"transaction_id": "nonexistent-tx-id"},
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False

    @pytest.mark.asyncio
    async def test_refund_after_spend(self, wallet_app):
        """Spend then refund restores balance."""
        app, api_key, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Spend first
            spend_resp = await client.post(
                "/v1/spend",
                json={"amount": 10.0, "description": "To refund", "skip_approval": True},
                headers={"X-API-Key": api_key},
            )
            tx_id = spend_resp.json().get("transaction_id")
            assert tx_id is not None

            # Refund
            refund_resp = await client.post(
                "/v1/refund",
                json={"transaction_id": tx_id},
                headers={"X-API-Key": api_key},
            )
            data = refund_resp.json()
            assert data["success"] is True
            assert data["amount_refunded"] == 10.0


# ═══════════════════════════════════════
# TRANSFER ENDPOINT
# ═══════════════════════════════════════

class TestTransferEndpoint:
    @pytest.mark.asyncio
    async def test_transfer_nonexistent_target(self, wallet_app):
        """POST /v1/transfer to nonexistent agent fails."""
        app, api_key, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/transfer",
                json={"to_agent_id": "nonexistent-agent-id", "amount": 5.0},
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False


# ═══════════════════════════════════════
# CARD ENDPOINT
# ═══════════════════════════════════════

class TestCardEndpoint:
    @pytest.mark.asyncio
    async def test_card_info_no_card(self, wallet_app):
        """GET /v1/card returns empty when no card exists."""
        app, api_key, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/card",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["last4"] is None

    @pytest.mark.asyncio
    async def test_card_transactions_empty(self, wallet_app):
        """GET /v1/card/transactions returns empty list."""
        app, api_key, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/card/transactions",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data == []


# ═══════════════════════════════════════
# WEBHOOK ENDPOINTS
# ═══════════════════════════════════════

class TestWebhookEndpoints:
    @pytest.mark.asyncio
    async def test_get_webhook_none(self, wallet_app):
        """GET /v1/webhook returns null when not set."""
        app, api_key, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/webhook",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["url"] is None

    @pytest.mark.asyncio
    async def test_set_webhook(self, wallet_app):
        """POST /v1/webhook sets URL and returns secret."""
        app, api_key, _ = wallet_app
        transport = ASGITransport(app=app)

        async def mock_register(agent_id, url, secret, events=None):
            pass  # skip DB write

        with patch("core.webhooks.register_webhook", side_effect=mock_register):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/webhook",
                    json={"url": "https://example.com/webhook", "events": ["spend", "refund"]},
                    headers={"X-API-Key": api_key},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["url"] == "https://example.com/webhook"
                assert data["secret"] is not None
                assert "spend" in data["events"]

    @pytest.mark.asyncio
    async def test_delete_webhook(self, wallet_app):
        """DELETE /v1/webhook succeeds."""
        app, api_key, _ = wallet_app
        transport = ASGITransport(app=app)

        async def mock_unregister(agent_id):
            pass  # skip DB write

        with patch("core.webhooks.unregister_webhook", side_effect=mock_unregister):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete(
                    "/v1/webhook",
                    headers={"X-API-Key": api_key},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["success"] is True


# ═══════════════════════════════════════
# X402 PROBE ENDPOINT
# ═══════════════════════════════════════

class TestX402Endpoints:
    @pytest.mark.asyncio
    async def test_x402_probe_invalid_url(self, wallet_app):
        """GET /v1/x402/probe with invalid URL returns 400."""
        app, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/x402/probe",
                params={"url": "not-a-url"},
            )
            assert resp.status_code == 400


# ═══════════════════════════════════════
# ADMIN ENDPOINTS
# ═══════════════════════════════════════

class TestAdminEndpoints:
    @pytest.mark.asyncio
    async def test_admin_revenue_no_auth(self, wallet_app):
        """GET /v1/admin/revenue without auth returns 401."""
        app, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/admin/revenue")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_admin_withdraw_no_auth(self, wallet_app):
        """POST /v1/admin/withdraw without auth returns 401."""
        app, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/admin/withdraw")
            assert resp.status_code == 401
