"""
Tests for wallet route endpoints — spend, refund, transfer, chains, webhook, card, x402.

Uses FastAPI TestClient with in-memory SQLite to test the full HTTP cycle.
"""
import os
import sys
from decimal import Decimal
from unittest.mock import patch, AsyncMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.database import Base
from models.schema import User, Agent
from core.wallet import hash_api_key


@pytest_asyncio.fixture
async def wallet_app():
    """Create a test app with fresh DB and a funded agent."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with factory() as session:
            yield session

    from api.main import app
    from models.database import get_db
    app.dependency_overrides[get_db] = override_get_db

    api_key = "ap_wallettest_key_1234567890abcdef1234567890"
    async with factory() as db:
        user = User(telegram_id=77777777, username="walletuser", first_name="Wallet")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        agent = Agent(
            user_id=user.id,
            name="wallet-agent",
            api_key_hash=hash_api_key(api_key),
            api_key_prefix="ap_wall...",
            balance_usd=Decimal("500.0000"),
            daily_limit_usd=Decimal("100.0000"),
            tx_limit_usd=Decimal("50.0000"),
            auto_approve_usd=Decimal("25.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

        # Second agent for transfers
        agent2 = Agent(
            user_id=user.id,
            name="target-agent",
            api_key_hash=hash_api_key("ap_target_key_zzzzzzzzzzzzzzzzzzzzzzzzzzzz"),
            api_key_prefix="ap_targ...",
            balance_usd=Decimal("10.0000"),
            is_active=True,
        )
        db.add(agent2)
        await db.commit()
        await db.refresh(agent2)

    yield app, api_key, agent, agent2

    app.dependency_overrides.clear()
    await engine.dispose()


def _headers(api_key: str) -> dict:
    return {"X-API-Key": api_key}


# ═══════════════════════════════════════
# SPEND
# ═══════════════════════════════════════

class TestSpendEndpoint:
    @pytest.mark.asyncio
    async def test_spend_success(self, wallet_app):
        app, key, agent, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/spend", json={
                "amount": 5.00,
                "description": "GPT-4 API call",
            }, headers=_headers(key))
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["amount"] == 5.00
            assert data["remaining_balance"] < 500.00
            assert data["transaction_id"] is not None

    @pytest.mark.asyncio
    async def test_spend_exceeds_tx_limit(self, wallet_app):
        app, key, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/spend", json={
                "amount": 75.00,  # tx_limit is 50
                "description": "Too expensive",
            }, headers=_headers(key))
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False
            assert "limit" in data.get("error", "").lower() or "approval" in str(data).lower()

    @pytest.mark.asyncio
    async def test_spend_with_idempotency(self, wallet_app):
        app, key, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            body = {
                "amount": 3.00,
                "description": "Idempotent call",
                "idempotency_key": "idem-test-123",
            }
            resp1 = await client.post("/v1/spend", json=body, headers=_headers(key))
            resp2 = await client.post("/v1/spend", json=body, headers=_headers(key))
            assert resp1.status_code == 200
            assert resp2.status_code == 200
            # Second call should return same transaction
            d1 = resp1.json()
            d2 = resp2.json()
            assert d1["transaction_id"] == d2["transaction_id"]
            assert d1["remaining_balance"] == d2["remaining_balance"]

    @pytest.mark.asyncio
    async def test_spend_no_auth(self, wallet_app):
        app, _, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/spend", json={"amount": 1.00})
            assert resp.status_code in (401, 403, 422)

    @pytest.mark.asyncio
    async def test_spend_invalid_key(self, wallet_app):
        app, _, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/spend", json={"amount": 1.00}, headers=_headers("ap_fake_key"))
            assert resp.status_code in (401, 403)


# ═══════════════════════════════════════
# REFUND
# ═══════════════════════════════════════

class TestRefundEndpoint:
    @pytest.mark.asyncio
    async def test_refund_success(self, wallet_app):
        app, key, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First spend
            spend_resp = await client.post("/v1/spend", json={
                "amount": 10.00, "description": "To refund"
            }, headers=_headers(key))
            tx_id = spend_resp.json()["transaction_id"]

            # Then refund
            refund_resp = await client.post("/v1/refund", json={
                "transaction_id": tx_id,
            }, headers=_headers(key))
            assert refund_resp.status_code == 200
            data = refund_resp.json()
            assert data["success"] is True
            assert data["amount_refunded"] == 10.00

    @pytest.mark.asyncio
    async def test_refund_nonexistent(self, wallet_app):
        app, key, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/refund", json={
                "transaction_id": "nonexistent-tx-id",
            }, headers=_headers(key))
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False

    @pytest.mark.asyncio
    async def test_double_refund(self, wallet_app):
        app, key, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            spend_resp = await client.post("/v1/spend", json={
                "amount": 5.00, "description": "Double refund test"
            }, headers=_headers(key))
            tx_id = spend_resp.json()["transaction_id"]

            # First refund — ok
            r1 = await client.post("/v1/refund", json={"transaction_id": tx_id}, headers=_headers(key))
            assert r1.json()["success"] is True

            # Second refund — should fail
            r2 = await client.post("/v1/refund", json={"transaction_id": tx_id}, headers=_headers(key))
            assert r2.json()["success"] is False


# ═══════════════════════════════════════
# TRANSFER
# ═══════════════════════════════════════

class TestTransferEndpoint:
    @pytest.mark.asyncio
    async def test_transfer_success(self, wallet_app):
        app, key, agent, agent2 = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/transfer", json={
                "to_agent_id": agent2.id,
                "amount": 20.00,
                "description": "Team funds",
            }, headers=_headers(key))
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["amount"] == 20.00

    @pytest.mark.asyncio
    async def test_transfer_insufficient_funds(self, wallet_app):
        app, key, _, agent2 = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/transfer", json={
                "to_agent_id": agent2.id,
                "amount": 9999.00,  # more than 500 balance but within 10000 model limit
            }, headers=_headers(key))
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False

    @pytest.mark.asyncio
    async def test_transfer_to_nonexistent_agent(self, wallet_app):
        app, key, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/transfer", json={
                "to_agent_id": "fake-agent-id",
                "amount": 5.00,
            }, headers=_headers(key))
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False


# ═══════════════════════════════════════
# CHAINS
# ═══════════════════════════════════════

class TestChainsEndpoint:
    @pytest.mark.asyncio
    async def test_list_chains(self, wallet_app):
        app, _, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/chains")
            assert resp.status_code == 200
            data = resp.json()
            assert "chains" in data
            chain_ids = [c["id"] for c in data["chains"]]
            assert "base" in chain_ids
            assert "solana" in chain_ids


# ═══════════════════════════════════════
# WEBHOOK
# ═══════════════════════════════════════

class TestWebhookEndpoints:
    @pytest.mark.asyncio
    async def test_set_webhook(self, wallet_app):
        app, key, agent, _ = wallet_app
        transport = ASGITransport(app=app)
        mock_secret = "whsec_testsecret1234567890"
        with patch("core.webhooks.register_webhook", new_callable=AsyncMock), \
             patch("core.webhooks.generate_webhook_secret", return_value=mock_secret):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/webhook", json={
                    "url": "https://example.com/webhook",
                    "events": ["spend", "deposit"],
                }, headers=_headers(key))
                assert resp.status_code == 200
                data = resp.json()
                assert data["url"] == "https://example.com/webhook"
                assert data["secret"] == mock_secret
                assert "spend" in data["events"]

    @pytest.mark.asyncio
    async def test_get_webhook(self, wallet_app):
        app, key, agent, _ = wallet_app
        transport = ASGITransport(app=app)
        mock_config = {
            "url": "https://example.com/hook",
            "secret": "whsec_abcdef12345678",
            "events": ["all"],
        }
        with patch("core.webhooks.get_webhook_config", return_value=mock_config):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/webhook", headers=_headers(key))
                assert resp.status_code == 200
                data = resp.json()
                assert data["url"] == "https://example.com/hook"
                assert "****" in data["secret"]

    @pytest.mark.asyncio
    async def test_delete_webhook(self, wallet_app):
        app, key, _, _ = wallet_app
        transport = ASGITransport(app=app)
        with patch("core.webhooks.unregister_webhook", new_callable=AsyncMock):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.delete("/v1/webhook", headers=_headers(key))
                assert resp.status_code == 200
                assert resp.json()["success"] is True

    @pytest.mark.asyncio
    async def test_get_webhook_empty(self, wallet_app):
        app, key, _, _ = wallet_app
        transport = ASGITransport(app=app)
        with patch("core.webhooks.get_webhook_config", return_value=None):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/webhook", headers=_headers(key))
                assert resp.status_code == 200
                data = resp.json()
                assert data["url"] is None


# ═══════════════════════════════════════
# CARD
# ═══════════════════════════════════════

class TestCardEndpoints:
    @pytest.mark.asyncio
    async def test_card_info_no_card(self, wallet_app):
        app, key, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/card", headers=_headers(key))
            assert resp.status_code == 200
            data = resp.json()
            assert data["last4"] is None

    @pytest.mark.asyncio
    async def test_card_transactions_empty(self, wallet_app):
        app, key, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/card/transactions", headers=_headers(key))
            assert resp.status_code == 200
            assert resp.json() == []


# ═══════════════════════════════════════
# x402
# ═══════════════════════════════════════

class TestX402Endpoints:
    @pytest.mark.asyncio
    async def test_x402_probe_invalid_url(self, wallet_app):
        app, _, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/x402/probe", params={"url": "not-a-url"})
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_x402_probe_valid_url(self, wallet_app):
        app, _, _, _ = wallet_app
        transport = ASGITransport(app=app)

        mock_result = {
            "gated": False,
            "url": "https://example.com/resource",
            "price_usd": 0,
        }
        with patch("providers.x402_protocol.estimate_x402_cost", return_value=mock_result):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/x402/probe", params={"url": "https://example.com/resource"})
                assert resp.status_code == 200


# ═══════════════════════════════════════
# BALANCE (via /v1/balance)
# ═══════════════════════════════════════

class TestBalanceEndpoint:
    @pytest.mark.asyncio
    async def test_balance(self, wallet_app):
        app, key, _, _ = wallet_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/balance", headers=_headers(key))
            assert resp.status_code == 200
            data = resp.json()
            assert "balance" in data or "balance_usd" in data
