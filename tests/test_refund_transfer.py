"""
Integration tests for refund and transfer API endpoints.

Covers:
- Successful refund of a completed spend
- Refund of non-existent transaction
- Refund of already-refunded transaction
- Successful transfer between agents (same owner)
- Transfer to non-existent agent
- Transfer with insufficient balance
- Transfer between different users (should fail)
"""
import pytest
import pytest_asyncio
import os
import sys
from decimal import Decimal
from unittest.mock import patch, AsyncMock

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite://")
os.environ.setdefault("BOT_TOKEN", "")
os.environ.setdefault("API_SECRET", "test-secret-key-for-tests-minimum-32-bytes-long")

from models.database import Base
from models.schema import User, Agent
from core.wallet import hash_api_key


@pytest_asyncio.fixture
async def app_with_two_agents():
    """App with two agents belonging to the same user + one agent from another user."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with sf() as session:
            yield session

    from api.main import app
    from models.database import get_db
    app.dependency_overrides[get_db] = override_get_db

    key_a = "ap_refund_test_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    key_b = "ap_refund_test_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    key_other = "ap_refund_test_cccccccccccccccccccccccccccccccccc"

    async with sf() as db:
        user = User(telegram_id=11111111, username="refunduser", first_name="Refund")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        agent_a = Agent(
            user_id=user.id,
            name="agent-alpha",
            api_key_hash=hash_api_key(key_a),
            api_key_prefix="ap_refun...",
            balance_usd=Decimal("100.0000"),
            daily_limit_usd=Decimal("500.0000"),
            tx_limit_usd=Decimal("200.0000"),
            auto_approve_usd=Decimal("200.0000"),
            is_active=True,
        )
        db.add(agent_a)
        await db.commit()
        await db.refresh(agent_a)

        agent_b = Agent(
            user_id=user.id,
            name="agent-beta",
            api_key_hash=hash_api_key(key_b),
            api_key_prefix="ap_refun...",
            balance_usd=Decimal("50.0000"),
            daily_limit_usd=Decimal("500.0000"),
            tx_limit_usd=Decimal("200.0000"),
            auto_approve_usd=Decimal("200.0000"),
            is_active=True,
        )
        db.add(agent_b)
        await db.commit()
        await db.refresh(agent_b)

        other_user = User(telegram_id=22222222, username="otheruser", first_name="Other")
        db.add(other_user)
        await db.commit()
        await db.refresh(other_user)

        agent_other = Agent(
            user_id=other_user.id,
            name="agent-other",
            api_key_hash=hash_api_key(key_other),
            api_key_prefix="ap_refun...",
            balance_usd=Decimal("200.0000"),
            daily_limit_usd=Decimal("500.0000"),
            tx_limit_usd=Decimal("200.0000"),
            auto_approve_usd=Decimal("200.0000"),
            is_active=True,
        )
        db.add(agent_other)
        await db.commit()
        await db.refresh(agent_other)

    yield app, key_a, key_b, key_other, agent_a, agent_b, agent_other

    app.dependency_overrides.clear()
    await engine.dispose()


# ═══════════════════════════════════════
# Refund Tests
# ═══════════════════════════════════════

class TestRefundEndpoint:
    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_refund_success(self, mock_wh, mock_notify, app_with_two_agents):
        app, key_a, *_ = app_with_two_agents
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First spend
            spend_resp = await client.post(
                "/v1/spend",
                json={"amount": 10.0, "description": "to be refunded"},
                headers={"X-API-Key": key_a},
            )
            assert spend_resp.status_code == 200
            tx_id = spend_resp.json()["transaction_id"]

            # Refund it
            refund_resp = await client.post(
                "/v1/refund",
                json={"transaction_id": tx_id},
                headers={"X-API-Key": key_a},
            )
            assert refund_resp.status_code == 200
            data = refund_resp.json()
            assert data["success"] is True
            assert data["amount_refunded"] > 0

    @pytest.mark.asyncio
    async def test_refund_nonexistent_tx(self, app_with_two_agents):
        app, key_a, *_ = app_with_two_agents
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/refund",
                json={"transaction_id": "nonexistent-tx-id"},
                headers={"X-API-Key": key_a},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False
            assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_refund_already_refunded(self, mock_wh, mock_notify, app_with_two_agents):
        app, key_a, *_ = app_with_two_agents
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Spend
            spend_resp = await client.post(
                "/v1/spend",
                json={"amount": 5.0, "description": "double refund test"},
                headers={"X-API-Key": key_a},
            )
            tx_id = spend_resp.json()["transaction_id"]

            # First refund — should succeed
            r1 = await client.post(
                "/v1/refund",
                json={"transaction_id": tx_id},
                headers={"X-API-Key": key_a},
            )
            assert r1.json()["success"] is True

            # Second refund — should fail
            r2 = await client.post(
                "/v1/refund",
                json={"transaction_id": tx_id},
                headers={"X-API-Key": key_a},
            )
            assert r2.json()["success"] is False
            assert "already refunded" in r2.json()["error"].lower()


# ═══════════════════════════════════════
# Transfer Tests
# ═══════════════════════════════════════

class TestTransferEndpoint:
    @pytest.mark.asyncio
    async def test_transfer_success(self, app_with_two_agents):
        app, key_a, key_b, _, agent_a, agent_b, _ = app_with_two_agents
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/transfer",
                json={
                    "to_agent_id": agent_b.id,
                    "amount": 25.0,
                    "description": "split costs",
                },
                headers={"X-API-Key": key_a},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["amount"] == 25.0

    @pytest.mark.asyncio
    async def test_transfer_nonexistent_target(self, app_with_two_agents):
        app, key_a, *_ = app_with_two_agents
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/transfer",
                json={"to_agent_id": "nonexistent-id", "amount": 5.0},
                headers={"X-API-Key": key_a},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False
            assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_transfer_insufficient_balance(self, app_with_two_agents):
        app, key_a, key_b, _, agent_a, agent_b, _ = app_with_two_agents
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Use an amount within validation range but above agent balance ($100)
            resp = await client.post(
                "/v1/transfer",
                json={"to_agent_id": agent_b.id, "amount": 9999.0},
                headers={"X-API-Key": key_a},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False
            assert "insufficient" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_transfer_cross_user_denied(self, app_with_two_agents):
        app, key_a, _, _, _, _, agent_other = app_with_two_agents
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/transfer",
                json={"to_agent_id": agent_other.id, "amount": 5.0},
                headers={"X-API-Key": key_a},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False
            assert "own agents" in data["error"].lower()
