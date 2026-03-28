"""
Tests for wallet API routes (spend, refund, transfer, chains, card, webhook, approvals, x402).
"""
import os
import sys
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"
os.environ["BOT_TOKEN"] = ""
os.environ["API_SECRET"] = "test-secret-key-for-tests-minimum-32-bytes-long"

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from models.database import Base, get_db
from models.schema import User, Agent
from core.wallet import hash_api_key


@pytest_asyncio.fixture
async def setup():
    """Create test app with seeded agents, overriding DB dependency."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with factory() as session:
            yield session

    from api.main import app
    app.dependency_overrides[get_db] = override_get_db

    api_key = "ap_test_key_1234567890abcdef1234567890abcdef"
    async with factory() as db:
        user = User(telegram_id=12345678, username="testuser", first_name="Test")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        agent = Agent(
            user_id=user.id, name="test-agent",
            api_key_hash=hash_api_key(api_key), api_key_prefix="ap_test_...",
            balance_usd=Decimal("100.0000"), daily_limit_usd=Decimal("50.0000"),
            tx_limit_usd=Decimal("25.0000"), auto_approve_usd=Decimal("10.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)
        agent_id = agent.id

        agent2 = Agent(
            user_id=user.id, name="second-agent",
            api_key_hash=hash_api_key("ap_second_key_abcdef1234567890abcdef1234567890"),
            api_key_prefix="ap_secon...",
            balance_usd=Decimal("50.0000"), daily_limit_usd=Decimal("50.0000"),
            tx_limit_usd=Decimal("25.0000"), is_active=True,
        )
        db.add(agent2)
        await db.commit()
        await db.refresh(agent2)
        agent2_id = agent2.id

    yield app, api_key, agent_id, agent2_id

    app.dependency_overrides.clear()
    await engine.dispose()


def _h(key: str) -> dict:
    return {"X-API-Key": key}


# ── Chains ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_chains(setup):
    app, *_ = setup
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/chains")
    assert r.status_code == 200
    chains = r.json()["chains"]
    ids = [ch["id"] for ch in chains]
    assert "solana" in ids
    assert any(ch["type"] == "evm" for ch in chains)


# ── Spend ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_spend_success(setup):
    app, key, *_ = setup
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/spend", json={"amount": 5.0, "description": "test", "skip_approval": True}, headers=_h(key))
    assert r.status_code == 200
    d = r.json()
    assert d["success"] is True
    assert d["amount"] == 5.0


@pytest.mark.asyncio
async def test_spend_no_auth(setup):
    app, *_ = setup
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/spend", json={"amount": 5.0})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_spend_bad_key(setup):
    app, *_ = setup
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/spend", json={"amount": 5.0, "description": "x"}, headers=_h("ap_bad_00000000000000000000000000000000000"))
    assert r.status_code == 401


# ── Refund ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_refund_nonexistent(setup):
    app, key, *_ = setup
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/refund", json={"transaction_id": "no-such-tx"}, headers=_h(key))
    assert r.status_code == 200
    assert r.json()["success"] is False


# ── Transfer ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transfer(setup):
    app, key, _, a2 = setup
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/transfer", json={"to_agent_id": a2, "amount": 10.0, "description": "xfer"}, headers=_h(key))
    assert r.status_code == 200
    d = r.json()
    assert d["success"] is True
    assert d["amount"] == 10.0


# ── Card ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_card_none(setup):
    app, key, *_ = setup
    with patch("providers.lithic_card.get_card_details", return_value=None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get("/v1/card", headers=_h(key))
    assert r.status_code == 200
    assert r.json()["last4"] is None


@pytest.mark.asyncio
async def test_card_transactions_empty(setup):
    app, key, *_ = setup
    with patch("providers.lithic_card.get_card_transactions", return_value=[]):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get("/v1/card/transactions", headers=_h(key))
    assert r.status_code == 200
    assert r.json() == []


# ── Webhook ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_webhook_get_none(setup):
    app, key, *_ = setup
    with patch("core.webhooks.get_webhook_config", return_value=None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get("/v1/webhook", headers=_h(key))
    assert r.status_code == 200
    assert r.json()["url"] is None


@pytest.mark.asyncio
async def test_webhook_set(setup):
    app, key, *_ = setup
    with patch("core.webhooks.register_webhook", new_callable=AsyncMock), \
         patch("core.webhooks.generate_webhook_secret", return_value="whsec_abc"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/v1/webhook", json={"url": "https://example.com/hook", "events": ["spend"]}, headers=_h(key))
    assert r.status_code == 200
    assert r.json()["url"] == "https://example.com/hook"


@pytest.mark.asyncio
async def test_webhook_delete(setup):
    app, key, *_ = setup
    with patch("core.webhooks.unregister_webhook", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.delete("/v1/webhook", headers=_h(key))
    assert r.status_code == 200
    assert r.json()["success"] is True


# ── Approvals ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_approval_not_found(setup):
    app, key, *_ = setup
    with patch("core.approvals.get_pending", return_value=None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get("/v1/approvals/fake-id", headers=_h(key))
    assert r.status_code == 404


# ── x402 probe ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_x402_probe_bad_url(setup):
    app, *_ = setup
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/x402/probe", params={"url": "not-a-url"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_x402_probe_ok(setup):
    app, *_ = setup
    with patch("providers.x402_protocol.estimate_x402_cost", return_value={"gated": True, "price_usd": "0.01", "network": "base"}):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.get("/v1/x402/probe", params={"url": "https://example.com/res"})
    assert r.status_code == 200
    assert r.json()["gated"] is True
