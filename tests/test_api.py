"""
Integration tests for API endpoints using FastAPI TestClient.

These test the full HTTP request/response cycle with mocked DB sessions.
"""
import pytest
import pytest_asyncio
import os
import sys
from decimal import Decimal
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must set env vars before importing app
os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"
os.environ["BOT_TOKEN"] = ""
os.environ["API_SECRET"] = "test-secret-key-for-tests"

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from models.database import Base
from models.schema import User, Agent
from core.wallet import hash_api_key


@pytest_asyncio.fixture
async def test_app():
    """Create a test app with an in-memory SQLite database."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Override the get_db dependency
    async def override_get_db():
        async with session_factory() as session:
            yield session

    from api.main import app
    from models.database import get_db
    app.dependency_overrides[get_db] = override_get_db

    # Create test data
    async with session_factory() as db:
        user = User(telegram_id=12345678, username="testuser", first_name="Test")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        api_key = "ap_test_integration_key_1234567890abcdef12345678"
        agent = Agent(
            user_id=user.id,
            name="integration-agent",
            api_key_hash=hash_api_key(api_key),
            api_key_prefix="ap_test_...",
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

    # Cleanup
    app.dependency_overrides.clear()
    await engine.dispose()


# ═══════════════════════════════════════
# Health Check
# ═══════════════════════════════════════

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, test_app):
        app, _, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["service"] == "agentpay"
            assert data["version"] == "0.1.0"


# ═══════════════════════════════════════
# Balance Endpoint
# ═══════════════════════════════════════

class TestBalanceEndpoint:
    @pytest.mark.asyncio
    async def test_get_balance_valid_key(self, test_app):
        app, api_key, agent = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/balance", headers={"X-API-Key": api_key})
            assert resp.status_code == 200
            data = resp.json()
            assert data["agent_name"] == "integration-agent"
            assert data["balance_usd"] == 100.0
            assert data["is_active"] is True

    @pytest.mark.asyncio
    async def test_get_balance_invalid_key(self, test_app):
        app, _, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/balance", headers={"X-API-Key": "ap_invalid_key"})
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_get_balance_no_key(self, test_app):
        app, _, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/balance")
            assert resp.status_code == 422  # Missing required header


# ═══════════════════════════════════════
# Spend Endpoint
# ═══════════════════════════════════════

class TestSpendEndpoint:
    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_spend_success(self, mock_wh, mock_notify, test_app):
        app, api_key, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/spend",
                json={"amount": 5.0, "description": "API test spend", "skip_approval": True},
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["amount"] == 5.0

    @pytest.mark.asyncio
    async def test_spend_invalid_amount(self, test_app):
        app, api_key, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/spend",
                json={"amount": -1, "description": "negative"},
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_spend_over_max(self, test_app):
        app, api_key, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/spend",
                json={"amount": 10001, "description": "too much"},
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 422


# ═══════════════════════════════════════
# Transactions Endpoint
# ═══════════════════════════════════════

class TestTransactionsEndpoint:
    @pytest.mark.asyncio
    async def test_list_transactions_empty(self, test_app):
        app, api_key, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/transactions", headers={"X-API-Key": api_key})
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)


# ═══════════════════════════════════════
# Chains Endpoint (Public)
# ═══════════════════════════════════════

class TestChainsEndpoint:
    @pytest.mark.asyncio
    async def test_list_chains(self, test_app):
        app, _, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/chains")
            assert resp.status_code == 200
            data = resp.json()
            assert "chains" in data
            chains = data["chains"]
            assert len(chains) > 0
            # Check Solana is in the list
            chain_ids = [c["id"] for c in chains]
            assert "solana" in chain_ids


# ═══════════════════════════════════════
# Security Headers
# ═══════════════════════════════════════

class TestSecurityHeaders:
    @pytest.mark.asyncio
    async def test_security_headers_present(self, test_app):
        app, _, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/health")
            assert resp.headers.get("x-content-type-options") == "nosniff"
            assert resp.headers.get("x-frame-options") == "DENY"
            assert "max-age" in resp.headers.get("strict-transport-security", "")
            assert resp.headers.get("x-xss-protection") == "1; mode=block"

    @pytest.mark.asyncio
    async def test_cors_headers(self, test_app):
        app, _, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options(
                "/v1/health",
                headers={
                    "Origin": "https://leofundmybot.dev",
                    "Access-Control-Request-Method": "GET",
                },
            )
            # CORS should allow this origin
            assert resp.status_code == 200


# ═══════════════════════════════════════
# Telegram Auth Endpoint (dev mode)
# ═══════════════════════════════════════

class TestTelegramAuth:
    @pytest.mark.asyncio
    async def test_auth_dev_mode(self, test_app):
        """Without BOT_TOKEN, auth should accept in dev mode."""
        app, _, _ = test_app
        transport = ASGITransport(app=app)
        with patch("api.routes.miniapp.BOT_TOKEN", ""):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/auth/telegram",
                    json={"init_data": "user=%7B%22id%22%3A12345%7D&auth_date=9999999999&hash=abc"},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert "token" in data
                assert data["telegram_id"] == 12345


# ═══════════════════════════════════════
# Export Endpoint
# ═══════════════════════════════════════

class TestExportEndpoint:
    @pytest.mark.asyncio
    async def test_export_csv(self, test_app):
        app, api_key, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/export", headers={"X-API-Key": api_key})
            assert resp.status_code == 200
            assert "text/csv" in resp.headers.get("content-type", "")
            # Check CSV has header row
            lines = resp.text.strip().split("\n")
            assert lines[0] == "date,type,amount,fee,description,status,id"


# ═══════════════════════════════════════
# Directory Endpoint (Public)
# ═══════════════════════════════════════

class TestDirectoryEndpoint:
    @pytest.mark.asyncio
    async def test_directory_empty(self, test_app):
        app, _, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/directory")
            assert resp.status_code == 200
            data = resp.json()
            assert "agents" in data
            assert "total" in data
            assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_directory_invalid_agent(self, test_app):
        app, _, _ = test_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/directory/nonexistent-id")
            assert resp.status_code == 404
