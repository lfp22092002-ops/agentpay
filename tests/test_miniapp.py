"""
Tests for Mini App endpoints (api/routes/miniapp.py).

Covers auth, agent listing, transactions, settings, wallet,
dashboard, analytics, identity, and card endpoints.
"""
import os
import sys
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.database import Base
from models.schema import User, Agent, Transaction, TransactionType, TransactionStatus, PaymentMethod, AgentIdentity
from core.wallet import hash_api_key

import jwt as pyjwt


# ═══════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════

@pytest_asyncio.fixture
async def miniapp_setup():
    """Create a test app with JWT-authenticated user and funded agent."""
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

    # Create user + agent
    async with factory() as db:
        user = User(telegram_id=12345, username="testuser", first_name="Test")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        key = "ap_miniapp_test_key_1234567890abcdef1234"
        agent = Agent(
            user_id=user.id,
            name="mini-agent",
            api_key_hash=hash_api_key(key),
            api_key_prefix="ap_mini...",
            balance_usd=Decimal("100.0000"),
            daily_limit_usd=Decimal("50.0000"),
            tx_limit_usd=Decimal("25.0000"),
            auto_approve_usd=Decimal("10.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

    # Create JWT token
    from config.settings import API_SECRET
    token = pyjwt.encode(
        {
            "telegram_id": 12345,
            "user": {"id": 12345, "first_name": "Test", "username": "testuser"},
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
        },
        API_SECRET,
        algorithm="HS256",
    )

    yield app, token, agent.id, factory

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest_asyncio.fixture
async def admin_miniapp_setup(miniapp_setup):
    """Same as miniapp_setup but with admin (G's telegram_id)."""
    app, _, agent_id, factory = miniapp_setup

    async with factory() as db:
        admin_user = User(telegram_id=5360481016, username="admin", first_name="Admin")
        db.add(admin_user)
        await db.commit()
        await db.refresh(admin_user)

        key = "ap_admin_test_key_abcdef1234567890abcdef"
        admin_agent = Agent(
            user_id=admin_user.id,
            name="admin-agent",
            api_key_hash=hash_api_key(key),
            api_key_prefix="ap_admi...",
            balance_usd=Decimal("500.0000"),
            is_active=True,
        )
        db.add(admin_agent)
        await db.commit()
        await db.refresh(admin_agent)

    from config.settings import API_SECRET
    admin_token = pyjwt.encode(
        {
            "telegram_id": 5360481016,
            "user": {"id": 5360481016, "first_name": "Admin"},
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
        },
        API_SECRET,
        algorithm="HS256",
    )

    return app, admin_token, admin_agent.id, factory


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ═══════════════════════════════════════
# AUTH
# ═══════════════════════════════════════

class TestMiniAppAuth:
    @pytest.mark.asyncio
    async def test_auth_no_header(self, miniapp_setup):
        app, _, _, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/agents")
            assert resp.status_code == 422 or resp.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_invalid_token(self, miniapp_setup):
        app, _, _, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents",
                headers={"Authorization": "Bearer invalid.token.here"},
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_valid_token(self, miniapp_setup):
        app, token, _, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/agents", headers=auth_headers(token))
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_telegram_auth_dev_mode(self, miniapp_setup):
        """With empty BOT_TOKEN, should accept init_data in dev mode."""
        app, _, _, _ = miniapp_setup
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

    @pytest.mark.asyncio
    async def test_telegram_auth_dev_mode_no_user(self, miniapp_setup):
        """Dev mode with empty init_data falls back to dev user."""
        app, _, _, _ = miniapp_setup
        transport = ASGITransport(app=app)
        with patch("api.routes.miniapp.BOT_TOKEN", ""):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/auth/telegram",
                    json={"init_data": ""},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["telegram_id"] == 0  # fallback dev user


# ═══════════════════════════════════════
# AGENT LISTING
# ═══════════════════════════════════════

class TestMiniAppAgents:
    @pytest.mark.asyncio
    async def test_list_agents(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/agents", headers=auth_headers(token))
            assert resp.status_code == 200
            data = resp.json()
            assert "agents" in data
            assert len(data["agents"]) == 1
            assert data["agents"][0]["name"] == "mini-agent"
            assert data["agents"][0]["balance_usd"] == 100.0

    @pytest.mark.asyncio
    async def test_list_agents_unknown_user(self, miniapp_setup):
        """User with no agents in DB gets empty list."""
        app, _, _, _ = miniapp_setup
        from config.settings import API_SECRET
        token = pyjwt.encode(
            {
                "telegram_id": 99999,
                "user": {"id": 99999},
                "exp": datetime.now(timezone.utc) + timedelta(hours=24),
            },
            API_SECRET,
            algorithm="HS256",
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/agents", headers=auth_headers(token))
            assert resp.status_code == 200
            assert resp.json()["agents"] == []


# ═══════════════════════════════════════
# TRANSACTIONS
# ═══════════════════════════════════════

class TestMiniAppTransactions:
    @pytest.mark.asyncio
    async def test_transactions_empty(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/transactions",
                headers=auth_headers(token),
            )
            assert resp.status_code == 200
            assert resp.json()["transactions"] == []

    @pytest.mark.asyncio
    async def test_transactions_with_data(self, miniapp_setup):
        app, token, agent_id, factory = miniapp_setup
        # Add a transaction
        async with factory() as db:
            tx = Transaction(
                agent_id=agent_id,
                tx_type=TransactionType.SPEND,
                amount_usd=Decimal("5.00"),
                fee_usd=Decimal("0.10"),
                description="Test spend",
                status=TransactionStatus.COMPLETED,
                payment_method=PaymentMethod.MANUAL,
            )
            db.add(tx)
            await db.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/transactions",
                headers=auth_headers(token),
            )
            assert resp.status_code == 200
            txs = resp.json()["transactions"]
            assert len(txs) == 1
            assert txs[0]["amount"] == 5.0
            assert txs[0]["type"] == "spend"

    @pytest.mark.asyncio
    async def test_transactions_not_found(self, miniapp_setup):
        app, token, _, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/nonexistent-id/transactions",
                headers=auth_headers(token),
            )
            assert resp.status_code == 404


# ═══════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════

class TestMiniAppSettings:
    @pytest.mark.asyncio
    async def test_update_daily_limit(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent_id}/settings",
                json={"daily_limit_usd": 200.0},
                headers=auth_headers(token),
            )
            assert resp.status_code == 200
            assert resp.json()["success"] is True

    @pytest.mark.asyncio
    async def test_update_tx_limit(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent_id}/settings",
                json={"tx_limit_usd": 50.0},
                headers=auth_headers(token),
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_update_auto_approve(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent_id}/settings",
                json={"auto_approve_usd": 25.0},
                headers=auth_headers(token),
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_toggle_active(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent_id}/settings",
                json={"is_active": False},
                headers=auth_headers(token),
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_daily_limit(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent_id}/settings",
                json={"daily_limit_usd": -10.0},
                headers=auth_headers(token),
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_settings_agent_not_found(self, miniapp_setup):
        app, token, _, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                "/v1/miniapp/agents/nonexistent/settings",
                json={"daily_limit_usd": 100.0},
                headers=auth_headers(token),
            )
            assert resp.status_code == 404


# ═══════════════════════════════════════
# CARD
# ═══════════════════════════════════════

class TestMiniAppCard:
    @pytest.mark.asyncio
    async def test_card_no_card(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        with patch("api.routes.miniapp.get_card_details", return_value=None), \
             patch("api.routes.miniapp.get_card_transactions", return_value=[]):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent_id}/card",
                    headers=auth_headers(token),
                )
                assert resp.status_code == 200
                assert resp.json()["card"] is None

    @pytest.mark.asyncio
    async def test_card_with_details(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        card_mock = {"last4": "1234", "state": "OPEN", "exp_month": 12, "exp_year": 2027, "spend_limit_cents": 5000}
        transport = ASGITransport(app=app)
        with patch("api.routes.miniapp.get_card_details", return_value=card_mock), \
             patch("api.routes.miniapp.get_card_transactions", return_value=[]):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent_id}/card",
                    headers=auth_headers(token),
                )
                assert resp.status_code == 200
                assert resp.json()["card"]["last4"] == "1234"

    @pytest.mark.asyncio
    async def test_card_toggle_pause(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        with patch("providers.lithic_card.update_card_state"):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    f"/v1/miniapp/agents/{agent_id}/card/pause",
                    headers=auth_headers(token),
                )
                assert resp.status_code == 200
                assert resp.json()["action"] == "pause"

    @pytest.mark.asyncio
    async def test_card_toggle_resume(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        with patch("providers.lithic_card.update_card_state"):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    f"/v1/miniapp/agents/{agent_id}/card/resume",
                    headers=auth_headers(token),
                )
                assert resp.status_code == 200
                assert resp.json()["action"] == "resume"

    @pytest.mark.asyncio
    async def test_card_toggle_invalid_action(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/v1/miniapp/agents/{agent_id}/card/destroy",
                headers=auth_headers(token),
            )
            assert resp.status_code == 400


# ═══════════════════════════════════════
# WALLET
# ═══════════════════════════════════════

class TestMiniAppWallet:
    @pytest.mark.asyncio
    async def test_wallet_evm(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        mock_balance = {
            "network": "base-sepolia",
            "native_token": "ETH",
            "balance_native": "0.001",
            "balance_eth": "0.001",
            "balance_usdc": "10.0",
        }
        with patch("api.routes.miniapp.get_wallet_address", return_value="0xabc123"), \
             patch("api.routes.miniapp.get_wallet_balance", return_value=mock_balance):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent_id}/wallet?chain=base",
                    headers=auth_headers(token),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["address"] == "0xabc123"
                assert data["chain"] == "base"

    @pytest.mark.asyncio
    async def test_wallet_solana(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        mock_balance = {"network": "solana-devnet", "balance_sol": "0.5", "balance_usdc": "25.0"}
        with patch("api.routes.miniapp.get_solana_wallet_address", return_value="SoLaddr123"), \
             patch("api.routes.miniapp.get_solana_balance", return_value=mock_balance):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent_id}/wallet?chain=solana",
                    headers=auth_headers(token),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["chain"] == "solana"
                assert data["address"] == "SoLaddr123"

    @pytest.mark.asyncio
    async def test_wallet_not_found(self, miniapp_setup):
        app, token, _, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/fake-agent/wallet",
                headers=auth_headers(token),
            )
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_wallet_all_chains(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        mock_evm_balance = {"chain": "base", "balance_usdc": "5.0", "balance_native": "0.01"}
        mock_sol_balance = {"chain": "solana", "balance_usdc": "3.0", "balance_sol": "0.1"}
        with patch("api.routes.miniapp.get_wallet_address", return_value="0xabc"), \
             patch("api.routes.miniapp.get_solana_wallet_address", return_value="SoL123"), \
             patch("api.routes.miniapp.get_wallet_balance", return_value=mock_evm_balance), \
             patch("api.routes.miniapp.get_solana_balance", return_value=mock_sol_balance):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent_id}/wallet/all",
                    headers=auth_headers(token),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["evm_address"] == "0xabc"
                assert data["solana_address"] == "SoL123"
                assert len(data["chains"]) > 0


# ═══════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════

class TestMiniAppDashboard:
    @pytest.mark.asyncio
    async def test_dashboard_with_agents(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/dashboard", headers=auth_headers(token))
            assert resp.status_code == 200
            data = resp.json()
            assert data["agent_count"] == 1
            assert data["total_balance_usd"] == 100.0
            assert "agents" in data
            assert "recent_transactions" in data

    @pytest.mark.asyncio
    async def test_dashboard_empty_user(self, miniapp_setup):
        """Unknown user gets empty dashboard."""
        app, _, _, _ = miniapp_setup
        from config.settings import API_SECRET
        token = pyjwt.encode(
            {
                "telegram_id": 88888,
                "user": {"id": 88888},
                "exp": datetime.now(timezone.utc) + timedelta(hours=24),
            },
            API_SECRET,
            algorithm="HS256",
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/dashboard", headers=auth_headers(token))
            assert resp.status_code == 200
            data = resp.json()
            assert data["agent_count"] == 0

    @pytest.mark.asyncio
    async def test_dashboard_admin_sees_platform_stats(self, admin_miniapp_setup):
        app, admin_token, _, _ = admin_miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/dashboard", headers=auth_headers(admin_token))
            assert resp.status_code == 200
            data = resp.json()
            assert data["platform_stats"] is not None
            assert "total_users" in data["platform_stats"]


# ═══════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════

class TestMiniAppAnalytics:
    @pytest.mark.asyncio
    async def test_analytics_empty(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/analytics",
                headers=auth_headers(token),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["daily_volume"]) == 30
            assert len(data["hourly_heatmap"]) == 24
            assert len(data["balance_history"]) == 30

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="SQLite doesn't support cast(DateTime, Date) — works on PostgreSQL in production")
    async def test_analytics_with_transactions(self, miniapp_setup):
        app, token, agent_id, factory = miniapp_setup
        async with factory() as db:
            for i in range(3):
                tx = Transaction(
                    agent_id=agent_id,
                    tx_type=TransactionType.SPEND,
                    amount_usd=Decimal(f"{(i + 1) * 5}.00"),
                    fee_usd=Decimal("0.10"),
                    description=f"Test {i}",
                    status=TransactionStatus.COMPLETED,
                    payment_method=PaymentMethod.MANUAL,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(tx)
            await db.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/analytics",
                headers=auth_headers(token),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert any(d["count"] > 0 for d in data["daily_volume"])

    @pytest.mark.asyncio
    async def test_analytics_not_found(self, miniapp_setup):
        app, token, _, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/nonexistent/analytics",
                headers=auth_headers(token),
            )
            assert resp.status_code == 404


# ═══════════════════════════════════════
# IDENTITY
# ═══════════════════════════════════════

class TestMiniAppIdentity:
    @pytest.mark.asyncio
    async def test_identity_not_set(self, miniapp_setup):
        app, token, agent_id, _ = miniapp_setup
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/identity",
                headers=auth_headers(token),
            )
            assert resp.status_code == 200
            assert resp.json()["identity"] is None

    @pytest.mark.asyncio
    async def test_identity_with_data(self, miniapp_setup):
        app, token, agent_id, factory = miniapp_setup
        async with factory() as db:
            identity = AgentIdentity(
                agent_id=agent_id,
                display_name="Mini Agent",
                description="A test agent",
                category="utility",
                verified=False,
                trust_score=50,
                total_transactions=10,
                total_volume_usd=Decimal("100.00"),
                first_seen=datetime.now(timezone.utc),
                last_active=datetime.now(timezone.utc),
            )
            db.add(identity)
            await db.commit()

        transport = ASGITransport(app=app)
        with patch("api.routes.identity._refresh_identity_counters"):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent_id}/identity",
                    headers=auth_headers(token),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["identity"] is not None
                assert data["identity"]["display_name"] == "Mini Agent"
                assert "trust_score_breakdown" in data
