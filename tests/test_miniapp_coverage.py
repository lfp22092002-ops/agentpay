"""
Additional miniapp tests to boost coverage from 41% toward 80%+.

Covers:
- miniapp_list_agents (lines 176-196)
- miniapp_agent_transactions with filters (lines 216-237)
- miniapp_update_agent_settings (lines 268-291)
- miniapp_agent_card (lines 308-314)
- miniapp_toggle_card (lines 335-346)
- miniapp_agent_wallet (lines 364-401)
- miniapp_agent_wallet_all (lines 426-440)
- miniapp_dashboard (lines 460-542)
- miniapp_agent_analytics (lines 572-676)
- miniapp_agent_identity (lines 703-721)
"""
import os
import sys
import time
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.database import Base
from models.schema import (
    User, Agent, Transaction, TransactionType, TransactionStatus,
    AgentIdentity, PlatformRevenue,
)
from core.wallet import hash_api_key

import jwt as pyjwt

MINIAPP_JWT_SECRET = os.environ.get("API_SECRET", "test-secret-key-for-tests")
MINIAPP_JWT_ALGO = "HS256"
ADMIN_TELEGRAM_ID = 5360481016


def _make_jwt(telegram_id: int, first_name: str = "Test", username: str = "tester") -> str:
    payload = {
        "sub": str(telegram_id),
        "telegram_id": telegram_id,
        "first_name": first_name,
        "username": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400,
        "type": "miniapp",
    }
    return pyjwt.encode(payload, MINIAPP_JWT_SECRET, algorithm=MINIAPP_JWT_ALGO)


def _auth(telegram_id: int) -> dict:
    return {"Authorization": f"Bearer {_make_jwt(telegram_id)}"}


@pytest_asyncio.fixture
async def miniapp_env():
    """Full miniapp test env with user, agents, transactions, identity."""
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

    tg_id = 11111111
    api_key = "ap_miniapp_cov_test_1234567890abcdef123"

    async with factory() as db:
        user = User(telegram_id=tg_id, username="miniapper", first_name="MiniApp")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        agent = Agent(
            user_id=user.id,
            name="mini-agent",
            api_key_hash=hash_api_key(api_key),
            api_key_prefix="ap_mini...",
            balance_usd=Decimal("500.0000"),
            daily_limit_usd=Decimal("100.0000"),
            tx_limit_usd=Decimal("50.0000"),
            auto_approve_usd=Decimal("10.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

        # Add some transactions with varied types
        now = datetime.now(timezone.utc)
        for i in range(5):
            tx = Transaction(
                agent_id=agent.id,
                tx_type=TransactionType.SPEND,
                amount_usd=Decimal("10.0000"),
                fee_usd=Decimal("0.2000"),
                description=f"API call {i}",
                status=TransactionStatus.COMPLETED,
                created_at=now - timedelta(hours=i),
            )
            db.add(tx)

        # Add a deposit
        dep = Transaction(
            agent_id=agent.id,
            tx_type=TransactionType.DEPOSIT,
            amount_usd=Decimal("100.0000"),
            fee_usd=Decimal("0.0000"),
            description="Stars deposit",
            status=TransactionStatus.COMPLETED,
            created_at=now - timedelta(days=1),
        )
        db.add(dep)

        # Add a refund
        refund = Transaction(
            agent_id=agent.id,
            tx_type=TransactionType.REFUND,
            amount_usd=Decimal("5.0000"),
            fee_usd=Decimal("0.0000"),
            description="Refund test",
            status=TransactionStatus.COMPLETED,
            created_at=now - timedelta(days=2),
        )
        db.add(refund)

        await db.commit()

        # Add identity
        identity = AgentIdentity(
            agent_id=agent.id,
            display_name="Mini Agent",
            description="Test agent for miniapp",
            category="defi",
            homepage_url="https://example.com",
            logo_url="https://example.com/logo.png",
            first_seen=agent.created_at,
            last_active=now,
        )
        db.add(identity)
        await db.commit()

    yield app, tg_id, agent

    app.dependency_overrides.pop(get_db, None)
    from api.middleware import limiter
    limiter.reset()
    await engine.dispose()


@pytest_asyncio.fixture
async def admin_miniapp_env():
    """Miniapp env with admin user for dashboard platform stats."""
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

    api_key = "ap_admin_miniapp_test_1234567890abcdef"

    async with factory() as db:
        user = User(telegram_id=ADMIN_TELEGRAM_ID, username="admin", first_name="Admin")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        agent = Agent(
            user_id=user.id,
            name="admin-agent",
            api_key_hash=hash_api_key(api_key),
            api_key_prefix="ap_admi...",
            balance_usd=Decimal("1000.0000"),
            daily_limit_usd=Decimal("500.0000"),
            tx_limit_usd=Decimal("100.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

        # Add a transaction first (PlatformRevenue needs transaction_id + agent_id)
        admin_tx = Transaction(
            agent_id=agent.id,
            tx_type=TransactionType.FEE,
            amount_usd=Decimal("25.0000"),
            fee_usd=Decimal("0.0000"),
            description="Platform fees",
            status=TransactionStatus.COMPLETED,
        )
        db.add(admin_tx)
        await db.commit()
        await db.refresh(admin_tx)

        # Add platform revenue record
        rev = PlatformRevenue(
            transaction_id=admin_tx.id,
            agent_id=agent.id,
            amount_usd=Decimal("25.0000"),
        )
        db.add(rev)

        # Add a transaction
        tx = Transaction(
            agent_id=agent.id,
            tx_type=TransactionType.SPEND,
            amount_usd=Decimal("50.0000"),
            fee_usd=Decimal("1.0000"),
            description="Admin spend",
            status=TransactionStatus.COMPLETED,
        )
        db.add(tx)
        await db.commit()

    yield app, ADMIN_TELEGRAM_ID, agent

    app.dependency_overrides.pop(get_db, None)
    from api.middleware import limiter
    limiter.reset()
    await engine.dispose()


# ═══════════════════════════════════════
# LIST AGENTS
# ═══════════════════════════════════════

class TestMiniappListAgents:
    @pytest.mark.asyncio
    async def test_list_agents_success(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/agents", headers=_auth(tg_id))
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["agents"]) == 1
            a = data["agents"][0]
            assert a["name"] == "mini-agent"
            assert a["balance_usd"] == 500.0
            assert a["daily_limit_usd"] == 100.0
            assert "daily_spent_usd" in a
            assert a["tx_limit_usd"] == 50.0
            assert a["auto_approve_usd"] == 10.0
            assert a["is_active"] is True

    @pytest.mark.asyncio
    async def test_list_agents_unknown_user(self, miniapp_env):
        app, _, _ = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/agents", headers=_auth(99999999))
            assert resp.status_code == 200
            assert resp.json()["agents"] == []


# ═══════════════════════════════════════
# AGENT TRANSACTIONS (with filters)
# ═══════════════════════════════════════

class TestMiniappTransactions:
    @pytest.mark.asyncio
    async def test_transactions_list(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent.id}/transactions",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["transactions"]) == 7  # 5 spends + 1 deposit + 1 refund
            tx = data["transactions"][0]
            assert "id" in tx
            assert "type" in tx
            assert "amount" in tx
            assert "created_at" in tx

    @pytest.mark.asyncio
    async def test_transactions_type_filter(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent.id}/transactions?type=deposit",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 200
            data = resp.json()
            # Only deposits
            for tx in data["transactions"]:
                assert tx["type"] == "deposit"

    @pytest.mark.asyncio
    async def test_transactions_date_filter(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent.id}/transactions?date={today}",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_transactions_invalid_date(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent.id}/transactions?date=bad-date",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 200  # Invalid date silently ignored

    @pytest.mark.asyncio
    async def test_transactions_not_found(self, miniapp_env):
        app, tg_id, _ = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/nonexistent/transactions",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 404


# ═══════════════════════════════════════
# UPDATE AGENT SETTINGS
# ═══════════════════════════════════════

class TestMiniappUpdateSettings:
    @pytest.mark.asyncio
    async def test_update_daily_limit(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent.id}/settings",
                headers=_auth(tg_id),
                json={"daily_limit_usd": 200.0},
            )
            assert resp.status_code == 200
            assert resp.json()["success"] is True

    @pytest.mark.asyncio
    async def test_update_tx_limit(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent.id}/settings",
                headers=_auth(tg_id),
                json={"tx_limit_usd": 75.0},
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_update_auto_approve(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent.id}/settings",
                headers=_auth(tg_id),
                json={"auto_approve_usd": 25.0},
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_update_is_active(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent.id}/settings",
                headers=_auth(tg_id),
                json={"is_active": False},
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_update_invalid_daily_limit(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent.id}/settings",
                headers=_auth(tg_id),
                json={"daily_limit_usd": -10.0},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_update_invalid_tx_limit(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent.id}/settings",
                headers=_auth(tg_id),
                json={"tx_limit_usd": -5.0},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_update_invalid_auto_approve(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent.id}/settings",
                headers=_auth(tg_id),
                json={"auto_approve_usd": -1.0},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_update_not_found(self, miniapp_env):
        app, tg_id, _ = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                "/v1/miniapp/agents/nonexistent/settings",
                headers=_auth(tg_id),
                json={"is_active": False},
            )
            assert resp.status_code == 404


# ═══════════════════════════════════════
# CARD ENDPOINTS
# ═══════════════════════════════════════

class TestMiniappCard:
    @pytest.mark.asyncio
    async def test_card_no_card(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("api.routes.miniapp.get_card_details", return_value=None):
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent.id}/card",
                    headers=_auth(tg_id),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["card"] is None
                assert data["transactions"] == []

    @pytest.mark.asyncio
    async def test_card_with_card(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        fake_card = {"last4": "1234", "exp_month": "12", "exp_year": "2028", "state": "OPEN"}
        fake_txns = [{"amount_cents": 500, "merchant": "TestMerchant", "status": "SETTLED"}]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("api.routes.miniapp.get_card_details", return_value=fake_card), \
                 patch("api.routes.miniapp.get_card_transactions", return_value=fake_txns):
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent.id}/card",
                    headers=_auth(tg_id),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["card"]["last4"] == "1234"
                assert len(data["transactions"]) == 1

    @pytest.mark.asyncio
    async def test_card_not_found(self, miniapp_env):
        app, tg_id, _ = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/nonexistent/card",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 404


class TestMiniappToggleCard:
    @pytest.mark.asyncio
    async def test_pause_card(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("providers.lithic_card.update_card_state", return_value={"success": True}):
                resp = await client.post(
                    f"/v1/miniapp/agents/{agent.id}/card/pause",
                    headers=_auth(tg_id),
                )
                assert resp.status_code == 200
                assert resp.json()["action"] == "pause"

    @pytest.mark.asyncio
    async def test_resume_card(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("providers.lithic_card.update_card_state", return_value={"success": True}):
                resp = await client.post(
                    f"/v1/miniapp/agents/{agent.id}/card/resume",
                    headers=_auth(tg_id),
                )
                assert resp.status_code == 200
                assert resp.json()["action"] == "resume"

    @pytest.mark.asyncio
    async def test_invalid_action(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/v1/miniapp/agents/{agent.id}/card/destroy",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_toggle_card_not_found(self, miniapp_env):
        app, tg_id, _ = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/miniapp/agents/nonexistent/card/pause",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_toggle_card_error(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("providers.lithic_card.update_card_state", side_effect=Exception("Card error")):
                resp = await client.post(
                    f"/v1/miniapp/agents/{agent.id}/card/pause",
                    headers=_auth(tg_id),
                )
                assert resp.status_code == 500


# ═══════════════════════════════════════
# WALLET ENDPOINTS
# ═══════════════════════════════════════

class TestMiniappWallet:
    @pytest.mark.asyncio
    async def test_wallet_evm_existing(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        fake_balance = {"network": "base-mainnet", "balance_native": "0.1", "balance_eth": "0.1", "balance_usdc": "50.0"}
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("api.routes.miniapp.get_wallet_address", return_value="0xABC123"), \
                 patch("api.routes.miniapp.get_wallet_balance", return_value=fake_balance):
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent.id}/wallet?chain=base",
                    headers=_auth(tg_id),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["address"] == "0xABC123"
                assert data["chain"] == "base"

    @pytest.mark.asyncio
    async def test_wallet_solana(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        fake_balance = {"network": "solana-devnet", "balance_sol": "1.5", "balance_usdc": "100.0"}
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("api.routes.miniapp.get_solana_wallet_address", return_value="SolAddr123"), \
                 patch("api.routes.miniapp.get_solana_balance", return_value=fake_balance):
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent.id}/wallet?chain=solana",
                    headers=_auth(tg_id),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["chain"] == "solana"
                assert data["address"] == "SolAddr123"

    @pytest.mark.asyncio
    async def test_wallet_solana_create(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        fake_balance = {"network": "solana-devnet", "balance_sol": "0", "balance_usdc": "0"}
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("api.routes.miniapp.get_solana_wallet_address", return_value=None), \
                 patch("api.routes.miniapp.create_solana_wallet", return_value={"address": "NewSolAddr"}), \
                 patch("api.routes.miniapp.get_solana_balance", return_value=fake_balance):
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent.id}/wallet?chain=solana",
                    headers=_auth(tg_id),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["address"] == "NewSolAddr"

    @pytest.mark.asyncio
    async def test_wallet_not_found(self, miniapp_env):
        app, tg_id, _ = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/nonexistent/wallet",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 404


class TestMiniappWalletAll:
    @pytest.mark.asyncio
    async def test_wallet_all_no_wallets(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("api.routes.miniapp.get_wallet_address", return_value=None), \
                 patch("api.routes.miniapp.get_solana_wallet_address", return_value=None):
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent.id}/wallet/all",
                    headers=_auth(tg_id),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["evm_address"] is None
                assert data["solana_address"] is None
                assert data["chains"] == []

    @pytest.mark.asyncio
    async def test_wallet_all_with_wallets(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        fake_evm = {"network": "base", "balance_native": "0.1", "balance_usdc": "50.0"}
        fake_sol = {"network": "solana-devnet", "balance_sol": "1.0", "balance_usdc": "25.0"}
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("api.routes.miniapp.get_wallet_address", return_value="0xABC"), \
                 patch("api.routes.miniapp.get_solana_wallet_address", return_value="SolABC"), \
                 patch("api.routes.miniapp.get_wallet_balance", return_value=fake_evm), \
                 patch("api.routes.miniapp.get_solana_balance", return_value=fake_sol):
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent.id}/wallet/all",
                    headers=_auth(tg_id),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["evm_address"] == "0xABC"
                assert data["solana_address"] == "SolABC"
                assert len(data["chains"]) > 0

    @pytest.mark.asyncio
    async def test_wallet_all_not_found(self, miniapp_env):
        app, tg_id, _ = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/nonexistent/wallet/all",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 404


# ═══════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════

class TestMiniappDashboard:
    @pytest.mark.asyncio
    async def test_dashboard_with_data(self, miniapp_env):
        app, tg_id, _ = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/dashboard", headers=_auth(tg_id))
            assert resp.status_code == 200
            data = resp.json()
            assert data["agent_count"] == 1
            assert data["total_balance_usd"] == 500.0
            assert data["total_transactions"] == 7
            assert data["total_volume_usd"] > 0
            assert len(data["recent_transactions"]) <= 5
            assert len(data["agents"]) == 1
            agent_info = data["agents"][0]
            assert "name" in agent_info
            assert "balance_usd" in agent_info
            assert "tx_count" in agent_info
            assert "is_active" in agent_info

    @pytest.mark.asyncio
    async def test_dashboard_no_user(self, miniapp_env):
        app, _, _ = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/dashboard", headers=_auth(99999999))
            assert resp.status_code == 200
            data = resp.json()
            assert data["agent_count"] == 0
            assert data["total_balance_usd"] == 0
            assert data["agents"] == []

    @pytest.mark.asyncio
    async def test_dashboard_admin_platform_stats(self, admin_miniapp_env):
        app, tg_id, _ = admin_miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/dashboard", headers=_auth(tg_id))
            assert resp.status_code == 200
            data = resp.json()
            assert data["platform_stats"] is not None
            assert "total_users" in data["platform_stats"]
            assert "total_agents" in data["platform_stats"]
            assert "total_revenue_usd" in data["platform_stats"]


# ═══════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════

class TestMiniappAnalytics:
    @pytest.mark.asyncio
    async def test_analytics_not_found(self, miniapp_env):
        """Agent not found returns 404."""
        app, tg_id, _ = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/nonexistent/analytics",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 404


# ═══════════════════════════════════════
# IDENTITY
# ═══════════════════════════════════════

class TestMiniappIdentity:
    @pytest.mark.asyncio
    async def test_identity_with_data(self, miniapp_env):
        app, tg_id, agent = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent.id}/identity",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["identity"] is not None
            assert data["identity"]["display_name"] == "Mini Agent"
            assert "trust_score_breakdown" in data

    @pytest.mark.asyncio
    async def test_identity_no_identity(self, admin_miniapp_env):
        """Agent without identity returns null."""
        app, tg_id, agent = admin_miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent.id}/identity",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["identity"] is None

    @pytest.mark.asyncio
    async def test_identity_not_found(self, miniapp_env):
        app, tg_id, _ = miniapp_env
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/nonexistent/identity",
                headers=_auth(tg_id),
            )
            assert resp.status_code == 404
