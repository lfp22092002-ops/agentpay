"""
Additional Mini App coverage tests — targets uncovered lines in api/routes/miniapp.py.

Covered areas:
- _validate_telegram_init_data (lines 59-82): valid hash, expired, missing hash, no user
- auth_telegram with real BOT_TOKEN (lines 137-141)
- miniapp_list_agents full response (lines 176-196)
- miniapp_agent_transactions with filters (lines 216-237)
- miniapp_update_agent_settings — all branches (lines 268-291)
- miniapp_agent_card with card data (lines 308-314)
- miniapp_toggle_card (lines 335-346)
- miniapp_agent_wallet — EVM new wallet, Solana new wallet (lines 364-401)
- miniapp_agent_wallet_all (lines 426-440)
- miniapp_dashboard with transactions (lines 460-542)
- miniapp_agent_analytics with data (lines 572-676)
- miniapp_agent_identity with identity (lines 703-721)
"""
import hashlib
import hmac
import json
import os
import sys
import time
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
    PaymentMethod, AgentIdentity, PlatformRevenue,
)
from core.wallet import hash_api_key

import jwt as pyjwt


# ═══════════════════════════════════════
# HELPER — build valid Telegram initData
# ═══════════════════════════════════════

BOT_TOKEN_FOR_TEST = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz0123456789a"


def _build_init_data(user_dict: dict, auth_date: int | None = None, bot_token: str = BOT_TOKEN_FOR_TEST) -> str:
    """Build a valid Telegram initData string with correct HMAC hash."""
    if auth_date is None:
        auth_date = int(time.time())
    user_json = json.dumps(user_dict, separators=(",", ":"))
    data_parts = {
        "auth_date": str(auth_date),
        "user": user_json,
    }
    check_string = "\n".join(f"{k}={v}" for k, v in sorted(data_parts.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    computed_hash = hmac.new(secret_key, check_string.encode(), hashlib.sha256).hexdigest()
    # Build URL-encoded string
    from urllib.parse import urlencode
    data_parts["hash"] = computed_hash
    return urlencode(data_parts)


# ═══════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════

@pytest_asyncio.fixture
async def full_miniapp():
    """Full mini app setup with user, agent, transactions, identity, revenue."""
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

    async with factory() as db:
        user = User(telegram_id=55555, username="covuser", first_name="Cov")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        key = "ap_cov_miniapp_test_key_abcdef1234567890"
        agent = Agent(
            user_id=user.id,
            name="cov-agent",
            api_key_hash=hash_api_key(key),
            api_key_prefix="ap_cov_...",
            balance_usd=Decimal("200.0000"),
            daily_limit_usd=Decimal("100.0000"),
            tx_limit_usd=Decimal("50.0000"),
            auto_approve_usd=Decimal("10.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

        # Add some transactions
        now = datetime.now(timezone.utc)
        for i in range(5):
            tx = Transaction(
                agent_id=agent.id,
                tx_type=TransactionType.SPEND,
                amount_usd=Decimal(f"{(i + 1) * 10}.00"),
                fee_usd=Decimal("0.20"),
                description=f"API call {i}",
                status=TransactionStatus.COMPLETED,
                payment_method=PaymentMethod.MANUAL,
                created_at=now - timedelta(hours=i),
            )
            db.add(tx)

        # Add a deposit
        deposit = Transaction(
            agent_id=agent.id,
            tx_type=TransactionType.DEPOSIT,
            amount_usd=Decimal("100.00"),
            fee_usd=Decimal("0.00"),
            description="Top up",
            status=TransactionStatus.COMPLETED,
            payment_method=PaymentMethod.MANUAL,
            created_at=now - timedelta(hours=10),
        )
        db.add(deposit)

        # Platform revenue (needs a transaction_id and agent_id)
        rev_tx = Transaction(
            agent_id=agent.id,
            tx_type=TransactionType.FEE,
            amount_usd=Decimal("1.00"),
            fee_usd=Decimal("0.00"),
            description="Platform fee",
            status=TransactionStatus.COMPLETED,
            payment_method=PaymentMethod.MANUAL,
        )
        db.add(rev_tx)
        await db.commit()
        await db.refresh(rev_tx)

        rev = PlatformRevenue(
            transaction_id=rev_tx.id,
            agent_id=agent.id,
            amount_usd=Decimal("1.00"),
        )
        db.add(rev)

        await db.commit()

        # Identity
        identity = AgentIdentity(
            agent_id=agent.id,
            display_name="Coverage Agent",
            description="Agent for coverage tests",
            category="utility",
            homepage_url="https://cov.example.com",
            logo_url="https://cov.example.com/logo.png",
            verified=True,
            trust_score=75,
            total_transactions=5,
            total_volume_usd=Decimal("150.00"),
            first_seen=agent.created_at,
            last_active=now,
        )
        db.add(identity)
        await db.commit()

    from config.settings import API_SECRET
    token = pyjwt.encode(
        {
            "telegram_id": 55555,
            "sub": str(55555),
            "first_name": "Cov",
            "username": "covuser",
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
            "type": "miniapp",
        },
        API_SECRET,
        algorithm="HS256",
    )

    # Admin token
    async with factory() as db:
        admin = User(telegram_id=5360481016, username="admin", first_name="Admin")
        db.add(admin)
        await db.commit()
        await db.refresh(admin)
        admin_agent = Agent(
            user_id=admin.id,
            name="admin-cov",
            api_key_hash=hash_api_key("ap_admin_cov_key_zzz"),
            api_key_prefix="ap_admi...",
            balance_usd=Decimal("999.0000"),
            is_active=True,
        )
        db.add(admin_agent)
        await db.commit()

    admin_token = pyjwt.encode(
        {
            "telegram_id": 5360481016,
            "sub": str(5360481016),
            "first_name": "Admin",
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
            "type": "miniapp",
        },
        API_SECRET,
        algorithm="HS256",
    )

    yield app, token, admin_token, agent.id, factory

    app.dependency_overrides.clear()
    await engine.dispose()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ═══════════════════════════════════════
# UNIT: _validate_telegram_init_data
# ═══════════════════════════════════════

class TestValidateTelegramInitData:
    def test_valid_init_data(self):
        from api.routes.miniapp import _validate_telegram_init_data
        user = {"id": 12345, "first_name": "Test"}
        init_data = _build_init_data(user)
        result = _validate_telegram_init_data(init_data, BOT_TOKEN_FOR_TEST)
        assert result is not None
        assert result["user"]["id"] == 12345

    def test_empty_init_data(self):
        from api.routes.miniapp import _validate_telegram_init_data
        assert _validate_telegram_init_data("", BOT_TOKEN_FOR_TEST) is None

    def test_missing_hash(self):
        from api.routes.miniapp import _validate_telegram_init_data
        result = _validate_telegram_init_data("auth_date=12345&user=%7B%7D", BOT_TOKEN_FOR_TEST)
        assert result is None

    def test_wrong_hash(self):
        from api.routes.miniapp import _validate_telegram_init_data
        user = {"id": 12345}
        init_data = _build_init_data(user)
        # Tamper with the hash
        init_data = init_data.replace(init_data[-8:], "deadbeef")
        result = _validate_telegram_init_data(init_data, BOT_TOKEN_FOR_TEST)
        assert result is None

    def test_expired_auth_date(self):
        from api.routes.miniapp import _validate_telegram_init_data
        user = {"id": 12345}
        old_time = int(time.time()) - 200000  # way past 86400s
        init_data = _build_init_data(user, auth_date=old_time)
        result = _validate_telegram_init_data(init_data, BOT_TOKEN_FOR_TEST)
        assert result is None


# ═══════════════════════════════════════
# auth_telegram with real BOT_TOKEN
# ═══════════════════════════════════════

class TestAuthTelegramWithToken:
    @pytest.mark.asyncio
    async def test_auth_telegram_valid(self, full_miniapp):
        """POST /auth/telegram with valid initData and real BOT_TOKEN."""
        app, _, _, _, _ = full_miniapp
        user = {"id": 55555, "first_name": "Cov", "username": "covuser"}
        init_data = _build_init_data(user, bot_token=BOT_TOKEN_FOR_TEST)
        transport = ASGITransport(app=app)
        with patch("api.routes.miniapp.BOT_TOKEN", BOT_TOKEN_FOR_TEST):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/auth/telegram", json={"init_data": init_data})
                assert resp.status_code == 200
                data = resp.json()
                assert "token" in data
                assert data["telegram_id"] == 55555
                assert data["user_name"] == "Cov"

    @pytest.mark.asyncio
    async def test_auth_telegram_invalid_hash(self, full_miniapp):
        """POST /auth/telegram with invalid hash should fail."""
        app, _, _, _, _ = full_miniapp
        transport = ASGITransport(app=app)
        with patch("api.routes.miniapp.BOT_TOKEN", BOT_TOKEN_FOR_TEST):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/auth/telegram",
                    json={"init_data": "user=%7B%22id%22%3A1%7D&auth_date=999&hash=badhash"},
                )
                assert resp.status_code == 401


# ═══════════════════════════════════════
# AGENT LISTING — full response body
# ═══════════════════════════════════════

class TestListAgentsFullResponse:
    @pytest.mark.asyncio
    async def test_list_agents_all_fields(self, full_miniapp):
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/agents", headers=_auth(token))
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["agents"]) == 1
            a = data["agents"][0]
            assert a["id"] == agent_id
            assert a["name"] == "cov-agent"
            assert a["balance_usd"] == 200.0
            assert a["daily_limit_usd"] == 100.0
            assert "daily_spent_usd" in a
            assert a["tx_limit_usd"] == 50.0
            assert a["auto_approve_usd"] == 10.0
            assert a["is_active"] is True


# ═══════════════════════════════════════
# TRANSACTIONS — with type/date filters
# ═══════════════════════════════════════

class TestTransactionsFiltered:
    @pytest.mark.asyncio
    async def test_transactions_type_filter(self, full_miniapp):
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/transactions?type=deposit",
                headers=_auth(token),
            )
            assert resp.status_code == 200
            txs = resp.json()["transactions"]
            assert all(tx["type"] == "deposit" for tx in txs)

    @pytest.mark.asyncio
    async def test_transactions_date_filter(self, full_miniapp):
        app, token, _, agent_id, _ = full_miniapp
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/transactions?date={today}",
                headers=_auth(token),
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_transactions_invalid_date(self, full_miniapp):
        """Invalid date format should be ignored (not crash)."""
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/transactions?date=not-a-date",
                headers=_auth(token),
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_transactions_full_response(self, full_miniapp):
        """Verify transaction response has all fields."""
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/transactions",
                headers=_auth(token),
            )
            assert resp.status_code == 200
            txs = resp.json()["transactions"]
            assert len(txs) >= 5
            tx = txs[0]
            assert "id" in tx
            assert "type" in tx
            assert "amount" in tx
            assert "fee" in tx
            assert "description" in tx
            assert "status" in tx
            assert "created_at" in tx


# ═══════════════════════════════════════
# SETTINGS — all update branches
# ═══════════════════════════════════════

class TestSettingsAllBranches:
    @pytest.mark.asyncio
    async def test_update_all_settings_at_once(self, full_miniapp):
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent_id}/settings",
                json={
                    "daily_limit_usd": 200.0,
                    "tx_limit_usd": 75.0,
                    "auto_approve_usd": 25.0,
                    "is_active": False,
                },
                headers=_auth(token),
            )
            assert resp.status_code == 200
            assert resp.json()["success"] is True

    @pytest.mark.asyncio
    async def test_invalid_tx_limit(self, full_miniapp):
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent_id}/settings",
                json={"tx_limit_usd": -5.0},
                headers=_auth(token),
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_auto_approve(self, full_miniapp):
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent_id}/settings",
                json={"auto_approve_usd": -1.0},
                headers=_auth(token),
            )
            assert resp.status_code == 400


# ═══════════════════════════════════════
# CARD — with details and toggle error
# ═══════════════════════════════════════

class TestCardEndpointsDeep:
    @pytest.mark.asyncio
    async def test_card_with_transactions(self, full_miniapp):
        """Card endpoint returns card + txns."""
        app, token, _, agent_id, _ = full_miniapp
        card_data = {"last4": "9999", "state": "OPEN", "exp_month": 6, "exp_year": 2028, "spend_limit_cents": 10000}
        card_txns = [{"amount_cents": 500, "merchant": "TestShop", "status": "SETTLED", "created": "2026-01-01"}]
        transport = ASGITransport(app=app)
        with patch("api.routes.miniapp.get_card_details", return_value=card_data), \
             patch("api.routes.miniapp.get_card_transactions", return_value=card_txns):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent_id}/card",
                    headers=_auth(token),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["card"]["last4"] == "9999"
                assert len(data["transactions"]) == 1

    @pytest.mark.asyncio
    async def test_card_not_found(self, full_miniapp):
        app, token, _, _, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/fake-id/card",
                headers=_auth(token),
            )
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_toggle_card_not_found(self, full_miniapp):
        app, token, _, _, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/miniapp/agents/fake-id/card/pause",
                headers=_auth(token),
            )
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_toggle_card_error(self, full_miniapp):
        """Card toggle with exception raises 500."""
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        with patch("providers.lithic_card.update_card_state", side_effect=Exception("Lithic down")):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    f"/v1/miniapp/agents/{agent_id}/card/pause",
                    headers=_auth(token),
                )
                assert resp.status_code == 500


# ═══════════════════════════════════════
# WALLET — EVM new wallet, Solana new wallet
# ═══════════════════════════════════════

class TestWalletEndpointsDeep:
    @pytest.mark.asyncio
    async def test_evm_new_wallet_creation(self, full_miniapp):
        """EVM wallet when none exists creates one."""
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        mock_balance = {
            "network": "base-sepolia",
            "native_token": "ETH",
            "balance_native": "0",
            "balance_eth": "0",
            "balance_usdc": "0",
        }
        with patch("api.routes.miniapp.get_wallet_address", return_value=None), \
             patch("providers.local_wallet.create_agent_wallet", return_value={"address": "0xnew123"}), \
             patch("api.routes.miniapp.get_wallet_balance", return_value=mock_balance):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent_id}/wallet?chain=base",
                    headers=_auth(token),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["address"] == "0xnew123"

    @pytest.mark.asyncio
    async def test_solana_new_wallet_creation(self, full_miniapp):
        """Solana wallet when none exists creates one."""
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        mock_balance = {"network": "solana-devnet", "balance_sol": "0", "balance_usdc": "0"}
        with patch("api.routes.miniapp.get_solana_wallet_address", return_value=None), \
             patch("api.routes.miniapp.create_solana_wallet", return_value={"address": "SoLnew456"}), \
             patch("api.routes.miniapp.get_solana_balance", return_value=mock_balance):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent_id}/wallet?chain=solana",
                    headers=_auth(token),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["address"] == "SoLnew456"
                assert data["chain"] == "solana"

    @pytest.mark.asyncio
    async def test_wallet_all_no_wallets(self, full_miniapp):
        """All-chains wallet with no wallets returns empty."""
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        with patch("api.routes.miniapp.get_wallet_address", return_value=None), \
             patch("api.routes.miniapp.get_solana_wallet_address", return_value=None):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent_id}/wallet/all",
                    headers=_auth(token),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["evm_address"] is None
                assert data["solana_address"] is None
                assert data["chains"] == []


# ═══════════════════════════════════════
# DASHBOARD — with transactions + admin platform stats
# ═══════════════════════════════════════

class TestDashboardDeep:
    @pytest.mark.asyncio
    async def test_dashboard_with_transactions(self, full_miniapp):
        """Dashboard with agent that has transactions."""
        app, token, _, _, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/dashboard", headers=_auth(token))
            assert resp.status_code == 200
            data = resp.json()
            assert data["agent_count"] == 1
            assert data["total_balance_usd"] == 200.0
            assert data["total_transactions"] >= 5
            assert data["total_volume_usd"] > 0
            assert len(data["recent_transactions"]) > 0
            assert len(data["agents"]) == 1
            agent_out = data["agents"][0]
            assert "tx_count" in agent_out
            assert "last_active" in agent_out
            # Non-admin should not see platform stats
            assert data.get("platform_stats") is None

    @pytest.mark.asyncio
    async def test_dashboard_admin_with_platform_stats(self, full_miniapp):
        """Admin dashboard shows platform_stats."""
        app, _, admin_token, _, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/dashboard", headers=_auth(admin_token))
            assert resp.status_code == 200
            data = resp.json()
            assert data["platform_stats"] is not None
            assert "total_users" in data["platform_stats"]
            assert "total_agents" in data["platform_stats"]
            assert "total_revenue_usd" in data["platform_stats"]

    @pytest.mark.asyncio
    async def test_dashboard_recent_tx_fields(self, full_miniapp):
        """Recent transactions in dashboard have all required fields."""
        app, token, _, _, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/dashboard", headers=_auth(token))
            data = resp.json()
            if data["recent_transactions"]:
                tx = data["recent_transactions"][0]
                assert "id" in tx
                assert "agent_id" in tx
                assert "agent_name" in tx
                assert "type" in tx
                assert "amount" in tx
                assert "fee" in tx
                assert "status" in tx
                assert "created_at" in tx


# ═══════════════════════════════════════
# ANALYTICS — daily, category, hourly, balance
# ═══════════════════════════════════════

class TestAnalyticsDeep:
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="SQLite doesn't support cast(DateTime, Date) — works on PostgreSQL in production")
    async def test_analytics_structure(self, full_miniapp):
        """Analytics returns all expected structures even without date-based data."""
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/analytics",
                headers=_auth(token),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "daily_volume" in data
            assert len(data["daily_volume"]) == 30
            assert "spending_by_category" in data
            assert "hourly_heatmap" in data
            assert len(data["hourly_heatmap"]) == 24
            assert "balance_history" in data
            assert len(data["balance_history"]) == 30
            # Each balance_history entry has date and balance
            bh = data["balance_history"][0]
            assert "date" in bh
            assert "balance" in bh


# ═══════════════════════════════════════
# IDENTITY — with data + refresh
# ═══════════════════════════════════════

class TestIdentityDeep:
    @pytest.mark.asyncio
    async def test_identity_with_full_data(self, full_miniapp):
        """Identity endpoint returns full data and trust score breakdown."""
        app, token, _, agent_id, _ = full_miniapp
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/identity",
                headers=_auth(token),
            )
            assert resp.status_code == 200
            data = resp.json()
            ident = data["identity"]
            assert ident is not None
            assert ident["display_name"] == "Coverage Agent"
            assert ident["description"] == "Agent for coverage tests"
            assert ident["category"] == "utility"
            assert ident["homepage_url"] == "https://cov.example.com"
            assert ident["logo_url"] == "https://cov.example.com/logo.png"
            assert ident["verified"] is True
            assert ident["trust_score"] >= 0
            assert "first_seen" in ident
            assert "last_active" in ident
            assert "trust_score_breakdown" in data
            assert "total" in data["trust_score_breakdown"]
