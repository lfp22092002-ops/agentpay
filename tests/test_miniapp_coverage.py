"""
Additional Mini App coverage tests — targeting uncovered lines.
Covers: validate_telegram_init_data, create/decode JWT, auth_telegram with real BOT_TOKEN,
wallet creation flows, dashboard with transactions, settings edge cases.
"""
import hashlib
import hmac
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock
from urllib.parse import urlencode

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.database import Base
from models.schema import (
    User, Agent, Transaction, TransactionType, TransactionStatus,
    PaymentMethod, Wallet, AgentIdentity,
)
from core.wallet import hash_api_key

import jwt as pyjwt


# ═══════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════

def _make_valid_init_data(bot_token: str, user_data: dict, auth_date: int = None) -> str:
    """Build a valid Telegram initData string with a correct hash."""
    if auth_date is None:
        auth_date = int(time.time())
    user_json = json.dumps(user_data, separators=(",", ":"))
    data_dict = {"user": user_json, "auth_date": str(auth_date)}
    check_string = "\n".join(f"{k}={v}" for k, v in sorted(data_dict.items()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    computed_hash = hmac.new(secret_key, check_string.encode(), hashlib.sha256).hexdigest()
    data_dict["hash"] = computed_hash
    return urlencode(data_dict)


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def miniapp_full():
    """Full miniapp test fixture with user, agent, and transactions."""
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
        user = User(telegram_id=55555, username="fulluser", first_name="Full")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        key = "ap_fulltest_key_1234567890abcdef1234567890"
        agent = Agent(
            user_id=user.id,
            name="full-agent",
            api_key_hash=hash_api_key(key),
            api_key_prefix="ap_full...",
            balance_usd=Decimal("300.0000"),
            daily_limit_usd=Decimal("100.0000"),
            tx_limit_usd=Decimal("50.0000"),
            auto_approve_usd=Decimal("10.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

        # Add transactions
        for i in range(5):
            tx = Transaction(
                agent_id=agent.id,
                tx_type=TransactionType.SPEND,
                amount_usd=Decimal("10.0000"),
                fee_usd=Decimal("0.2000"),
                description=f"Spend {i}",
                status=TransactionStatus.COMPLETED,
                payment_method=PaymentMethod.MANUAL,
            )
            db.add(tx)
        # Add a deposit
        tx_dep = Transaction(
            agent_id=agent.id,
            tx_type=TransactionType.DEPOSIT,
            amount_usd=Decimal("50.0000"),
            fee_usd=Decimal("0.0000"),
            description="Deposit test",
            status=TransactionStatus.COMPLETED,
            payment_method=PaymentMethod.MANUAL,
        )
        db.add(tx_dep)
        await db.commit()

    from config.settings import API_SECRET
    token = pyjwt.encode(
        {
            "sub": "55555",
            "telegram_id": 55555,
            "first_name": "Full",
            "username": "fulluser",
            "iat": int(time.time()),
            "exp": int(time.time()) + 86400,
            "type": "miniapp",
        },
        API_SECRET,
        algorithm="HS256",
    )

    yield app, token, agent.id, factory

    app.dependency_overrides.clear()
    from api.middleware import limiter
    limiter.reset()
    await engine.dispose()


# ═══════════════════════════════════════
# UNIT TESTS: _validate_telegram_init_data
# ═══════════════════════════════════════

class TestValidateTelegramInitData:
    def test_valid_init_data(self):
        from api.routes.miniapp import _validate_telegram_init_data
        bot_token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        user_data = {"id": 12345, "first_name": "Test"}
        init_data = _make_valid_init_data(bot_token, user_data)
        result = _validate_telegram_init_data(init_data, bot_token)
        assert result is not None
        assert result["user"]["id"] == 12345

    def test_empty_init_data(self):
        from api.routes.miniapp import _validate_telegram_init_data
        result = _validate_telegram_init_data("", "token")
        assert result is None

    def test_no_hash(self):
        from api.routes.miniapp import _validate_telegram_init_data
        result = _validate_telegram_init_data("auth_date=123", "token")
        assert result is None

    def test_wrong_hash(self):
        from api.routes.miniapp import _validate_telegram_init_data
        result = _validate_telegram_init_data("auth_date=123&hash=badhash", "token")
        assert result is None

    def test_expired_auth_date(self):
        from api.routes.miniapp import _validate_telegram_init_data
        bot_token = "123456:testtoken"
        user_data = {"id": 99}
        # auth_date far in the past (>86400 seconds ago)
        old_time = int(time.time()) - 100000
        init_data = _make_valid_init_data(bot_token, user_data, auth_date=old_time)
        result = _validate_telegram_init_data(init_data, bot_token)
        assert result is None


# ═══════════════════════════════════════
# UNIT TESTS: JWT helpers
# ═══════════════════════════════════════

class TestJwtHelpers:
    def test_create_and_decode_jwt(self):
        from api.routes.miniapp import _create_miniapp_jwt, _decode_miniapp_jwt
        token = _create_miniapp_jwt(12345, {"first_name": "Test", "username": "test"})
        assert isinstance(token, str)
        payload = _decode_miniapp_jwt(token)
        assert payload is not None
        assert payload["telegram_id"] == 12345
        assert payload["type"] == "miniapp"

    def test_decode_expired_jwt(self):
        from api.routes.miniapp import _decode_miniapp_jwt, MINIAPP_JWT_SECRET, MINIAPP_JWT_ALGO
        expired_token = pyjwt.encode(
            {"sub": "1", "exp": int(time.time()) - 3600},
            MINIAPP_JWT_SECRET,
            algorithm=MINIAPP_JWT_ALGO,
        )
        result = _decode_miniapp_jwt(expired_token)
        assert result is None

    def test_decode_invalid_jwt(self):
        from api.routes.miniapp import _decode_miniapp_jwt
        result = _decode_miniapp_jwt("not.a.valid.token")
        assert result is None


# ═══════════════════════════════════════
# get_miniapp_user
# ═══════════════════════════════════════

class TestGetMiniappUser:
    @pytest.mark.asyncio
    async def test_no_bearer_prefix(self, miniapp_full):
        app, _, _, _ = miniapp_full
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents",
                headers={"Authorization": "Token abc123"},
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_token_rejected(self, miniapp_full):
        app, _, _, _ = miniapp_full
        from config.settings import API_SECRET
        expired = pyjwt.encode(
            {"telegram_id": 55555, "exp": int(time.time()) - 100},
            API_SECRET,
            algorithm="HS256",
        )
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents",
                headers={"Authorization": f"Bearer {expired}"},
            )
            assert resp.status_code == 401


# ═══════════════════════════════════════
# auth_telegram with real BOT_TOKEN
# ═══════════════════════════════════════

class TestAuthTelegramEndpoint:
    @pytest.mark.asyncio
    async def test_auth_with_valid_bot_token(self, miniapp_full):
        """Test /auth/telegram with a proper HMAC-validated initData."""
        app, _, _, _ = miniapp_full
        bot_token = "123456:ValidTestToken"
        user_data = {"id": 55555, "first_name": "Full", "username": "fulluser"}
        init_data = _make_valid_init_data(bot_token, user_data)
        transport = ASGITransport(app=app)
        with patch("api.routes.miniapp.BOT_TOKEN", bot_token):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/auth/telegram",
                    json={"init_data": init_data},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["telegram_id"] == 55555
                assert "token" in data

    @pytest.mark.asyncio
    async def test_auth_invalid_init_data(self, miniapp_full):
        """Test /auth/telegram rejects bad initData when BOT_TOKEN is set."""
        app, _, _, _ = miniapp_full
        transport = ASGITransport(app=app)
        with patch("api.routes.miniapp.BOT_TOKEN", "123456:ValidTestToken"):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/auth/telegram",
                    json={"init_data": "bad=data&hash=wrong"},
                )
                assert resp.status_code == 401


# ═══════════════════════════════════════
# TRANSACTIONS FILTERING
# ═══════════════════════════════════════

class TestTransactionFiltering:
    @pytest.mark.asyncio
    async def test_filter_by_type(self, miniapp_full):
        app, token, agent_id, _ = miniapp_full
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/transactions?type=deposit",
                headers=auth_headers(token),
            )
            assert resp.status_code == 200
            txs = resp.json()["transactions"]
            assert len(txs) == 1
            assert txs[0]["type"] == "deposit"

    @pytest.mark.asyncio
    async def test_filter_by_date(self, miniapp_full):
        app, token, agent_id, _ = miniapp_full
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/transactions?date={today}",
                headers=auth_headers(token),
            )
            assert resp.status_code == 200
            txs = resp.json()["transactions"]
            assert len(txs) == 6  # 5 spends + 1 deposit

    @pytest.mark.asyncio
    async def test_filter_by_invalid_date(self, miniapp_full):
        app, token, agent_id, _ = miniapp_full
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/v1/miniapp/agents/{agent_id}/transactions?date=not-a-date",
                headers=auth_headers(token),
            )
            assert resp.status_code == 200  # invalid date just ignored


# ═══════════════════════════════════════
# SETTINGS EDGE CASES
# ═══════════════════════════════════════

class TestSettingsEdgeCases:
    @pytest.mark.asyncio
    async def test_invalid_tx_limit(self, miniapp_full):
        app, token, agent_id, _ = miniapp_full
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent_id}/settings",
                json={"tx_limit_usd": -5.0},
                headers=auth_headers(token),
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_auto_approve(self, miniapp_full):
        app, token, agent_id, _ = miniapp_full
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                f"/v1/miniapp/agents/{agent_id}/settings",
                json={"auto_approve_usd": -1.0},
                headers=auth_headers(token),
            )
            assert resp.status_code == 400


# ═══════════════════════════════════════
# WALLET CREATION FLOWS
# ═══════════════════════════════════════

class TestWalletCreation:
    @pytest.mark.asyncio
    async def test_evm_wallet_creation_on_first_access(self, miniapp_full):
        """When no EVM wallet exists, one is created on first access."""
        app, token, agent_id, _ = miniapp_full
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
                    headers=auth_headers(token),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["address"] == "0xnew123"
                assert data["chain"] == "base"

    @pytest.mark.asyncio
    async def test_solana_wallet_creation_on_first_access(self, miniapp_full):
        """When no Solana wallet exists, one is created on first access."""
        app, token, agent_id, _ = miniapp_full
        transport = ASGITransport(app=app)
        mock_balance = {"network": "solana-devnet", "balance_sol": "0", "balance_usdc": "0"}
        with patch("api.routes.miniapp.get_solana_wallet_address", return_value=None), \
             patch("api.routes.miniapp.create_solana_wallet", return_value={"address": "SoLnew456"}), \
             patch("api.routes.miniapp.get_solana_balance", return_value=mock_balance):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent_id}/wallet?chain=solana",
                    headers=auth_headers(token),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["address"] == "SoLnew456"

    @pytest.mark.asyncio
    async def test_wallet_all_no_wallets(self, miniapp_full):
        """When no wallets exist, /wallet/all returns empty chains."""
        app, token, agent_id, _ = miniapp_full
        transport = ASGITransport(app=app)
        with patch("api.routes.miniapp.get_wallet_address", return_value=None), \
             patch("api.routes.miniapp.get_solana_wallet_address", return_value=None):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/v1/miniapp/agents/{agent_id}/wallet/all",
                    headers=auth_headers(token),
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["chains"] == []
                assert data["evm_address"] is None
                assert data["solana_address"] is None


# ═══════════════════════════════════════
# CARD TOGGLE EDGE CASES
# ═══════════════════════════════════════

class TestCardToggleEdgeCases:
    @pytest.mark.asyncio
    async def test_card_toggle_agent_not_found(self, miniapp_full):
        app, token, _, _ = miniapp_full
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/miniapp/agents/fake-id/card/pause",
                headers=auth_headers(token),
            )
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_card_toggle_error(self, miniapp_full):
        """Card state update error returns 500."""
        app, token, agent_id, _ = miniapp_full
        transport = ASGITransport(app=app)
        with patch("providers.lithic_card.update_card_state", side_effect=Exception("Lithic API error")):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    f"/v1/miniapp/agents/{agent_id}/card/resume",
                    headers=auth_headers(token),
                )
                assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_card_agent_not_found(self, miniapp_full):
        app, token, _, _ = miniapp_full
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/fake-id/card",
                headers=auth_headers(token),
            )
            assert resp.status_code == 404


# ═══════════════════════════════════════
# DASHBOARD WITH TRANSACTIONS
# ═══════════════════════════════════════

class TestDashboardWithTxs:
    @pytest.mark.asyncio
    async def test_dashboard_with_transactions(self, miniapp_full):
        """Dashboard includes recent transactions and agent stats."""
        app, token, agent_id, _ = miniapp_full
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/miniapp/dashboard", headers=auth_headers(token))
            assert resp.status_code == 200
            data = resp.json()
            assert data["agent_count"] == 1
            assert data["total_balance_usd"] == 300.0
            assert data["total_transactions"] == 6
            assert data["total_volume_usd"] > 0
            assert len(data["recent_transactions"]) > 0
            assert data["agents"][0]["tx_count"] == 6

    @pytest.mark.asyncio
    async def test_dashboard_no_user_in_db(self, miniapp_full):
        """User with a valid JWT but no DB record gets empty dashboard."""
        app, _, _, _ = miniapp_full
        from config.settings import API_SECRET
        token = pyjwt.encode(
            {
                "telegram_id": 77777,
                "exp": int(time.time()) + 86400,
                "type": "miniapp",
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


# ═══════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════

class TestAnalyticsWithTxs:
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="SQLite doesn't support cast(DateTime, Date) — works on PostgreSQL in production")
    async def test_analytics_with_transactions(self, miniapp_full):
        """Analytics endpoint returns data when transactions exist."""
        app, token, agent_id, _ = miniapp_full
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
            assert len(data["spending_by_category"]) > 0


# ═══════════════════════════════════════
# IDENTITY
# ═══════════════════════════════════════

class TestIdentityEndpoint:
    @pytest.mark.asyncio
    async def test_identity_agent_not_found(self, miniapp_full):
        app, token, _, _ = miniapp_full
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/fake-id/identity",
                headers=auth_headers(token),
            )
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_wallet_all_agent_not_found(self, miniapp_full):
        app, token, _, _ = miniapp_full
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/miniapp/agents/fake-id/wallet/all",
                headers=auth_headers(token),
            )
            assert resp.status_code == 404
