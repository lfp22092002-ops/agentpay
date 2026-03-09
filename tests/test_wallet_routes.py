"""
Tests for wallet route endpoints — /wallet, /wallet/all, /wallet/send-usdc,
/wallet/send-native, /card, /card/transactions (HTTP layer coverage).

Targets coverage gaps in api/routes/wallets.py lines 57-58, 74, 89-91,
106-108, 152-184, 195-206, 217-224, 231-238, 251, 303-313, 331-334, 347.
"""
import os
import sys
from decimal import Decimal
from unittest.mock import patch, MagicMock, AsyncMock

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
async def app_env():
    """Create a test app with fresh DB, a funded agent, and wallet fixtures."""
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

    api_key = "ap_walletroute_test_1234567890abcdef123456"

    async with factory() as db:
        user = User(telegram_id=88888888, username="walletrouteuser", first_name="WR")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        agent = Agent(
            user_id=user.id,
            name="wr-agent",
            api_key_hash=hash_api_key(api_key),
            api_key_prefix="ap_wall...",
            balance_usd=Decimal("200.0000"),
            daily_limit_usd=Decimal("100.0000"),
            tx_limit_usd=Decimal("50.0000"),
            auto_approve_usd=Decimal("25.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

    yield app, api_key, agent

    app.dependency_overrides.clear()
    await engine.dispose()


def _h(key: str) -> dict:
    return {"X-API-Key": key}


# ═══════════════════════════════════════
# /wallet — EVM (base, polygon, bnb)
# ═══════════════════════════════════════

class TestWalletEVM:
    @pytest.mark.asyncio
    async def test_wallet_existing_address(self, app_env):
        """When the agent already has an EVM wallet, return its info."""
        app, key, agent = app_env

        mock_balance = {
            "network": "Base Mainnet",
            "chain": "base",
            "balance_eth": "0.01",
            "balance_usdc": "5.50",
        }

        with patch("api.routes.wallets.get_wallet_address", return_value="0x" + "aa" * 20), \
             patch("api.routes.wallets.get_wallet_balance", return_value=mock_balance):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/wallet?chain=base", headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["chain"] == "base"
        assert data["address"] == "0x" + "aa" * 20
        assert data["network"] == "Base Mainnet"

    @pytest.mark.asyncio
    async def test_wallet_creates_new_evm(self, app_env):
        """When no wallet exists, creates one and persists to DB."""
        app, key, agent = app_env

        mock_balance = {
            "network": "Polygon PoS",
            "chain": "polygon",
            "balance_eth": "0.0",
            "balance_usdc": "0.0",
        }
        new_addr = "0x" + "bb" * 20

        with patch("api.routes.wallets.get_wallet_address", return_value=None), \
             patch("providers.local_wallet.create_agent_wallet", return_value={"address": new_addr}), \
             patch("api.routes.wallets.get_wallet_balance", return_value=mock_balance):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/wallet?chain=polygon", headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["chain"] == "polygon"
        assert data["address"] == new_addr


# ═══════════════════════════════════════
# /wallet — Solana
# ═══════════════════════════════════════

class TestWalletSolana:
    @pytest.mark.asyncio
    async def test_wallet_solana_existing(self, app_env):
        """Solana wallet already exists."""
        app, key, agent = app_env

        mock_balance = {
            "network": "Solana Mainnet",
            "balance_sol": "1.5",
            "balance_usdc": "10.0",
        }

        with patch("api.routes.wallets.get_solana_wallet_address", return_value="So1" + "x" * 40), \
             patch("api.routes.wallets.get_solana_balance", return_value=mock_balance):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/wallet?chain=solana", headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["chain"] == "solana"
        assert data["balance_sol"] == "1.5"
        assert data["balance_usdc"] == "10.0"

    @pytest.mark.asyncio
    async def test_wallet_solana_creates_new(self, app_env):
        """Solana wallet doesn't exist — creates one."""
        app, key, agent = app_env

        mock_balance = {
            "network": "Solana Mainnet",
            "balance_sol": "0",
            "balance_usdc": "0",
        }
        new_addr = "NewSolAddr" + "z" * 34

        with patch("api.routes.wallets.get_solana_wallet_address", return_value=None), \
             patch("api.routes.wallets.create_solana_wallet", return_value={"address": new_addr}), \
             patch("api.routes.wallets.get_solana_balance", return_value=mock_balance):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/wallet?chain=solana", headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["chain"] == "solana"
        assert data["address"] == new_addr


# ═══════════════════════════════════════
# /wallet/all
# ═══════════════════════════════════════

class TestWalletAll:
    @pytest.mark.asyncio
    async def test_wallet_all_both(self, app_env):
        """Both EVM and Solana wallets exist."""
        app, key, agent = app_env

        evm_balance = {"chain": "base", "balance_usdc": "5", "balance_native": "0.01"}
        sol_balance = {"chain": "solana", "balance_sol": "2", "balance_usdc": "0"}

        with patch("api.routes.wallets.get_wallet_address", return_value="0x" + "cc" * 20), \
             patch("api.routes.wallets.get_solana_wallet_address", return_value="SolAddr123"), \
             patch("api.routes.wallets.get_wallet_balance", return_value=evm_balance), \
             patch("api.routes.wallets.get_solana_balance", return_value=sol_balance):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/wallet/all", headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["evm_address"] == "0x" + "cc" * 20
        assert data["solana_address"] == "SolAddr123"
        # EVM chains (3) + Solana (1) = 4 chain entries
        assert len(data["chains"]) == 4

    @pytest.mark.asyncio
    async def test_wallet_all_none(self, app_env):
        """No wallets exist yet."""
        app, key, agent = app_env

        with patch("api.routes.wallets.get_wallet_address", return_value=None), \
             patch("api.routes.wallets.get_solana_wallet_address", return_value=None):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/wallet/all", headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["evm_address"] is None
        assert data["solana_address"] is None
        assert data["chains"] == []


# ═══════════════════════════════════════
# /wallet/send-usdc
# ═══════════════════════════════════════

class TestSendUsdcEndpoint:
    @pytest.mark.asyncio
    async def test_send_usdc_evm(self, app_env):
        """Send USDC on an EVM chain."""
        app, key, agent = app_env

        mock_result = {"success": True, "tx_hash": "0xabc123", "amount": 10.0, "chain": "base"}

        with patch("api.routes.wallets.send_usdc", new_callable=AsyncMock, return_value=mock_result):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/wallet/send-usdc", json={
                    "to_address": "0x" + "dd" * 20,
                    "amount": 10.0,
                    "chain": "base",
                }, headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["tx_hash"] == "0xabc123"

    @pytest.mark.asyncio
    async def test_send_usdc_solana(self, app_env):
        """Send USDC on Solana."""
        app, key, agent = app_env

        mock_result = {"success": True, "tx_hash": "SolTx456", "amount": 5.0, "chain": "solana"}

        with patch("api.routes.wallets.send_solana_usdc", new_callable=AsyncMock, return_value=mock_result):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/wallet/send-usdc", json={
                    "to_address": "SolRecipient" + "x" * 32,
                    "amount": 5.0,
                    "chain": "solana",
                }, headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_send_usdc_unsupported_chain(self, app_env):
        """Unsupported chain returns error."""
        app, key, agent = app_env

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/wallet/send-usdc", json={
                "to_address": "0x" + "00" * 20,
                "amount": 1.0,
                "chain": "avalanche",
            }, headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "Unsupported chain" in data.get("error", "")


# ═══════════════════════════════════════
# /wallet/send-native
# ═══════════════════════════════════════

class TestSendNativeEndpoint:
    @pytest.mark.asyncio
    async def test_send_native_evm(self, app_env):
        """Send native token on EVM chain."""
        app, key, agent = app_env

        mock_result = {"success": True, "tx_hash": "0xnative123", "amount": 0.01, "chain": "polygon",
                       "native_token": "POL"}

        with patch("api.routes.wallets.send_native", new_callable=AsyncMock, return_value=mock_result):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/wallet/send-native", json={
                    "to_address": "0x" + "ee" * 20,
                    "amount": 0.01,
                    "chain": "polygon",
                }, headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_send_native_solana(self, app_env):
        """Send SOL."""
        app, key, agent = app_env

        mock_result = {"success": True, "tx_hash": "SolNative789", "amount": 0.5, "chain": "solana",
                       "native_token": "SOL"}

        with patch("api.routes.wallets.send_sol", new_callable=AsyncMock, return_value=mock_result):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/wallet/send-native", json={
                    "to_address": "SolRecipient" + "y" * 32,
                    "amount": 0.5,
                    "chain": "solana",
                }, headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_send_native_unsupported_chain(self, app_env):
        """Unsupported chain returns error."""
        app, key, agent = app_env

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/wallet/send-native", json={
                "to_address": "0x" + "00" * 20,
                "amount": 0.01,
                "chain": "fantom",
            }, headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "Unsupported chain" in data.get("error", "")


# ═══════════════════════════════════════
# /card and /card/transactions with data
# ═══════════════════════════════════════

class TestCardWithData:
    @pytest.mark.asyncio
    async def test_card_info_with_card(self, app_env):
        """When agent has a Lithic card, return details."""
        app, key, agent = app_env

        mock_details = {
            "last4": "4242",
            "exp_month": "12",
            "exp_year": "2027",
            "state": "OPEN",
            "spend_limit_cents": 50000,
        }

        with patch("api.routes.wallets.get_card_details", return_value=mock_details):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/card", headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["last4"] == "4242"
        assert data["state"] == "OPEN"
        assert data["spend_limit_cents"] == 50000

    @pytest.mark.asyncio
    async def test_card_transactions_with_data(self, app_env):
        """Card transactions return list."""
        app, key, agent = app_env

        mock_txns = [
            {"merchant": "OpenAI", "amount_cents": 2000, "status": "SETTLED", "created": "2026-03-08T12:00:00Z"},
            {"merchant": "AWS", "amount_cents": 500, "status": "PENDING", "created": "2026-03-08T13:00:00Z"},
        ]

        with patch("api.routes.wallets.get_card_transactions", return_value=mock_txns):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/card/transactions?limit=5", headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_card_transactions_limit_capped(self, app_env):
        """Limit is capped at 50."""
        app, key, agent = app_env

        with patch("api.routes.wallets.get_card_transactions", return_value=[]) as mock_get:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/card/transactions?limit=999", headers=_h(key))

        assert resp.status_code == 200
        # Verify the limit was capped to 50
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["limit"] == 50 or call_kwargs[0][-1] == 50


# ═══════════════════════════════════════
# /approvals/{approval_id}
# ═══════════════════════════════════════

class TestApprovalEndpoint:
    @pytest.mark.asyncio
    async def test_approval_not_found(self, app_env):
        """Non-existent approval returns 404."""
        app, key, agent = app_env

        with patch("core.approvals.get_pending", return_value=None):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/approvals/fake-id", headers=_h(key))

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_approval_wrong_agent(self, app_env):
        """Approval belonging to a different agent returns 403."""
        app, key, agent = app_env

        mock_approval = MagicMock()
        mock_approval.agent_id = "some-other-agent-id"

        with patch("core.approvals.get_pending", return_value=mock_approval):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/approvals/some-approval", headers=_h(key))

        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_approval_pending(self, app_env):
        """Pending approval returns status."""
        app, key, agent = app_env

        mock_approval = MagicMock()
        mock_approval.id = "appr-123"
        mock_approval.agent_id = agent.id
        mock_approval.amount_usd = Decimal("15.00")
        mock_approval.description = "Big purchase"
        mock_approval.resolved = False
        mock_approval.result = None

        with patch("core.approvals.get_pending", return_value=mock_approval):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/approvals/appr-123", headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["approval_id"] == "appr-123"
        assert data["resolved"] is False
        assert data["result"] is None

    @pytest.mark.asyncio
    async def test_approval_resolved(self, app_env):
        """Resolved approval returns result."""
        app, key, agent = app_env

        mock_future = MagicMock()
        mock_future.done.return_value = True
        mock_future.result.return_value = {"approved": True}

        mock_approval = MagicMock()
        mock_approval.id = "appr-456"
        mock_approval.agent_id = agent.id
        mock_approval.amount_usd = Decimal("20.00")
        mock_approval.description = "Approved purchase"
        mock_approval.resolved = True
        mock_approval.result = mock_future

        with patch("core.approvals.get_pending", return_value=mock_approval):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/approvals/appr-456", headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["resolved"] is True
        assert data["result"] == "approved"


# ═══════════════════════════════════════
# /x402/pay
# ═══════════════════════════════════════

class TestX402PayEndpoint:
    @pytest.mark.asyncio
    async def test_x402_pay_success(self, app_env):
        """Pay for an x402-gated resource."""
        app, key, agent = app_env

        mock_result = {
            "success": True,
            "status": 200,
            "data": '{"data": "premium content"}',
            "paid_usd": 0.01,
        }

        with patch("providers.x402_protocol.pay_x402_resource", new_callable=AsyncMock, return_value=mock_result):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/x402/pay", json={
                    "url": "https://api.example.com/resource",
                    "method": "GET",
                    "max_price_usd": 0.05,
                }, headers=_h(key))

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["paid_usd"] == 0.01
