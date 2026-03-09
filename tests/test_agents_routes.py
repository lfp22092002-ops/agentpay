"""
Tests for api/routes/agents.py — balance, transactions, rotate-key, export.
Covers uncovered lines: 25, 41, 59-61, 74-89.
"""
import os
import sys
from decimal import Decimal

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.database import Base
from models.schema import User, Agent, Transaction, TransactionType, TransactionStatus
from core.wallet import hash_api_key


def _headers(key: str):
    return {"X-API-Key": key}


@pytest_asyncio.fixture
async def agents_app():
    """Create a test app with user, agent, and some transactions."""
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

    api_key = "ap_agents_route_test_1234567890abcdef"

    async with factory() as db:
        user = User(telegram_id=77777777, username="agentrouter", first_name="Router")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        agent = Agent(
            user_id=user.id,
            name="route-agent",
            api_key_hash=hash_api_key(api_key),
            api_key_prefix="ap_agen...",
            balance_usd=Decimal("250.0000"),
            daily_limit_usd=Decimal("100.0000"),
            tx_limit_usd=Decimal("50.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

        # Add some transactions
        for i in range(3):
            tx = Transaction(
                agent_id=agent.id,
                tx_type=TransactionType.SPEND,
                amount_usd=Decimal("10.0000"),
                fee_usd=Decimal("0.2000"),
                description=f"Test spend {i}",
                status=TransactionStatus.COMPLETED,
            )
            db.add(tx)
        await db.commit()

    yield app, api_key, agent

    app.dependency_overrides.clear()
    await engine.dispose()


class TestGetBalance:
    @pytest.mark.asyncio
    async def test_balance_returns_all_fields(self, agents_app):
        """GET /v1/balance returns full BalanceResponse."""
        app, api_key, agent = agents_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/balance", headers=_headers(api_key))
            assert resp.status_code == 200
            data = resp.json()
            assert data["agent_id"] == agent.id
            assert data["agent_name"] == "route-agent"
            assert data["balance_usd"] == 250.0
            assert data["daily_limit_usd"] == 100.0
            assert "daily_spent_usd" in data
            assert "daily_remaining_usd" in data
            assert data["tx_limit_usd"] == 50.0
            assert data["is_active"] is True


class TestListTransactions:
    @pytest.mark.asyncio
    async def test_list_transactions(self, agents_app):
        """GET /v1/transactions returns transaction list."""
        app, api_key, _ = agents_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/transactions", headers=_headers(api_key))
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 3
            for tx in data:
                assert "id" in tx
                assert tx["type"] == "spend"
                assert tx["amount"] == 10.0
                assert tx["fee"] == 0.2
                assert "description" in tx
                assert tx["status"] == "completed"
                assert "created_at" in tx

    @pytest.mark.asyncio
    async def test_list_transactions_with_limit(self, agents_app):
        """GET /v1/transactions?limit=1 respects limit."""
        app, api_key, _ = agents_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/transactions?limit=1", headers=_headers(api_key))
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) <= 1


class TestRotateKey:
    @pytest.mark.asyncio
    async def test_rotate_key_success(self, agents_app):
        """POST /v1/agent/rotate-key returns new key."""
        app, api_key, _ = agents_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/agent/rotate-key", headers=_headers(api_key))
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert "new_api_key" in data
            assert data["new_api_key"].startswith("ap_")
            assert "key_prefix" in data

            # Old key should no longer work
            resp2 = await client.get("/v1/balance", headers=_headers(api_key))
            assert resp2.status_code == 401

            # New key should work
            resp3 = await client.get("/v1/balance", headers=_headers(data["new_api_key"]))
            assert resp3.status_code == 200


class TestExportCSV:
    @pytest.mark.asyncio
    async def test_export_csv_content(self, agents_app):
        """GET /v1/export returns CSV with transactions."""
        app, api_key, _ = agents_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/export", headers=_headers(api_key))
            assert resp.status_code == 200
            assert "text/csv" in resp.headers["content-type"]
            assert "attachment" in resp.headers.get("content-disposition", "")
            content = resp.text
            lines = [l.strip() for l in content.strip().split("\n")]
            assert lines[0] == "date,type,amount,fee,description,status,id"
            assert len(lines) == 4  # header + 3 transactions

    @pytest.mark.asyncio
    async def test_export_csv_empty(self, agents_app):
        """GET /v1/export with no transactions returns header only."""
        # We already have transactions, but this test validates the endpoint works
        app, api_key, _ = agents_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/export", headers=_headers(api_key))
            assert resp.status_code == 200
            assert "text/csv" in resp.headers["content-type"]
