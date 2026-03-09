"""
Additional agent route tests — covers rotate-key and transactions list body (lines 25, 41, 61, 74-89).
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


@pytest_asyncio.fixture
async def agents_app():
    """App with user, agent, and transactions."""
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

    api_key = "ap_agcov_test_1234567890abcdef1234567890"
    async with factory() as db:
        user = User(telegram_id=66666666, username="agcovuser", first_name="AgCov")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        agent = Agent(
            user_id=user.id,
            name="agcov-agent",
            api_key_hash=hash_api_key(api_key),
            api_key_prefix="ap_agco...",
            balance_usd=Decimal("200.0000"),
            daily_limit_usd=Decimal("100.0000"),
            tx_limit_usd=Decimal("50.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

        # Add transactions for list/export tests
        for i in range(3):
            tx = Transaction(
                agent_id=agent.id,
                tx_type=TransactionType.SPEND,
                amount_usd=Decimal("10.0000"),
                fee_usd=Decimal("0.2000"),
                description=f"Agent tx {i}",
                status=TransactionStatus.COMPLETED,
            )
            db.add(tx)
        await db.commit()

    yield app, api_key, agent

    app.dependency_overrides.clear()
    await engine.dispose()


class TestTransactionsList:
    """Tests covering list_transactions body (lines 38-53 in agents.py)."""

    @pytest.mark.asyncio
    async def test_list_transactions_with_data(self, agents_app):
        """GET /transactions returns transaction list with correct fields."""
        app, api_key, _ = agents_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/transactions",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 3
            tx = data[0]
            assert "id" in tx
            assert tx["type"] == "spend"
            assert tx["amount"] == 10.0
            assert tx["fee"] == 0.2
            assert "description" in tx
            assert tx["status"] == "completed"
            assert "created_at" in tx

    @pytest.mark.asyncio
    async def test_list_transactions_limit(self, agents_app):
        """GET /transactions respects limit parameter."""
        app, api_key, _ = agents_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/transactions?limit=2",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2


class TestRotateKey:
    """Tests covering rotate-key endpoint (lines 55-66 in agents.py)."""

    @pytest.mark.asyncio
    async def test_rotate_key_success(self, agents_app):
        """POST /agent/rotate-key returns new key and invalidates old."""
        app, api_key, _ = agents_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/agent/rotate-key",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert "new_api_key" in data
            assert data["new_api_key"].startswith("ap_")
            assert data["key_prefix"].endswith("...")

            # Old key should no longer work
            resp2 = await client.get(
                "/v1/balance",
                headers={"X-API-Key": api_key},
            )
            assert resp2.status_code == 401

            # New key should work
            resp3 = await client.get(
                "/v1/balance",
                headers={"X-API-Key": data["new_api_key"]},
            )
            assert resp3.status_code == 200


class TestExportWithData:
    """Tests covering export CSV with actual data (lines 74-89 in agents.py)."""

    @pytest.mark.asyncio
    async def test_export_csv_with_transactions(self, agents_app):
        """GET /export returns CSV with transaction rows."""
        app, api_key, _ = agents_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/export",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            assert "text/csv" in resp.headers.get("content-type", "")
            content = resp.text
            lines = [line.strip() for line in content.strip().split("\n")]
            # Header + 3 data rows
            assert len(lines) == 4
            assert lines[0] == "date,type,amount,fee,description,status,id"
            # Check a data row has correct fields
            parts = lines[1].split(",")
            assert parts[1] == "spend"
            assert parts[2] == "10.0"
            assert parts[3] == "0.2"
