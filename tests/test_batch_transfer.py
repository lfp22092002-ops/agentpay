"""
Tests for api/routes/batch.py — batch/transfer endpoint.
Covers orchestrator→sub-agent parallel funding pattern.
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
async def batch_app():
    """Create a test app with one parent agent and three sub-agents."""
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

    parent_key = "ap_batch_test_parent_1234567890abc"
    sub_keys = [f"ap_batch_test_sub{i}_1234567890def" for i in range(1, 4)]

    async with factory() as db:
        user = User(telegram_id=99999999, username="batchuser", first_name="Batch")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        # Parent agent with balance
        parent = Agent(
            user_id=user.id,
            name="orchestrator-agent",
            api_key_hash=hash_api_key(parent_key),
            api_key_prefix="ap_batch...",
            balance_usd=Decimal("500.0000"),
            daily_limit_usd=Decimal("1000.0000"),
            tx_limit_usd=Decimal("200.0000"),
            is_active=True,
        )
        db.add(parent)

        # Three sub-agents
        sub_agents = []
        for i in range(3):
            agent = Agent(
                user_id=user.id,
                name=f"sub-agent-{i+1}",
                api_key_hash=hash_api_key(sub_keys[i]),
                api_key_prefix="ap_sub...",
                balance_usd=Decimal("0.0000"),
                daily_limit_usd=Decimal("50.0000"),
                tx_limit_usd=Decimal("10.0000"),
                is_active=True,
            )
            db.add(agent)
            sub_agents.append(agent)

        await db.commit()
        for a in sub_agents:
            await db.refresh(a)

    # Extract IDs as strings
    sub_agent_ids = [str(sub_agents[i].id) for i in range(3)]
    yield app, parent_key, parent, sub_agent_ids, sub_agent_ids

    app.dependency_overrides.clear()
    await engine.dispose()


class TestBatchTransferSuccess:
    @pytest.mark.asyncio
    async def test_batch_transfer_all_success(self, batch_app):
        """Batch transfer to 3 agents — all succeed."""
        app, parent_key, parent, _, sub_ids = batch_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/batch/transfer",
                headers=_headers(parent_key),
                json={
                    "payments": [
                        {"agent_id": sub_ids[0], "amount_usd": 50.0},
                        {"agent_id": sub_ids[1], "amount_usd": 75.0},
                        {"agent_id": sub_ids[2], "amount_usd": 25.0},
                    ]
                }
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_requested_usd"] == 150.0
        assert data["total_successful_usd"] == 150.0
        assert data["total_failed_usd"] == 0.0
        assert len(data["items"]) == 3
        for item in data["items"]:
            assert item["success"] is True
            assert item["tx_id"] is not None

    @pytest.mark.asyncio
    async def test_batch_transfer_single_agent(self, batch_app):
        """Single-agent batch transfer works (edge case of multi-agent)."""
        app, parent_key, _, _, sub_ids = batch_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/batch/transfer",
                headers=_headers(parent_key),
                json={"payments": [{"agent_id": sub_ids[0], "amount_usd": 30.0}]}
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_requested_usd"] == 30.0
        assert data["total_successful_usd"] == 30.0


class TestBatchTransferFailure:
    @pytest.mark.asyncio
    async def test_batch_transfer_insufficient_balance(self, batch_app):
        """Request more than agent balance — returns 409."""
        app, parent_key, _, _, sub_ids = batch_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/batch/transfer",
                headers=_headers(parent_key),
                json={"payments": [{"agent_id": sub_ids[0], "amount_usd": 9999.0}]}
            )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_batch_transfer_no_payments(self, batch_app):
        """Empty payments list returns 400."""
        app, parent_key, _, _, _ = batch_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/batch/transfer",
                headers=_headers(parent_key),
                json={"payments": []}
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_transfer_missing_field(self, batch_app):
        """Missing agent_id field returns 400."""
        app, parent_key, _, _, _ = batch_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/batch/transfer",
                headers=_headers(parent_key),
                json={"payments": [{"amount_usd": 10.0}]}
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_transfer_invalid_amount(self, batch_app):
        """Invalid amount returns 400."""
        app, parent_key, _, _, _ = batch_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/batch/transfer",
                headers=_headers(parent_key),
                json={"payments": [{"agent_id": "abc-123", "amount_usd": "notanumber"}]}
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_transfer_invalid_key(self, batch_app):
        """Bad API key returns 401."""
        app, _, _, _, sub_ids = batch_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/batch/transfer",
                headers={"X-API-Key": "ap_invalid_0000000000000000000000000000"},
                json={"payments": [{"agent_id": sub_ids[0], "amount_usd": 10.0}]}
            )
        assert resp.status_code == 401


class TestBatchTransferPartial:
    @pytest.mark.asyncio
    async def test_batch_transfer_one_invalid_target(self, batch_app):
        """One good + one bad agent_id — partial success."""
        app, parent_key, _, _, sub_ids = batch_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/batch/transfer",
                headers=_headers(parent_key),
                json={
                    "payments": [
                        {"agent_id": sub_ids[0], "amount_usd": 25.0},
                        {"agent_id": "nonexistent-agent-uuid", "amount_usd": 25.0},
                    ]
                }
            )
        assert resp.status_code == 201
        data = resp.json()
        assert len(data["items"]) == 2
        # First should succeed, second should fail
        good_item = next(i for i in data["items"] if i["agent_id"] == sub_ids[0])
        bad_item = next(i for i in data["items"] if i["success"] is False)
        assert good_item["success"] is True
        assert bad_item["error"] == "Target agent not found"


class TestAuthRejections:
    @pytest.mark.asyncio
    async def test_batch_transfer_no_auth(self, batch_app):
        """No auth header returns 422 (missing required X-API-Key)."""
        app, _, _, _, sub_ids = batch_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/batch/transfer",
                json={"payments": [{"agent_id": sub_ids[0], "amount_usd": 10.0}]}
            )
        assert resp.status_code == 422  # missing required X-API-Key header → FastAPI validation error
