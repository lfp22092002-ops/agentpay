"""
Additional identity tests to cover uncovered lines:
- _refresh_identity_counters (lines 88-91)
- get_own_identity with existing identity (lines 104-113)
- upsert update path (lines 142-172)
- trust score endpoint (lines 198-207)
- directory null-byte / edge cases (lines 231, 242-247)
- public profile output (lines 277-281)
"""
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.database import Base
from models.schema import User, Agent, AgentIdentity, Transaction, TransactionType, TransactionStatus
from core.wallet import hash_api_key


@pytest_asyncio.fixture
async def identity_app_with_txs():
    """App with a user, agent, identity, and some completed transactions."""
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

    api_key = "ap_idcov_test_1234567890abcdef1234567890"
    async with factory() as db:
        user = User(telegram_id=77777777, username="covuser", first_name="Coverage")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        agent = Agent(
            user_id=user.id,
            name="cov-agent",
            api_key_hash=hash_api_key(api_key),
            api_key_prefix="ap_idco...",
            balance_usd=Decimal("500.0000"),
            daily_limit_usd=Decimal("100.0000"),
            tx_limit_usd=Decimal("50.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

        # Add completed transactions so _refresh_identity_counters has data
        for i in range(5):
            tx = Transaction(
                agent_id=agent.id,
                tx_type=TransactionType.SPEND,
                amount_usd=Decimal("10.0000"),
                fee_usd=Decimal("0.2000"),
                description=f"Test tx {i}",
                status=TransactionStatus.COMPLETED,
            )
            db.add(tx)
        await db.commit()

        # Pre-create identity so GET tests hit the existing-identity path
        identity = AgentIdentity(
            agent_id=agent.id,
            display_name="Existing Agent",
            description="Already created",
            category="defi",
            homepage_url="https://example.com",
            logo_url="https://example.com/logo.png",
            first_seen=agent.created_at,
            last_active=datetime.now(timezone.utc),
        )
        db.add(identity)
        await db.commit()

    yield app, api_key, agent

    app.dependency_overrides.pop(get_db, None)
    # Reset rate limiter to avoid cross-test bleed
    from api.middleware import limiter
    limiter.reset()
    await engine.dispose()


class TestRefreshCounters:
    """Tests that hit _refresh_identity_counters (lines 88-91)."""

    @pytest.mark.asyncio
    async def test_get_identity_refreshes_counters(self, identity_app_with_txs):
        """GET identity refreshes tx counts from actual transaction data."""
        app, api_key, _ = identity_app_with_txs
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            ident = data["identity"]
            assert ident is not None
            # Should reflect the 5 completed transactions
            assert ident["total_transactions"] == 5
            assert ident["total_volume_usd"] == 50.0
            assert ident["display_name"] == "Existing Agent"
            assert "first_seen" in ident
            assert "last_active" in ident


class TestGetExistingIdentity:
    """Tests that hit the full GET identity return path (lines 104-113)."""

    @pytest.mark.asyncio
    async def test_get_identity_returns_all_fields(self, identity_app_with_txs):
        """GET identity returns full AgentIdentityOut with all fields populated."""
        app, api_key, _ = identity_app_with_txs
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            ident = data["identity"]
            assert ident["description"] == "Already created"
            assert ident["homepage_url"] == "https://example.com"
            assert ident["logo_url"] == "https://example.com/logo.png"
            assert ident["category"] == "defi"
            assert ident["verified"] is False
            assert ident["trust_score"] >= 0
            assert ident["metadata_json"] is None


class TestUpsertIdentityUpdate:
    """Tests that hit the UPDATE branch of upsert (lines 142-172)."""

    @pytest.mark.asyncio
    async def test_update_existing_identity(self, identity_app_with_txs):
        """PUT identity when identity exists updates fields."""
        app, api_key, _ = identity_app_with_txs
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
                json={
                    "display_name": "Updated Agent",
                    "description": "Updated description",
                    "homepage_url": "https://updated.com",
                    "logo_url": "https://updated.com/logo.png",
                    "category": "trading",
                    "metadata_json": '{"v": 2}',
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            ident = data["identity"]
            assert ident["display_name"] == "Updated Agent"
            assert ident["description"] == "Updated description"
            assert ident["homepage_url"] == "https://updated.com"
            assert ident["logo_url"] == "https://updated.com/logo.png"
            assert ident["category"] == "trading"
            assert ident["metadata_json"] == '{"v": 2}'
            # Counters should be refreshed
            assert ident["total_transactions"] == 5
            assert ident["total_volume_usd"] == 50.0

    @pytest.mark.asyncio
    async def test_update_partial_fields(self, identity_app_with_txs):
        """PUT identity with partial fields updates only provided fields."""
        app, api_key, _ = identity_app_with_txs
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
                json={
                    "display_name": "Partial Update",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["identity"]["display_name"] == "Partial Update"


class TestTrustScoreEndpoint:
    """Tests that hit the trust score endpoint with existing identity (lines 198-207)."""

    @pytest.mark.asyncio
    async def test_score_with_transactions(self, identity_app_with_txs):
        """GET score with transactions returns non-zero breakdown."""
        app, api_key, _ = identity_app_with_txs
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/agent/identity/score",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] >= 0
            assert data["total"] <= 100
            assert "details" in data
            assert data["details"]["total_transactions"] == 5
            assert data["details"]["total_volume_usd"] == 50.0


class TestDirectoryEdgeCases:
    """Tests for directory null-byte sanitization and edge cases (lines 231, 242-247)."""

    @pytest.mark.asyncio
    async def test_directory_null_byte_category(self, identity_app_with_txs):
        """Directory with null byte in category sanitizes it."""
        app, _, _ = identity_app_with_txs
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/directory?category=%00bad")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_directory_empty_category_after_strip(self, identity_app_with_txs):
        """Directory with category that becomes empty after sanitization."""
        app, _, _ = identity_app_with_txs
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/directory?category=%20%20")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_directory_long_category(self, identity_app_with_txs):
        """Directory with overly long category treats as None."""
        app, _, _ = identity_app_with_txs
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            long_cat = "x" * 100
            resp = await client.get(f"/v1/directory?category={long_cat}")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_directory_with_category_filter_match(self, identity_app_with_txs):
        """Directory with matching category filter returns the agent."""
        app, _, _ = identity_app_with_txs
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/directory?category=defi")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 1
            assert data["agents"][0]["category"] == "defi"

    @pytest.mark.asyncio
    async def test_directory_page_bounds(self, identity_app_with_txs):
        """Directory with out-of-range page/page_size clamps correctly."""
        app, _, _ = identity_app_with_txs
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/directory?page=0&page_size=0")
            assert resp.status_code == 200
            data = resp.json()
            assert data["page"] == 1
            assert data["page_size"] == 1


class TestPublicProfile:
    """Tests for GET /directory/{agent_id} full output (lines 277-281)."""

    @pytest.mark.asyncio
    async def test_public_profile_all_fields(self, identity_app_with_txs):
        """Public profile returns all AgentIdentityOut fields."""
        app, _, agent = identity_app_with_txs
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/v1/directory/{agent.id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["display_name"] == "Existing Agent"
            assert data["description"] == "Already created"
            assert data["homepage_url"] == "https://example.com"
            assert data["logo_url"] == "https://example.com/logo.png"
            assert data["category"] == "defi"
            assert data["verified"] is False
            assert "first_seen" in data
            assert "last_active" in data
            assert "metadata_json" in data
