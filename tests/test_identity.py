"""
Tests for identity endpoints — CRUD, trust score, directory.
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
from models.schema import User, Agent
from core.wallet import hash_api_key


@pytest_asyncio.fixture
async def identity_app():
    """Create a test app with a fresh DB, user, and agent."""
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

    api_key = "ap_identity_test_1234567890abcdef1234567890"
    async with factory() as db:
        user = User(telegram_id=88888888, username="iduser", first_name="Identity")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        agent = Agent(
            user_id=user.id,
            name="id-agent",
            api_key_hash=hash_api_key(api_key),
            api_key_prefix="ap_iden...",
            balance_usd=Decimal("100.0000"),
            daily_limit_usd=Decimal("50.0000"),
            tx_limit_usd=Decimal("25.0000"),
            is_active=True,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

    yield app, api_key, agent

    app.dependency_overrides.clear()
    await engine.dispose()


class TestIdentityGet:
    @pytest.mark.asyncio
    async def test_get_no_identity(self, identity_app):
        """GET identity when none exists returns None."""
        app, api_key, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["identity"] is None

    @pytest.mark.asyncio
    async def test_get_identity_unauthorized(self, identity_app):
        """GET identity without key returns 401/403."""
        app, _, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/agent/identity")
            assert resp.status_code in (401, 403, 422)


class TestIdentityUpsert:
    @pytest.mark.asyncio
    async def test_create_identity(self, identity_app):
        """PUT identity creates a new profile."""
        app, api_key, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
                json={
                    "display_name": "Test Agent",
                    "description": "A test agent for testing",
                    "category": "defi",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["identity"]["display_name"] == "Test Agent"
            assert data["identity"]["category"] == "defi"
            assert data["identity"]["trust_score"] >= 0

    @pytest.mark.asyncio
    async def test_update_identity(self, identity_app):
        """PUT identity twice updates existing profile."""
        app, api_key, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create
            await client.put(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
                json={"display_name": "Original"},
            )
            # Update
            resp = await client.put(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
                json={"display_name": "Updated", "description": "Now with description"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["identity"]["display_name"] == "Updated"
            assert data["identity"]["description"] == "Now with description"

    @pytest.mark.asyncio
    async def test_invalid_category(self, identity_app):
        """PUT identity with invalid category returns 400."""
        app, api_key, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
                json={"display_name": "Bad", "category": "not-a-real-category"},
            )
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_full_profile(self, identity_app):
        """PUT identity with all fields."""
        app, api_key, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
                json={
                    "display_name": "Full Agent",
                    "description": "Comprehensive profile",
                    "homepage_url": "https://example.com",
                    "logo_url": "https://example.com/logo.png",
                    "category": "trading",
                    "metadata_json": '{"version": "1.0"}',
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            ident = data["identity"]
            assert ident["homepage_url"] == "https://example.com"
            assert ident["logo_url"] == "https://example.com/logo.png"
            # Profile completeness should boost trust score
            assert ident["trust_score"] >= 0


class TestTrustScore:
    @pytest.mark.asyncio
    async def test_score_no_identity(self, identity_app):
        """GET score without identity returns 404."""
        app, api_key, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/agent/identity/score",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_score_with_identity(self, identity_app):
        """GET score returns breakdown after creating identity."""
        app, api_key, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create identity first
            await client.put(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
                json={"display_name": "Scored Agent"},
            )
            resp = await client.get(
                "/v1/agent/identity/score",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "total" in data
            assert "account_age_pts" in data
            assert "transaction_count_pts" in data
            assert "volume_pts" in data
            assert "profile_completeness_pts" in data
            assert "verified_pts" in data
            assert data["total"] >= 0
            assert data["total"] <= 100


class TestGetIdentityAfterCreate:
    @pytest.mark.asyncio
    async def test_create_returns_full_profile(self, identity_app):
        """PUT identity returns the created profile with all fields."""
        app, api_key, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
                json={"display_name": "Readable Agent", "category": "analytics"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["identity"]["display_name"] == "Readable Agent"
            assert data["identity"]["category"] == "analytics"
            assert "trust_score" in data["identity"]
            assert "first_seen" in data["identity"]
            assert "last_active" in data["identity"]


class TestDirectory:
    @pytest.mark.asyncio
    async def test_directory_empty(self, identity_app):
        """GET directory when no identities exist returns empty list."""
        app, _, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/directory")
            assert resp.status_code == 200
            data = resp.json()
            assert data["agents"] == []
            assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_directory_with_agents(self, identity_app):
        """GET directory after creating identity shows agent."""
        app, api_key, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create identity
            await client.put(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
                json={"display_name": "Listed Agent", "category": "defi"},
            )
            resp = await client.get("/v1/directory")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 1
            assert data["agents"][0]["display_name"] == "Listed Agent"

    @pytest.mark.asyncio
    async def test_directory_filter_category(self, identity_app):
        """GET directory with category filter."""
        app, api_key, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.put(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
                json={"display_name": "DeFi Agent", "category": "defi"},
            )
            # Matching category
            resp = await client.get("/v1/directory?category=defi")
            assert resp.status_code == 200
            assert resp.json()["total"] == 1

            # Non-matching category
            resp = await client.get("/v1/directory?category=trading")
            assert resp.status_code == 200
            assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_directory_pagination(self, identity_app):
        """GET directory respects page_size."""
        app, _, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/directory?page=1&page_size=5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["page"] == 1
            assert data["page_size"] == 5

    @pytest.mark.asyncio
    async def test_directory_public_profile(self, identity_app):
        """GET directory/{agent_id} returns public profile."""
        app, api_key, agent = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.put(
                "/v1/agent/identity",
                headers={"X-API-Key": api_key},
                json={"display_name": "Public Agent"},
            )
            resp = await client.get(f"/v1/directory/{agent.id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["display_name"] == "Public Agent"

    @pytest.mark.asyncio
    async def test_directory_not_found(self, identity_app):
        """GET directory/{agent_id} with bad ID returns 404."""
        app, _, _ = identity_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/directory/nonexistent-id-12345")
            assert resp.status_code == 404
