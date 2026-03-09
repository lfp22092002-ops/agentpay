"""
Tests for api/dependencies.py — auth flow: rate limit, invalid key, deactivated agent.
"""
import os
import sys
from decimal import Decimal

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.schema import User, Agent
from core.wallet import hash_api_key


@pytest_asyncio.fixture
async def deps_app(engine, db):
    """Create test app with dependency injection overrides."""
    from api.main import app
    from api.middleware import limiter
    from models.database import get_db

    # Create a user and agent
    user = User(telegram_id=77777777, username="depuser", first_name="Dep")
    db.add(user)
    await db.commit()
    await db.refresh(user)

    api_key = "ap_deps_test_key_abcdef1234567890abcdef1234567890"
    agent = Agent(
        user_id=user.id,
        name="deps-agent",
        api_key_hash=hash_api_key(api_key),
        api_key_prefix="ap_deps_...",
        balance_usd=Decimal("100.0000"),
        daily_limit_usd=Decimal("50.0000"),
        tx_limit_usd=Decimal("25.0000"),
        is_active=True,
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    agent._test_api_key = api_key

    # Create a deactivated agent
    deact_key = "ap_deact_key_zzzzzz1234567890abcdef1234567890"
    deact_agent = Agent(
        user_id=user.id,
        name="deactivated-agent",
        api_key_hash=hash_api_key(deact_key),
        api_key_prefix="ap_deact...",
        balance_usd=Decimal("10.0000"),
        daily_limit_usd=Decimal("50.0000"),
        tx_limit_usd=Decimal("25.0000"),
        is_active=False,
    )
    db.add(deact_agent)
    await db.commit()

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    limiter.reset()
    yield app, api_key, deact_key
    app.dependency_overrides.clear()


class TestDependenciesAuth:
    @pytest.mark.asyncio
    async def test_valid_key_passes(self, deps_app):
        """Valid API key returns 200."""
        app, api_key, _ = deps_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/balance", headers={"X-API-Key": api_key})
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_key_401(self, deps_app):
        """Invalid API key returns 401."""
        app, _, _ = deps_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/balance", headers={"X-API-Key": "ap_totally_bogus"})
            assert resp.status_code == 401
            assert "Invalid" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_deactivated_agent_rejected(self, deps_app):
        """Deactivated agent is rejected (401 — get_agent_by_api_key filters is_active)."""
        app, _, deact_key = deps_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/balance", headers={"X-API-Key": deact_key})
            # wallet.get_agent_by_api_key filters is_active=True, so deactivated → None → 401
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_rate_limit_429(self, deps_app):
        """Rate limited API key returns 429."""
        app, api_key, _ = deps_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("api.dependencies.check_api_key_rate_limit", return_value=False):
                resp = await client.get("/v1/balance", headers={"X-API-Key": api_key})
                assert resp.status_code == 429
                assert "Rate limit" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_missing_key_422(self, deps_app):
        """Missing API key returns 422."""
        app, _, _ = deps_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/balance")
            assert resp.status_code == 422


class TestMiniappUserDep:
    def test_get_miniapp_user_dep_returns_callable(self):
        """get_miniapp_user_dep returns the get_miniapp_user function."""
        from api.dependencies import get_miniapp_user_dep
        result = get_miniapp_user_dep()
        assert callable(result)
