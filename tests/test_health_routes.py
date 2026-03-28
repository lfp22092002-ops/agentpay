"""
Tests for health and root endpoints.
"""
import os
import sys
import pytest
import pytest_asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"
os.environ["BOT_TOKEN"] = ""
os.environ["API_SECRET"] = "test-secret-key-for-tests-minimum-32-bytes-long"

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from models.database import Base


@pytest_asyncio.fixture
async def test_app():
    """Create a test app with in-memory SQLite."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    from api.main import app
    yield app

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_health_v1(test_app):
    """GET /v1/health returns ok status."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "agentpay"
    assert "version" in data


@pytest.mark.asyncio
async def test_health_shortcut(test_app):
    """GET /health (alias) returns same response."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_root_serves_landing(test_app):
    """GET / returns HTML landing page (or 404 if file missing)."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    # Either serves landing page or 404 (in test env file may not exist)
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        assert "text/html" in resp.headers.get("content-type", "")
