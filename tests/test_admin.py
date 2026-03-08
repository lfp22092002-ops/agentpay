"""
Tests for admin API routes — revenue tracking and withdrawal.
"""
import os
import sys
from decimal import Decimal

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import jwt as pyjwt
from models.schema import PlatformRevenue
from config.settings import API_SECRET


def _admin_token():
    """Create a valid admin JWT."""
    return pyjwt.encode(
        {"telegram_id": 5360481016, "exp": 9999999999},
        API_SECRET,
        algorithm="HS256",
    )


def _non_admin_token():
    """Create a non-admin JWT."""
    return pyjwt.encode(
        {"telegram_id": 99999, "exp": 9999999999},
        API_SECRET,
        algorithm="HS256",
    )


@pytest_asyncio.fixture
async def admin_app(engine, db):
    """Create a test FastAPI app with admin routes."""
    from api.main import app
    from api.middleware import limiter
    from models.database import get_db

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    # Reset rate limiter between tests
    limiter.reset()
    yield app
    app.dependency_overrides.clear()


class TestRevenueEndpoint:
    @pytest.mark.asyncio
    async def test_revenue_no_auth(self, admin_app):
        """Revenue endpoint rejects unauthenticated requests."""
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/admin/revenue")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_revenue_non_admin(self, admin_app):
        """Revenue endpoint rejects non-admin users."""
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/admin/revenue",
                headers={"Authorization": f"Bearer {_non_admin_token()}"},
            )
            assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_revenue_admin_empty(self, admin_app, db):
        """Admin sees zero revenue when no fees collected."""
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/admin/revenue",
                headers={"Authorization": f"Bearer {_admin_token()}"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_revenue_usd"] == 0.0
            assert data["today_revenue_usd"] == 0.0
            assert data["total_fee_transactions"] == 0

    @pytest.mark.asyncio
    async def test_revenue_admin_with_data(self, admin_app, db):
        """Admin sees correct revenue totals."""
        rev1 = PlatformRevenue(
            transaction_id="tx-test-001",
            agent_id="agent-test-001",
            amount_usd=Decimal("0.50"),
        )
        rev2 = PlatformRevenue(
            transaction_id="tx-test-002",
            agent_id="agent-test-001",
            amount_usd=Decimal("1.25"),
        )
        db.add(rev1)
        db.add(rev2)
        await db.commit()

        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/admin/revenue",
                headers={"Authorization": f"Bearer {_admin_token()}"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_revenue_usd"] == 1.75
            assert data["total_fee_transactions"] == 2


class TestWithdrawEndpoint:
    @pytest.mark.asyncio
    async def test_withdraw_no_auth(self, admin_app):
        """Withdraw rejects unauthenticated requests."""
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/admin/withdraw")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_withdraw_non_admin(self, admin_app):
        """Withdraw rejects non-admin users."""
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/admin/withdraw",
                headers={"Authorization": f"Bearer {_non_admin_token()}"},
            )
            assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_withdraw_empty(self, admin_app, db):
        """Withdraw with no revenue returns error."""
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/admin/withdraw",
                headers={"Authorization": f"Bearer {_admin_token()}"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False
            assert "No revenue" in data["error"]

    @pytest.mark.asyncio
    async def test_withdraw_with_revenue(self, admin_app, db):
        """Withdraw with revenue returns total and wallet address."""
        rev = PlatformRevenue(
            transaction_id="tx-test-003",
            agent_id="agent-test-001",
            amount_usd=Decimal("5.00"),
        )
        db.add(rev)
        await db.commit()

        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/admin/withdraw",
                headers={"Authorization": f"Bearer {_admin_token()}"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["total_revenue_usd"] == 5.0
            assert "0xD51B" in data["withdraw_to"]
