"""Tests for api/routes/stripe.py — Stripe checkout & webhook."""
import os
import sys
from decimal import Decimal

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from unittest.mock import patch, MagicMock, AsyncMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.database import Base
from models.schema import User, Agent, PlatformRevenue, Transaction, TransactionType, TransactionStatus
from core.wallet import hash_api_key


def _headers(key: str):
    return {"X-API-Key": key}


@pytest_asyncio.fixture
async def stripe_app():
    """Create a test app with user and two agents."""
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

    parent_key = "ap_stripe_parent_1234567890abc"

    async with factory() as db:
        user = User(telegram_id=88888888, username="stripeuser", first_name="Stripe")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        parent_agent = Agent(
            user_id=user.id,
            name="parent-agent",
            api_key_hash=hash_api_key(parent_key),
            api_key_prefix="ap_strip...",
            balance_usd=Decimal("10.0000"),
            daily_limit_usd=Decimal("500.0000"),
            tx_limit_usd=Decimal("200.0000"),
            is_active=True,
        )
        db.add(parent_agent)

        child_agent = Agent(
            user_id=user.id,
            name="child-agent",
            api_key_hash=hash_api_key("ap_stripe_child_1234567890def"),
            api_key_prefix="ap_strip...",
            balance_usd=Decimal("0.0000"),
            daily_limit_usd=Decimal("50.0000"),
            tx_limit_usd=Decimal("10.0000"),
            is_active=True,
        )
        db.add(child_agent)

        await db.commit()
        await db.refresh(parent_agent)
        await db.refresh(child_agent)

    yield app, parent_key, parent_agent.id, child_agent.id

    app.dependency_overrides.clear()
    await engine.dispose()


class TestCheckoutSession:
    @pytest.mark.asyncio
    async def test_checkout_success(self, stripe_app):
        """Stripe checkout returns URL when configured."""
        app, parent_key, _, child_id = stripe_app

        fake_session = MagicMock()
        fake_session.url = "https://checkout.stripe.com/fake"
        fake_session.id = "cs_test_fake123"

        with patch("stripe.checkout.Session.create", return_value=fake_session):
            old_key = os.environ.get("STRIPE_SECRET_KEY")
            old_wh = os.environ.get("STRIPE_WEBHOOK_SECRET")
            os.environ["STRIPE_SECRET_KEY"] = "sk_test_fake"
            os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_fake"

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    f"/v1/funding/stripe/create-checkout?agent_id={child_id}&amount_usd=25.0",
                    headers=_headers(parent_key)
                )
            
            # Restore env vars
            if old_key is not None:
                os.environ["STRIPE_SECRET_KEY"] = old_key
            else:
                os.environ.pop("STRIPE_SECRET_KEY", None)
            if old_wh is not None:
                os.environ["STRIPE_WEBHOOK_SECRET"] = old_wh
            else:
                os.environ.pop("STRIPE_WEBHOOK_SECRET", None)

        assert resp.status_code == 200
        data = resp.json()
        assert "checkout_url" in data
        assert data["checkout_url"].startswith("https://checkout.stripe.com")
        assert data["amount_usd"] == 25.0
        assert data["session_id"] == "cs_test_fake123"

    @pytest.mark.asyncio
    async def test_checkout_stripe_not_configured(self, stripe_app):
        """Returns 503 when Stripe key is missing."""
        app, parent_key, _, child_id = stripe_app

        # Ensure no Stripe key
        old_key = os.environ.pop("STRIPE_SECRET_KEY", None)
        old_wh = os.environ.pop("STRIPE_WEBHOOK_SECRET", None)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/v1/funding/stripe/create-checkout?agent_id={child_id}&amount_usd=25.0",
                headers=_headers(parent_key)
            )
        
        # Restore if needed
        if old_key:
            os.environ["STRIPE_SECRET_KEY"] = old_key
        if old_wh:
            os.environ["STRIPE_WEBHOOK_SECRET"] = old_wh

        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_checkout_invalid_agent(self, stripe_app):
        """Returns 403 for non-existent agent_id."""
        app, parent_key, _, child_id = stripe_app

        fake_session = MagicMock()
        fake_session.url = "https://checkout.stripe.com/fake"
        fake_session.id = "cs_test_fake123"

        with patch("stripe.checkout.Session.create", return_value=fake_session):
            os.environ["STRIPE_SECRET_KEY"] = "sk_test_fake"
            os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_fake"

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    f"/v1/funding/stripe/create-checkout?agent_id=nonexistent-agent-uuid&amount_usd=25.0",
                    headers=_headers(parent_key)
                )

            os.environ.pop("STRIPE_SECRET_KEY", None)
            os.environ.pop("STRIPE_WEBHOOK_SECRET", None)

        assert resp.status_code == 403


class TestWebhook:
    @pytest.mark.asyncio
    async def test_webhook_session_completed(self, stripe_app):
        """Stripe webhook credits agent balance on success."""
        from sqlalchemy import select as sa_select
        from models.database import get_db
        
        app, _, parent_id, child_id = stripe_app
        
        # Get initial balance
        override_db = app.dependency_overrides[get_db]
        override_db = app.dependency_overrides[get_db]
        async for db in override_db():
            result = await db.execute(sa_select(Agent).where(Agent.id == child_id))
            initial_balance = result.scalar_one().balance_usd
            break
        
        # Mock a completed checkout session event
        import stripe
        fake_session = MagicMock()
        fake_session.id = "cs_test_fake123"
        fake_session.metadata = {
            "agentpay_agent_id": child_id,
            "amount_usd": "50.0",
        }
        fake_event = type('Event', (), {
            'type': 'checkout.session.completed',
            'data': MagicMock(object=fake_session)
        })
        
        with patch("stripe.Webhook.construct_event", return_value=fake_event):
            old_key = os.environ.get("STRIPE_WEBHOOK_SECRET")
            os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_fake"

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/funding/stripe/webhook", json={})

            if old_key is not None:
                os.environ["STRIPE_WEBHOOK_SECRET"] = old_key
            else:
                os.environ.pop("STRIPE_WEBHOOK_SECRET", None)

        assert resp.status_code == 200
        
        # Verify balance increased (50 - 0.50 fee = 49.50)
        async for db in override_db():
            result = await db.execute(sa_select(Agent).where(Agent.id == child_id))
            new_balance = result.scalar_one().balance_usd
            assert new_balance - initial_balance == Decimal("49.50")
            break

    @pytest.mark.asyncio
    async def test_webhook_invalid_signature(self, stripe_app):
        """Returns 400 for invalid webhook signature."""
        app, _, _, _ = stripe_app
        
        old_key = os.environ.get("STRIPE_WEBHOOK_SECRET")
        os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_fake"

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/funding/stripe/webhook", json={})

        if old_key is not None:
            os.environ["STRIPE_WEBHOOK_SECRET"] = old_key
        else:
            os.environ.pop("STRIPE_WEBHOOK_SECRET", None)

        assert resp.status_code == 400


class TestAuthRejections:
    @pytest.mark.asyncio
    async def test_checkout_no_auth(self, stripe_app):
        """No API key returns 401."""
        app, _, child_id, _ = stripe_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                f"/v1/funding/stripe/create-checkout?agent_id={child_id}&amount_usd=25.0"
            )
        assert resp.status_code == 401
