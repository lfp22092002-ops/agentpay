"""
Tests for payee rules endpoints — CRUD and evaluation.
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
async def payee_app():
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

    api_key = "ap_payee_test_1234567890abcdef1234567890ab"
    async with factory() as db:
        user = User(telegram_id=99999999, username="payeeuser", first_name="Payee")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        agent = Agent(
            user_id=user.id,
            name="payee-agent",
            api_key_hash=hash_api_key(api_key),
            api_key_prefix="ap_paye...",
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


class TestPayeeRulesCRUD:
    @pytest.mark.asyncio
    async def test_list_empty(self, payee_app):
        """GET payee-rules with no rules returns empty list."""
        app, api_key, _ = payee_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/v1/agent/payee-rules",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["rules"] == []
            assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_create_allow_rule(self, payee_app):
        """POST payee-rules creates an allow rule."""
        app, api_key, _ = payee_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/agent/payee-rules",
                headers={"X-API-Key": api_key},
                json={
                    "rule_type": "allow",
                    "payee_type": "domain",
                    "payee_value": "api.openai.com",
                    "max_amount_usd": 5.00,
                    "note": "Allow OpenAI API calls",
                },
            )
            assert resp.status_code == 201
            data = resp.json()
            assert data["success"] is True
            assert data["rule"]["payee_value"] == "api.openai.com"
            assert data["rule"]["max_amount_usd"] == 5.0

    @pytest.mark.asyncio
    async def test_create_deny_rule(self, payee_app):
        """POST payee-rules creates a deny rule."""
        app, api_key, _ = payee_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/agent/payee-rules",
                headers={"X-API-Key": api_key},
                json={
                    "rule_type": "deny",
                    "payee_type": "category",
                    "payee_value": "gambling",
                },
            )
            assert resp.status_code == 201
            assert resp.json()["rule"]["rule_type"] == "deny"

    @pytest.mark.asyncio
    async def test_duplicate_rule_rejected(self, payee_app):
        """Duplicate active rules are rejected with 409."""
        app, api_key, _ = payee_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "rule_type": "allow",
                "payee_type": "agent_id",
                "payee_value": "some-agent-id",
            }
            resp1 = await client.post(
                "/v1/agent/payee-rules",
                headers={"X-API-Key": api_key},
                json=payload,
            )
            assert resp1.status_code == 201

            resp2 = await client.post(
                "/v1/agent/payee-rules",
                headers={"X-API-Key": api_key},
                json=payload,
            )
            assert resp2.status_code == 409

    @pytest.mark.asyncio
    async def test_list_after_create(self, payee_app):
        """GET payee-rules returns created rules."""
        app, api_key, _ = payee_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/v1/agent/payee-rules",
                headers={"X-API-Key": api_key},
                json={
                    "rule_type": "allow",
                    "payee_type": "domain",
                    "payee_value": "example.com",
                },
            )
            resp = await client.get(
                "/v1/agent/payee-rules",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_delete_rule(self, payee_app):
        """DELETE payee-rules/{id} deactivates the rule."""
        app, api_key, _ = payee_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_resp = await client.post(
                "/v1/agent/payee-rules",
                headers={"X-API-Key": api_key},
                json={
                    "rule_type": "allow",
                    "payee_type": "address",
                    "payee_value": "0x1234567890abcdef",
                },
            )
            rule_id = create_resp.json()["rule"]["id"]

            resp = await client.delete(
                f"/v1/agent/payee-rules/{rule_id}",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 200
            assert resp.json()["success"] is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, payee_app):
        """DELETE payee-rules with bad ID returns 404."""
        app, api_key, _ = payee_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.delete(
                "/v1/agent/payee-rules/nonexistent-id",
                headers={"X-API-Key": api_key},
            )
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_rule_type(self, payee_app):
        """POST with invalid rule_type returns 422."""
        app, api_key, _ = payee_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/agent/payee-rules",
                headers={"X-API-Key": api_key},
                json={
                    "rule_type": "maybe",
                    "payee_type": "domain",
                    "payee_value": "example.com",
                },
            )
            assert resp.status_code == 422


class TestPayeeEvaluation:
    @pytest.mark.asyncio
    async def test_open_by_default(self, payee_app):
        """No rules = all payments allowed."""
        app, _, agent = payee_app
        from models.database import get_db
        from api.routes.payee_rules import check_payee_allowed

        override_db = app.dependency_overrides[get_db]
        async for db in override_db():
            allowed, reason = await check_payee_allowed(
                db, agent.id, "domain", "anything.com", Decimal("10.00")
            )
            assert allowed is True
            assert reason is None
            break

    @pytest.mark.asyncio
    async def test_deny_blocks(self, payee_app):
        """A deny rule blocks matching payee."""
        app, api_key, agent = payee_app
        from models.database import get_db
        from api.routes.payee_rules import check_payee_allowed
        from models.schema import PayeeRule

        override_db = app.dependency_overrides[get_db]
        async for db in override_db():
            rule = PayeeRule(
                agent_id=agent.id,
                rule_type="deny",
                payee_type="domain",
                payee_value="evil.com",
            )
            db.add(rule)
            await db.commit()

            allowed, reason = await check_payee_allowed(
                db, agent.id, "domain", "evil.com", Decimal("1.00")
            )
            assert allowed is False
            assert "denied" in reason
            break

    @pytest.mark.asyncio
    async def test_allowlist_blocks_unlisted(self, payee_app):
        """When allow rules exist, unlisted payees are blocked."""
        app, api_key, agent = payee_app
        from models.database import get_db
        from api.routes.payee_rules import check_payee_allowed
        from models.schema import PayeeRule

        override_db = app.dependency_overrides[get_db]
        async for db in override_db():
            rule = PayeeRule(
                agent_id=agent.id,
                rule_type="allow",
                payee_type="domain",
                payee_value="api.openai.com",
            )
            db.add(rule)
            await db.commit()

            # Allowed payee passes
            allowed, _ = await check_payee_allowed(
                db, agent.id, "domain", "api.openai.com", Decimal("5.00")
            )
            assert allowed is True

            # Unlisted payee blocked
            allowed, reason = await check_payee_allowed(
                db, agent.id, "domain", "unknown.com", Decimal("5.00")
            )
            assert allowed is False
            assert "not in allowlist" in reason
            break

    @pytest.mark.asyncio
    async def test_amount_cap_enforcement(self, payee_app):
        """Per-payee amount cap is enforced."""
        app, api_key, agent = payee_app
        from models.database import get_db
        from api.routes.payee_rules import check_payee_allowed
        from models.schema import PayeeRule

        override_db = app.dependency_overrides[get_db]
        async for db in override_db():
            rule = PayeeRule(
                agent_id=agent.id,
                rule_type="allow",
                payee_type="domain",
                payee_value="api.openai.com",
                max_amount_usd=Decimal("2.00"),
            )
            db.add(rule)
            await db.commit()

            # Under cap → allowed
            allowed, _ = await check_payee_allowed(
                db, agent.id, "domain", "api.openai.com", Decimal("1.50")
            )
            assert allowed is True

            # Over cap → blocked
            allowed, reason = await check_payee_allowed(
                db, agent.id, "domain", "api.openai.com", Decimal("3.00")
            )
            assert allowed is False
            assert "exceeds" in reason
            break

    @pytest.mark.asyncio
    async def test_list_rule_with_max_amount(self, payee_app):
        """Listed rule with max_amount_usd returns float value."""
        app, api_key, _ = payee_app
        from httpx import AsyncClient, ASGITransport

        # Create rule with max_amount
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await c.post(
                "/v1/agent/payee-rules",
                json={"rule_type": "allow", "payee_type": "domain", "payee_value": "capped.com", "max_amount_usd": 5.0},
                headers={"X-API-Key": api_key},
            )
            r = await c.get("/v1/agent/payee-rules", headers={"X-API-Key": api_key})
        assert r.status_code == 200
        rules = r.json()["rules"]
        capped = next((x for x in rules if x["payee_value"] == "capped.com"), None)
        assert capped is not None
        assert capped["max_amount_usd"] == 5.0

    @pytest.mark.asyncio
    async def test_deny_no_matching_allow_rule(self, payee_app):
        """check_payee_allowed: only-deny rules + no match = allowed."""
        app, api_key, agent = payee_app
        from models.database import get_db
        from api.routes.payee_rules import check_payee_allowed
        from models.schema import PayeeRule

        override_db = app.dependency_overrides[get_db]
        async for db in override_db():
            rule = PayeeRule(
                agent_id=agent.id,
                rule_type="deny",
                payee_type="domain",
                payee_value="blocked.com",
            )
            db.add(rule)
            await db.commit()

            # Different domain — only deny rules exist, no match → allowed
            allowed, reason = await check_payee_allowed(
                db, agent.id, "domain", "other.com", Decimal("1.00")
            )
            assert allowed is True
            assert reason is None
            break
