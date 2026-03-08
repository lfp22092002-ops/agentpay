"""
Tests for the approval workflow system.
"""
from decimal import Decimal

import pytest

from core.approvals import (
    request_approval,
    resolve_approval,
    get_pending,
    get_agent_pending,
    get_user_pending,
    create_approval_id,
    _pending,
)


class TestCreateApprovalId:
    def test_prefix(self):
        aid = create_approval_id()
        assert aid.startswith("apr_")

    def test_uniqueness(self):
        ids = {create_approval_id() for _ in range(20)}
        assert len(ids) == 20

    def test_length(self):
        aid = create_approval_id()
        # apr_ (4) + 16 hex chars = 20
        assert len(aid) == 20


class TestRequestApproval:
    def setup_method(self):
        _pending.clear()

    def teardown_method(self):
        _pending.clear()

    @pytest.mark.asyncio
    async def test_creates_pending(self):
        approval = await request_approval(
            agent_id="agent_1",
            agent_name="test-agent",
            user_id="user_1",
            telegram_id=12345,
            amount_usd=Decimal("25.00"),
            description="Buy API credits",
        )
        assert approval.id in _pending
        assert approval.agent_id == "agent_1"
        assert approval.amount_usd == Decimal("25.00")
        assert approval.resolved is False
        assert approval.result is not None

    @pytest.mark.asyncio
    async def test_multiple_pending(self):
        a1 = await request_approval("a1", "agent-1", "u1", 111, Decimal("10"))
        a2 = await request_approval("a2", "agent-2", "u2", 222, Decimal("20"))
        assert len(_pending) == 2
        assert a1.id != a2.id


class TestResolveApproval:
    def setup_method(self):
        _pending.clear()

    def teardown_method(self):
        _pending.clear()

    @pytest.mark.asyncio
    async def test_approve(self):
        approval = await request_approval("a1", "agent", "u1", 111, Decimal("50"))
        result = resolve_approval(approval.id, approved=True, reason="Looks good")
        assert result is True
        assert approval.resolved is True
        # Future should be resolved
        assert approval.result.done()
        r = approval.result.result()
        assert r["approved"] is True
        assert r["reason"] == "Looks good"

    @pytest.mark.asyncio
    async def test_deny(self):
        approval = await request_approval("a1", "agent", "u1", 111, Decimal("50"))
        result = resolve_approval(approval.id, approved=False, reason="Too expensive")
        assert result is True
        r = approval.result.result()
        assert r["approved"] is False

    @pytest.mark.asyncio
    async def test_resolve_nonexistent(self):
        result = resolve_approval("apr_doesnotexist", approved=True)
        assert result is False

    @pytest.mark.asyncio
    async def test_double_resolve(self):
        approval = await request_approval("a1", "agent", "u1", 111, Decimal("50"))
        resolve_approval(approval.id, approved=True)
        # Second resolve should fail
        result = resolve_approval(approval.id, approved=False)
        assert result is False
        # Original decision stands
        r = approval.result.result()
        assert r["approved"] is True

    @pytest.mark.asyncio
    async def test_default_reason_approve(self):
        approval = await request_approval("a1", "agent", "u1", 111, Decimal("5"))
        resolve_approval(approval.id, approved=True)
        r = approval.result.result()
        assert r["reason"] == "Approved by user"

    @pytest.mark.asyncio
    async def test_default_reason_deny(self):
        approval = await request_approval("a1", "agent", "u1", 111, Decimal("5"))
        resolve_approval(approval.id, approved=False)
        r = approval.result.result()
        assert r["reason"] == "Denied by user"


class TestGetPending:
    def setup_method(self):
        _pending.clear()

    def teardown_method(self):
        _pending.clear()

    @pytest.mark.asyncio
    async def test_get_by_id(self):
        approval = await request_approval("a1", "agent", "u1", 111, Decimal("10"))
        found = get_pending(approval.id)
        assert found is not None
        assert found.id == approval.id

    @pytest.mark.asyncio
    async def test_get_missing(self):
        assert get_pending("apr_nope") is None

    @pytest.mark.asyncio
    async def test_get_agent_pending(self):
        await request_approval("a1", "agent-1", "u1", 111, Decimal("10"))
        await request_approval("a1", "agent-1", "u1", 111, Decimal("20"))
        await request_approval("a2", "agent-2", "u2", 222, Decimal("30"))
        pending = get_agent_pending("a1")
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_get_agent_pending_excludes_resolved(self):
        a = await request_approval("a1", "agent", "u1", 111, Decimal("10"))
        await request_approval("a1", "agent", "u1", 111, Decimal("20"))
        resolve_approval(a.id, approved=True)
        pending = get_agent_pending("a1")
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_get_user_pending(self):
        await request_approval("a1", "agent-1", "u1", 111, Decimal("10"))
        await request_approval("a2", "agent-2", "u1", 111, Decimal("20"))
        await request_approval("a3", "agent-3", "u2", 222, Decimal("30"))
        pending = get_user_pending(111)
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_get_user_pending_excludes_resolved(self):
        a = await request_approval("a1", "agent", "u1", 111, Decimal("10"))
        await request_approval("a2", "agent-2", "u1", 111, Decimal("20"))
        resolve_approval(a.id, approved=False)
        pending = get_user_pending(111)
        assert len(pending) == 1
