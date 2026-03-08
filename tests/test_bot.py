"""
Tests for bot/main.py — command handlers and callback logic.

Uses mocked aiogram Message/CallbackQuery objects to test handler behavior
without a running Telegram bot. Tests verify database side effects and
response text patterns.
"""
import asyncio
import os
import sys
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.schema import User, Agent, Transaction, TransactionType, TransactionStatus, PaymentMethod
from core.wallet import hash_api_key, get_or_create_user, create_agent, get_user_agents


# ═══════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════

def make_message(text: str, user_id: int = 12345678, username: str = "testuser", first_name: str = "Test"):
    """Create a mock aiogram Message."""
    msg = AsyncMock()
    msg.text = text
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.from_user.username = username
    msg.from_user.first_name = first_name
    msg.answer = AsyncMock()
    msg.answer_invoice = AsyncMock()
    msg.answer_document = AsyncMock()
    return msg


def make_callback(data: str, user_id: int = 12345678, username: str = "testuser"):
    """Create a mock aiogram CallbackQuery."""
    cb = AsyncMock()
    cb.data = data
    cb.from_user = MagicMock()
    cb.from_user.id = user_id
    cb.from_user.username = username
    cb.message = AsyncMock()
    cb.message.answer = AsyncMock()
    cb.message.edit_text = AsyncMock()
    cb.answer = AsyncMock()
    return cb


# ═══════════════════════════════════════
# START COMMAND
# ═══════════════════════════════════════

class TestStartCommand:
    """Test /start for new and returning users."""

    @pytest.mark.asyncio
    async def test_start_new_user(self, db: AsyncSession):
        """New user sees welcome message with 'Create My First Agent' button."""
        from bot.main import cmd_start

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/start")
            await cmd_start(msg)

        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "AgentPay" in call_text
        assert "wallet" in call_text.lower() or "agent" in call_text.lower()

    @pytest.mark.asyncio
    async def test_start_existing_user_with_agents(self, db: AsyncSession, test_user, test_agent):
        """Returning user with agents sees dashboard summary."""
        from bot.main import cmd_start

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/start", user_id=test_user.telegram_id)
            await cmd_start(msg)

        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Welcome back" in call_text
        assert "$100.00" in call_text  # test_agent balance


# ═══════════════════════════════════════
# HELP COMMAND
# ═══════════════════════════════════════

class TestHelpCommand:
    @pytest.mark.asyncio
    async def test_help_shows_commands(self):
        """Help lists key commands."""
        from bot.main import send_help

        msg = make_message("/help")
        await send_help(msg)

        call_text = msg.answer.call_args[0][0]
        assert "/newagent" in call_text
        assert "/fund" in call_text
        assert "/balance" in call_text
        assert "/history" in call_text


# ═══════════════════════════════════════
# AGENT MANAGEMENT
# ═══════════════════════════════════════

class TestNewAgentCommand:
    @pytest.mark.asyncio
    async def test_newagent_no_name(self, db: AsyncSession):
        """No name shows usage hint."""
        from bot.main import cmd_new_agent

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/newagent")
            await cmd_new_agent(msg)

        call_text = msg.answer.call_args[0][0]
        assert "/newagent" in call_text

    @pytest.mark.asyncio
    async def test_newagent_creates_agent(self, db: AsyncSession):
        """Valid name creates agent and shows API key."""
        from bot.main import cmd_new_agent

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/newagent my-bot")
            await cmd_new_agent(msg)

        call_text = msg.answer.call_args[0][0]
        assert "my-bot" in call_text
        assert "API Key" in call_text
        assert "ap_" in call_text  # Key prefix

    @pytest.mark.asyncio
    async def test_newagent_max_agents_free(self, db: AsyncSession):
        """Free users can't create more than 5 agents."""
        from bot.main import cmd_new_agent

        # Create user + 5 agents
        user = User(telegram_id=55555, username="maxed", first_name="Max")
        db.add(user)
        await db.commit()
        await db.refresh(user)

        for i in range(5):
            key = f"ap_max_key_{i:032d}"
            agent = Agent(
                user_id=user.id,
                name=f"agent-{i}",
                api_key_hash=hash_api_key(key),
                api_key_prefix=f"ap_max_{i}...",
                balance_usd=Decimal("0"),
            )
            db.add(agent)
        await db.commit()

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/newagent agent-6", user_id=55555)
            await cmd_new_agent(msg)

        call_text = msg.answer.call_args[0][0]
        assert "max 5" in call_text.lower() or "Free tier" in call_text


class TestDeleteCommand:
    @pytest.mark.asyncio
    async def test_delete_no_args_shows_picker(self, db: AsyncSession, test_user, test_agent):
        """No agent name shows agent picker."""
        from bot.main import cmd_delete

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/delete", user_id=test_user.telegram_id)
            await cmd_delete(msg)

        call_text = msg.answer.call_args[0][0]
        assert "Delete which agent" in call_text

    @pytest.mark.asyncio
    async def test_delete_by_name(self, db: AsyncSession, test_user, test_agent):
        """Deleting by name removes agent."""
        from bot.main import cmd_delete

        with patch("bot.main.async_session", return_value=db), \
             patch("bot.main.cleanup_agent", new_callable=AsyncMock):
            msg = make_message("/delete test-agent", user_id=test_user.telegram_id)
            await cmd_delete(msg)

        call_text = msg.answer.call_args[0][0]
        assert "deleted" in call_text.lower()

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, db: AsyncSession, test_user):
        """Deleting nonexistent agent shows error."""
        from bot.main import cmd_delete

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/delete ghost-agent", user_id=test_user.telegram_id)
            await cmd_delete(msg)

        call_text = msg.answer.call_args[0][0]
        assert "not found" in call_text.lower()


# ═══════════════════════════════════════
# AGENTS LIST / BALANCE
# ═══════════════════════════════════════

class TestAgentsCommand:
    @pytest.mark.asyncio
    async def test_agents_empty(self, db: AsyncSession):
        """No agents shows create hint."""
        from bot.main import cmd_agents_for_user

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/agents")
            await cmd_agents_for_user(msg, 12345678)

        call_text = msg.answer.call_args[0][0]
        assert "/newagent" in call_text

    @pytest.mark.asyncio
    async def test_agents_shows_list(self, db: AsyncSession, test_user, test_agent):
        """Lists agents with balances."""
        from bot.main import cmd_agents_for_user

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/agents")
            await cmd_agents_for_user(msg, test_user.telegram_id)

        call_text = msg.answer.call_args[0][0]
        assert "test-agent" in call_text
        assert "$100.00" in call_text


# ═══════════════════════════════════════
# API KEY COMMANDS
# ═══════════════════════════════════════

class TestApiKeyCommand:
    @pytest.mark.asyncio
    async def test_apikey_no_agents(self, db: AsyncSession):
        """No agents shows create hint."""
        from bot.main import cmd_apikey

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/apikey")
            await cmd_apikey(msg)

        call_text = msg.answer.call_args[0][0]
        assert "/newagent" in call_text

    @pytest.mark.asyncio
    async def test_apikey_shows_prefixes(self, db: AsyncSession, test_user, test_agent):
        """Shows API key prefixes for agents."""
        from bot.main import cmd_apikey

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/apikey", user_id=test_user.telegram_id)
            await cmd_apikey(msg)

        call_text = msg.answer.call_args[0][0]
        assert "ap_test_" in call_text


class TestRotateKeyCommand:
    @pytest.mark.asyncio
    async def test_rotatekey_no_args_shows_picker(self, db: AsyncSession, test_user, test_agent):
        """No agent name shows picker."""
        from bot.main import cmd_rotatekey

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/rotatekey", user_id=test_user.telegram_id)
            await cmd_rotatekey(msg)

        call_text = msg.answer.call_args[0][0]
        assert "Rotate key" in call_text

    @pytest.mark.asyncio
    async def test_rotatekey_by_name(self, db: AsyncSession, test_user, test_agent):
        """Rotating key returns new key."""
        from bot.main import cmd_rotatekey

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/rotatekey test-agent", user_id=test_user.telegram_id)
            await cmd_rotatekey(msg)

        call_text = msg.answer.call_args[0][0]
        assert "rotated" in call_text.lower()
        assert "ap_" in call_text


# ═══════════════════════════════════════
# FUNDING
# ═══════════════════════════════════════

class TestFundCommand:
    @pytest.mark.asyncio
    async def test_fund_no_agents(self, db: AsyncSession):
        """Fund with no agents shows create hint."""
        from bot.main import cmd_fund

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/fund")
            await cmd_fund(msg)

        call_text = msg.answer.call_args[0][0]
        assert "/newagent" in call_text

    @pytest.mark.asyncio
    async def test_fund_shows_agent_picker(self, db: AsyncSession, test_user, test_agent):
        """Fund shows agent selection."""
        from bot.main import cmd_fund

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/fund", user_id=test_user.telegram_id)
            await cmd_fund(msg)

        call_text = msg.answer.call_args[0][0]
        assert "Fund Agent" in call_text


# ═══════════════════════════════════════
# HISTORY & STATS
# ═══════════════════════════════════════

class TestHistoryCommand:
    @pytest.mark.asyncio
    async def test_history_no_agents(self, db: AsyncSession):
        """No agents shows create hint."""
        from bot.main import cmd_history_for_user

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/history")
            await cmd_history_for_user(msg, 12345678)

        call_text = msg.answer.call_args[0][0]
        assert "/newagent" in call_text

    @pytest.mark.asyncio
    async def test_history_no_transactions(self, db: AsyncSession, test_user, test_agent):
        """Agent with no transactions shows empty state."""
        from bot.main import cmd_history_for_user

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/history")
            await cmd_history_for_user(msg, test_user.telegram_id)

        call_text = msg.answer.call_args[0][0]
        assert "No transactions" in call_text or "Recent Activity" in call_text


class TestStatsCommand:
    @pytest.mark.asyncio
    async def test_stats_no_agents(self, db: AsyncSession):
        """No agents shows hint."""
        from bot.main import cmd_stats

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/stats")
            await cmd_stats(msg)

        call_text = msg.answer.call_args[0][0]
        assert "No agents" in call_text

    @pytest.mark.asyncio
    async def test_stats_shows_summary(self, db: AsyncSession, test_user, test_agent):
        """Stats shows account summary."""
        from bot.main import cmd_stats

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/stats", user_id=test_user.telegram_id)
            await cmd_stats(msg)

        call_text = msg.answer.call_args[0][0]
        assert "Account" in call_text
        assert "$100.00" in call_text
        assert "1 active" in call_text or "1 total" in call_text


# ═══════════════════════════════════════
# LIMITS
# ═══════════════════════════════════════

class TestLimitsCommands:
    @pytest.mark.asyncio
    async def test_limits_shows_values(self, db: AsyncSession, test_user, test_agent):
        """Limits shows daily/per-tx values."""
        from bot.main import cmd_limits_for_user

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/limits")
            await cmd_limits_for_user(msg, test_user.telegram_id)

        call_text = msg.answer.call_args[0][0]
        assert "Spending Limits" in call_text
        assert "test-agent" in call_text

    @pytest.mark.asyncio
    async def test_setlimit_no_args(self, db: AsyncSession):
        """No args shows usage."""
        from bot.main import cmd_setlimit

        msg = make_message("/setlimit")
        await cmd_setlimit(msg)

        call_text = msg.answer.call_args[0][0]
        assert "Usage" in call_text

    @pytest.mark.asyncio
    async def test_setlimit_daily(self, db: AsyncSession, test_user, test_agent):
        """Sets daily limit."""
        from bot.main import cmd_setlimit

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/setlimit test-agent daily 200", user_id=test_user.telegram_id)
            await cmd_setlimit(msg)

        call_text = msg.answer.call_args[0][0]
        assert "Daily limit" in call_text
        assert "$200.00" in call_text

    @pytest.mark.asyncio
    async def test_setlimit_invalid_type(self, db: AsyncSession):
        """Invalid limit type shows error."""
        from bot.main import cmd_setlimit

        msg = make_message("/setlimit bot weekly 100")
        await cmd_setlimit(msg)

        call_text = msg.answer.call_args[0][0]
        assert "daily" in call_text or "tx" in call_text


# ═══════════════════════════════════════
# APPROVAL WORKFLOW
# ═══════════════════════════════════════

class TestApprovalCommands:
    @pytest.mark.asyncio
    async def test_setapprove_no_args(self, db: AsyncSession):
        """No args shows usage."""
        from bot.main import cmd_set_approve

        msg = make_message("/setapprove")
        await cmd_set_approve(msg)

        call_text = msg.answer.call_args[0][0]
        assert "Usage" in call_text

    @pytest.mark.asyncio
    async def test_setapprove_valid(self, db: AsyncSession, test_user, test_agent):
        """Sets auto-approve threshold."""
        from bot.main import cmd_set_approve

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/setapprove test-agent 50", user_id=test_user.telegram_id)
            await cmd_set_approve(msg)

        call_text = msg.answer.call_args[0][0]
        assert "auto-approve" in call_text
        assert "$50" in call_text

    @pytest.mark.asyncio
    async def test_approvals_empty(self, db: AsyncSession):
        """No pending approvals."""
        from bot.main import cmd_approvals

        with patch("core.approvals.get_user_pending", return_value=[]):
            msg = make_message("/approvals")
            await cmd_approvals(msg)

        call_text = msg.answer.call_args[0][0]
        assert "No pending" in call_text


# ═══════════════════════════════════════
# REFUND & TRANSFER
# ═══════════════════════════════════════

class TestRefundCommand:
    @pytest.mark.asyncio
    async def test_refund_no_args(self, db: AsyncSession):
        """No args shows usage."""
        from bot.main import cmd_refund

        msg = make_message("/refund")
        await cmd_refund(msg)

        call_text = msg.answer.call_args[0][0]
        assert "Usage" in call_text


class TestTransferCommand:
    @pytest.mark.asyncio
    async def test_transfer_no_args(self, db: AsyncSession):
        """No args shows usage."""
        from bot.main import cmd_transfer

        msg = make_message("/transfer")
        await cmd_transfer(msg)

        call_text = msg.answer.call_args[0][0]
        assert "Usage" in call_text

    @pytest.mark.asyncio
    async def test_transfer_invalid_amount(self, db: AsyncSession):
        """Invalid amount shows error."""
        from bot.main import cmd_transfer

        msg = make_message("/transfer Bot1 Bot2 abc")
        await cmd_transfer(msg)

        call_text = msg.answer.call_args[0][0]
        assert "Invalid" in call_text

    @pytest.mark.asyncio
    async def test_transfer_same_agent(self, db: AsyncSession, test_user, test_agent):
        """Can't transfer to same agent."""
        from bot.main import cmd_transfer

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/transfer test-agent test-agent 10", user_id=test_user.telegram_id)
            await cmd_transfer(msg)

        call_text = msg.answer.call_args[0][0]
        assert "same agent" in call_text.lower()


# ═══════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════

class TestExportCommand:
    @pytest.mark.asyncio
    async def test_export_no_agents(self, db: AsyncSession):
        """No agents shows hint."""
        from bot.main import cmd_export

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/export")
            await cmd_export(msg)

        call_text = msg.answer.call_args[0][0]
        assert "No agents" in call_text


# ═══════════════════════════════════════
# REVENUE (admin)
# ═══════════════════════════════════════

class TestRevenueCommand:
    @pytest.mark.asyncio
    async def test_revenue_non_admin_silent(self, db: AsyncSession):
        """Non-admin user gets no response."""
        from bot.main import cmd_revenue

        msg = make_message("/revenue", user_id=99999)
        await cmd_revenue(msg)

        msg.answer.assert_not_called()

    @pytest.mark.asyncio
    async def test_revenue_admin(self, db: AsyncSession):
        """Admin (G) sees revenue stats."""
        from bot.main import cmd_revenue

        with patch("bot.main.async_session", return_value=db):
            msg = make_message("/revenue", user_id=5360481016)
            await cmd_revenue(msg)

        call_text = msg.answer.call_args[0][0]
        assert "Revenue" in call_text
        assert "$" in call_text


# ═══════════════════════════════════════
# DEMO
# ═══════════════════════════════════════

class TestDemoCommand:
    @pytest.mark.asyncio
    async def test_demo_starts(self, db: AsyncSession):
        """Demo command starts interactive demo."""
        from bot.main import cmd_demo

        msg = make_message("/demo")
        # Speed up sleep
        with patch("bot.main.asyncio.sleep", new_callable=AsyncMock):
            await cmd_demo(msg)

        # Should have initial answer + edits
        msg.answer.assert_called_once()
        call_text = msg.answer.call_args[0][0]
        assert "Live Demo" in call_text


# ═══════════════════════════════════════
# CATCH-ALL
# ═══════════════════════════════════════

class TestCatchAll:
    @pytest.mark.asyncio
    async def test_unknown_command(self, db: AsyncSession):
        """Unknown command shows help hint."""
        from bot.main import catch_all

        msg = make_message("/blahblah")
        await catch_all(msg)

        call_text = msg.answer.call_args[0][0]
        assert "Unknown command" in call_text or "/help" in call_text

    @pytest.mark.asyncio
    async def test_regular_text_ignored(self, db: AsyncSession):
        """Regular text doesn't trigger unknown command reply."""
        from bot.main import catch_all

        msg = make_message("hello there")
        await catch_all(msg)

        msg.answer.assert_not_called()


# ═══════════════════════════════════════
# CALLBACK: DEMO APPROVE/DENY
# ═══════════════════════════════════════

class TestDemoCallbacks:
    @pytest.mark.asyncio
    async def test_demo_approve(self):
        """Demo approve shows final summary."""
        from bot.main import callback_demo_approve

        cb = make_callback("demo:approve")
        await callback_demo_approve(cb)

        call_text = cb.message.edit_text.call_args[0][0]
        assert "Demo Complete" in call_text
        assert "approved" in call_text.lower()

    @pytest.mark.asyncio
    async def test_demo_deny(self):
        """Demo deny shows no funds deducted."""
        from bot.main import callback_demo_deny

        cb = make_callback("demo:deny")
        await callback_demo_deny(cb)

        call_text = cb.message.edit_text.call_args[0][0]
        assert "Demo Complete" in call_text
        assert "denied" in call_text.lower()
