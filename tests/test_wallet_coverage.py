"""
Tests for core/wallet.py coverage gaps — deposit notifications,
execute_approved_spend, _get_agent_user, _check_idempotency standalone,
and transfer webhook paths.
"""
import pytest
from decimal import Decimal
from unittest.mock import patch, AsyncMock

from models.schema import TransactionType, TransactionStatus, PaymentMethod, Agent
from core.wallet import (
    deposit, spend, execute_approved_spend, get_daily_spent,
    _check_idempotency, _get_agent_user, transfer_between_agents,
    get_or_create_user, create_agent, get_agent_transactions,
)


# ═══════════════════════════════════════
# Deposit with notification fire
# ═══════════════════════════════════════

class TestDepositNotifications:
    """Cover lines 94-98, 110 — deposit notification + webhook branches."""

    @pytest.mark.asyncio
    @patch("core.wallet.notify_deposit", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_deposit_fires_notification_and_webhook(self, mock_wh, mock_notify, db, test_user, test_agent):
        """Deposit triggers both notification and webhook."""
        tx = await deposit(db, test_agent, Decimal("25.00"), PaymentMethod.TELEGRAM_STARS, external_ref="stars_123")
        assert tx.amount_usd == Decimal("25.00")
        assert test_agent.balance_usd == Decimal("125.00")

        # Verify notification was scheduled (asyncio.create_task called)
        mock_notify.assert_called_once()
        call_args = mock_notify.call_args
        assert call_args[0][0] == test_agent.name
        assert call_args[0][1] == test_user.telegram_id
        assert call_args[0][2] == 25.0

        mock_wh.assert_called_once()
        wh_args = mock_wh.call_args
        assert wh_args[0][1] == "deposit"

    @pytest.mark.asyncio
    @patch("core.wallet.notify_deposit", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_deposit_custom_description(self, mock_wh, mock_notify, db, test_agent):
        """Deposit with custom description."""
        tx = await deposit(db, test_agent, Decimal("5.00"), PaymentMethod.USDC, description="Manual top-up")
        assert tx.description == "Manual top-up"


# ═══════════════════════════════════════
# _get_agent_user (line 140)
# ═══════════════════════════════════════

class TestGetAgentUser:
    @pytest.mark.asyncio
    async def test_get_agent_user_returns_user(self, db, test_user, test_agent):
        """Should find the user who owns this agent."""
        user = await _get_agent_user(db, test_agent)
        assert user is not None
        assert user.id == test_user.id

    @pytest.mark.asyncio
    async def test_get_agent_user_orphan_agent(self, db):
        """Agent with bad user_id returns None."""
        from core.wallet import hash_api_key
        orphan = Agent(
            user_id="nonexistent-user-id",
            name="orphan",
            api_key_hash=hash_api_key("ap_orphan_key"),
            api_key_prefix="ap_orph...",
            balance_usd=Decimal("0"),
        )
        db.add(orphan)
        await db.flush()
        user = await _get_agent_user(db, orphan)
        assert user is None


# ═══════════════════════════════════════
# _check_idempotency (line 130)
# ═══════════════════════════════════════

class TestCheckIdempotency:
    @pytest.mark.asyncio
    async def test_empty_key_returns_none(self, db):
        """Empty idempotency key short-circuits."""
        result = await _check_idempotency(db, "any-agent", "")
        assert result is None

    @pytest.mark.asyncio
    async def test_none_key_returns_none(self, db):
        """None idempotency key short-circuits."""
        result = await _check_idempotency(db, "any-agent", None)
        assert result is None

    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_existing_key_returns_transaction(self, mock_wh, mock_notify, db, test_agent):
        """If idempotency key exists, return the original transaction."""
        tx, _ = await spend(db, test_agent, Decimal("5.00"), idempotency_key="idem-001", require_approval=False)
        assert tx is not None

        found = await _check_idempotency(db, test_agent.id, "idem-001")
        assert found is not None
        assert found.id == tx.id


# ═══════════════════════════════════════
# Spend notification paths (lines 229, 185)
# ═══════════════════════════════════════

class TestSpendNotifications:
    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_spend_fires_notification_and_webhook(self, mock_wh, mock_notify, db, test_user, test_agent):
        """Spend triggers notification + webhook for owner."""
        tx, err = await spend(db, test_agent, Decimal("10.00"), description="API call", require_approval=False)
        assert err is None
        assert tx is not None

        mock_notify.assert_called_once()
        mock_wh.assert_called_once()
        wh_payload = mock_wh.call_args[0][2]
        assert wh_payload["description"] == "API call"

    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_spend_with_approval_required_above_threshold(self, mock_wh, mock_notify, db, test_user, test_agent):
        """Spend above auto-approve threshold returns approval_pending."""
        test_agent.auto_approve_usd = Decimal("5.00")
        await db.commit()

        with patch("core.wallet.request_approval", new_callable=AsyncMock) as mock_approval, \
             patch("core.wallet.notify_approval_request", new_callable=AsyncMock) as mock_ar_notify:
            from core.approvals import PendingApproval
            mock_approval.return_value = PendingApproval(
                id="approval-123",
                agent_id=test_agent.id,
                agent_name=test_agent.name,
                user_id=test_user.id,
                telegram_id=test_user.telegram_id,
                amount_usd=Decimal("20.00"),
                description="Big purchase",
            )
            tx, err = await spend(db, test_agent, Decimal("20.00"), description="Big purchase", require_approval=True)

        assert tx is None
        assert "approval_pending" in err


# ═══════════════════════════════════════
# execute_approved_spend (lines 264-266, 270-271)
# ═══════════════════════════════════════

class TestExecuteApprovedSpend:
    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_approved_spend_success(self, mock_wh, mock_notify, db, test_agent):
        """Execute a pre-approved spend."""
        tx, err = await execute_approved_spend(db, test_agent.id, Decimal("15.00"), "Approved purchase")
        assert err is None
        assert tx is not None
        assert tx.amount_usd == Decimal("15.00")

    @pytest.mark.asyncio
    async def test_approved_spend_nonexistent_agent(self, db):
        """Approved spend for nonexistent agent returns error."""
        tx, err = await execute_approved_spend(db, "fake-agent-id", Decimal("5.00"))
        assert tx is None
        assert "not found" in err.lower()

    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_approved_spend_insufficient_balance(self, mock_wh, mock_notify, db, test_agent):
        """Approved spend with insufficient balance."""
        test_agent.balance_usd = Decimal("2.00")
        await db.commit()

        tx, err = await execute_approved_spend(db, test_agent.id, Decimal("50.00"))
        assert tx is None
        assert "insufficient balance" in err.lower()


# ═══════════════════════════════════════
# get_daily_spent (line 113)
# ═══════════════════════════════════════

class TestGetDailySpent:
    @pytest.mark.asyncio
    async def test_daily_spent_zero(self, db, test_agent):
        """No spends today → $0."""
        spent = await get_daily_spent(db, test_agent)
        assert spent == Decimal("0")

    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_daily_spent_accumulates(self, mock_wh, mock_notify, db, test_agent):
        """Multiple spends today accumulate."""
        await spend(db, test_agent, Decimal("3.00"), require_approval=False)
        await spend(db, test_agent, Decimal("7.00"), require_approval=False)
        spent = await get_daily_spent(db, test_agent)
        assert spent == Decimal("10.00")


# ═══════════════════════════════════════
# Transfer with webhook (lines 361-394, 398-401)
# ═══════════════════════════════════════

class TestTransferFull:
    @pytest.mark.asyncio
    async def test_transfer_creates_paired_transactions(self, db, test_agent, second_agent):
        """Transfer creates debit + credit transactions."""
        tx, err = await transfer_between_agents(db, test_agent, second_agent.id, Decimal("20.00"), "Split funds")
        assert err is None

        # Debit on source
        assert tx.tx_type == TransactionType.SPEND
        assert tx.description == "Split funds"

        # Credit on target
        target_txs = await get_agent_transactions(db, second_agent, limit=1)
        assert len(target_txs) >= 1
        credit_tx = target_txs[0]
        assert credit_tx.tx_type == TransactionType.DEPOSIT
        assert credit_tx.amount_usd == Decimal("20.00")

    @pytest.mark.asyncio
    async def test_transfer_default_description(self, db, test_agent, second_agent):
        """Transfer without description uses default."""
        tx, err = await transfer_between_agents(db, test_agent, second_agent.id, Decimal("5.00"))
        assert err is None
        assert second_agent.name in tx.description  # "Transfer to <name>"


# ═══════════════════════════════════════
# get_or_create_user edge cases (lines 31-34)
# ═══════════════════════════════════════

class TestGetOrCreateUserEdge:
    @pytest.mark.asyncio
    async def test_create_user_with_all_fields(self, db):
        """Creating user with all optional fields."""
        user = await get_or_create_user(db, 999888777, username="allfields", first_name="Full")
        assert user.username == "allfields"
        assert user.first_name == "Full"

    @pytest.mark.asyncio
    async def test_create_user_minimal(self, db):
        """Creating user with just telegram_id."""
        user = await get_or_create_user(db, 777666555)
        assert user.telegram_id == 777666555
        assert user.username is None
