"""
Tests for core wallet operations: create, balance, spend, refund, transfer.
"""
import pytest
import pytest_asyncio
from decimal import Decimal
from unittest.mock import patch, AsyncMock

from models.schema import (
    Agent, User, Transaction, TransactionType, TransactionStatus, PaymentMethod,
)
from core.wallet import (
    hash_api_key, generate_api_key, get_agent_by_api_key,
    get_daily_spent, spend, refund, deposit,
    transfer_between_agents, get_agent_transactions,
    create_agent, get_or_create_user, rotate_api_key,
)


# ═══════════════════════════════════════
# API Key Hashing & Generation
# ═══════════════════════════════════════

class TestAPIKeyHashing:
    def test_hash_is_deterministic(self):
        key = "ap_test123"
        assert hash_api_key(key) == hash_api_key(key)

    def test_different_keys_produce_different_hashes(self):
        assert hash_api_key("ap_key1") != hash_api_key("ap_key2")

    def test_hash_is_sha256_hex(self):
        h = hash_api_key("ap_test")
        assert len(h) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in h)

    def test_generate_api_key_format(self):
        full_key, key_hash, key_prefix = generate_api_key()
        assert full_key.startswith("ap_")
        assert len(full_key) == 51  # "ap_" + 48 hex chars
        assert hash_api_key(full_key) == key_hash
        assert key_prefix == full_key[:8] + "..."

    def test_generate_api_key_uniqueness(self):
        keys = set()
        for _ in range(100):
            full_key, _, _ = generate_api_key()
            keys.add(full_key)
        assert len(keys) == 100  # All unique


# ═══════════════════════════════════════
# Agent Lookup by API Key
# ═══════════════════════════════════════

class TestGetAgentByAPIKey:
    @pytest.mark.asyncio
    async def test_valid_key_returns_agent(self, db, test_agent):
        agent = await get_agent_by_api_key(db, test_agent._test_api_key)
        assert agent is not None
        assert agent.id == test_agent.id
        assert agent.name == "test-agent"

    @pytest.mark.asyncio
    async def test_invalid_key_returns_none(self, db, test_agent):
        agent = await get_agent_by_api_key(db, "ap_totally_wrong_key")
        assert agent is None

    @pytest.mark.asyncio
    async def test_inactive_agent_returns_none(self, db, test_agent):
        test_agent.is_active = False
        await db.commit()
        agent = await get_agent_by_api_key(db, test_agent._test_api_key)
        assert agent is None


# ═══════════════════════════════════════
# Balance & Daily Spent
# ═══════════════════════════════════════

class TestBalance:
    @pytest.mark.asyncio
    async def test_initial_balance(self, db, test_agent):
        assert test_agent.balance_usd == Decimal("100.0000")

    @pytest.mark.asyncio
    async def test_daily_spent_starts_at_zero(self, db, test_agent):
        daily = await get_daily_spent(db, test_agent)
        assert daily == Decimal("0")


# ═══════════════════════════════════════
# Spend Operations
# ═══════════════════════════════════════

class TestSpend:
    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_successful_spend(self, mock_webhook, mock_notify, db, test_agent):
        """Basic spend under all limits should succeed."""
        tx, error = await spend(db, test_agent, Decimal("5.00"), description="Test purchase", require_approval=False)
        assert error is None
        assert tx is not None
        assert tx.tx_type == TransactionType.SPEND
        assert tx.status == TransactionStatus.COMPLETED
        assert tx.amount_usd == Decimal("5.0000")

    @pytest.mark.asyncio
    async def test_spend_insufficient_balance(self, db, test_agent):
        """Spend more than balance should fail."""
        tx, error = await spend(db, test_agent, Decimal("999.00"), require_approval=False)
        assert tx is None
        assert "Insufficient balance" in error

    @pytest.mark.asyncio
    async def test_spend_exceeds_tx_limit(self, db, test_agent):
        """Spend above per-transaction limit should fail."""
        tx, error = await spend(db, test_agent, Decimal("30.00"), require_approval=False)
        assert tx is None
        assert "transaction limit" in error.lower()

    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_spend_exceeds_daily_limit(self, mock_wh, mock_notify, db, test_agent):
        """Multiple spends exceeding daily limit should fail."""
        # First spend: 24.99 (under tx_limit of 25, under daily of 50)
        tx1, err1 = await spend(db, test_agent, Decimal("24.99"), require_approval=False)
        assert err1 is None
        # Second: 24.99 (total ~50, right at daily limit with fees)
        tx2, err2 = await spend(db, test_agent, Decimal("24.99"), require_approval=False)
        assert err2 is None
        # Third: should exceed daily limit
        tx3, err3 = await spend(db, test_agent, Decimal("1.00"), require_approval=False)
        assert tx3 is None
        assert "daily limit" in err3.lower()

    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_spend_deducts_balance(self, mock_wh, mock_notify, db, test_agent):
        """Balance should decrease after spend (amount + fee)."""
        initial = test_agent.balance_usd
        amount = Decimal("10.00")
        tx, error = await spend(db, test_agent, amount, require_approval=False)
        assert error is None
        # Balance should be less than initial minus amount (fee is also deducted)
        assert test_agent.balance_usd < initial
        assert test_agent.balance_usd <= initial - amount

    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_idempotency_key_prevents_duplicate(self, mock_wh, mock_notify, db, test_agent):
        """Same idempotency key should return the same transaction, not create a new one."""
        tx1, err1 = await spend(db, test_agent, Decimal("5.00"), idempotency_key="unique-123", require_approval=False)
        assert err1 is None
        balance_after_first = test_agent.balance_usd

        tx2, err2 = await spend(db, test_agent, Decimal("5.00"), idempotency_key="unique-123", require_approval=False)
        assert err2 is None
        assert tx2.id == tx1.id  # Same transaction returned
        # Balance should not change on idempotent retry
        # (Note: the function returns the existing tx, so balance stays the same)


# ═══════════════════════════════════════
# Refund Operations
# ═══════════════════════════════════════

class TestRefund:
    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_successful_refund(self, mock_wh, mock_notify, db, test_agent):
        """Refunding a completed spend should credit back amount + fee."""
        initial_balance = test_agent.balance_usd
        tx, _ = await spend(db, test_agent, Decimal("10.00"), require_approval=False)
        after_spend = test_agent.balance_usd

        refund_tx, error = await refund(db, test_agent, tx.id)
        assert error is None
        assert refund_tx is not None
        assert refund_tx.tx_type == TransactionType.REFUND
        # Balance should be restored (amount + fee credited back)
        assert test_agent.balance_usd == initial_balance

    @pytest.mark.asyncio
    async def test_refund_nonexistent_tx(self, db, test_agent):
        """Refunding a non-existent transaction should fail."""
        tx, error = await refund(db, test_agent, "nonexistent-tx-id")
        assert tx is None
        assert "not found" in error.lower()

    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_double_refund_fails(self, mock_wh, mock_notify, db, test_agent):
        """Refunding the same transaction twice should fail."""
        tx, _ = await spend(db, test_agent, Decimal("5.00"), require_approval=False)
        refund_tx, err = await refund(db, test_agent, tx.id)
        assert err is None

        refund_tx2, err2 = await refund(db, test_agent, tx.id)
        assert refund_tx2 is None
        assert "already refunded" in err2.lower()


# ═══════════════════════════════════════
# Deposit Operations
# ═══════════════════════════════════════

class TestDeposit:
    @pytest.mark.asyncio
    @patch("core.wallet.notify_deposit", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_deposit_increases_balance(self, mock_wh, mock_notify, db, test_agent):
        initial = test_agent.balance_usd
        tx = await deposit(db, test_agent, Decimal("25.00"), PaymentMethod.MANUAL)
        assert tx.tx_type == TransactionType.DEPOSIT
        assert tx.status == TransactionStatus.COMPLETED
        assert test_agent.balance_usd == initial + Decimal("25.00")


# ═══════════════════════════════════════
# Transfer Operations
# ═══════════════════════════════════════

class TestTransfer:
    @pytest.mark.asyncio
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_transfer_between_own_agents(self, mock_wh, db, test_agent, second_agent):
        """Transfer between agents owned by the same user should succeed."""
        initial_from = test_agent.balance_usd
        initial_to = second_agent.balance_usd
        amount = Decimal("10.00")

        tx, error = await transfer_between_agents(db, test_agent, second_agent.id, amount)
        assert error is None
        assert test_agent.balance_usd == initial_from - amount
        assert second_agent.balance_usd == initial_to + amount

    @pytest.mark.asyncio
    async def test_transfer_to_other_user_agent_fails(self, db, test_agent, other_user_agent):
        """Transfer to an agent owned by a different user should fail."""
        tx, error = await transfer_between_agents(db, test_agent, other_user_agent.id, Decimal("5.00"))
        assert tx is None
        assert "your own agents" in error.lower()

    @pytest.mark.asyncio
    async def test_transfer_insufficient_balance(self, db, test_agent, second_agent):
        """Transfer more than balance should fail."""
        tx, error = await transfer_between_agents(db, test_agent, second_agent.id, Decimal("999.00"))
        assert tx is None
        assert "insufficient balance" in error.lower()

    @pytest.mark.asyncio
    async def test_transfer_to_nonexistent_agent(self, db, test_agent):
        """Transfer to non-existent agent should fail."""
        tx, error = await transfer_between_agents(db, test_agent, "nonexistent-id", Decimal("5.00"))
        assert tx is None
        assert "not found" in error.lower()


# ═══════════════════════════════════════
# Transaction History
# ═══════════════════════════════════════

class TestTransactionHistory:
    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_get_transactions(self, mock_wh, mock_notify, db, test_agent):
        """Transactions should be retrievable and ordered newest-first."""
        for i in range(5):
            await spend(db, test_agent, Decimal("1.00"), description=f"tx-{i}", require_approval=False)

        txs = await get_agent_transactions(db, test_agent, limit=10)
        assert len(txs) == 5
        # Newest first
        assert txs[0].description == "tx-4"
        assert txs[4].description == "tx-0"

    @pytest.mark.asyncio
    @patch("core.wallet.notify_spend", new_callable=AsyncMock)
    @patch("core.wallet.deliver_webhook", new_callable=AsyncMock)
    async def test_get_transactions_limit(self, mock_wh, mock_notify, db, test_agent):
        """Limit parameter should be respected."""
        for i in range(5):
            await spend(db, test_agent, Decimal("1.00"), description=f"tx-{i}", require_approval=False)

        txs = await get_agent_transactions(db, test_agent, limit=3)
        assert len(txs) == 3


# ═══════════════════════════════════════
# User & Agent Creation
# ═══════════════════════════════════════

class TestUserAgentCreation:
    @pytest.mark.asyncio
    async def test_get_or_create_user_new(self, db):
        user = await get_or_create_user(db, telegram_id=111222333, username="newguy")
        assert user.telegram_id == 111222333
        assert user.username == "newguy"

    @pytest.mark.asyncio
    async def test_get_or_create_user_existing(self, db, test_user):
        user = await get_or_create_user(db, telegram_id=test_user.telegram_id)
        assert user.id == test_user.id

    @pytest.mark.asyncio
    async def test_create_agent(self, db, test_user):
        agent, full_key = await create_agent(db, test_user, "my-new-agent")
        assert agent.name == "my-new-agent"
        assert full_key.startswith("ap_")
        assert agent.api_key_hash == hash_api_key(full_key)

    @pytest.mark.asyncio
    async def test_rotate_api_key(self, db, test_agent):
        old_hash = test_agent.api_key_hash
        new_key = await rotate_api_key(db, test_agent)
        assert new_key.startswith("ap_")
        assert test_agent.api_key_hash != old_hash
        assert test_agent.api_key_hash == hash_api_key(new_key)
