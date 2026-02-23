import secrets
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from models.schema import User, Agent, Wallet, Transaction, TransactionType, TransactionStatus, PaymentMethod, PlatformRevenue
from config.settings import PLATFORM_FEE_PERCENT, DEFAULT_DAILY_LIMIT_USD, DEFAULT_TRANSACTION_LIMIT_USD
from core.webhooks import deliver_webhook, notify_spend, notify_deposit, notify_approval_request
from core.approvals import request_approval, DEFAULT_AUTO_APPROVE_USD


def generate_api_key() -> str:
    return f"ap_{secrets.token_hex(24)}"


async def get_or_create_user(db: AsyncSession, telegram_id: int, username: str = None, first_name: str = None) -> User:
    result = await db.execute(select(User).where(User.telegram_id == telegram_id))
    user = result.scalar_one_or_none()
    if not user:
        user = User(telegram_id=telegram_id, username=username, first_name=first_name)
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user


async def create_agent(db: AsyncSession, user: User, name: str) -> Agent:
    agent = Agent(
        user_id=user.id,
        name=name,
        api_key=generate_api_key(),
        daily_limit_usd=Decimal(str(DEFAULT_DAILY_LIMIT_USD)),
        tx_limit_usd=Decimal(str(DEFAULT_TRANSACTION_LIMIT_USD)),
    )
    db.add(agent)
    await db.flush()

    wallet = Wallet(agent_id=agent.id, wallet_type="internal")
    db.add(wallet)

    await db.commit()
    await db.refresh(agent)
    return agent


async def get_user_agents(db: AsyncSession, user: User) -> list[Agent]:
    result = await db.execute(select(Agent).where(Agent.user_id == user.id).order_by(Agent.created_at))
    return list(result.scalars().all())


async def get_agent_by_api_key(db: AsyncSession, api_key: str) -> Agent | None:
    result = await db.execute(select(Agent).where(Agent.api_key == api_key, Agent.is_active == True))
    return result.scalar_one_or_none()


async def deposit(db: AsyncSession, agent: Agent, amount_usd: Decimal, method: PaymentMethod, external_ref: str = None, description: str = None) -> Transaction:
    tx = Transaction(
        agent_id=agent.id,
        tx_type=TransactionType.DEPOSIT,
        status=TransactionStatus.COMPLETED,
        amount_usd=amount_usd,
        payment_method=method,
        external_ref=external_ref,
        description=description or f"Deposit via {method.value}",
    )
    agent.balance_usd += amount_usd
    db.add(tx)
    await db.commit()
    await db.refresh(agent)

    # Get user for notifications
    user = await _get_agent_user(db, agent)
    if user:
        # Fire-and-forget notifications
        asyncio.create_task(notify_deposit(
            agent.name, user.telegram_id, float(amount_usd), method.value, float(agent.balance_usd)
        ))
        asyncio.create_task(deliver_webhook(agent.id, "deposit", {
            "amount_usd": str(amount_usd),
            "method": method.value,
            "new_balance": str(agent.balance_usd),
            "transaction_id": tx.id,
        }))

    return tx


async def get_daily_spent(db: AsyncSession, agent: Agent) -> Decimal:
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(func.coalesce(func.sum(Transaction.amount_usd), 0))
        .where(
            Transaction.agent_id == agent.id,
            Transaction.tx_type == TransactionType.SPEND,
            Transaction.status == TransactionStatus.COMPLETED,
            Transaction.created_at >= today_start,
        )
    )
    return Decimal(str(result.scalar()))


async def _check_idempotency(db: AsyncSession, agent_id: str, idempotency_key: str) -> Transaction | None:
    """Check if a transaction with this idempotency key already exists."""
    if not idempotency_key:
        return None
    result = await db.execute(
        select(Transaction).where(
            Transaction.agent_id == agent_id,
            Transaction.idempotency_key == idempotency_key,
        )
    )
    return result.scalar_one_or_none()


async def _get_agent_user(db: AsyncSession, agent: Agent) -> User | None:
    """Get the user who owns this agent."""
    result = await db.execute(select(User).where(User.id == agent.user_id))
    return result.scalar_one_or_none()


async def spend(
    db: AsyncSession,
    agent: Agent,
    amount_usd: Decimal,
    description: str = None,
    idempotency_key: str = None,
    require_approval: bool = True,
) -> tuple[Transaction | None, str | None]:
    """
    Execute a spend. If amount > auto_approve_usd, requires human approval via Telegram.
    
    Returns (transaction, error_string).
    If approval needed, returns (None, "approval_pending:{approval_id}").
    """

    # Idempotency check
    if idempotency_key:
        existing = await _check_idempotency(db, agent.id, idempotency_key)
        if existing:
            return existing, None  # Return the same transaction

    # Check balance
    if agent.balance_usd < amount_usd:
        return None, f"Insufficient balance. Have ${agent.balance_usd}, need ${amount_usd}"

    # Check transaction limit
    if amount_usd > agent.tx_limit_usd:
        return None, f"Exceeds transaction limit of ${agent.tx_limit_usd}"

    # Check daily limit
    daily_spent = await get_daily_spent(db, agent)
    if daily_spent + amount_usd > agent.daily_limit_usd:
        remaining = agent.daily_limit_usd - daily_spent
        return None, f"Exceeds daily limit. Remaining today: ${remaining}"

    # Approval check
    auto_approve = agent.auto_approve_usd if agent.auto_approve_usd else DEFAULT_AUTO_APPROVE_USD
    if require_approval and amount_usd > auto_approve:
        user = await _get_agent_user(db, agent)
        if user:
            approval = await request_approval(
                agent_id=agent.id,
                agent_name=agent.name,
                user_id=user.id,
                telegram_id=user.telegram_id,
                amount_usd=amount_usd,
                description=description,
            )
            # Send Telegram notification
            asyncio.create_task(notify_approval_request(
                agent.name, user.telegram_id, float(amount_usd), description, approval.id
            ))
            return None, f"approval_pending:{approval.id}"

    # Execute spend
    return await _execute_spend(db, agent, amount_usd, description, idempotency_key)


async def _execute_spend(
    db: AsyncSession,
    agent: Agent,
    amount_usd: Decimal,
    description: str = None,
    idempotency_key: str = None,
) -> tuple[Transaction, None]:
    """Actually execute the spend (after approval if needed)."""
    fee = amount_usd * Decimal(str(PLATFORM_FEE_PERCENT / 100))

    tx = Transaction(
        agent_id=agent.id,
        tx_type=TransactionType.SPEND,
        status=TransactionStatus.COMPLETED,
        amount_usd=amount_usd,
        fee_usd=fee,
        description=description or "Agent spend",
        idempotency_key=idempotency_key,
    )
    agent.balance_usd -= (amount_usd + fee)
    db.add(tx)
    await db.flush()

    # Track platform revenue
    if fee > 0:
        revenue = PlatformRevenue(
            transaction_id=tx.id,
            agent_id=agent.id,
            amount_usd=fee,
        )
        db.add(revenue)

    await db.commit()
    await db.refresh(agent)

    # Fire-and-forget notifications
    user = await _get_agent_user(db, agent)
    if user:
        asyncio.create_task(notify_spend(
            agent.name, user.telegram_id,
            float(amount_usd), float(fee), description, float(agent.balance_usd)
        ))
        asyncio.create_task(deliver_webhook(agent.id, "spend", {
            "amount_usd": str(amount_usd),
            "fee_usd": str(fee),
            "description": description,
            "remaining_balance": str(agent.balance_usd),
            "transaction_id": tx.id,
        }))

    return tx, None


async def execute_approved_spend(
    db: AsyncSession,
    agent_id: str,
    amount_usd: Decimal,
    description: str = None,
) -> tuple[Transaction | None, str | None]:
    """Execute a spend that was approved via Telegram."""
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        return None, "Agent not found"

    # Re-check balance (might have changed since approval request)
    if agent.balance_usd < amount_usd:
        return None, f"Insufficient balance. Have ${agent.balance_usd}, need ${amount_usd}"

    return await _execute_spend(db, agent, amount_usd, description)


async def get_agent_transactions(db: AsyncSession, agent: Agent, limit: int = 20) -> list[Transaction]:
    result = await db.execute(
        select(Transaction)
        .where(Transaction.agent_id == agent.id)
        .order_by(Transaction.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def refund(
    db: AsyncSession,
    agent: Agent,
    transaction_id: str,
) -> tuple[Transaction | None, str | None]:
    """
    Refund a completed spend transaction. Credits the amount back to the agent.
    Only spend transactions can be refunded, and only once.
    """
    # Find the original transaction
    result = await db.execute(
        select(Transaction).where(
            Transaction.id == transaction_id,
            Transaction.agent_id == agent.id,
        )
    )
    original = result.scalar_one_or_none()
    if not original:
        return None, "Transaction not found"
    if original.tx_type != TransactionType.SPEND:
        return None, "Only spend transactions can be refunded"
    if original.status != TransactionStatus.COMPLETED:
        return None, "Transaction is not completed"

    # Check if already refunded (look for a refund tx referencing this one)
    result = await db.execute(
        select(Transaction).where(
            Transaction.agent_id == agent.id,
            Transaction.tx_type == TransactionType.REFUND,
            Transaction.external_ref == transaction_id,
        )
    )
    existing_refund = result.scalar_one_or_none()
    if existing_refund:
        return None, "Transaction already refunded"

    # Create refund (credit back amount + fee)
    refund_amount = original.amount_usd + original.fee_usd
    tx = Transaction(
        agent_id=agent.id,
        tx_type=TransactionType.REFUND,
        status=TransactionStatus.COMPLETED,
        amount_usd=refund_amount,
        fee_usd=Decimal("0"),
        description=f"Refund for tx {transaction_id[:8]}",
        external_ref=transaction_id,
    )
    agent.balance_usd += refund_amount
    db.add(tx)
    await db.commit()
    await db.refresh(agent)

    # Notifications
    user = await _get_agent_user(db, agent)
    if user:
        asyncio.create_task(deliver_webhook(agent.id, "refund", {
            "amount_usd": str(refund_amount),
            "original_transaction_id": transaction_id,
            "new_balance": str(agent.balance_usd),
            "transaction_id": tx.id,
        }))

    return tx, None


async def transfer_between_agents(
    db: AsyncSession,
    from_agent: Agent,
    to_agent_id: str,
    amount_usd: Decimal,
    description: str = None,
) -> tuple[Transaction | None, str | None]:
    """Transfer funds from one agent to another (same user only)."""
    # Find target agent
    result = await db.execute(select(Agent).where(Agent.id == to_agent_id))
    to_agent = result.scalar_one_or_none()
    if not to_agent:
        return None, "Target agent not found"

    # Verify same owner
    if from_agent.user_id != to_agent.user_id:
        return None, "Can only transfer between your own agents"

    # Check balance
    if from_agent.balance_usd < amount_usd:
        return None, f"Insufficient balance. Have ${from_agent.balance_usd}, need ${amount_usd}"

    # Debit source
    from_tx = Transaction(
        agent_id=from_agent.id,
        tx_type=TransactionType.SPEND,
        status=TransactionStatus.COMPLETED,
        amount_usd=amount_usd,
        fee_usd=Decimal("0"),  # No fee on internal transfers
        description=description or f"Transfer to {to_agent.name}",
    )
    from_agent.balance_usd -= amount_usd
    db.add(from_tx)

    # Credit destination
    to_tx = Transaction(
        agent_id=to_agent.id,
        tx_type=TransactionType.DEPOSIT,
        status=TransactionStatus.COMPLETED,
        amount_usd=amount_usd,
        fee_usd=Decimal("0"),
        description=description or f"Transfer from {from_agent.name}",
    )
    to_agent.balance_usd += amount_usd
    db.add(to_tx)

    await db.commit()
    await db.refresh(from_agent)
    await db.refresh(to_agent)

    return from_tx, None
