"""
Approval workflow for AgentPay.

When a spend exceeds auto_approve_usd threshold, it goes into pending state.
User gets a Telegram notification with Approve/Deny buttons.
Approved → executes the spend. Denied → refunds nothing (never charged). Timeout → auto-deny after 5 min.
"""
import asyncio
import logging
import time
from decimal import Decimal
from dataclasses import dataclass, field

logger = logging.getLogger("agentpay.approvals")

# Pending approvals: { approval_id: PendingApproval }
_pending: dict[str, "PendingApproval"] = {}

# Default: auto-approve up to this amount. Above → needs human approval.
DEFAULT_AUTO_APPROVE_USD = Decimal("10.00")
APPROVAL_TIMEOUT_SECONDS = 300  # 5 minutes


@dataclass
class PendingApproval:
    id: str
    agent_id: str
    agent_name: str
    user_id: str
    telegram_id: int
    amount_usd: Decimal
    description: str | None
    created_at: float = field(default_factory=time.time)
    result: asyncio.Future | None = None
    resolved: bool = False


def create_approval_id() -> str:
    import secrets
    return f"apr_{secrets.token_hex(8)}"


async def request_approval(
    agent_id: str,
    agent_name: str,
    user_id: str,
    telegram_id: int,
    amount_usd: Decimal,
    description: str | None = None,
) -> "PendingApproval":
    """Create a pending approval and return it. Caller should await the result future."""
    approval_id = create_approval_id()
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    approval = PendingApproval(
        id=approval_id,
        agent_id=agent_id,
        agent_name=agent_name,
        user_id=user_id,
        telegram_id=telegram_id,
        amount_usd=amount_usd,
        description=description,
        result=future,
    )
    _pending[approval_id] = approval

    # Auto-deny after timeout
    async def _timeout():
        await asyncio.sleep(APPROVAL_TIMEOUT_SECONDS)
        if not approval.resolved:
            resolve_approval(approval_id, approved=False, reason="Timed out (5 min)")

    asyncio.create_task(_timeout())

    logger.info(f"Approval requested: {approval_id} — ${amount_usd} for {agent_name}")
    return approval


def resolve_approval(approval_id: str, approved: bool, reason: str | None = None) -> bool:
    """Resolve a pending approval. Returns True if it existed and was pending."""
    approval = _pending.get(approval_id)
    if not approval or approval.resolved:
        return False

    approval.resolved = True

    result = {
        "approved": approved,
        "reason": reason or ("Approved by user" if approved else "Denied by user"),
        "resolved_at": time.time(),
    }

    if approval.result and not approval.result.done():
        approval.result.set_result(result)

    # Clean up after a short delay
    async def _cleanup():
        await asyncio.sleep(10)
        _pending.pop(approval_id, None)

    try:
        asyncio.create_task(_cleanup())
    except RuntimeError:
        _pending.pop(approval_id, None)

    logger.info(f"Approval {approval_id}: {'APPROVED' if approved else 'DENIED'} — {reason}")
    return True


def get_pending(approval_id: str) -> PendingApproval | None:
    return _pending.get(approval_id)


def get_agent_pending(agent_id: str) -> list[PendingApproval]:
    return [a for a in _pending.values() if a.agent_id == agent_id and not a.resolved]


def get_user_pending(telegram_id: int) -> list[PendingApproval]:
    return [a for a in _pending.values() if a.telegram_id == telegram_id and not a.resolved]
