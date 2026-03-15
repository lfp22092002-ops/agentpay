"""AgentPay tool wrappers for HuggingFace Smolagents.

Each tool is a @tool-decorated function compatible with CodeAgent / ToolCallingAgent.
"""

from __future__ import annotations

import os
from typing import Optional

from smolagents import tool

from agentpay import AgentPayClient

_DEFAULT_KEY = os.getenv("AGENTPAY_API_KEY", "")
_DEFAULT_URL = os.getenv("AGENTPAY_BASE_URL", "https://leofundmybot.dev")

# Module-level client — set via get_agentpay_tools() or env vars
_client: Optional[AgentPayClient] = None


def _get_client() -> AgentPayClient:
    global _client
    if _client is None:
        if not _DEFAULT_KEY:
            raise RuntimeError(
                "Set AGENTPAY_API_KEY env var or call get_agentpay_tools(api_key=...)"
            )
        _client = AgentPayClient(api_key=_DEFAULT_KEY, base_url=_DEFAULT_URL)
    return _client


# ─── Tools ────────────────────────────────────────────

@tool
def agentpay_balance() -> str:
    """Check the agent's current wallet balance, daily limits, and spending."""
    b = _get_client().get_balance()
    return (
        f"Balance: ${b.balance_usd:.2f} | "
        f"Daily limit: ${b.daily_limit_usd:.2f} | "
        f"Remaining today: ${b.daily_remaining_usd:.2f}"
    )


@tool
def agentpay_spend(amount: float, description: str) -> str:
    """Spend money from the agent's wallet.

    Args:
        amount: Amount in USD to spend (e.g. 0.50, 5.00)
        description: What the money is being spent on
    """
    tx = _get_client().spend(amount=amount, description=description)
    return (
        f"Spent ${tx.amount:.2f} on '{tx.description}' | "
        f"TX: {tx.id} | Remaining: ${tx.remaining_balance:.2f}"
    )


@tool
def agentpay_transfer(to_agent_id: str, amount: float, description: str = "") -> str:
    """Transfer funds to another agent.

    Args:
        to_agent_id: Target agent ID
        amount: Amount in USD
        description: Transfer description
    """
    tx = _get_client().transfer(
        to_agent_id=to_agent_id, amount=amount, description=description
    )
    return f"Transferred ${tx.amount:.2f} to {tx.to_agent_id} | TX: {tx.id}"


@tool
def agentpay_transactions(limit: int = 5) -> str:
    """Get recent transaction history.

    Args:
        limit: Number of transactions to return (1-100)
    """
    txs = _get_client().get_transactions(limit=limit)
    if not txs:
        return "No transactions yet."
    lines = []
    for tx in txs:
        lines.append(
            f"• {tx.type} ${tx.amount:.2f} — {tx.description} ({tx.created_at})"
        )
    return "\n".join(lines)


@tool
def agentpay_x402_pay(url: str, max_amount: float = 1.0) -> str:
    """Pay for an x402-gated API resource.

    Args:
        url: The x402-gated resource URL
        max_amount: Maximum price in USD (default 1.0)
    """
    result = _get_client().x402_pay(url=url, max_amount=max_amount)
    return f"Paid ${result.amount:.2f} for {url} | TX: {result.tx_hash}"


# ─── Loader ───────────────────────────────────────────

def get_agentpay_tools(
    api_key: Optional[str] = None,
    base_url: str = "https://leofundmybot.dev",
) -> list:
    """Get all AgentPay tools configured with the given API key.

    Args:
        api_key: AgentPay API key (or set AGENTPAY_API_KEY env var)
        base_url: API base URL (default: production)

    Returns:
        List of @tool-decorated functions ready for CodeAgent/ToolCallingAgent.
    """
    global _client
    key = api_key or _DEFAULT_KEY
    if not key:
        raise ValueError(
            "Provide api_key or set AGENTPAY_API_KEY environment variable"
        )
    _client = AgentPayClient(api_key=key, base_url=base_url)
    return [
        agentpay_balance,
        agentpay_spend,
        agentpay_transfer,
        agentpay_transactions,
        agentpay_x402_pay,
    ]
