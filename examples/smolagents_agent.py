"""
AgentPay — Smolagents (HuggingFace) Integration

Give your Smolagents agent budget governance and payment capabilities.
Smolagents is HuggingFace's lightweight agent framework — growing fast
as a simple, code-first alternative to LangChain.

Requirements:
    pip install agentpay smolagents
"""

from __future__ import annotations

import os

from smolagents import CodeAgent, HfApiModel, tool

from agentpay import AgentPayClient

# ─── Client ──────────────────────────────────────────

_client = AgentPayClient(
    api_key=os.getenv("AGENTPAY_API_KEY", "ap_your_key_here"),
    base_url="https://leofundmybot.dev",
)


# ─── AgentPay Tools ──────────────────────────────────

@tool
def check_balance() -> str:
    """Check the agent's current wallet balance, daily limits, and spending.
    Call this before making any purchase."""
    b = _client.get_balance()
    return (
        f"Balance: ${b.balance_usd:.2f} | "
        f"Daily limit: ${b.daily_limit_usd:.2f} | "
        f"Remaining today: ${b.daily_remaining_usd:.2f}"
    )


@tool
def spend_money(amount: float, description: str) -> str:
    """Spend money from the agent's wallet for a specific purpose.

    Args:
        amount: Amount in USD to spend (e.g. 0.50, 5.00)
        description: What the money is being spent on (e.g. "GPT-4 API call")
    """
    tx = _client.spend(amount=amount, description=description)
    return (
        f"✅ Spent ${tx.amount:.2f} on '{tx.description}' | "
        f"TX: {tx.id} | Remaining: ${tx.remaining_balance:.2f}"
    )


@tool
def get_transactions(limit: int = 5) -> str:
    """Get recent transaction history.

    Args:
        limit: Number of transactions to return (default 5, max 100)
    """
    txs = _client.get_transactions(limit=limit)
    if not txs:
        return "No transactions yet."
    lines = []
    for tx in txs:
        lines.append(
            f"• {tx.type} ${tx.amount:.2f} — {tx.description} "
            f"({tx.created_at}) [{tx.id}]"
        )
    return "\n".join(lines)


@tool
def transfer_funds(to_agent_id: str, amount: float, description: str = "") -> str:
    """Transfer funds to another agent.

    Args:
        to_agent_id: Target agent ID
        amount: Amount in USD
        description: Optional transfer description
    """
    tx = _client.transfer(
        to_agent_id=to_agent_id, amount=amount, description=description
    )
    return f"✅ Transferred ${tx.amount:.2f} to {tx.to_agent_id} | TX: {tx.id}"


@tool
def pay_x402(url: str, max_amount: float = 1.0) -> str:
    """Pay for an x402-gated API resource using the agent's on-chain wallet.

    Args:
        url: The x402-gated resource URL
        max_amount: Maximum price willing to pay in USD (default 1.0)
    """
    result = _client.x402_pay(url=url, max_amount=max_amount)
    return f"✅ Paid ${result.amount:.2f} for {url} | TX: {result.tx_hash}"


# ─── Agent ────────────────────────────────────────────

def main():
    model = HfApiModel()  # Uses HF_TOKEN env var

    agent = CodeAgent(
        tools=[check_balance, spend_money, get_transactions, transfer_funds, pay_x402],
        model=model,
        system_prompt=(
            "You are a helpful assistant with a budget. You can check your balance, "
            "spend money on tasks, transfer to other agents, and pay for x402-gated APIs. "
            "Always check your balance before spending. Never exceed your daily limit."
        ),
    )

    result = agent.run("Check my balance, then spend $0.10 on a test transaction.")
    print(result)


if __name__ == "__main__":
    main()
