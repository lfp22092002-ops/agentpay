"""
AgentPay — Pydantic AI Agent Integration

Give your Pydantic AI agent budget governance and payment capabilities.
Pydantic AI is the type-safe, production-ready agent framework from the
creators of Pydantic — growing fast in 2026.

Requirements:
    pip install agentpay pydantic-ai
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext, Tool

from agentpay import AgentPayClient


# ─── Dependencies (injected into agent context) ──────

@dataclass
class AgentDeps:
    """Dependencies injected into the agent at runtime."""
    pay: AgentPayClient
    user_name: str = "User"


# ─── AgentPay Tools ──────────────────────────────────

async def check_balance(ctx: RunContext[AgentDeps]) -> str:
    """Check the agent's current wallet balance, daily limits, and spending info.
    Call this before making any purchase or when the user asks about budget."""
    b = ctx.deps.pay.get_balance()
    return (
        f"Balance: ${b.balance_usd:.2f} | "
        f"Daily limit: ${b.daily_limit_usd:.2f} | "
        f"Remaining today: ${b.daily_remaining_usd:.2f}"
    )


async def spend(ctx: RunContext[AgentDeps], amount: float, description: str) -> str:
    """Spend money from the agent's wallet for a specific purpose.

    Args:
        amount: Amount in USD to spend (e.g. 0.50, 5.00)
        description: What the money is being spent on (e.g. "GPT-4 API call")
    """
    tx = ctx.deps.pay.spend(amount=amount, description=description)
    return (
        f"✅ Spent ${tx.amount:.2f} on '{tx.description}' | "
        f"TX: {tx.id} | Remaining: ${tx.remaining_balance:.2f}"
    )


async def get_transactions(ctx: RunContext[AgentDeps], limit: int = 5) -> str:
    """Get recent transaction history.

    Args:
        limit: Number of transactions to return (default 5, max 100)
    """
    txs = ctx.deps.pay.get_transactions(limit=limit)
    if not txs:
        return "No transactions yet."
    lines = []
    for tx in txs:
        lines.append(
            f"• {tx.type} ${tx.amount:.2f} — {tx.description} "
            f"({tx.created_at}) [{tx.id}]"
        )
    return "\n".join(lines)


async def pay_x402(ctx: RunContext[AgentDeps], url: str, max_amount: float = 1.0) -> str:
    """Pay for an x402-gated API resource using the agent's on-chain wallet.

    Args:
        url: The x402-gated resource URL
        max_amount: Maximum price willing to pay in USD (default 1.0)
    """
    result = ctx.deps.pay.x402_pay(url=url, max_amount=max_amount)
    return f"✅ Paid ${result.amount:.2f} for {url} | TX: {result.tx_hash}"


# ─── Agent Definition ────────────────────────────────

agent = Agent(
    "openai:gpt-4o",
    deps_type=AgentDeps,
    system_prompt=(
        "You are a helpful assistant with a budget. You can check your balance, "
        "spend money on tasks, and pay for x402-gated APIs. Always check your "
        "balance before spending. Never exceed your daily limit. Report all "
        "transactions clearly."
    ),
    tools=[
        Tool(check_balance, takes_ctx=True),
        Tool(spend, takes_ctx=True),
        Tool(get_transactions, takes_ctx=True),
        Tool(pay_x402, takes_ctx=True),
    ],
)


# ─── Run ──────────────────────────────────────────────

async def main():
    api_key = os.getenv("AGENTPAY_API_KEY", "ap_your_key_here")
    client = AgentPayClient(api_key, base_url="https://leofundmybot.dev")

    deps = AgentDeps(pay=client, user_name="Developer")

    result = await agent.run(
        "Check my balance, then spend $0.10 on a test transaction.",
        deps=deps,
    )
    print(result.data)
    print(f"\n--- Usage: {result.usage()} ---")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
