"""
AgentPay — OpenAI Agents SDK Integration

Give your OpenAI agent autonomous spending via AgentPay's MCP server.
The OpenAI Agents SDK has built-in MCP support, so AgentPay tools
(balance, spend, transfer, x402) appear natively alongside other tools.

Two approaches shown:
  1. Remote MCP (connects to leofundmybot.dev — zero setup)
  2. Function tools (direct SDK calls — works offline)

Requirements:
    pip install openai-agents agentpay
"""

import asyncio

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHTTP


# ─── Approach 1: MCP Server (Recommended) ────────────
# Agent discovers all AgentPay tools automatically via MCP.
# Zero boilerplate — just point at the server.

async def mcp_agent_demo():
    """Agent with AgentPay tools via MCP — automatic discovery."""

    agentpay_mcp = MCPServerStreamableHTTP(
        url="https://leofundmybot.dev/mcp",
        headers={"x-api-key": "ap_your_key_here"},
    )

    agent = Agent(
        name="Budget-Aware Agent",
        instructions=(
            "You are an AI agent with your own wallet via AgentPay. "
            "You can check your balance, spend money on APIs, transfer "
            "funds to other agents, and pay for x402-gated resources. "
            "Always check your budget before spending. Be frugal."
        ),
        mcp_servers=[agentpay_mcp],
    )

    # Agent auto-discovers: agentpay_balance, agentpay_spend,
    # agentpay_transfer, agentpay_x402_pay, agentpay_wallet, etc.

    result = await Runner.run(
        agent,
        "Check my balance, then spend $0.02 on a test transaction.",
    )
    print(f"Result: {result.final_output}")


# ─── Approach 2: Function Tools (Direct SDK) ─────────
# For when you want explicit control or offline operation.

from agents import function_tool
from agentpay import AgentPayClient

AGENTPAY_KEY = "ap_your_key_here"
pay = AgentPayClient(AGENTPAY_KEY, base_url="https://leofundmybot.dev")


@function_tool
def check_budget() -> str:
    """Check the agent's current balance and spending limits."""
    b = pay.get_balance()
    return (
        f"Balance: ${b.balance_usd:.2f} | "
        f"Daily limit: ${b.daily_limit_usd:.2f} | "
        f"Remaining today: ${b.daily_remaining_usd:.2f}"
    )


@function_tool
def spend_money(amount: float, description: str) -> str:
    """Spend money from the agent's wallet for an API or service."""
    try:
        tx = pay.spend(amount, description)
        if tx.success:
            return f"✅ Spent ${tx.amount:.2f} — {description}. Remaining: ${tx.remaining_balance:.2f}"
        if tx.approval_id:
            return f"⏳ Needs approval (id: {tx.approval_id}). Amount exceeds auto-approve limit."
        return f"❌ Failed: {tx.error}"
    except Exception as e:
        return f"❌ Error: {e}"


@function_tool
def pay_x402_resource(url: str, max_price: float = 1.0) -> str:
    """Pay for an x402-gated API resource using the agent's wallet."""
    try:
        result = pay.x402_pay(url, max_amount=max_price)
        if result.success:
            return f"✅ Paid ${result.paid_usd:.4f} for {url}"
        return f"❌ Payment failed: {result.error}"
    except Exception as e:
        return f"❌ Error: {e}"


async def function_tools_demo():
    """Agent with AgentPay via direct function tools."""

    agent = Agent(
        name="Frugal Agent",
        instructions=(
            "You have a wallet. Check budget before spending. "
            "Use pay_x402_resource for paywalled APIs."
        ),
        tools=[check_budget, spend_money, pay_x402_resource],
    )

    result = await Runner.run(
        agent,
        "What's my budget? Then spend $0.01 on testing.",
    )
    print(f"Result: {result.final_output}")


# ─── Run ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== MCP Server Approach ===")
    asyncio.run(mcp_agent_demo())

    print("\n=== Function Tools Approach ===")
    asyncio.run(function_tools_demo())
