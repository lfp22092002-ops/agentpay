"""
AgentPay — LlamaIndex Agent Integration

Give your LlamaIndex agent autonomous spending via AgentPay.
Uses FunctionTool to expose balance, spend, and transfer as
callable tools the agent discovers automatically.

Requirements:
    pip install llama-index agentpay
"""

import asyncio
from agentpay import AgentPay
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI


# ─── Setup ────────────────────────────────────────────
ap = AgentPay(api_key="ap_your_key_here")


# ─── Tool definitions ────────────────────────────────
def check_balance() -> str:
    """Check the agent's current AgentPay wallet balance."""
    bal = asyncio.get_event_loop().run_until_complete(ap.balance())
    return f"Balance: {bal.available} {bal.currency} (held: {bal.held})"


def spend_money(amount: float, description: str, recipient: str = "") -> str:
    """Spend money from the agent's wallet for a task or API call.

    Args:
        amount: Amount to spend (e.g. 0.50)
        description: What the spend is for
        recipient: Optional recipient agent or service
    """
    result = asyncio.get_event_loop().run_until_complete(
        ap.spend(amount=amount, description=description, metadata={"to": recipient})
    )
    return f"Spent {result.amount} {result.currency}. TX: {result.transaction_id}"


def transfer_funds(to_agent: str, amount: float, reason: str) -> str:
    """Transfer funds to another agent's AgentPay wallet.

    Args:
        to_agent: Target agent's API key or wallet ID
        amount: Amount to transfer
        reason: Why the transfer is being made
    """
    result = asyncio.get_event_loop().run_until_complete(
        ap.transfer(to=to_agent, amount=amount, description=reason)
    )
    return f"Transferred {result.amount} to {to_agent}. TX: {result.transaction_id}"


# ─── Build agent ──────────────────────────────────────
tools = [
    FunctionTool.from_defaults(fn=check_balance),
    FunctionTool.from_defaults(fn=spend_money),
    FunctionTool.from_defaults(fn=transfer_funds),
]

agent = ReActAgent.from_tools(
    tools=tools,
    llm=OpenAI(model="gpt-4o-mini"),
    verbose=True,
    system_prompt=(
        "You are a budget-conscious AI agent with your own wallet. "
        "Always check your balance before spending. Be frugal and "
        "explain each transaction before making it."
    ),
)


# ─── Run ──────────────────────────────────────────────
if __name__ == "__main__":
    response = agent.chat(
        "I need to call a weather API that costs $0.01 per request. "
        "Check if I have enough budget, then make the purchase."
    )
    print(response)
