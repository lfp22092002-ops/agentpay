"""
AgentPay — AutoGen Integration

Give your AutoGen agents autonomous spending capabilities.
Budget-aware tool use with balance checks, spending, and x402 micropayments.

Requirements:
    pip install agentpay pyautogen
"""

from typing import Annotated
from autogen import ConversableAgent, register_function

from agentpay import AgentPayClient


# ─── Configuration ────────────────────────────────────

AGENTPAY_KEY = "ap_your_key_here"
AGENTPAY_URL = "https://leofundmybot.dev"

pay = AgentPayClient(AGENTPAY_KEY, base_url=AGENTPAY_URL)


# ─── Tool Functions ───────────────────────────────────

def check_balance() -> str:
    """Check the agent's current balance and spending limits."""
    b = pay.get_balance()
    return (
        f"Balance: ${b.balance_usd:.2f} | "
        f"Daily limit: ${b.daily_limit_usd:.2f} | "
        f"Remaining today: ${b.daily_remaining_usd:.2f}"
    )


def spend(
    amount: Annotated[float, "Amount in USD to spend"],
    description: Annotated[str, "What the money is for"],
) -> str:
    """Spend from the agent's wallet. Returns transaction receipt."""
    b = pay.get_balance()
    if amount > b.daily_remaining_usd:
        return f"BLOCKED: ${amount:.2f} exceeds remaining daily budget (${b.daily_remaining_usd:.2f})"
    tx = pay.spend(amount=amount, description=description)
    return (
        f"✅ Spent ${tx.amount:.2f} on: {description} | "
        f"TX: {tx.transaction_id} | "
        f"Remaining: ${tx.remaining_balance:.2f}"
    )


def get_transactions(
    limit: Annotated[int, "Number of recent transactions"] = 5,
) -> str:
    """Get recent transaction history."""
    txs = pay.get_transactions(limit=limit)
    if not txs:
        return "No transactions yet."
    lines = []
    for tx in txs:
        lines.append(
            f"  {tx.created_at} | ${tx.amount:.2f} | {tx.type} | {tx.description}"
        )
    return "Recent transactions:\n" + "\n".join(lines)


def pay_x402_resource(
    url: Annotated[str, "The x402-gated URL to access"],
    max_amount: Annotated[float, "Maximum willing to pay in USD"] = 1.0,
) -> str:
    """Pay for an x402-gated HTTP resource using the agent's wallet."""
    result = pay.x402_pay(url=url, max_amount=max_amount)
    return f"✅ Accessed {url} | Paid: ${result.amount_paid:.4f} | Status: {result.status}"


# ─── Agent Setup ──────────────────────────────────────

assistant = ConversableAgent(
    name="research_assistant",
    system_message=(
        "You are a research assistant with a budget. "
        "Always check your balance before spending. "
        "Use x402 payments for gated APIs. "
        "Report costs after each task."
    ),
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": "YOUR_OPENAI_KEY"}]},
)

user_proxy = ConversableAgent(
    name="user",
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TASK_COMPLETE" in msg.get("content", ""),
)

# Register tools with both agents
for func in [check_balance, spend, get_transactions, pay_x402_resource]:
    register_function(
        func,
        caller=assistant,
        executor=user_proxy,
        description=func.__doc__,
    )


# ─── Run ──────────────────────────────────────────────

if __name__ == "__main__":
    user_proxy.initiate_chat(
        assistant,
        message=(
            "Check your budget, then research the latest x402 protocol news. "
            "Use the x402 payment tool if you find any gated resources. "
            "Report your findings and remaining budget."
        ),
    )
