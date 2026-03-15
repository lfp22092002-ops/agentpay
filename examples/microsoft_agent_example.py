"""
Microsoft Agent Framework + AgentPay example.

Shows how to create a budget-aware agent using Microsoft Agent Framework
with AgentPay tools for spending governance.

Requirements:
    pip install agent-framework agentpay

Usage:
    export AGENTPAY_API_KEY="ap_your_key_here"
    python microsoft_agent_example.py
"""

import asyncio
import os
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIResponsesClient
from agentpay import AgentPayClient
from agentpay.exceptions import InsufficientFundsError, AgentPayError

API_KEY = os.getenv("AGENTPAY_API_KEY", "ap_your_key_here")
BASE_URL = os.getenv("AGENTPAY_BASE_URL", "https://leofundmybot.dev")

client = AgentPayClient(API_KEY, base_url=BASE_URL)


@tool
def check_budget() -> dict:
    """Check the agent's current balance and remaining daily budget."""
    try:
        balance = client.get_balance()
        return {
            "balance_usd": str(balance.balance_usd),
            "daily_remaining_usd": str(balance.daily_remaining_usd),
            "daily_limit_usd": str(balance.daily_limit_usd),
        }
    except AgentPayError as e:
        return {"error": str(e)}


@tool
def spend_funds(amount: float, description: str) -> dict:
    """Spend funds from the agent's wallet.

    Args:
        amount: USD amount to spend
        description: What the funds are for
    """
    try:
        tx = client.spend(amount, description)
        return {
            "success": True,
            "transaction_id": tx.transaction_id,
            "amount_spent": str(tx.amount),
            "remaining_balance": str(tx.remaining_balance),
        }
    except InsufficientFundsError:
        return {"error": "Insufficient funds", "success": False}
    except AgentPayError as e:
        return {"error": str(e), "success": False}


@tool
def get_transactions(limit: int = 5) -> dict:
    """Get recent transaction history."""
    try:
        txs = client.get_transactions(limit=limit)
        return {
            "transactions": [
                {"id": tx.transaction_id, "amount": str(tx.amount),
                 "description": tx.description, "status": tx.status}
                for tx in txs
            ]
        }
    except AgentPayError as e:
        return {"error": str(e)}


async def main():
    agent = OpenAIResponsesClient().as_agent(
        name="BudgetBot",
        instructions="""You are a budget-aware assistant. Before any action that costs money:
1. Check your budget with check_budget
2. Only proceed if you have sufficient funds
3. Use spend_funds to record expenses
4. Report remaining balance after spending""",
        tools=[check_budget, spend_funds, get_transactions],
    )

    result = await agent.run("Check my budget and tell me how much I can spend today.")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
