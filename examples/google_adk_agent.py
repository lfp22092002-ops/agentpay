"""
AgentPay — Google ADK Agent with Budget Control

An AI agent built with Google's Agent Development Kit (ADK) that uses
AgentPay to manage its own spending on external tool calls.

Requirements:
    pip install agentpay google-adk
"""

import os
from agentpay import AgentPayClient
from agentpay.exceptions import InsufficientFundsError, AgentPayError

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


# ─── Configuration ───

AGENTPAY_KEY = os.getenv("AGENTPAY_API_KEY", "ap_your_key_here")
AGENTPAY_URL = os.getenv("AGENTPAY_URL", "https://leofundmybot.dev")

pay = AgentPayClient(AGENTPAY_KEY, base_url=AGENTPAY_URL)


# ─── AgentPay Tools (callable by the agent) ───

def check_budget() -> dict:
    """Check the agent's current balance and remaining daily budget."""
    try:
        balance = pay.get_balance()
        return {
            "balance_usd": str(balance.balance_usd),
            "daily_remaining_usd": str(balance.daily_remaining_usd),
            "daily_limit_usd": str(balance.daily_limit_usd),
        }
    except AgentPayError as e:
        return {"error": str(e)}


def spend_funds(amount: float, description: str) -> dict:
    """Spend funds from the agent's wallet for a specific purpose.

    Args:
        amount: USD amount to spend (e.g. 0.50)
        description: What the funds are being used for
    """
    try:
        tx = pay.spend(amount, description)
        return {
            "success": True,
            "transaction_id": tx.transaction_id,
            "amount_spent": str(tx.amount),
            "remaining_balance": str(tx.remaining_balance),
        }
    except InsufficientFundsError:
        return {"error": "Insufficient funds", "suggestion": "Request a top-up or reduce spend"}
    except AgentPayError as e:
        return {"error": str(e)}


def get_transactions(limit: int = 5) -> dict:
    """Get recent transaction history.

    Args:
        limit: Number of recent transactions to retrieve (default 5)
    """
    try:
        txs = pay.get_transactions(limit=limit)
        return {
            "transactions": [
                {
                    "id": tx.transaction_id,
                    "amount": str(tx.amount),
                    "description": tx.description,
                    "status": tx.status,
                    "created_at": tx.created_at,
                }
                for tx in txs
            ]
        }
    except AgentPayError as e:
        return {"error": str(e)}


# ─── ADK Agent Definition ───

budget_agent = Agent(
    name="budget_agent",
    model="gemini-2.0-flash",
    description="An AI agent that manages its own budget using AgentPay",
    instruction="""You are an autonomous agent with your own wallet managed by AgentPay.

Before doing anything that costs money:
1. Check your budget with check_budget()
2. If sufficient, spend with spend_funds(amount, description)
3. Always describe what you're spending on

Be transparent about your spending. If funds are low, inform the user
and suggest they top up via the Telegram bot @FundmyAIbot.

You can also review your transaction history to track spending patterns.""",
    tools=[check_budget, spend_funds, get_transactions],
)


# ─── Run the Agent ───

def main():
    session_service = InMemorySessionService()
    runner = Runner(
        agent=budget_agent,
        app_name="agentpay_demo",
        session_service=session_service,
    )

    session = session_service.create_session(
        app_name="agentpay_demo",
        user_id="demo_user",
    )

    print("🤖 ADK Agent with AgentPay Budget Control")
    print("=" * 50)
    print("Try: 'Check my balance', 'Spend $1 on data analysis', 'Show my transactions'")
    print()

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        from google.adk.events import Event

        response = runner.run(
            session_id=session.id,
            user_id="demo_user",
            new_message=Event.user_message(user_input),
        )

        for event in response:
            if event.is_final_response():
                print(f"Agent: {event.content}")
                print()


if __name__ == "__main__":
    main()
