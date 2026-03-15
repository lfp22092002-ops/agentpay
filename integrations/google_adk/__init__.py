"""
AgentPay — Google ADK integration

Provides AgentPay tools for Google Agent Development Kit agents:
- check_budget: Current balance and daily remaining
- spend_funds: Spend from agent wallet
- get_transactions: Transaction history

Usage with ADK:
    from integrations.google_adk import get_agentpay_tools

    tools = get_agentpay_tools("ap_your_key", "https://leofundmybot.dev")
    agent = Agent(name="my_agent", tools=tools, ...)
"""

from agentpay import AgentPayClient
from agentpay.exceptions import InsufficientFundsError, AgentPayError


def get_agentpay_tools(api_key: str, base_url: str = "https://leofundmybot.dev") -> list:
    """Create AgentPay tools for a Google ADK agent.

    Args:
        api_key: AgentPay API key (ap_xxx...)
        base_url: AgentPay API base URL

    Returns:
        List of callable tools compatible with google.adk.agents.Agent
    """
    client = AgentPayClient(api_key, base_url=base_url)

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
            return {"error": "Insufficient funds"}
        except AgentPayError as e:
            return {"error": str(e)}

    def get_transactions(limit: int = 5) -> dict:
        """Get recent transaction history.

        Args:
            limit: Number of transactions to retrieve
        """
        try:
            txs = client.get_transactions(limit=limit)
            return {
                "transactions": [
                    {
                        "id": tx.transaction_id,
                        "amount": str(tx.amount),
                        "description": tx.description,
                        "status": tx.status,
                    }
                    for tx in txs
                ]
            }
        except AgentPayError as e:
            return {"error": str(e)}

    return [check_budget, spend_funds, get_transactions]
