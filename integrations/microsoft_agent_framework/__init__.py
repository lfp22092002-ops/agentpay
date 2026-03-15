"""
AgentPay — Microsoft Agent Framework integration

Provides AgentPay tools decorated with @tool for the Microsoft Agent Framework
(successor to Semantic Kernel + AutoGen). Tools can be passed directly to
Agent(..., tools=[...]).

Usage:
    from integrations.microsoft_agent_framework import get_agentpay_tools

    tools = get_agentpay_tools("ap_your_key")
    agent = Agent(
        name="my_agent",
        instructions="You are a budget-aware assistant.",
        tools=tools,
    )
"""

from agentpay import AgentPayClient
from agentpay.exceptions import InsufficientFundsError, AgentPayError

try:
    from agent_framework import tool
except ImportError:
    # Fallback: define a no-op decorator for import-time compatibility
    def tool(fn=None, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f


def get_agentpay_tools(api_key: str, base_url: str = "https://leofundmybot.dev") -> list:
    """Create AgentPay tools for a Microsoft Agent Framework agent.

    Args:
        api_key: AgentPay API key (ap_xxx...)
        base_url: AgentPay API base URL

    Returns:
        List of @tool decorated callables for Agent(tools=[...])
    """
    client = AgentPayClient(api_key, base_url=base_url)

    @tool
    def check_budget() -> dict:
        """Check the agent's current balance and remaining daily budget.
        Returns balance_usd, daily_remaining_usd, daily_limit_usd."""
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
            amount: USD amount to spend (e.g., 0.50)
            description: What the funds are for (e.g., "API call to OpenAI")
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
        """Get recent transaction history.

        Args:
            limit: Number of transactions to retrieve (default 5)
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

    @tool
    def list_payee_rules() -> dict:
        """List the agent's payee whitelist/blocklist rules."""
        try:
            rules = client.list_payee_rules()
            return {"rules": rules.get("rules", []), "total": rules.get("total", 0)}
        except AgentPayError as e:
            return {"error": str(e)}

    return [check_budget, spend_funds, get_transactions, list_payee_rules]
