"""AgentPay CrewAI tool definitions.

Each tool extends CrewAI's BaseTool with proper schemas.

Requires: pip install crewai agentpay
"""

from __future__ import annotations

from typing import List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class SpendInput(BaseModel):
    amount: float = Field(..., description="Amount in USD to spend")
    description: str = Field(..., description="What the spend is for")
    category: Optional[str] = Field(None, description="Spending category")


class TransferInput(BaseModel):
    to_agent_id: str = Field(..., description="Target agent ID")
    amount: float = Field(..., description="Amount in USD to transfer")
    description: Optional[str] = Field(None, description="Transfer description")


class TransactionsInput(BaseModel):
    limit: int = Field(10, description="Max transactions to return")
    tx_type: Optional[str] = Field(None, description="Filter: spend, transfer, refund, deposit")


class X402Input(BaseModel):
    url: str = Field(..., description="URL that returned HTTP 402")
    max_amount: float = Field(..., description="Max USD willing to pay")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class AgentPayBalanceTool(BaseTool):
    name: str = "Check AgentPay Balance"
    description: str = "Check the agent's wallet balance in USD. Returns balance, total spent, and total received."
    api_key: str = ""
    base_url: str = "https://leofundmybot.dev"

    def _run(self) -> str:
        from agentpay import AgentPayClient
        client = AgentPayClient(self.api_key, base_url=self.base_url)
        b = client.get_balance()
        return f"Balance: ${b.balance_usd:.2f} | Spent: ${b.total_spent:.2f} | Received: ${b.total_received:.2f}"


class AgentPaySpendTool(BaseTool):
    name: str = "Spend from AgentPay Wallet"
    description: str = "Spend funds from the agent's wallet. Provide amount (USD) and description."
    args_schema: Type[BaseModel] = SpendInput
    api_key: str = ""
    base_url: str = "https://leofundmybot.dev"

    def _run(self, amount: float, description: str, category: Optional[str] = None) -> str:
        from agentpay import AgentPayClient
        client = AgentPayClient(self.api_key, base_url=self.base_url)
        tx = client.spend(amount=amount, description=description, category=category)
        return f"Spent ${tx.amount:.2f} — {description} | TX: {tx.transaction_id} | Remaining: ${tx.remaining_balance:.2f}"


class AgentPayTransferTool(BaseTool):
    name: str = "Transfer via AgentPay"
    description: str = "Transfer USD to another agent. Requires target agent_id and amount."
    args_schema: Type[BaseModel] = TransferInput
    api_key: str = ""
    base_url: str = "https://leofundmybot.dev"

    def _run(self, to_agent_id: str, amount: float, description: Optional[str] = None) -> str:
        from agentpay import AgentPayClient
        client = AgentPayClient(self.api_key, base_url=self.base_url)
        tx = client.transfer(to_agent_id=to_agent_id, amount=amount, description=description)
        return f"Transferred ${tx.amount:.2f} to {to_agent_id} | TX: {tx.transaction_id}"


class AgentPayTransactionsTool(BaseTool):
    name: str = "List AgentPay Transactions"
    description: str = "List recent transactions. Optionally filter by type (spend, transfer, refund, deposit)."
    args_schema: Type[BaseModel] = TransactionsInput
    api_key: str = ""
    base_url: str = "https://leofundmybot.dev"

    def _run(self, limit: int = 10, tx_type: Optional[str] = None) -> str:
        from agentpay import AgentPayClient
        client = AgentPayClient(self.api_key, base_url=self.base_url)
        txs = client.list_transactions(limit=limit, tx_type=tx_type)
        if not txs:
            return "No transactions found."
        lines = [f"  {tx.created_at} | {tx.tx_type} | ${tx.amount:.2f} | {tx.description}" for tx in txs]
        return f"Transactions ({len(txs)}):\n" + "\n".join(lines)


class AgentPayX402PayTool(BaseTool):
    name: str = "Pay via x402 Protocol"
    description: str = "Pay for a resource that returned HTTP 402. Provide URL and max amount in USD."
    args_schema: Type[BaseModel] = X402Input
    api_key: str = ""
    base_url: str = "https://leofundmybot.dev"

    def _run(self, url: str, max_amount: float) -> str:
        from agentpay import AgentPayClient
        client = AgentPayClient(self.api_key, base_url=self.base_url)
        result = client.x402_pay(url=url, max_amount=max_amount)
        return f"Paid ${result.amount_paid:.4f} for {url} | TX: {result.transaction_id}"


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def get_agentpay_tools(
    api_key: str,
    base_url: str = "https://leofundmybot.dev",
) -> List[BaseTool]:
    """Return all AgentPay tools configured with the given API key.

    Example::

        from agentpay.integrations.crewai import get_agentpay_tools
        from crewai import Agent

        tools = get_agentpay_tools("ap_xxxx...")
        agent = Agent(role="Buyer", tools=tools, goal="Purchase compute resources")
    """
    common = {"api_key": api_key, "base_url": base_url}
    return [
        AgentPayBalanceTool(**common),
        AgentPaySpendTool(**common),
        AgentPayTransferTool(**common),
        AgentPayTransactionsTool(**common),
        AgentPayX402PayTool(**common),
    ]
