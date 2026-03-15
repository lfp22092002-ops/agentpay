"""AgentPay LangChain tool definitions.

Each tool wraps a single AgentPay SDK method with proper schemas so
LangChain agents can discover and invoke them automatically.

Requires: pip install langchain-core agentpay
"""

from __future__ import annotations

from typing import Any, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class BalanceInput(BaseModel):
    """No input required."""
    pass


class SpendInput(BaseModel):
    amount: float = Field(..., description="Amount in USD to spend")
    description: str = Field(..., description="What the spend is for")
    category: Optional[str] = Field(None, description="Spending category")
    idempotency_key: Optional[str] = Field(None, description="Unique key to prevent duplicate charges")


class TransferInput(BaseModel):
    to_agent_id: str = Field(..., description="Target agent ID to transfer to")
    amount: float = Field(..., description="Amount in USD to transfer")
    description: Optional[str] = Field(None, description="Transfer description")


class TransactionsInput(BaseModel):
    limit: int = Field(10, description="Max transactions to return")
    offset: int = Field(0, description="Pagination offset")
    tx_type: Optional[str] = Field(None, description="Filter by type: spend, transfer, refund, deposit")


class X402PayInput(BaseModel):
    url: str = Field(..., description="URL that returned HTTP 402")
    max_amount: float = Field(..., description="Max USD willing to pay")


class IdentityInput(BaseModel):
    """No input required."""
    pass


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class AgentPayBalanceTool(BaseTool):
    """Check the agent's current wallet balance."""

    name: str = "agentpay_balance"
    description: str = (
        "Check the agent's AgentPay wallet balance. Returns balance in USD, "
        "total spent, total received, and pending amounts."
    )
    args_schema: Type[BaseModel] = BalanceInput
    api_key: str = ""
    base_url: str = "https://leofundmybot.dev"

    def _run(self, **kwargs: Any) -> str:
        from agentpay import AgentPayClient
        client = AgentPayClient(self.api_key, base_url=self.base_url)
        b = client.get_balance()
        return (
            f"Balance: ${b.balance_usd:.2f} | "
            f"Spent: ${b.total_spent:.2f} | "
            f"Received: ${b.total_received:.2f} | "
            f"Pending: ${b.pending_amount:.2f}"
        )


class AgentPaySpendTool(BaseTool):
    """Spend funds from the agent's wallet."""

    name: str = "agentpay_spend"
    description: str = (
        "Spend funds from the agent's AgentPay wallet. Provide amount in USD "
        "and a description of what the spend is for. Returns transaction ID "
        "and remaining balance."
    )
    args_schema: Type[BaseModel] = SpendInput
    api_key: str = ""
    base_url: str = "https://leofundmybot.dev"

    def _run(
        self,
        amount: float,
        description: str,
        category: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        from agentpay import AgentPayClient
        client = AgentPayClient(self.api_key, base_url=self.base_url)
        tx = client.spend(
            amount=amount,
            description=description,
            category=category,
            idempotency_key=idempotency_key,
        )
        return (
            f"Spent ${tx.amount:.2f} — {description} | "
            f"TX: {tx.transaction_id} | "
            f"Remaining: ${tx.remaining_balance:.2f}"
        )


class AgentPayTransferTool(BaseTool):
    """Transfer funds to another agent."""

    name: str = "agentpay_transfer"
    description: str = (
        "Transfer USD from this agent's wallet to another agent. "
        "Requires the target agent_id and amount."
    )
    args_schema: Type[BaseModel] = TransferInput
    api_key: str = ""
    base_url: str = "https://leofundmybot.dev"

    def _run(
        self,
        to_agent_id: str,
        amount: float,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        from agentpay import AgentPayClient
        client = AgentPayClient(self.api_key, base_url=self.base_url)
        tx = client.transfer(
            to_agent_id=to_agent_id,
            amount=amount,
            description=description,
        )
        return (
            f"Transferred ${tx.amount:.2f} to {to_agent_id} | "
            f"TX: {tx.transaction_id} | "
            f"Remaining: ${tx.remaining_balance:.2f}"
        )


class AgentPayTransactionsTool(BaseTool):
    """List recent transactions."""

    name: str = "agentpay_transactions"
    description: str = (
        "List recent AgentPay transactions. Optionally filter by type "
        "(spend, transfer, refund, deposit) and paginate with limit/offset."
    )
    args_schema: Type[BaseModel] = TransactionsInput
    api_key: str = ""
    base_url: str = "https://leofundmybot.dev"

    def _run(
        self,
        limit: int = 10,
        offset: int = 0,
        tx_type: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        from agentpay import AgentPayClient
        client = AgentPayClient(self.api_key, base_url=self.base_url)
        txs = client.list_transactions(limit=limit, offset=offset, tx_type=tx_type)
        if not txs:
            return "No transactions found."
        lines = []
        for tx in txs:
            lines.append(
                f"  {tx.created_at} | {tx.tx_type} | ${tx.amount:.2f} | {tx.description}"
            )
        return f"Transactions ({len(txs)}):\n" + "\n".join(lines)


class AgentPayX402PayTool(BaseTool):
    """Pay for a resource using x402 protocol."""

    name: str = "agentpay_x402_pay"
    description: str = (
        "Pay for a resource that returned HTTP 402 Payment Required. "
        "Provide the URL and max amount willing to pay in USD."
    )
    args_schema: Type[BaseModel] = X402PayInput
    api_key: str = ""
    base_url: str = "https://leofundmybot.dev"

    def _run(self, url: str, max_amount: float, **kwargs: Any) -> str:
        from agentpay import AgentPayClient
        client = AgentPayClient(self.api_key, base_url=self.base_url)
        result = client.x402_pay(url=url, max_amount=max_amount)
        return (
            f"Paid ${result.amount_paid:.4f} for {url} | "
            f"TX: {result.transaction_id} | "
            f"Status: {result.status}"
        )


class AgentPayIdentityTool(BaseTool):
    """Get the agent's identity and trust score."""

    name: str = "agentpay_identity"
    description: str = (
        "Get the agent's AgentPay identity profile and trust score (0-100). "
        "Trust score factors: account age, transaction count, volume, profile, verification."
    )
    args_schema: Type[BaseModel] = IdentityInput
    api_key: str = ""
    base_url: str = "https://leofundmybot.dev"

    def _run(self, **kwargs: Any) -> str:
        from agentpay import AgentPayClient
        client = AgentPayClient(self.api_key, base_url=self.base_url)
        identity = client.get_identity()
        return (
            f"Agent: {identity.agent_id} | "
            f"Name: {identity.name or 'unnamed'} | "
            f"Trust: {identity.trust_score}/100 | "
            f"Verified: {identity.verified}"
        )


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def get_agentpay_tools(
    api_key: str,
    base_url: str = "https://leofundmybot.dev",
) -> List[BaseTool]:
    """Return all AgentPay tools configured with the given API key.

    Example::

        from agentpay.integrations.langchain import get_agentpay_tools
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_react_agent

        tools = get_agentpay_tools("ap_xxxx...")
        llm = ChatOpenAI(model="gpt-4o")
        agent = create_react_agent(llm, tools)
    """
    common = {"api_key": api_key, "base_url": base_url}
    return [
        AgentPayBalanceTool(**common),
        AgentPaySpendTool(**common),
        AgentPayTransferTool(**common),
        AgentPayTransactionsTool(**common),
        AgentPayX402PayTool(**common),
        AgentPayIdentityTool(**common),
    ]
