"""
AgentPay — LangChain Tool Integration

Give your LangChain agent the ability to spend money, check budgets,
and pay for x402-gated APIs autonomously.

Works with any LangChain agent type (ReAct, OpenAI functions, etc.)

Requirements:
    pip install agentpay langchain langchain-openai
"""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from agentpay import AgentPayClient


# ─── Configuration ────────────────────────────────────

AGENTPAY_KEY = "ap_your_key_here"
AGENTPAY_URL = "https://leofundmybot.dev"

client = AgentPayClient(AGENTPAY_KEY, base_url=AGENTPAY_URL)


# ─── Tool: Check Balance ─────────────────────────────

class CheckBalanceTool(BaseTool):
    name: str = "check_budget"
    description: str = (
        "Check the agent's current balance. Use before making purchases "
        "or when asked about remaining budget."
    )

    def _run(self) -> str:
        balance = client.get_balance()
        return (
            f"Balance: ${balance.balance_usd:.2f} | "
            f"Daily limit: ${balance.daily_limit_usd:.2f} | "
            f"Remaining today: ${balance.daily_remaining_usd:.2f} | "
            f"Per-tx limit: ${balance.tx_limit_usd:.2f}"
        )


# ─── Tool: Spend Money ───────────────────────────────

class SpendInput(BaseModel):
    amount: float = Field(description="Amount in USD to spend (e.g. 0.50)")
    description: str = Field(description="What the money is being spent on")


class SpendMoneyTool(BaseTool):
    name: str = "spend_money"
    description: str = (
        "Spend money from the agent's balance. Use when you need to pay "
        "for an API call, service, or resource. Always check budget first."
    )
    args_schema: type[BaseModel] = SpendInput

    def _run(self, amount: float, description: str) -> str:
        try:
            tx = client.spend(amount, description)
            if tx.success:
                return (
                    f"✅ Spent ${tx.amount:.2f} — {description}. "
                    f"Remaining: ${tx.remaining_balance:.2f}"
                )
            if tx.approval_id:
                return (
                    f"⏳ Requires approval (id: {tx.approval_id}). "
                    f"Amount ${amount:.2f} exceeds auto-approve limit."
                )
            return f"❌ Failed: {tx.error}"
        except Exception as e:
            return f"❌ Error: {e}"


# ─── Tool: Pay for x402 Resource ─────────────────────

class X402Input(BaseModel):
    url: str = Field(description="URL of the x402-gated resource")
    max_price: float = Field(
        default=1.0, description="Maximum price willing to pay in USD"
    )


class X402PayTool(BaseTool):
    name: str = "pay_for_resource"
    description: str = (
        "Pay for an x402-gated API or resource using the agent's wallet. "
        "Use when a service requires payment via the x402 protocol."
    )
    args_schema: type[BaseModel] = X402Input

    def _run(self, url: str, max_price: float = 1.0) -> str:
        try:
            result = client.x402_pay(url, max_amount=max_price)
            if result.success:
                return (
                    f"✅ Paid ${result.paid_usd:.4f} for {url}. "
                    f"Response data: {result.data[:500] if result.data else 'N/A'}"
                )
            return f"❌ Payment failed: {result.error}"
        except Exception as e:
            return f"❌ Error: {e}"


# ─── Tool: Transaction History ────────────────────────

class TransactionHistoryTool(BaseTool):
    name: str = "transaction_history"
    description: str = (
        "Get recent transaction history. Use when asked about past "
        "spending or to audit what the agent has paid for."
    )

    def _run(self) -> str:
        txs = client.get_transactions(limit=5)
        if not txs:
            return "No transactions yet."
        lines = [f"Last {len(txs)} transactions:"]
        for tx in txs:
            lines.append(
                f"  {tx.type}: ${tx.amount:.2f} — {tx.description or 'N/A'} "
                f"({tx.status})"
            )
        return "\n".join(lines)


# ─── Build the Agent ──────────────────────────────────

def build_agent():
    """Build a LangChain agent with AgentPay tools."""
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    tools = [
        CheckBalanceTool(),
        SpendMoneyTool(),
        X402PayTool(),
        TransactionHistoryTool(),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an AI agent with your own wallet. You can check your "
         "balance, spend money on APIs, and pay for x402-gated resources. "
         "Always check your budget before spending. Be frugal."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


if __name__ == "__main__":
    executor = build_agent()

    # Example: agent checks budget and decides whether to spend
    result = executor.invoke({
        "input": "Check my budget, then spend $0.05 on a test API call"
    })
    print(f"\n📄 Result: {result['output']}")
