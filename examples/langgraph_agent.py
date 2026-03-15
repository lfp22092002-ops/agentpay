"""
AgentPay — LangGraph Agent Integration

A budget-aware ReAct agent built with LangGraph that can autonomously
manage its wallet, spend on APIs, and pay for x402-gated resources.

LangGraph gives you full control over the agent loop — conditional edges,
state persistence, human-in-the-loop approval for large purchases.

Requirements:
    pip install agentpay langgraph langchain-openai
"""

from typing import Annotated, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agentpay import AgentPayClient

# ─── Configuration ────────────────────────────────────

AGENTPAY_KEY = "ap_your_key_here"
AGENTPAY_URL = "https://leofundmybot.dev"

client = AgentPayClient(AGENTPAY_KEY, base_url=AGENTPAY_URL)

# Maximum amount the agent can spend without human approval
AUTO_APPROVE_LIMIT = 1.00


# ─── Tools ────────────────────────────────────────────

@tool
def check_balance() -> str:
    """Check the agent's current wallet balance, daily limit, and remaining budget."""
    balance = client.get_balance()
    return (
        f"Balance: ${balance.balance_usd:.2f} | "
        f"Daily limit: ${balance.daily_limit_usd:.2f} | "
        f"Remaining today: ${balance.daily_remaining_usd:.2f} | "
        f"Per-tx limit: ${balance.tx_limit_usd:.2f}"
    )


@tool
def spend(amount: float, description: str) -> str:
    """Spend money from the agent's wallet. Always check balance first.

    Args:
        amount: Amount in USD to spend (e.g. 0.50)
        description: What the money is for (e.g. "GPT-4 API call")
    """
    try:
        tx = client.spend(amount, description)
        if tx.success:
            return (
                f"✅ Spent ${tx.amount:.2f} on '{description}'. "
                f"Remaining: ${tx.remaining_balance:.2f}"
            )
        if tx.approval_id:
            return (
                f"⏳ Requires human approval (id: {tx.approval_id}). "
                f"${amount:.2f} exceeds auto-approve limit."
            )
        return f"❌ Transaction failed: {tx.error}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def x402_pay(url: str, max_price: float = 1.0) -> str:
    """Pay for an x402-gated API or resource using the agent's wallet.

    Args:
        url: URL of the x402-gated resource
        max_price: Maximum price willing to pay in USD (default: $1.00)
    """
    try:
        result = client.x402_pay(url, max_amount=max_price)
        if result.success:
            return (
                f"✅ Paid ${result.paid_usd:.4f} for {url}\n"
                f"Response: {result.data[:500] if result.data else 'N/A'}"
            )
        return f"❌ Payment failed: {result.error}"
    except Exception as e:
        return f"❌ Error: {e}"


@tool
def transaction_history(limit: int = 5) -> str:
    """Get recent transaction history.

    Args:
        limit: Number of transactions to return (default: 5)
    """
    txs = client.get_transactions(limit=limit)
    if not txs:
        return "No transactions yet."
    lines = [f"Last {len(txs)} transactions:"]
    for tx in txs:
        lines.append(
            f"  {tx.type}: ${tx.amount:.2f} — {tx.description or 'N/A'} ({tx.status})"
        )
    return "\n".join(lines)


@tool
def transfer(to_agent_id: str, amount: float, memo: str = "") -> str:
    """Transfer funds to another agent's wallet.

    Args:
        to_agent_id: Target agent ID (e.g. "agent_456")
        amount: Amount in USD to transfer
        memo: Optional note for the transfer
    """
    try:
        tx = client.transfer(to_agent_id=to_agent_id, amount=amount, memo=memo)
        if tx.success:
            return f"✅ Transferred ${tx.amount:.2f} to {to_agent_id}. Remaining: ${tx.remaining_balance:.2f}"
        return f"❌ Transfer failed: {tx.error}"
    except Exception as e:
        return f"❌ Error: {e}"


# ─── Graph State ──────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ─── Graph Construction ──────────────────────────────

tools = [check_balance, spend, x402_pay, transaction_history, transfer]
tool_node = ToolNode(tools)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


def agent(state: AgentState) -> dict:
    """The LLM reasoning node."""
    system = (
        "You are an AI agent with your own crypto wallet powered by AgentPay. "
        "You can check your balance, spend money on APIs, pay for x402-gated "
        "resources, view transaction history, and transfer funds to other agents. "
        "Rules:\n"
        "1. ALWAYS check your balance before spending\n"
        f"2. Never spend more than ${AUTO_APPROVE_LIMIT:.2f} without confirming\n"
        "3. Be frugal — prefer cheaper alternatives when available\n"
        "4. Log what you spend and why"
    )
    messages = [{"role": "system", "content": system}] + state["messages"]
    return {"messages": [llm.invoke(messages)]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Route to tools or end based on LLM output."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ─── Build Graph ──────────────────────────────────────

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()


# ─── Run ──────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 AgentPay + LangGraph Agent\n")

    # Example 1: Budget check + spend
    result = app.invoke({
        "messages": [
            HumanMessage("Check my balance, then spend $0.05 on a test API call")
        ]
    })
    print(f"📄 {result['messages'][-1].content}\n")

    # Example 2: x402 payment
    result = app.invoke({
        "messages": [
            HumanMessage(
                "I need data from https://api.example.com/crypto-prices — "
                "it's an x402-gated endpoint. Pay up to $0.10 for it."
            )
        ]
    })
    print(f"📄 {result['messages'][-1].content}\n")

    # Example 3: Multi-step workflow
    result = app.invoke({
        "messages": [
            HumanMessage(
                "Show my last 3 transactions, check my balance, "
                "and transfer $1.00 to agent_research_bot"
            )
        ]
    })
    print(f"📄 {result['messages'][-1].content}")
