"""
AgentPay — CrewAI Integration

Give your CrewAI agents autonomous spending capabilities.
Each crew member gets budget-aware tools for spending, balance checks,
and x402 micropayments.

Requirements:
    pip install agentpay crewai crewai-tools
"""

from crewai import Agent, Crew, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from agentpay import AgentPayClient
from agentpay.exceptions import AgentPayError


# ─── Configuration ────────────────────────────────────

AGENTPAY_KEY = "ap_your_key_here"
AGENTPAY_URL = "https://leofundmybot.dev"

pay = AgentPayClient(AGENTPAY_KEY, base_url=AGENTPAY_URL)


# ─── Tools ────────────────────────────────────────────

class CheckBudgetTool(BaseTool):
    name: str = "check_budget"
    description: str = (
        "Check the agent's remaining budget before spending. "
        "Returns balance, daily limit, and per-transaction limit."
    )

    def _run(self) -> str:
        b = pay.get_balance()
        return (
            f"Balance: ${b.balance_usd:.2f} | "
            f"Daily remaining: ${b.daily_remaining_usd:.2f} | "
            f"Per-tx limit: ${b.tx_limit_usd:.2f}"
        )


class SpendArgs(BaseModel):
    amount: float = Field(description="Amount in USD")
    description: str = Field(description="What the payment is for")


class SpendTool(BaseTool):
    name: str = "spend"
    description: str = (
        "Spend money from the agent's wallet. Always check_budget first. "
        "Use for API calls, data purchases, or service fees."
    )
    args_schema: type[BaseModel] = SpendArgs

    def _run(self, amount: float, description: str) -> str:
        try:
            tx = pay.spend(amount, description)
            if tx.success:
                return f"✅ Spent ${tx.amount:.2f} — remaining ${tx.remaining_balance:.2f}"
            if tx.approval_id:
                return f"⏳ Needs approval (id: {tx.approval_id})"
            return f"❌ Failed: {tx.error}"
        except AgentPayError as e:
            return f"❌ {e}"


class X402PayArgs(BaseModel):
    url: str = Field(description="x402-gated resource URL")
    max_price: float = Field(default=0.50, description="Max price in USD")


class X402PayTool(BaseTool):
    name: str = "pay_for_resource"
    description: str = (
        "Pay for an x402-gated API endpoint. Use when a service "
        "returns HTTP 402 and requires micropayment."
    )
    args_schema: type[BaseModel] = X402PayArgs

    def _run(self, url: str, max_price: float = 0.50) -> str:
        try:
            result = pay.x402_pay(url, max_amount=max_price)
            if result.success:
                data_preview = (result.data[:300] + "...") if result.data and len(result.data) > 300 else result.data
                return f"✅ Paid ${result.paid_usd:.4f}. Data: {data_preview}"
            return f"❌ {result.error}"
        except AgentPayError as e:
            return f"❌ {e}"


# ─── Crew Definition ─────────────────────────────────

budget_tools = [CheckBudgetTool(), SpendTool(), X402PayTool()]

researcher = Agent(
    role="Research Analyst",
    goal="Find and summarize the latest information on a given topic",
    backstory=(
        "You are a meticulous researcher with access to a spending wallet. "
        "You can pay for premium data sources and APIs. Always check your "
        "budget before spending and be cost-conscious."
    ),
    tools=budget_tools,
    verbose=True,
)

writer = Agent(
    role="Content Writer",
    goal="Transform research into clear, engaging content",
    backstory=(
        "You are a skilled writer who turns raw research into polished "
        "content. You have wallet access for purchasing stock images "
        "or additional data if needed."
    ),
    tools=budget_tools,
    verbose=True,
)


def run_crew(topic: str) -> str:
    """Run a research + writing crew with budget-aware agents."""
    research_task = Task(
        description=(
            f"Research the topic: '{topic}'. Check your budget first. "
            f"If you need premium data, spend up to $0.25 total. "
            f"Provide a structured summary with key findings."
        ),
        expected_output="A structured research summary with 3-5 key findings",
        agent=researcher,
    )

    writing_task = Task(
        description=(
            "Using the research provided, write a concise article (300 words). "
            "Check budget before any additional spending. "
            "Include a compelling headline and 3 key takeaways."
        ),
        expected_output="A 300-word article with headline and key takeaways",
        agent=writer,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    output = run_crew("AI agent payments and the x402 protocol")
    print(f"\n{'='*60}")
    print(f"📄 Final Output:\n{output}")
