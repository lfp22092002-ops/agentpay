"""
AgentPay — Autonomous Agent Budget Manager

A practical example showing how an AI agent manages its own spending budget:
  1. Check available balance before expensive operations
  2. Request approval for large spends
  3. Track cumulative costs across a session
  4. Gracefully degrade when funds run low

This pattern works with any LLM framework (LangChain, CrewAI, AutoGen, etc.)

Requirements:
    pip install agentpay openai
"""

import os
from dataclasses import dataclass, field

from agentpay import AgentPayClient
from agentpay.exceptions import AgentPayError, InsufficientBalanceError


API_KEY = os.getenv("AGENTPAY_API_KEY", "ap_your_key_here")
BASE_URL = os.getenv("AGENTPAY_BASE_URL", "https://leofundmybot.dev")

# Estimated costs per operation (adjust for your models)
COST_TABLE = {
    "gpt-4o": 0.03,         # per 1K tokens (blended)
    "gpt-4o-mini": 0.002,
    "claude-sonnet": 0.015,
    "web-search": 0.01,
    "image-gen": 0.04,
}


@dataclass
class BudgetTracker:
    """Track session spending against a budget."""
    session_budget: float = 5.00
    spent: float = 0.0
    operations: list = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return self.session_budget - self.spent

    def can_afford(self, cost: float) -> bool:
        return self.remaining >= cost

    def record(self, operation: str, cost: float) -> None:
        self.spent += cost
        self.operations.append({"op": operation, "cost": cost})

    def summary(self) -> str:
        lines = [f"Session budget: ${self.session_budget:.2f}"]
        lines.append(f"Total spent:    ${self.spent:.2f}")
        lines.append(f"Remaining:      ${self.remaining:.2f}")
        lines.append(f"Operations:     {len(self.operations)}")
        return "\n".join(lines)


def run_agent_task(task: str) -> None:
    """Simulate an agent that manages its budget while completing a task."""
    budget = BudgetTracker(session_budget=5.00)

    with AgentPayClient(API_KEY, base_url=BASE_URL) as pay:
        # Pre-flight: verify we have funds
        balance = pay.get_balance()
        print(f"Wallet balance: ${balance.balance_usd:.2f}")

        if balance.balance_usd < 1.0:
            print("⚠ Low balance — requesting minimal-cost mode")

        # Step 1: Plan (cheap model)
        model = "gpt-4o-mini"
        cost = COST_TABLE[model]
        if not budget.can_afford(cost):
            print("Budget exhausted at planning stage")
            return

        tx = pay.spend(
            amount=cost,
            description=f"Planning: {task[:50]}",
            metadata={"model": model, "stage": "plan"},
        )
        budget.record(f"plan ({model})", cost)
        print(f"✓ Plan generated (${cost})")

        # Step 2: Research (web search — medium cost)
        cost = COST_TABLE["web-search"]
        if budget.can_afford(cost):
            tx = pay.spend(
                amount=cost,
                description=f"Web search for: {task[:50]}",
                metadata={"stage": "research"},
            )
            budget.record("web-search", cost)
            print(f"✓ Research complete (${cost})")
        else:
            print("⏭ Skipping research (budget)")

        # Step 3: Execute (expensive model)
        model = "gpt-4o"
        cost = COST_TABLE[model]
        if budget.can_afford(cost):
            try:
                tx = pay.spend(
                    amount=cost,
                    description=f"Execute: {task[:50]}",
                    metadata={"model": model, "stage": "execute"},
                )
                budget.record(f"execute ({model})", cost)
                print(f"✓ Task executed (${cost})")
            except InsufficientBalanceError:
                # Fallback to cheaper model
                print("⚠ Wallet insufficient — falling back to mini")
                fallback_cost = COST_TABLE["gpt-4o-mini"]
                tx = pay.spend(
                    amount=fallback_cost,
                    description=f"Execute (fallback): {task[:50]}",
                    metadata={"model": "gpt-4o-mini", "stage": "execute"},
                )
                budget.record("execute (fallback)", fallback_cost)
        else:
            print("⏭ Using cheap model (budget)")
            cost = COST_TABLE["gpt-4o-mini"]
            tx = pay.spend(amount=cost, description=f"Execute (budget): {task[:50]}")
            budget.record("execute (budget)", cost)

        # Summary
        print(f"\n{'='*40}")
        print(budget.summary())


if __name__ == "__main__":
    run_agent_task("Research competitors and write a market analysis report")
