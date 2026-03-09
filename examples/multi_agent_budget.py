"""
Multi-Agent Budget Manager — AgentPay

This example shows how to manage a fleet of agents with shared budgets:
1. Create multiple agents for different tasks
2. Set appropriate spending limits per agent
3. Transfer funds between agents based on workload
4. Monitor spending across all agents

Requirements:
    pip install agentpay
"""

from agentpay import AgentPayClient


def main():
    # Master agent that manages sub-agents
    master = AgentPayClient(
        api_key="ap_master_key_here",
        base_url="https://leofundmybot.dev",
    )

    # Check master balance
    balance = master.get_balance()
    print(f"🏦 Master agent: ${balance.balance_usd}")
    print(f"   Daily limit: ${balance.daily_limit_usd}")
    print(f"   Spent today: ${balance.daily_spent_usd}")
    print(f"   Remaining:   ${balance.daily_remaining_usd}")
    print()

    # Sub-agents for different tasks
    agents = {
        "research": AgentPayClient("ap_research_key", base_url="https://leofundmybot.dev"),
        "coding": AgentPayClient("ap_coding_key", base_url="https://leofundmybot.dev"),
        "data": AgentPayClient("ap_data_key", base_url="https://leofundmybot.dev"),
    }

    # Check all agent balances
    print("📊 Agent Fleet Status:")
    for name, client in agents.items():
        b = client.get_balance()
        usage_pct = (b.daily_spent_usd / b.daily_limit_usd * 100) if b.daily_limit_usd else 0
        status = "🟢" if usage_pct < 50 else "🟡" if usage_pct < 80 else "🔴"
        print(f"  {status} {name}: ${b.balance_usd:.2f} | "
              f"spent ${b.daily_spent_usd:.2f}/${b.daily_limit_usd:.2f} "
              f"({usage_pct:.0f}%)")

    print()

    # Example: Research agent needs more funds
    research_balance = agents["research"].get_balance()
    if research_balance.balance_usd < 5.00:
        print("⚡ Research agent low on funds — transferring $10 from master")
        tx = master.transfer(
            to_agent=str(research_balance.agent_id),
            amount=10.00,
        )
        print(f"   ✅ Transfer complete: tx_{tx.transaction_id}")

    # Example: Spending with description tracking
    print("\n💸 Executing tasks:")
    try:
        tx = agents["research"].spend(0.05, "Brave Search API — market analysis query")
        print(f"  research: spent ${tx.amount} (${tx.remaining_balance} left)")
    except Exception as e:
        print(f"  research: failed — {e}")

    try:
        tx = agents["coding"].spend(0.25, "Claude API — code review 500 lines")
        print(f"  coding:   spent ${tx.amount} (${tx.remaining_balance} left)")
    except Exception as e:
        print(f"  coding: failed — {e}")


if __name__ == "__main__":
    main()
