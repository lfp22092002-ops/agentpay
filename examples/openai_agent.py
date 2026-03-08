"""
AgentPay — OpenAI Agent with Budget Control

An AI agent that uses AgentPay to pay for its own API calls,
with spending limits and approval workflows.

Requirements:
    pip install agentpay openai
"""
import asyncio
from agentpay import AgentPayClient, AgentPayError


AGENTPAY_KEY = "ap_your_key_here"
OPENAI_KEY = "sk-your-openai-key"
AGENTPAY_URL = "https://leofundmybot.dev"

# Cost estimates (per 1K tokens, approximate)
COST_PER_1K_INPUT = 0.01   # GPT-4o-mini
COST_PER_1K_OUTPUT = 0.03


async def ai_agent_task(task: str):
    """An agent that checks budget before making API calls."""
    async with AgentPayClient(api_key=AGENTPAY_KEY, base_url=AGENTPAY_URL) as pay:
        # Check if we have enough budget
        balance = await pay.get_balance()
        available = float(balance["balance_usd"])
        print(f"💰 Budget: ${available:.2f}")

        if available < 0.10:
            print("⚠️ Low budget — skipping task. Top up via @FundmyAIbot")
            return None

        # Estimate cost and pre-authorize the spend
        estimated_cost = 0.05  # Conservative estimate for a typical call
        try:
            tx = await pay.spend(
                amount=estimated_cost,
                description=f"OpenAI GPT-4o-mini: {task[:50]}",
                idempotency_key=f"task-{hash(task) % 10000}",
            )
            print(f"✅ Pre-authorized ${estimated_cost}")
        except AgentPayError as e:
            print(f"❌ Can't spend: {e}")
            return None

        # Make the actual API call
        import openai
        client = openai.OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": task}],
            max_tokens=500,
        )

        result = response.choices[0].message.content
        print(f"🤖 Result: {result[:200]}...")

        # Check remaining budget
        new_balance = await pay.get_balance()
        print(f"💰 Remaining: ${new_balance['balance_usd']}")

        return result


async def main():
    result = await ai_agent_task("Summarize the key benefits of autonomous AI agents in 3 bullet points")
    if result:
        print(f"\n📄 Full response:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
