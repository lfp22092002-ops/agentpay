"""
AgentPay — Basic Usage Example

Shows how to check balance, make a spend, view transactions, and handle errors.

Requirements:
    pip install agentpay
    # or: pip install httpx  (and use raw HTTP)
"""
import asyncio
from agentpay import AgentPayClient, AgentPayError


API_KEY = "ap_your_key_here"  # Get from @FundmyAIbot on Telegram
BASE_URL = "https://leofundmybot.dev"  # Or your self-hosted instance


async def main():
    async with AgentPayClient(api_key=API_KEY, base_url=BASE_URL) as client:

        # 1. Check balance
        balance = await client.get_balance()
        print(f"Balance: ${balance['balance_usd']}")
        print(f"Daily spent: ${balance['daily_spent_usd']} / ${balance['daily_limit_usd']}")

        # 2. Spend money (e.g., pay for an API call)
        try:
            tx = await client.spend(
                amount=2.50,
                description="GPT-4 API call — summarize document",
                idempotency_key="doc-summary-001",  # Prevents double-charging
            )
            print(f"Spent ${tx['amount_usd']} — TX: {tx['id']}")
        except AgentPayError as e:
            if "insufficient" in str(e).lower():
                print("Not enough funds — top up via @FundmyAIbot")
            elif "approval" in str(e).lower():
                print("Amount exceeds auto-approve limit — waiting for owner approval")
            else:
                print(f"Spend failed: {e}")

        # 3. View recent transactions
        txs = await client.get_transactions(limit=5)
        for tx in txs:
            icon = "↙" if tx["tx_type"] == "deposit" else "↗"
            print(f"  {icon} ${tx['amount_usd']} — {tx['description']}")

        # 4. Refund a transaction
        try:
            refund = await client.refund(transaction_id="some-tx-id")
            print(f"Refunded: ${refund['amount_usd']}")
        except AgentPayError as e:
            print(f"Refund failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
