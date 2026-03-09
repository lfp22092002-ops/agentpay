"""
AgentPay — Basic Usage Example

Shows how to check balance, make a spend, view transactions, and handle errors.

Requirements:
    pip install agentpay
"""

from agentpay import AgentPayClient
from agentpay.exceptions import AgentPayError, InsufficientBalanceError


API_KEY = "ap_your_key_here"  # Get from @FundmyAIbot on Telegram
BASE_URL = "https://leofundmybot.dev"  # Or your self-hosted instance


def main() -> None:
    with AgentPayClient(API_KEY, base_url=BASE_URL) as client:

        # 1. Check balance
        balance = client.get_balance()
        print(f"Balance: ${balance.balance_usd:.2f}")
        print(f"Daily spent: ${balance.daily_spent_usd:.2f} / ${balance.daily_limit_usd:.2f}")

        # 2. Spend money (e.g., pay for an API call)
        try:
            tx = client.spend(
                amount=2.50,
                description="GPT-4 API call — summarize document",
                idempotency_key="doc-summary-001",  # Prevents double-charging
            )
            print(f"Spent ${tx.amount} — TX: {tx.transaction_id}")
            print(f"Remaining: ${tx.remaining_balance:.2f}")
        except InsufficientBalanceError:
            print("Not enough funds — top up via @FundmyAIbot")
        except AgentPayError as e:
            if "approval" in str(e).lower():
                print("Amount exceeds auto-approve limit — waiting for owner approval")
            else:
                print(f"Spend failed: {e}")

        # 3. View recent transactions
        txs = client.get_transactions(limit=5)
        for tx in txs:
            icon = "↙" if tx.type == "deposit" else "↗"
            print(f"  {icon} ${tx.amount} — {tx.description}")

        # 4. Refund a transaction
        try:
            refund = client.refund("some-tx-id")
            print(f"Refunded: ${refund.amount_refunded}")
        except AgentPayError as e:
            print(f"Refund failed: {e}")


if __name__ == "__main__":
    main()
