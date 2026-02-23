# AgentPay Python SDK

Official Python client for the [AgentPay](https://leofundmybot.dev) API — fund your AI agents and let them spend.

## Installation

```bash
pip install agentpay
```

## Quick Start

```python
from agentpay import AgentPayClient

client = AgentPayClient("ap_your_api_key")

# Check balance
balance = client.get_balance()
print(f"Balance: ${balance['balance']}")

# Spend funds
tx = client.spend(0.50, "GPT-4 API call")
print(f"Transaction: {tx['id']}")

# Refund a transaction
refund = client.refund(tx["id"])

# Get transaction history
transactions = client.get_transactions(limit=10)
```

## Webhooks

```python
# Register a webhook
webhook = client.register_webhook(
    url="https://your-server.com/webhook",
    events=["spend", "refund", "transfer"],
)
print(f"Secret: {webhook['secret']}")

# Delete webhook
client.delete_webhook()
```

## Wallet & Card

```python
# On-chain USDC wallet (Base network)
wallet = client.get_wallet()
print(f"Address: {wallet['address']}")

# Send USDC
client.send_usdc("0x...", 5.00)

# Virtual Visa card
card = client.get_card()
```

## Export

```python
# Download transactions as CSV
csv_data = client.export_csv()
with open("transactions.csv", "w") as f:
    f.write(csv_data)
```

## Error Handling

```python
from agentpay.exceptions import AgentPayError

try:
    client.spend(1000.00, "Big purchase")
except AgentPayError as e:
    print(f"Error {e.status_code}: {e.detail}")
```

## Links

- 📊 **Dashboard**: [leofundmybot.dev/app](https://leofundmybot.dev/app/)
- 📖 **API Docs**: [leofundmybot.dev/docs-site](https://leofundmybot.dev/docs-site/)
- 🤖 **Telegram Bot**: [@FundmyAIbot](https://t.me/FundmyAIbot)

## License

MIT
