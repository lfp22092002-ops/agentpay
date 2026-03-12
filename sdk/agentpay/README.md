# AgentPay Python SDK

Python client for the [AgentPay](https://leofundmybot.dev) API — payment infrastructure for AI agents.

## Install

```bash
pip install agentpay
```

## Quick Start

```python
from agentpay import AgentPayClient

client = AgentPayClient("ap_your_api_key")

# Check balance
balance = client.get_balance()
print(f"Balance: ${balance.balance_usd}")

# Spend
tx = client.spend(0.50, description="GPT-4 API call")
print(f"Spent ${tx.amount}, remaining ${tx.remaining_balance}")

# Transfer between agents
client.transfer("agent_xyz", 5.00, description="Data enrichment")

# Refund a transaction
client.refund(tx.transaction_id, reason="Duplicate charge")
```

## Async Usage

```python
from agentpay import AgentPayAsyncClient

async def main():
    client = AgentPayAsyncClient("ap_your_api_key")
    balance = await client.get_balance()
    print(f"Balance: ${balance.balance_usd}")
```

## Wallets

```python
# Get multi-chain wallet info
wallets = client.get_wallets()
for chain in wallets.chains:
    print(f"{chain['chain']}: {chain['address']}")

# Send USDC on Base
result = client.send_usdc("0x...", 10.0, chain="base")
```

## Webhooks

```python
# Register a webhook
client.set_webhook("https://example.com/webhook", events=["spend", "deposit"])

# Verify incoming webhook signatures
from agentpay import verify_webhook_signature
is_valid = verify_webhook_signature(payload, signature, secret)
```

## x402 Protocol

```python
# Pay for an x402-gated resource
result = client.x402_pay("https://api.example.com/premium", max_price=2.00)
print(result.response_body)  # The actual API response
```

## Error Handling

```python
from agentpay import AgentPayError, InsufficientBalanceError

try:
    client.spend(1000.00, description="Big purchase")
except InsufficientBalanceError as e:
    print(f"Not enough funds: {e}")
except AgentPayError as e:
    print(f"API error: {e}")
```

## Configuration

```python
client = AgentPayClient(
    api_key="ap_...",
    base_url="https://leofundmybot.dev",  # default
    timeout=30.0,  # seconds
)
```

## Requirements

- Python 3.9+
- httpx >= 0.25.0

## Links

- [Documentation](https://leofundmybot.dev/docs-site/)
- [API Reference](https://leofundmybot.dev/docs)
- [Telegram Bot](https://t.me/FundmyAIbot)
- [GitHub](https://github.com/lfp22092002-ops/agentpay)
