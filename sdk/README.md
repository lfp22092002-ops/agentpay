# AgentPay Python SDK

[![PyPI version](https://img.shields.io/pypi/v/agentpay)](https://pypi.org/project/agentpay/)
[![Python](https://img.shields.io/pypi/pyversions/agentpay)](https://pypi.org/project/agentpay/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The official Python SDK for [AgentPay](https://leofundmybot.dev) — payment infrastructure for AI agents. Fund your agents, let them spend, and track every transaction.

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
print(f"Balance: ${balance.balance_usd}")
print(f"Daily remaining: ${balance.daily_remaining_usd}")

# Spend funds
tx = client.spend(0.50, "GPT-4 API call")
print(f"Transaction: {tx.transaction_id}")
print(f"Remaining: ${tx.remaining_balance}")
```

## Async Usage

```python
import asyncio
from agentpay import AgentPayAsyncClient

async def main():
    async with AgentPayAsyncClient("ap_your_api_key") as client:
        balance = await client.get_balance()
        print(f"Balance: ${balance.balance_usd}")

        tx = await client.spend(0.25, "Embedding generation")
        print(f"Spent: ${tx.amount}")

asyncio.run(main())
```

## API Reference

### Initialization

```python
# Sync client
client = AgentPayClient(
    api_key="ap_your_api_key",
    base_url="https://leofundmybot.dev",  # default
    timeout=30.0,                          # default
)

# Async client
client = AgentPayAsyncClient(
    api_key="ap_your_api_key",
    base_url="https://leofundmybot.dev",
    timeout=30.0,
)
```

### Balance

```python
balance = client.get_balance()
# Balance(
#     agent_id="agent_abc123",
#     agent_name="my-agent",
#     balance_usd=42.50,
#     daily_limit_usd=50.0,
#     daily_spent_usd=7.50,
#     daily_remaining_usd=42.50,
#     tx_limit_usd=25.0,
#     is_active=True,
# )
```

### Spend

```python
tx = client.spend(
    amount=1.50,
    description="Claude API call",
    idempotency_key="req_unique_123",  # optional, prevents duplicates
)
# SpendResponse(success=True, transaction_id="tx_...", amount=1.50, ...)
```

### Transactions

```python
transactions = client.get_transactions(limit=10)
for tx in transactions:
    print(f"{tx.type}: ${tx.amount} — {tx.description}")
```

### Refund

```python
result = client.refund("tx_abc123")
print(f"Refunded: ${result.amount_refunded}")
print(f"New balance: ${result.new_balance}")
```

### Transfer

```python
result = client.transfer(
    to_agent_id="agent_xyz",
    amount=5.00,
    description="Payment for data processing",
)
```

### On-Chain Wallet

```python
# Get wallet for a specific chain
wallet = client.get_wallet(chain="base")  # base, polygon, bnb, solana
print(f"Address: {wallet.address}")
print(f"USDC Balance: {wallet.balance_usdc}")

# List all supported chains
chains = client.list_chains()
for chain in chains:
    print(f"{chain.name} ({chain.id}) — {chain.native_token}")
```

### Webhooks

```python
webhook = client.register_webhook(
    url="https://your-server.com/webhook",
    events=["spend", "refund", "transfer"],
)
print(f"Signing secret: {webhook.secret}")
```

### x402 Protocol

```python
# Pay for an x402-gated resource
result = client.x402_pay(
    url="https://api.example.com/premium-data",
    max_amount=2.00,
)
print(result.data)
print(f"Paid: ${result.paid_usd}")
```

## Error Handling

```python
from agentpay import (
    AgentPayError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
)

try:
    tx = client.spend(1000.00, "Big purchase")
except AuthenticationError:
    print("Invalid API key")
except InsufficientBalanceError:
    print("Not enough funds — top up via Telegram Stars")
except RateLimitError:
    print("Too many requests — slow down")
except AgentPayError as e:
    print(f"API error {e.status_code}: {e.detail}")
```

## Context Manager

Both clients support context managers for automatic cleanup:

```python
# Sync
with AgentPayClient("ap_key") as client:
    client.spend(0.10, "test")

# Async
async with AgentPayAsyncClient("ap_key") as client:
    await client.spend(0.10, "test")
```

## Type Safety

The SDK is fully typed with [PEP 561](https://peps.python.org/pep-0561/) support. All response models are Pydantic v2 models with full IDE autocomplete.

```python
from agentpay import Balance, Transaction, Wallet
```

## Links

| Resource | URL |
|----------|-----|
| 📊 Dashboard | [leofundmybot.dev/app](https://leofundmybot.dev/app/) |
| 📖 API Docs | [leofundmybot.dev/docs-site](https://leofundmybot.dev/docs-site/) |
| 🤖 Telegram Bot | [@FundmyAIbot](https://t.me/FundmyAIbot) |
| 💻 GitHub | [github.com/lfp22092002-ops/agentpay](https://github.com/lfp22092002-ops/agentpay) |
| 📦 PyPI | [pypi.org/project/agentpay](https://pypi.org/project/agentpay/) |

## License

MIT — see [LICENSE](LICENSE) for details.
