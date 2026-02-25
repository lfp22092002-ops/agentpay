# AgentPay 💳🤖 — Fund Your AI Agents

Give your AI agents their own wallet. Let them spend autonomously via API. Built on Telegram, powered by on-chain USDC on Base.

[![Telegram Bot](https://img.shields.io/badge/Telegram-@FundmyAIbot-blue?logo=telegram)](https://t.me/FundmyAIbot)
[![Website](https://img.shields.io/badge/Website-leofundmybot.dev-green)](https://leofundmybot.dev)
[![Docs](https://img.shields.io/badge/Docs-API%20Reference-orange)](https://leofundmybot.dev/docs-site/)

---

## The Problem

AI agents can browse the web, write code, manage servers — but the moment they need to **spend money**, a human has to step in. Every paid API call, every purchase, every transaction requires manual approval. That's a bottleneck that kills agent autonomy.

## The Solution

AgentPay is a Telegram-native payment platform that gives any AI agent its own wallet. Fund it, set spending rules, and let your agent operate independently.

- **Fund with Telegram Stars** → Agent gets a balance → Agent spends via API
- **On-chain USDC wallets** on Base network for crypto-native workflows
- **2% fee** per transaction — free to set up and try

---

## Quick Start

### 1. Create an Agent

Open [@FundmyAIbot](https://t.me/FundmyAIbot) in Telegram and create a new agent. You'll get an API key.

### 2. Fund the Wallet

Send Telegram Stars to your agent's wallet through the bot, or deposit USDC to the on-chain Base wallet.

### 3. Use the SDK

```bash
pip install agentpay
```

```python
from agentpay import AgentPay

# Initialize with your API key
client = AgentPay(api_key="your_api_key_here")

# Check balance
balance = client.get_balance()
print(f"Agent balance: {balance.amount} {balance.currency}")

# Spend funds
tx = client.spend(
    amount=5.00,
    recipient="api-provider-wallet",
    description="GPT-4 API call batch",
    metadata={"batch_id": "abc123"}
)
print(f"Transaction: {tx.id} — Status: {tx.status}")

# Set up a webhook for balance changes
client.create_webhook(
    url="https://your-server.com/webhook",
    events=["balance.updated", "transaction.completed"]
)

# Transfer between agents
client.transfer(
    to_agent="agent_456",
    amount=10.00,
    description="Splitting compute costs"
)
```

### 4. Use the REST API Directly

```bash
# Check balance
curl -X GET https://leofundmybot.dev/api/balance \
  -H "Authorization: Bearer YOUR_API_KEY"

# Create a spend transaction
curl -X POST https://leofundmybot.dev/api/spend \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 5.00,
    "recipient": "provider-id",
    "description": "API usage"
  }'
```

---

## Features

| Feature | Description |
|---------|-------------|
| 🔑 **API-First** | Full REST API — agents spend programmatically |
| ⭐ **Telegram Stars** | Fund wallets with Telegram's native currency |
| 🔗 **USDC on Base** | On-chain wallets for crypto workflows |
| 🔔 **Webhooks** | Real-time notifications for balance changes and transactions |
| ✅ **Approval Workflows** | Set spend limits, require human approval above thresholds |
| 💸 **Transfers** | Move funds between agents |
| ↩️ **Refunds** | Reverse transactions when needed |
| 📊 **CSV Export** | Download transaction history for accounting |
| 🌐 **x402 Protocol** | HTTP-native payments for the open web |
| 📱 **Mini App Dashboard** | Manage everything inside Telegram |

---

## Architecture

```
┌─────────────────────┐
│   Telegram Bot       │  ← User creates agent, funds wallet
│   (@FundmyAIbot)     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   AgentPay API       │  ← REST API for agent operations
│   (leofundmybot.dev) │
└──────┬────────┬─────┘
       │        │
┌──────▼──┐ ┌──▼──────────┐
│ Telegram │ │ Base Network │
│ Stars    │ │ (USDC)       │
│ Ledger   │ │ On-chain     │
└──────────┘ └──────────────┘
```

1. **Telegram Bot** handles user interaction — create agents, fund wallets, view dashboards
2. **AgentPay API** processes all programmatic operations — balance checks, spends, transfers
3. **Dual settlement** — Telegram Stars for instant in-app payments, USDC on Base for on-chain settlement
4. **Webhooks** push transaction events to your agent's server in real-time

---

## Links

- 🤖 **Bot**: [@FundmyAIbot](https://t.me/FundmyAIbot)
- 🌐 **Website**: [leofundmybot.dev](https://leofundmybot.dev)
- 📚 **API Docs**: [leofundmybot.dev/docs-site](https://leofundmybot.dev/docs-site/)
- 🐍 **Python SDK**: `pip install agentpay` *(coming soon)*

---

## Pricing

- **Setup**: Free
- **Transactions**: 2% fee per spend
- **No monthly fees**, no minimums

---

## Contributing

Contributions welcome! Open an issue or submit a PR.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built with ❤️ for the agent economy. Because agents deserve wallets too.
