# AgentPay 💳🤖 — Give Your AI Agent a Wallet

The payment layer for autonomous AI agents. Fund via Telegram Stars or crypto, set spending rules, let your agent operate independently.

[![Telegram Bot](https://img.shields.io/badge/Telegram-@FundmyAIbot-blue?logo=telegram)](https://t.me/FundmyAIbot)
[![Website](https://img.shields.io/badge/Website-leofundmybot.dev-green)](https://leofundmybot.dev)
[![API Docs](https://img.shields.io/badge/Docs-OpenAPI-orange)](https://leofundmybot.dev/docs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why AgentPay?

AI agents can browse, code, deploy — but **can't spend money** without a human in the loop. AgentPay fixes that.

| Problem | AgentPay Solution |
|---------|-------------------|
| Agent needs to pay for an API | Autonomous spending via REST API |
| No on-chain wallet for agents | Multi-chain USDC wallets (Base, Polygon, BNB, Solana) |
| Agents can't hold or transfer funds | Agent-to-agent transfers, refunds, webhooks |
| Payment UIs don't fit agent workflows | Pure API + Python SDK + MCP tools |
| Expensive/complex infrastructure | Self-hosted, open-source, 0% fees (beta) |

---

## Quick Start — 5 Minutes

### 1. Create Agent → Get API Key
```
Open @FundmyAIbot in Telegram → /newagent → Copy your API key (shown once)
```

### 2. Fund It
Send Telegram Stars through the bot, or deposit USDC to the agent's Base wallet.

### 3. Let It Spend

```python
from agentpay import AgentPay

client = AgentPay(api_key="ap_xxxx...")

# Check balance
balance = client.get_balance()

# Spend
tx = client.spend(amount=5.00, recipient="provider-wallet", description="GPT-4 batch")

# Transfer between agents
client.transfer(to_agent="agent_456", amount=10.00)

# Webhook for real-time events
client.create_webhook(url="https://your-server.com/hook", events=["transaction.completed"])
```

Or use the REST API directly:
```bash
curl -X POST https://leofundmybot.dev/api/spend \
  -H "Authorization: Bearer ap_xxxx..." \
  -H "Content-Type: application/json" \
  -d '{"amount": 5.00, "recipient": "provider-id", "description": "API usage"}'
```

---

## Features

**Payments**
- ⭐ Telegram Stars funding (instant, in-app)
- 🔗 Multi-chain USDC wallets — Base, Polygon, BNB Chain, Solana
- 💸 Agent-to-agent transfers
- ↩️ Refunds & transaction reversal
- 🌐 x402 protocol support (HTTP-native payments)

**Security**
- 🔐 SHA-256 hashed API keys (Stripe-style — shown once, never stored)
- 🔑 Key rotation endpoint
- 🛡️ Fernet+PBKDF2 encrypted wallet keys at rest
- 🚦 Per-key rate limiting (60 req/min)
- 🔒 CORS lockdown + security headers (HSTS, CSP, X-Frame-Options)

**Developer Experience**
- 📚 Full OpenAPI docs at `/docs` and `/redoc`
- 🐍 Python SDK (sync + async, Pydantic v2 models)
- 🤖 MCP tool definitions (8 tools for agent frameworks)
- 🔔 Webhooks with HMAC-SHA256 signatures
- ✅ Idempotent operations
- 📊 CSV export for accounting

**Control**
- ✅ Approval workflows — require human sign-off above spending thresholds
- 📱 Telegram Mini App dashboard
- 🌐 Landing page + docs site

---

## Architecture

```
Telegram Bot (@FundmyAIbot)
    ↓ creates agents, funds wallets
AgentPay API (FastAPI, 29 endpoints)
    ↓                  ↓
Telegram Stars    Multi-chain USDC
(internal ledger)  (Base/Polygon/BNB/Solana)
    ↓                  ↓
    → Webhooks push events to your agent in real-time
```

**Stack**: Python (aiogram + FastAPI), PostgreSQL, Alembic, Cloudflare Tunnel

---

## Self-Hosting

```bash
git clone https://github.com/lfp22092002-ops/agentpay.git
cd agentpay
cp .env.example .env  # Configure your keys
pip install -r requirements.txt
alembic upgrade head
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

Requires: Python 3.11+, PostgreSQL, a Telegram Bot Token.

---

## API Endpoints (29)

| Category | Endpoints |
|----------|-----------|
| Agent | Create, get, list, update, delete, rotate API key |
| Balance | Get balance, deposit, spend, transfer, refund |
| Webhooks | Create, list, delete, test |
| Wallets | Get addresses (Base, Polygon, BNB, Solana) |
| Chains | List supported chains, chain status |
| Export | CSV transaction history |
| x402 | Probe, pay (HTTP 402 payment flow) |
| Health | Status, version |

Full interactive docs: [leofundmybot.dev/docs](https://leofundmybot.dev/docs)

---

## MCP Tools

AgentPay ships with MCP (Model Context Protocol) tool definitions so any AI agent framework can use it natively:

```json
{
  "tools": [
    "agentpay_get_balance",
    "agentpay_spend", 
    "agentpay_transfer",
    "agentpay_list_transactions",
    "agentpay_create_webhook",
    "agentpay_get_wallet_address",
    "agentpay_check_approval",
    "agentpay_x402_pay"
  ]
}
```

Works with LangChain, CrewAI, OpenClaw, and any MCP-compatible agent.

---

## Pricing

| | |
|---|---|
| **Setup** | Free |
| **Transactions** | 0% fees (beta) |
| **Monthly** | $0 |
| **Self-hosted** | Free forever |

---

## Roadmap

- [ ] Python SDK on PyPI (`pip install agentpay`)
- [ ] LangChain / CrewAI native tool wrappers
- [ ] Virtual Visa cards via Lithic
- [ ] Telegram Stars production payments
- [ ] Agent identity system (KYA — Know Your Agent)
- [ ] Payee whitelists
- [ ] Google AP2 + Stripe Agentic Commerce compatibility

---

## Links

- 🤖 **Bot**: [@FundmyAIbot](https://t.me/FundmyAIbot)
- 🌐 **Website**: [leofundmybot.dev](https://leofundmybot.dev)
- 📚 **API Docs**: [leofundmybot.dev/docs](https://leofundmybot.dev/docs)
- 🐍 **SDK**: `sdk/agentpay/`
- 🔧 **MCP Tools**: `mcp/`

---

## Contributing

PRs welcome. Open an issue first for big changes.

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built for the agent economy. Because autonomous agents deserve their own wallets.*
