# I Built a Payment Layer So AI Agents Can Spend Money Without Asking Permission

*Your AI agent can write code, browse the web, and deploy apps. But the moment it needs to pay $0.05 for an API call, you have to step in with your credit card. That's broken.*

---

## The Problem

Every developer building with AI agents hits the same wall: your agent is autonomous until it needs money. Then it's not.

- Agent needs to call a paid API → human enters credit card
- Agent wants to pay for compute → manual billing dashboard
- Agent-to-agent transfer → doesn't exist in traditional finance

The x402 protocol (Coinbase) solved the *how* — HTTP-native micropayments in stablecoins. But agents still need a **wallet**, **budget rules**, and an **identity** to actually use it. That's what I built.

## What AgentPay Does

AgentPay is an open-source payment layer for AI agents. Three concepts:

1. **Agent Wallet** — USDC wallets on Base, Polygon, BNB Chain, and Solana. Fund via Telegram Stars or direct USDC deposit.
2. **Budget Governance** — Spending limits, approval workflows, idempotency. Your agent can spend within rules you define.
3. **Agent Identity (KYA)** — Know Your Agent. Trust scores based on account age, transaction history, and verification status.

## Quick Start (5 minutes)

### 1. Create an agent

Open [@FundmyAIbot](https://t.me/FundmyAIbot) in Telegram → `/newagent` → get your API key.

### 2. Install the SDK

```bash
pip install agentpay
```

or for TypeScript:

```bash
npm install agentpay
```

### 3. Let your agent spend

```python
from agentpay import AgentPayClient

client = AgentPayClient("ap_your_key")

# Check balance
balance = client.get_balance()
print(f"Balance: ${balance.balance_usd}")

# Spend autonomously
tx = client.spend(amount=0.50, description="GPT-4 API call")
print(f"Spent ${tx.amount}, remaining: ${tx.remaining_balance}")
```

### 4. Use as a LangChain tool

```python
from langchain.tools import Tool
from agentpay import AgentPayClient

client = AgentPayClient("ap_your_key")

pay_tool = Tool(
    name="pay",
    description="Pay for a service or API. Input: amount (float) and description (string)",
    func=lambda x: client.spend(
        amount=float(x.split(",")[0]),
        description=x.split(",")[1].strip()
    )
)
```

Your LangChain/CrewAI/AutoGen agent now has a wallet.

### 5. Pay for x402 resources

```python
# Agent automatically pays for x402-gated API
result = client.x402_pay(
    url="https://some-api.com/data",
    max_price_usd=1.00
)
```

## Architecture

```
Agent (Python/TS/curl)
    │
    ▼
AgentPay API (REST)
    │
    ├── Balance Engine (Stars + USDC)
    ├── Multi-chain Wallets (Base, Polygon, BNB, Solana)
    ├── x402 Provider
    ├── KYA Identity + Trust Scoring
    └── Webhooks (HMAC-SHA256)
```

The API runs on FastAPI. Everything is open source under MIT.

## MCP Integration

AgentPay ships as an MCP (Model Context Protocol) server. Any MCP-compatible agent can discover and use it:

```json
{
  "mcpServers": {
    "agentpay": {
      "url": "https://leofundmybot.dev/mcp",
      "headers": {
        "x-api-key": "YOUR_API_KEY"
      }
    }
  }
}
```

7 tools exposed: `get_balance`, `spend`, `transfer`, `get_transactions`, `create_agent`, `get_identity`, `register_webhook`.

## Why Open Source?

The agent payment space is heating up fast. Coinbase launched x402, Stripe is building Tempo, Visa and Mastercard have agent payment protocols, and every exchange (Binance, OKX, Kraken) is racing to be agent infrastructure.

But most solutions are enterprise-focused or closed. AgentPay is:

- **Self-hostable** — run it on your own server
- **Free** — 0% fees during beta
- **Telegram-native** — fund and manage via bot, no dashboards
- **Developer-first** — Python SDK, TypeScript SDK, REST API, MCP tools

## What's Next

- [ ] npm publish (TypeScript SDK ready, pending registry)
- [ ] PyPI publish (Python SDK packaged, pending token)
- [ ] ERC-8183 escrow integration (trustless agent-to-agent conditional payments)
- [ ] More chain support (Stellar, Arbitrum)
- [ ] Credit system (agents with good trust scores can "borrow")

## Try It

- **Telegram Bot**: [@FundmyAIbot](https://t.me/FundmyAIbot)
- **GitHub**: [lfp22092002-ops/agentpay](https://github.com/lfp22092002-ops/agentpay)
- **Docs**: [leofundmybot.dev/docs](https://leofundmybot.dev/docs)
- **API**: [leofundmybot.dev/docs](https://leofundmybot.dev/docs) (OpenAPI)
- **MCP**: `https://leofundmybot.dev/mcp`

Star the repo if this is useful. PRs welcome.

---

*Built by a solo dev who got tired of manually approving every API payment for his AI agents.*

---

**Tags**: `ai`, `agents`, `payments`, `crypto`, `open-source`, `mcp`, `x402`
