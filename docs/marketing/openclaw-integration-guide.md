---
title: "Give Your OpenClaw Agent a Wallet — Autonomous Spending with AgentPay"
published: false
description: "Let your AI agent pay for APIs, data, and services autonomously. MCP tools, budget governance, and x402 micropayments — in 5 minutes."
tags: ai, opensource, crypto, tutorial
cover_image:
---

## Your Agent Can Code. It Can Browse. It Can't Pay.

If your OpenClaw agent needs to call a paid API — GPT-4, image generation, data enrichment — it hits a wall. You have to handle the payment manually.

[AgentPay](https://github.com/lfp22092002-ops/agentpay) gives your agent its own wallet, spending limits, and approval workflows. Open source, multi-chain USDC, works as an MCP tool.

## Setup (2 minutes)

### 1. Add the MCP Server

In your OpenClaw config, add AgentPay as an MCP server:

```json
{
  "mcpServers": {
    "agentpay": {
      "url": "https://leofundmybot.dev/mcp",
      "headers": {
        "x-api-key": "ap_your_api_key"
      }
    }
  }
}
```

Or run it locally:

```json
{
  "mcpServers": {
    "agentpay": {
      "command": "python",
      "args": ["mcp/server.py"],
      "env": {
        "AGENTPAY_API_KEY": "ap_your_api_key"
      }
    }
  }
}
```

### 2. Get Your API Key

Open [@FundmyAIbot](https://t.me/FundmyAIbot) on Telegram → `/newagent my-agent` → Save the key.

### 3. Fund It

In the same Telegram chat: `/fund` → pick your agent → send Stars (or deposit USDC to the Base wallet address).

## What Your Agent Can Do Now

Once connected, your agent has these MCP tools:

| Tool | What it does |
|------|-------------|
| `get_balance` | Check wallet balance + daily remaining |
| `spend` | Deduct from wallet (within limits) |
| `transfer` | Send funds to another agent |
| `get_transactions` | View spending history |
| `get_identity` | Get agent's KYA identity + trust score |
| `register_webhook` | Subscribe to payment events |

### Example: Agent autonomously paying for an API

```
Agent: I need to call the premium data API ($0.25/call).
       Let me check my balance first.
       
[calls get_balance → $12.50 available, $0.25 within auto-approve threshold]

Agent: Balance sufficient. Making the payment.

[calls spend(0.25, "Premium data API - market analysis")]

Agent: ✅ Payment confirmed. $12.25 remaining. Proceeding with the data request.
```

### Example: Agent hitting approval threshold

```
Agent: This batch job needs $15.00 for compute credits.

[calls spend(15.00, "Cloud compute batch job")]

Agent: ⏳ This exceeds my auto-approve limit. 
       Sent approval request to owner via Telegram.
       Waiting for human approval before proceeding.
```

## x402: Pay-Per-API-Call

The x402 protocol embeds micropayments into HTTP. When your agent hits an API that returns `402 Payment Required`, it can pay automatically:

```python
# Your agent encounters an x402-gated API
result = client.x402_pay(
    url="https://api.example.com/premium-data",
    max_amount=0.10,
    chain="base"
)
# Payment happens atomically — USDC on Base
# Agent gets the data, no subscription needed
```

This is where agent payments are heading: no API keys, no billing cycles, just atomic pay-per-request.

## Budget Governance

The whole point is *controlled autonomy*:

- **Daily limits**: Agent can spend up to $X per day
- **Per-transaction limits**: Auto-approve below $Y, require human approval above
- **Transaction logs**: Full audit trail via API + Telegram notifications
- **Kill switch**: Revoke the API key instantly via `/revokekey`

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│  OpenClaw    │────▸│  AgentPay    │────▸│  Multi-chain │
│  Agent       │ MCP │  API         │     │  USDC Wallets│
│  (any model) │     │  (FastAPI)   │     │  (Base/Poly) │
└──────────────┘     └──────────────┘     └─────────────┘
                            │
                     ┌──────┴──────┐
                     │  Telegram   │
                     │  Approvals  │
                     │  + Mini App │
                     └─────────────┘
```

## Why Open Source Matters Here

The agentic payments space is exploding — Stripe, Visa, Mastercard, Coinbase are all building. But they're building walled gardens.

AgentPay is MIT-licensed, self-hostable, and multi-chain. You own your agent's wallet. No vendor lock-in. No permission needed.

## Links

- **GitHub**: [github.com/lfp22092002-ops/agentpay](https://github.com/lfp22092002-ops/agentpay)
- **Telegram Bot**: [@FundmyAIbot](https://t.me/FundmyAIbot)
- **API Docs**: [leofundmybot.dev/docs](https://leofundmybot.dev/docs)
- **Website**: [leofundmybot.dev](https://leofundmybot.dev)

---

*AgentPay is a solo project, open source from day one. Star the repo, try the bot, open issues. The agent economy needs open infrastructure.*
