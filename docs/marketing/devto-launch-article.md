---
title: "I Built a Payment Layer for AI Agents — Here's What I Learned"
published: false
description: "AgentPay gives AI agents their own wallets, spending limits, and approval workflows. Open source. Multi-chain. Telegram-native."
tags: ai, python, crypto, opensource
cover_image:
---

## The Problem

Your AI agent can browse the web, write code, and analyze data. But it can't pay for things.

If your agent needs to call a paid API, buy cloud credits, or subscribe to a service — it needs *you* to manually handle the payment. Every time. That breaks autonomy.

I built [AgentPay](https://github.com/lfp22092002-ops/agentpay) to fix this.

## What AgentPay Does

AgentPay is an open-source payment layer that gives AI agents:

- **Their own USDC wallets** across Base, Polygon, BNB Chain, and Solana
- **Spending limits** — daily caps, per-transaction limits, auto-approve thresholds
- **Human approval workflows** — agents request permission before spending above a threshold
- **API keys with Stripe-style security** — hashed, prefixed, shown once
- **x402 protocol support** — pay-per-API-call using HTTP 402
- **Telegram bot + Mini App dashboard** for managing everything

The core idea: *agents should have bank accounts, not credit cards.* You fund the wallet, set the rules, and the agent operates within those boundaries.

## Architecture

```
┌────────────┐     ┌──────────────┐     ┌─────────────┐
│  AI Agent  │────▸│  AgentPay    │────▸│  On-chain   │
│  (any LLM) │     │  API (REST)  │     │  Wallets    │
└────────────┘     └──────────────┘     └─────────────┘
                          │
                   ┌──────┴──────┐
                   │  Telegram   │
                   │  Approvals  │
                   └─────────────┘
```

When an agent wants to spend money:
1. It calls `POST /v1/spend` with amount and description
2. AgentPay checks limits — if under auto-approve threshold, it goes through
3. If above threshold, it sends a Telegram notification asking the human to approve
4. On approval, the balance is deducted and the transaction is logged

## Quick Start

```python
from agentpay import AgentPayClient

client = AgentPayClient("ap_your_api_key")

# Check balance
balance = client.get_balance()
print(f"Available: ${balance.balance_usd}")

# Spend money (within agent's limits)
result = client.spend(
    amount=2.50,
    description="GPT-4 API call",
)
print(f"TX: {result.transaction_id}")
```

Async works too:

```python
from agentpay import AgentPayAsyncClient

async with AgentPayAsyncClient("ap_your_key") as client:
    balance = await client.get_balance()
```

## x402: Pay-Per-Request

The x402 protocol embeds payments directly into HTTP. Instead of billing cycles and API keys, every request carries a stablecoin payment:

```python
# As a client (agent paying for an API)
result = client.x402_pay(
    url="https://api.example.com/data",
    max_amount=0.01,
    chain="base",
)

# As a server (accepting x402 payments)
from agentpay.x402 import X402Server
server = X402Server(wallet_address="0x...", price_usd=0.001)
```

This is where agent payments are heading — no subscriptions, no invoices, just atomic pay-per-call.

## Trust & Identity

How does a merchant know if an agent is trustworthy? AgentPay includes a KYA (Know Your Agent) system:

- **Trust scores** based on transaction history, age, volume, and verification status
- **Agent identity cards** — display name, category, homepage, logo
- **On-chain attestation** — agent identity anchored to blockchain

```python
identity = client.get_identity()
trust = client.get_trust_score()
print(f"Trust score: {trust['total']}/100")
```

## Why Open Source?

The agentic payments space is heating up fast. Stripe launched SPTs. Visa is doing "Intelligent Commerce." Google dropped UCP. Mastercard completed live agent payments.

All of these are walled gardens.

AgentPay takes the opposite approach: open source, self-hostable, multi-chain. You own your agent's wallet. You set the rules. No vendor lock-in.

## What's Built

- **API**: 24+ REST endpoints (FastAPI + SQLAlchemy async)
- **Telegram Bot**: 19 commands for agent management
- **Mini App**: 5-tab dashboard with analytics
- **Python SDK**: Sync + async clients, retry logic, webhook verification
- **TypeScript SDK**: Full client with types
- **CI/CD**: 544 tests, GitHub Actions, Docker deployment
- **Security**: HMAC-SHA256 webhooks, Fernet-encrypted wallet keys, rate limiting

## What's Next

- PyPI publish (`pip install agentpay`)
- npm publish (`npm install agentpay`)
- Virtual Visa cards via Lithic (pending production keys)
- MCP tool integration for agent frameworks
- Product Hunt launch

## Try It

- **GitHub**: [github.com/lfp22092002-ops/agentpay](https://github.com/lfp22092002-ops/agentpay)
- **Telegram Bot**: [@FundmyAIbot](https://t.me/FundmyAIbot)
- **Docs**: [leofundmybot.dev/docs-site](https://leofundmybot.dev/docs-site/)

Star the repo, try the bot, break things, open issues. The agent economy needs open infrastructure.

---

*Building this solo. Feedback and contributions welcome. If you're working on AI agents and need payment capabilities, let's talk.*
