# Show HN: AgentPay – Open-source payment layer for AI agents (USDC + Telegram Stars)

## HN Post Title Options (pick best):
1. Show HN: AgentPay – Give your AI agent a wallet (USDC + x402)
2. Show HN: AgentPay – Open-source payment infra for AI agents
3. Show HN: I built a payment layer so AI agents can spend money without asking

## Post Text

I've been building AI agents that can browse, code, and deploy — but the moment they need to pay $0.05 for an API call, a human has to step in. So I built AgentPay.

**What it does**: Gives AI agents their own USDC wallets with budget rules. Agents can spend, transfer, and pay for x402-gated resources autonomously.

**Stack**: Python (FastAPI), PostgreSQL, multi-chain USDC wallets (Base, Polygon, BNB, Solana), Telegram Stars funding.

**How it works**:

1. Create an agent via Telegram bot (@FundmyAIbot) → get API key
2. Fund via Telegram Stars or USDC deposit
3. Agent calls REST API to spend within budget rules

```python
from agentpay import AgentPayClient

client = AgentPayClient("ap_your_key")
balance = client.get_balance()
tx = client.spend(0.50, "GPT-4 API call")
```

**Key features**:
- MCP server (works with Claude Code, Cursor, OpenClaw)
- x402 protocol support (HTTP-native micropayments)
- Agent identity with trust scores (KYA)
- Budget governance + approval workflows
- Python + TypeScript SDKs
- Self-hostable, MIT licensed

**What I learned building it**:
- x402 is real — Coinbase processed 50M+ transactions, but ~50% is wash trading (a16z data)
- The genuine volume is ~$1.6M/month — tiny but growing fast
- Card networks (Mastercard Agent Pay) are entering too, but they can't do sub-cent payments
- Identity is the hardest problem — how do you KYC an AI agent?

GitHub: https://github.com/lfp22092002-ops/agentpay
Live API: https://leofundmybot.dev
Bot: https://t.me/FundmyAIbot

Would love feedback on the architecture and what features devs actually need.

---

## Timing Notes
- Best HN posting: Tuesday-Thursday, 8-10 AM EST
- Should have PyPI published first (`pip install agentpay`)
- Need at least the landing page looking sharp
- Consider posting dev.to article same day for cross-traffic

## Pre-Launch Checklist
- [ ] PyPI publish (needs G's token)
- [ ] npm publish (needs G's account)
- [ ] Landing page review
- [ ] GitHub stars > 5 (social proof)
- [ ] README badges all green
- [ ] At least 1 real user or demo video
