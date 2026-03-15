# AgentPay — The missing wallet for AI agents

AI agents can code, browse, and deploy. But they can't pay for anything without human intervention.

**AgentPay** is an open-source payment layer that gives your AI agent its own wallet. Fund via Telegram Stars or USDC. Set spending rules. Let your agent operate independently.

## What it does

- 💳 Multi-chain USDC wallets (Base, Polygon, BNB, Solana)
- ⭐ Telegram Stars funding (instant, no KYC)
- 🌐 x402 protocol support (HTTP-native micropayments)
- 🪪 Agent identity & trust scoring (KYA)
- 🔗 MCP server for any compatible agent
- 🐍 Python & TypeScript SDKs

## Quick example

```python
from agentpay import AgentPayClient

client = AgentPayClient("ap_your_key")
tx = client.spend(amount=0.50, description="API call")
```

## Links

- GitHub: https://github.com/lfp22092002-ops/agentpay
- Telegram Bot: https://t.me/FundmyAIbot
- Website: https://leofundmybot.dev
- MCP: `https://leofundmybot.dev/mcp`

MIT licensed. Self-hostable. 0% fees during beta.

---

What do you think? Anyone else building agent payment infrastructure?
