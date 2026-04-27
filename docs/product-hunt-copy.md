# AgentPay — Product Hunt Launch Copy

**Tagline:**
> Give your AI agent a wallet. Autonomous payments for autonomous agents.

---

**Short Description (260 chars):**
AgentPay is the open-source payment layer for AI agents. Fund via Telegram Stars or crypto (USDC on Base/Polygon/BNB/Solana), set spending rules, and let your agent spend, transfer, and receive payments autonomously — no human in the loop.

---

**Long Description:**

AI agents can browse the web, write code, and deploy software — but they still can't pay for anything without stopping to ask a human. AgentPay fixes that.

**What it does:**
- 🤖 Each agent gets its own wallet with an API key
- 💳 Fund via Telegram Stars or crypto (USDC on Base, Polygon, BNB, Solana)
- 💸 Agent spends via REST API — no UI needed
- 🔁 Agent-to-agent transfers, refunds, webhooks (HMAC-signed)
- 🔐 Spending limits, approval workflows, rate limiting built-in
- 📦 Python SDK (`pip install agentpay`), MCP tools, x402 protocol support
- 🆓 Self-hosted, open-source, 0% fees in beta

**Who it's for:**
- Developers building autonomous agents that need to pay for APIs, services, or other agents
- Bot builders who want to monetize via Telegram Stars
- Anyone tired of hardcoding payment logic into every agent

**Quick start:**
Open @FundmyAIbot on Telegram → `/newagent` → get your API key → `client.spend(0.10, "openai")` and you're done.

**Built with:** Python, FastAPI, aiogram, PostgreSQL, Cloudflare Tunnel

**Links:**
- GitHub: https://github.com/lfp22092002-ops/agentpay
- Telegram Bot: https://t.me/FundmyAIbot
- API Docs: https://leofundmybot.dev/docs

---

**First comment (to post on launch day):**

Hey PH! 👋 Built this after realizing every autonomous agent project I worked on had the same gap — agents can *do* almost anything but can't *pay* for anything. Had to hardcode API keys, manually top up credits, babysit every transaction.

AgentPay started as a weekend hack and turned into something I actually use. It's open-source, self-hostable, and built API-first so agents (not humans) are the primary users.

Would love feedback — especially from anyone building with Claude/GPT agents or MCP tools. What payment flows are you currently hacking around?

---

**Topics/Tags:** Artificial Intelligence, Developer Tools, Payments, Bots, Open Source

---

*Draft prepared: 2026-04-27*
