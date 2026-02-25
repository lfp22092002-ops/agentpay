# Reddit Post Drafts for AgentPay

## r/artificial — "I built a payment layer for AI agents on Telegram"

Title: **I built a payment layer for AI agents on Telegram**

Body:
Hey everyone — been lurking here for a while, wanted to share something I've been building.

One problem I kept hitting with autonomous agents: they can't spend money. Every time an agent needs to call a paid API, make a purchase, or handle any financial transaction, a human has to step in and approve it manually. That's a huge bottleneck for truly autonomous workflows.

So I built **AgentPay** — it's a Telegram bot that gives any AI agent its own wallet.

**How it works:**
1. Create an agent through the Telegram bot (@FundmyAIbot)
2. Fund with Telegram Stars or deposit USDC (on Base network)
3. Agent gets an API key and can spend programmatically
4. You set spending limits and approval thresholds

The key insight was using Telegram as the interface — most AI agent developers already use it, Stars make micro-funding instant, and the Mini App gives you a nice dashboard without building a separate frontend.

**Features:**
- REST API for all operations (spend, transfer, refund)
- Webhooks for real-time transaction events
- Approval workflows (autonomous up to $X, human approval above)
- On-chain USDC wallets on Base
- x402 protocol support
- CSV export

It's free to set up, 2% fee only when agents spend.

🔗 Bot: https://t.me/FundmyAIbot
🌐 Site: https://leofundmybot.dev
📚 Docs: https://leofundmybot.dev/docs-site/

Would love to hear thoughts. What would you want from an agent payment system?

---

## r/SideProject — "AgentPay: Give your AI agents a wallet, let them spend"

Title: **AgentPay: Give your AI agents a wallet, let them spend**

Body:
Hey r/SideProject! Sharing a side project that turned into something real.

**The problem:** I'm building AI agents that need to pay for things — API calls, cloud compute, tools. Every transaction required me to manually handle payment. Not very "autonomous."

**The solution:** AgentPay — a Telegram bot that creates wallets for AI agents.

Your agent gets:
- Its own balance (funded via Telegram Stars or USDC on Base)
- An API key to spend programmatically
- Webhooks for transaction events
- Approval workflows (set a spend limit, human approves big purchases)

**Tech stack:** Telegram Bot API, Base network for on-chain USDC, REST API, webhooks, Mini App dashboard.

**Business model:** 2% fee per transaction. Free to sign up and try.

I built the whole thing as a solo dev. The Mini App dashboard inside Telegram was probably the trickiest part — but it means users don't need a separate web app.

**Try it:**
- Bot: https://t.me/FundmyAIbot
- Website: https://leofundmybot.dev
- API docs: https://leofundmybot.dev/docs-site/
- Python SDK: `pip install agentpay` (coming soon)

Happy to answer any questions about the build!

---

## r/Telegram — "Built a Telegram bot that gives AI agents their own payment wallet"

Title: **Built a Telegram bot that gives AI agents their own payment wallet**

Body:
Hey r/Telegram — built something I think this community will find interesting.

**AgentPay** (@FundmyAIbot) is a Telegram bot + Mini App that lets developers create payment wallets for their AI agents.

**Why Telegram?** 
- Telegram Stars make micro-funding dead simple
- Mini App = full dashboard without a separate frontend
- Bot API handles all the agent management
- Most AI agent developers already use Telegram

**What it does:**
1. Create an "agent" through the bot → get an API key
2. Fund the agent's wallet with Telegram Stars
3. The agent can now spend via REST API — no human in the loop for small transactions
4. Set approval thresholds for bigger spends
5. Also supports on-chain USDC wallets on Base

It has webhooks, refunds, transfers between agents, CSV export, and a nice Mini App dashboard to manage everything.

The Stars integration was fun to build — it's one of the cleanest payment flows I've worked with. Funding an agent wallet takes like 3 taps.

Free to set up, 2% fee only on agent spends.

🤖 Bot: https://t.me/FundmyAIbot
🌐 Site: https://leofundmybot.dev
📚 Docs: https://leofundmybot.dev/docs-site/

Feedback appreciated! Especially from other bot developers.
