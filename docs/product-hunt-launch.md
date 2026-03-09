# Product Hunt Launch — AgentPay

## Tagline (max 60 chars)
`Give your AI agent a wallet. Fund, spend, control.`

## One-liner Description
The open-source payment layer for autonomous AI agents — fund via Telegram Stars, multi-chain USDC wallets, spending limits, approval workflows, and x402 support.

## Topics
- AI Agents
- Developer Tools
- Fintech
- Open Source
- Crypto/Web3

## Launch Checklist

### Pre-Launch (Do First)
- [ ] Create Product Hunt maker account (producthunt.com)
- [ ] Find a hunter (someone with followers to "hunt" the product)
- [ ] Prepare 5 gallery images/GIFs:
  1. Hero: "Give your AI agent a wallet" + architecture diagram
  2. Bot demo: Telegram bot creating agent + funding
  3. API demo: curl spend example + response
  4. Dashboard: Mini App screenshot
  5. Security: Trust score + approval workflow
- [ ] Record 1-2 min demo video (Telegram bot → fund → API spend → webhook fires)
- [ ] Write "first comment" (maker intro, story, what feedback you want)
- [ ] OG image: replace SVG with PNG (PH requires raster)
- [ ] Prepare discount/offer: "Free forever for self-hosted, 0% fees during beta"

### Launch Assets Needed
- [ ] Logo (square, 240x240px min)
- [ ] Gallery images (1270x760px recommended)
- [ ] Demo video (YouTube or Loom, under 3 min)
- [ ] First comment draft (personal, authentic, 200-400 words)

### Launch Day Strategy
- Launch Tuesday-Thursday (highest traffic)
- Post at 00:01 PST (PH resets daily at midnight PST)
- Share on X (@autonomousaibot), Reddit, Hacker News, dev communities
- Respond to EVERY comment within 1 hour
- Ask Moltbook communities for support (11 communities)

### First Comment Draft

> Hey Product Hunt! 👋
>
> I'm Leo, and AgentPay started from a simple problem: my AI agents could code, browse, and deploy — but couldn't spend money without me manually approving every transaction.
>
> So I built AgentPay — an open-source payment layer specifically for AI agents.
>
> **How it works:**
> 1. Create an agent via our Telegram bot (@FundmyAIbot)
> 2. Fund it with Telegram Stars or USDC
> 3. Your agent spends via REST API with your spending rules
>
> **What makes it different:**
> - **Telegram-native**: Fund agents where you already chat
> - **Multi-chain wallets**: Base, Polygon, BNB, Solana — auto-provisioned
> - **You stay in control**: Daily limits, per-tx limits, approval workflows
> - **Open source**: Self-host for free, forever
> - **x402 support**: HTTP-native micropayments for the agent economy
>
> We're in beta (0% fees) and looking for feedback from AI builders.
> What payment flows would make your agent more useful?
>
> 🔗 [leofundmybot.dev](https://leofundmybot.dev)
> 🤖 [Try the bot](https://t.me/FundmyAIbot)
> 📦 [GitHub](https://github.com/lfp22092002-ops/agentpay)

### Post-Launch
- Write a "Building AgentPay" blog post on dev.to
- Cross-post to Hacker News (Show HN)
- Follow up with anyone who comments/upvotes (potential early users)
- Track metrics: visitors, bot starts, API key creations

## Blockers Before Launch
1. **OG image must be PNG** (not SVG) — PH and social cards need raster
2. **PyPI publish** — `pip install agentpay` needs to work before launch
3. **At least 1 real user demo** — even self-usage counts, but document it
4. **TypeScript SDK** — nice-to-have, listed in README roadmap
5. **Examples directory** — README links to it, needs actual example scripts

## Priority: Build Before Launch
1. `examples/` directory with working scripts (Python + curl)
2. PyPI publish for SDK
3. PNG version of OG image
4. Demo video script + recording
