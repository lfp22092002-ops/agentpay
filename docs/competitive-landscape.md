# AgentPay Competitive Landscape — March 2026

## The Market (Q1 2026 Snapshot)
- **x402 volume**: $50M+ cumulative (Coinbase), but ~$1.6M/30d genuine after filtering wash trades (a16z/Allium Labs)
- **ERC-8004 registrations**: 24,000+ agents on mainnet since Jan 2026 launch
- **McKinsey projection**: $3-5T agentic commerce globally by 2030

## Direct Competitors

### FinAI (Closest Competitor)
- **Stack**: x402 (payments) + ERC-8004 (KYA identity) + credit scoring — nearly identical to AgentPay
- **Team**: Ex-leading internet companies, seed funded by blockchain VCs
- **Revenue model**: API subscriptions (B2B Web2 devs) + low tx fees (Web3)
- **Status**: First autonomous payment completed Q1 2026, expects profitability within year
- **Strengths**: VC-backed, compliance focus, ERC-8138 integration planned
- **Weaknesses**: No Telegram-native UX, no Stars integration, enterprise B2B focus
- **Our differentiation**: Telegram-first, indie dev UX, Stars funding, self-hostable open source

### MoonPay Agents (Enterprise Infrastructure)
- **Stack**: Non-custodial wallets, fiat on-ramp, x402 support, cross-chain swaps
- **Scale**: 500+ enterprise customers, 30M+ users, full US money transmitter licenses
- **Model**: One-time human KYC → autonomous agent operations
- **Status**: Live (launched Mar 2026), also building PYUSDx stablecoin issuance
- **Strengths**: Massive distribution, regulatory moat (BitLicense, MiCA), fiat-to-crypto rails
- **Weaknesses**: Enterprise pricing/complexity, custodial feel, no bot-native UX
- **Our differentiation**: Open source, free tier, Telegram bot-native, developer-first API

## Infrastructure Players (Not Direct Competitors)

### Coinbase (x402 Protocol Owner)
- Facilitator on Base, Solana, Polygon; 50M+ tx processed
- AgentKit for wallet provisioning; infrastructure layer, not end-product
- **Relationship**: We build ON x402, not against it

### Stripe (Vertical Integration)
- Session Payment Tokens (SPTs), Tempo blockchain (mainnet 2026)
- Partners: Affirm, Klarna, Etsy — targeting established commerce
- **Relationship**: Different market (card-based established merchants vs crypto-native agents)

### Card Networks
- Mastercard Agent Pay + Santander completed Europe's first live AI agent payment (Mar 2)
- Visa exploring agent virtual cards
- **Relationship**: Cards win where refunds/chargebacks matter; we win where speed/micro-payments matter

### CEXs (Trading Infrastructure)
- Binance: 12 AI Agent Skills (trading execution)
- OKX: OnchainOS (60+ chains, 1B+ daily API calls)
- Bitget/BingX: Agent Hub + AI Skills Hub
- **Relationship**: They're execution venues; we're the payment/wallet layer

## x402 Ecosystem Growth
- **Chains**: Base, Solana, Polygon, Stellar, Etherlink (Tezos), Algorand, TRON (planned)
- **Integrations**: Stripe, Cloudflare, Vercel, Google AP2
- **Developer adoption**: Indie devs building 402 paywalled APIs (Express.js middleware, ~15 lines)

## AgentPay Positioning

| Dimension | AgentPay | FinAI | MoonPay | Coinbase |
|-----------|----------|-------|---------|----------|
| Target user | Indie devs, bot builders | Enterprise B2B | Enterprise/fintech | Everyone (infra) |
| Distribution | Telegram bot, GitHub | API sales team | Enterprise sales | SDK/protocol |
| Funding method | Stars + crypto | Crypto only | Fiat + crypto | Crypto only |
| Open source | Yes (MIT) | No | No | Partial (protocol) |
| Self-hostable | Yes | No | No | N/A |
| Identity (KYA) | Built-in | Built-in | Via KYC | Via AgentKit |
| Pricing | Free (beta) | API subscription | Enterprise | Free (protocol) |

## Strategic Priorities
1. **Speed to PyPI**: SDK publishable → instant developer onboarding via 
2. **First 10 users**: Dev community outreach (dev.to, Reddit, HN), Moltbook, X threads
3. **LangChain/CrewAI tools**: Framework-native integrations → agents discover AgentPay automatically
4. **npm SDK**: TypeScript SDK on npm → JS/Node.js developer capture
5. **Product Hunt launch**: Time for maximum visibility once 10+ active users
