# AgentPay TypeScript SDK

TypeScript client for the [AgentPay](https://leofundmybot.dev) API — payment infrastructure for AI agents.

## Install

```bash
npm install agentpay
```

## Quick Start

```ts
import { AgentPayClient } from "agentpay";

const client = new AgentPayClient("ap_your_api_key");

// Check balance
const balance = await client.getBalance();
console.log(`Balance: $${balance.balance_usd}`);

// Spend
const tx = await client.spend(0.5, "GPT-4 API call");
console.log(`Spent $${tx.amount}, remaining $${tx.remaining_balance}`);

// Transfer between agents
await client.transfer("agent_xyz", 5.0, "Data enrichment payment");

// x402 paywall
const result = await client.x402Pay("https://api.example.com/premium", 2.0);
console.log(result.data);
```

## Error Handling

```ts
import { AgentPayClient, InsufficientBalanceError, RateLimitError } from "agentpay";

try {
  await client.spend(100, "Big purchase");
} catch (e) {
  if (e instanceof InsufficientBalanceError) {
    console.log("Top up via @FundmyAIbot");
  } else if (e instanceof RateLimitError) {
    console.log("Slow down — 60 req/min");
  }
}
```

## Links

- [Website](https://leofundmybot.dev)
- [API Docs](https://leofundmybot.dev/docs)
- [GitHub](https://github.com/lfp22092002-ops/agentpay)
- [Telegram Bot](https://t.me/FundmyAIbot)
