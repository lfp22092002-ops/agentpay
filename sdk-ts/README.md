# AgentPay TypeScript SDK

TypeScript/JavaScript SDK for [AgentPay](https://leofundmybot.dev) — payment infrastructure for AI agents.

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

// Spend funds
const tx = await client.spend(0.50, "GPT-4 API call");
console.log(`Spent $${tx.amount}, remaining $${tx.remaining_balance}`);

// Transfer between agents
const result = await client.transfer("agent_xyz", 5.0, "Payment for services");

// Pay for x402-gated resources
const data = await client.x402Pay("https://api.example.com/premium");

// Get identity & trust score
const identity = await client.getIdentity();
console.log(`Trust score: ${identity.trust_score}`);
```

## Configuration

```ts
const client = new AgentPayClient("ap_your_api_key", {
  baseUrl: "https://leofundmybot.dev", // default
  timeoutMs: 30_000,                    // 30s default
  maxRetries: 3,                        // retries on 429/5xx/network errors
});
```

## API Methods

| Method | Description |
|--------|-------------|
| `getBalance()` | Get agent wallet balance |
| `getWallet(chain?)` | Get on-chain wallet (base/polygon/bnb/solana) |
| `listChains()` | List supported blockchains |
| `spend(amount, description, idempotencyKey?)` | Spend from agent balance |
| `refund(transactionId)` | Refund a transaction |
| `transfer(toAgentId, amount, description?)` | Agent-to-agent transfer |
| `getTransactions(limit?)` | Get transaction history |
| `registerWebhook(url, events?)` | Register webhook endpoint |
| `x402Pay(url, maxAmount?)` | Pay for x402-gated resource |
| `getIdentity()` | Get agent identity profile (KYA) |
| `getTrustScore()` | Get trust score breakdown |

## Error Handling

```ts
import {
  AgentPayClient,
  AuthenticationError,
  InsufficientBalanceError,
  RateLimitError,
} from "agentpay";

try {
  await client.spend(100, "Big purchase");
} catch (err) {
  if (err instanceof InsufficientBalanceError) {
    console.log("Not enough funds");
  } else if (err instanceof AuthenticationError) {
    console.log("Invalid API key");
  } else if (err instanceof RateLimitError) {
    console.log("Rate limited — slow down");
  }
}
```

## Features

- ✅ Zero dependencies (uses native `fetch`)
- ✅ ESM + CJS dual builds
- ✅ Full TypeScript types
- ✅ Automatic retries with exponential backoff
- ✅ Respects `Retry-After` headers
- ✅ Works in Node.js 18+, Deno, Bun, edge runtimes

## License

MIT
