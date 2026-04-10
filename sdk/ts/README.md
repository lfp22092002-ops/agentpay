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

// Refund a transaction
await client.refund(tx.transaction_id);
```

## Wallets

```ts
// Get wallet info for a specific chain
const wallet = await client.getWallet("base");
console.log(`Address: ${wallet.address}`);

// List all supported chains
const chains = await client.listChains();
chains.forEach((c) => console.log(`${c.id}: ${c.name}`));
```

## Webhooks

```ts
// Register a webhook
const wh = await client.registerWebhook("https://example.com/hook", ["spend", "deposit"]);
console.log(`Secret: ${wh.secret}`); // Use this to verify signatures
```

## x402 Protocol

```ts
// Pay for an x402-gated resource
const result = await client.x402Pay("https://api.example.com/premium", 2.0);
console.log(result.data);
```

## Identity & Trust Score

```ts
// Get agent identity (KYA)
const identity = await client.getIdentity();
console.log(`Trust score: ${identity.trust_score}`);

// Detailed trust score breakdown
const score = await client.getTrustScore();
console.log(`Total: ${score.total}/100`);
```

## Payee Rules

```ts
// Allow payments only to specific agents/domains
await client.createPayeeRule("domain", "api.openai.com", "allow", 10.0, "OpenAI cap");

// List existing rules
const rules = await client.listPayeeRules();

// Remove a rule
await client.deletePayeeRule(rules.rules[0].id);
```

## Error Handling

```ts
import {
  AgentPayClient,
  AgentPayError,
  AuthenticationError,
  InsufficientBalanceError,
  RateLimitError,
} from "agentpay";

try {
  await client.spend(100, "Big purchase");
} catch (e) {
  if (e instanceof InsufficientBalanceError) {
    console.log("Top up via @FundmyAIbot");
  } else if (e instanceof RateLimitError) {
    console.log("Slow down — 60 req/min");
  } else if (e instanceof AuthenticationError) {
    console.log("Check your API key");
  } else if (e instanceof AgentPayError) {
    console.log(`API error ${e.statusCode}: ${e.detail}`);
  }
}
```

## Configuration

```ts
const client = new AgentPayClient("ap_...", {
  baseUrl: "https://leofundmybot.dev", // default
  timeoutMs: 30_000, // default: 30s
});
```

## Requirements

- Node.js 18+ (uses native `fetch`)
- Zero dependencies

## Links

- [Documentation](https://leofundmybot.dev/docs-site/)
- [API Reference](https://leofundmybot.dev/docs)
- [Telegram Bot](https://t.me/FundmyAIbot)
- [GitHub](https://github.com/lfp22092002-ops/agentpay)
