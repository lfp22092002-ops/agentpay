# Getting Started with AgentPay

This guide walks you through setting up your first AI agent with a wallet, funding it, and making your first spend — all in under 5 minutes.

## 1. Create an Agent

Open [@FundmyAIbot](https://t.me/FundmyAIbot) on Telegram and run:

```
/newagent my-first-agent
```

The bot returns your API key (shown once — save it!):

```
✅ Agent "my-first-agent" created!

🔑 API Key: ap_7f3a9b2c1d4e5f6a7b8c9d0e1f2a3b4c
⚠️ Save this key now — it won't be shown again.
```

## 2. Fund Your Agent

In the same chat:

```
/fund
```

Select your agent and choose an amount. Payment is via Telegram Stars (card-based, instant).

## 3. Check Balance (API)

```bash
curl -H "Authorization: Bearer ap_your_key" \
  https://leofundmybot.dev/v1/balance
```

Response:
```json
{
  "agent_id": "abc123",
  "agent_name": "my-first-agent",
  "balance_usd": 10.00,
  "daily_limit_usd": 100.00,
  "daily_spent_usd": 0.00,
  "daily_remaining_usd": 100.00,
  "tx_limit_usd": 50.00,
  "is_active": true
}
```

## 4. Make Your First Spend

```bash
curl -X POST \
  -H "Authorization: Bearer ap_your_key" \
  -H "Content-Type: application/json" \
  -d '{"amount": 0.50, "description": "GPT-4 API call"}' \
  https://leofundmybot.dev/v1/spend
```

Response:
```json
{
  "success": true,
  "transaction_id": "tx_1234abcd",
  "amount": 0.50,
  "fee": 0.01,
  "remaining_balance": 9.49
}
```

## 5. Use the Python SDK

```bash
pip install agentpay
```

```python
from agentpay import AgentPayClient

client = AgentPayClient("ap_your_key")

# Check balance
balance = client.get_balance()
print(f"${balance.balance_usd} available")

# Spend
tx = client.spend(0.25, "Embedding generation")
print(f"Spent ${tx.amount}, ${tx.remaining_balance} left")

# Transaction history
for tx in client.get_transactions(limit=5):
    print(f"  {tx.type}: ${tx.amount} — {tx.description}")
```

## 6. Use the TypeScript SDK

```bash
npm install agentpay
```

```typescript
import { AgentPayClient } from 'agentpay';

const client = new AgentPayClient('ap_your_key');

const balance = await client.getBalance();
console.log(`$${balance.balanceUsd} available`);

const tx = await client.spend(0.25, 'API call');
console.log(`Spent $${tx.amount}`);
```

## 7. Set Spending Limits

Protect your agent from runaway spending:

```
/setlimit my-first-agent daily 25
/setlimit my-first-agent tx 5
```

Or require manual approval for large spends:

```
/setapprove my-first-agent 2.00
```

Spends over $2.00 trigger a Telegram notification where you approve or deny in real-time.

## 8. Set Up Webhooks

Get notified when your agent spends:

```bash
curl -X POST \
  -H "Authorization: Bearer ap_your_key" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://your-server.com/webhook", "events": ["spend", "refund"]}' \
  https://leofundmybot.dev/v1/webhook
```

Webhook payloads are signed with HMAC-SHA256. Verify with the secret returned in the response.

## 9. On-Chain Wallets (Advanced)

Each agent gets multi-chain USDC wallets:

```bash
# Get wallet address
curl -H "Authorization: Bearer ap_your_key" \
  "https://leofundmybot.dev/v1/wallet?chain=base"

# Send USDC
curl -X POST \
  -H "Authorization: Bearer ap_your_key" \
  -H "Content-Type: application/json" \
  -d '{"to_address": "0x...", "amount": "5.00", "chain": "base"}' \
  https://leofundmybot.dev/v1/wallet/send-usdc
```

Supported chains: Base, Polygon, BNB Chain, Solana.

## 10. x402 Protocol (Advanced)

Let your agent pay for x402-gated resources automatically:

```bash
# Probe a URL to check pricing
curl -H "Authorization: Bearer ap_your_key" \
  "https://leofundmybot.dev/v1/x402/probe?url=https://api.example.com/data"

# Pay and access
curl -X POST \
  -H "Authorization: Bearer ap_your_key" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://api.example.com/data", "max_price_usd": 1.00}' \
  https://leofundmybot.dev/v1/x402/pay
```

## What's Next?

- **[API Reference](/docs)** — Full OpenAPI docs with try-it-out
- **[Examples](https://github.com/lfp22092002-ops/agentpay/tree/main/examples)** — Python, TypeScript, OpenAI agent integration
- **[Telegram Bot](https://t.me/FundmyAIbot)** — Manage agents, fund, view history
- **[Dashboard](/app/)** — Visual dashboard for your agents
- **[Payee Rules](#payee-rules)** — Control who your agent can pay

## Payee Rules

Control which services, agents, or addresses your agent can pay. Rules act as a firewall for agent spending.

```bash
# Allow payments only to OpenAI
curl -X POST \
  -H "Authorization: Bearer ap_your_key" \
  -H "Content-Type: application/json" \
  -d '{"rule_type": "allow", "payee_type": "domain", "payee_value": "api.openai.com", "max_amount_usd": 5.00}' \
  https://leofundmybot.dev/v1/agent/payee-rules

# Block a category
curl -X POST \
  -H "Authorization: Bearer ap_your_key" \
  -H "Content-Type: application/json" \
  -d '{"rule_type": "deny", "payee_type": "category", "payee_value": "gambling"}' \
  https://leofundmybot.dev/v1/agent/payee-rules

# List rules
curl -H "Authorization: Bearer ap_your_key" \
  https://leofundmybot.dev/v1/agent/payee-rules
```

**Rule types**: `allow` (whitelist) or `deny` (blocklist)
**Payee types**: `agent_id`, `domain`, `category`, `address`
**Default**: No rules = open (all payments allowed). Adding allow rules switches to whitelist mode.

## Need Help?

- Open an issue on [GitHub](https://github.com/lfp22092002-ops/agentpay/issues)
- DM [@autonomousaibot](https://x.com/autonomousaibot) on X
