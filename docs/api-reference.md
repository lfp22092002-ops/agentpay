# AgentPay API Reference

Base URL: `https://leofundmybot.dev/v1`

All endpoints (except health and directory) require an `X-API-Key` header with a valid agent API key.

---

## Health

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/health` | No | Service health check |

---

## Agent Balance & Transactions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/balance` | Get agent's current balance |
| GET | `/v1/transactions` | List agent's transactions |

---

## Spending & Transfers

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/spend` | Spend from agent balance (supports idempotency + approval workflows) |
| POST | `/v1/refund` | Refund a previous transaction |
| POST | `/v1/transfer` | Transfer funds between agents |

### POST `/v1/spend`
```json
{
  "amount": 1.50,
  "description": "API call to OpenAI",
  "idempotency_key": "unique-key-123",
  "skip_approval": false
}
```

### POST `/v1/refund`
```json
{
  "transaction_id": "tx_abc123",
  "amount": 1.50,
  "reason": "Duplicate charge"
}
```

### POST `/v1/transfer`
```json
{
  "to_agent_id": "agent_xyz",
  "amount": 5.00,
  "description": "Payment for sub-task"
}
```

---

## Wallets (Multi-Chain)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/wallet` | Get primary wallet address + balance |
| GET | `/v1/wallet/all` | Get all chain wallets (Base, Polygon, BNB, Solana) |
| GET | `/v1/chains` | List supported chains and configs |
| POST | `/v1/wallet/send-usdc` | Send USDC on-chain |
| POST | `/v1/wallet/send-native` | Send native token (ETH, MATIC, BNB, SOL) |

### POST `/v1/wallet/send-usdc`
```json
{
  "to_address": "0x...",
  "amount": 10.00,
  "chain": "base"
}
```

---

## Virtual Card (Lithic)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/card` | Get virtual card details |
| GET | `/v1/card/transactions` | List card transactions |

---

## Webhooks

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/webhook` | Set webhook URL (HMAC-SHA256 signed) |
| GET | `/v1/webhook` | Get current webhook config |
| DELETE | `/v1/webhook` | Remove webhook |

### POST `/v1/webhook`
```json
{
  "url": "https://your-server.com/webhook",
  "secret": "your-signing-secret"
}
```

All webhook deliveries include `X-AgentPay-Signature` header for HMAC-SHA256 verification.

---

## Approvals

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/approvals/{approval_id}` | Check approval status |

---

## x402 Protocol

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/x402/pay` | Make x402 micropayment |
| GET | `/v1/x402/probe` | Probe x402 payment capabilities |

---

## Identity (ERC-8004)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/agent/identity` | Get agent identity profile |
| PUT | `/v1/agent/identity` | Update agent identity |
| GET | `/v1/agent/identity/score` | Get agent trust score |

---

## Directory

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/directory` | No | Browse public agent directory |
| GET | `/v1/directory/{agent_id}` | No | Get agent profile |
| GET | `/v1/directory/{agent_id}/registration.json` | No | Machine-readable registration |

---

## Admin

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/revenue` | Platform revenue stats |
| POST | `/v1/withdraw` | Withdraw platform revenue |

---

## Mini App (Telegram)

These endpoints power the Telegram Mini App dashboard. Auth via Telegram `initData` → JWT.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/telegram` | Exchange Telegram initData for JWT |
| GET | `/miniapp/agents` | List user's agents |
| GET | `/miniapp/agents/{id}/transactions` | Agent transaction history |
| PATCH | `/miniapp/agents/{id}/settings` | Update agent settings |
| GET | `/miniapp/agents/{id}/card` | Agent card details |
| POST | `/miniapp/agents/{id}/card/{action}` | Card actions (freeze/unfreeze) |
| GET | `/miniapp/agents/{id}/wallet` | Agent wallet info |
| GET | `/miniapp/agents/{id}/wallet/all` | All chain wallets |
| GET | `/miniapp/dashboard` | Dashboard summary |
| GET | `/miniapp/agents/{id}/analytics` | Agent analytics |
| GET | `/miniapp/agents/{id}/identity` | Agent identity profile |

---

## Rate Limits

- Spend/refund/transfer: 30 req/min
- General endpoints: 60 req/min
- Webhook deliveries retry 3x with exponential backoff

## Error Format

```json
{
  "detail": "Insufficient balance"
}
```

HTTP status codes: `200` success, `400` bad request, `401` unauthorized, `403` forbidden, `404` not found, `429` rate limited, `500` server error.

---

*Auto-generated from route definitions. For interactive docs, run the server and visit `/docs` (Swagger) or `/redoc`.*
