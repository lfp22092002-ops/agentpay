# Troubleshooting

Common issues and solutions for AgentPay.

## Authentication

### `401 Unauthorized` on all requests
- Ensure `X-API-Key` header is set (not `Authorization: Bearer`)
- Check the key matches what was generated via `/v1/agents` POST
- API keys are SHA-256 hashed at rest — you can't recover them, only regenerate

### `403 Forbidden` on agent endpoints
- The API key is scoped to a specific agent. You can only access your own agent's wallets/transactions
- Admin endpoints (`/v1/admin/*`) require the master `API_SECRET`, not an agent key

## Wallets & Transactions

### `400 Bad Request` on spend
- Minimum spend is chain-dependent. Check your balance first: `GET /v1/wallets/{agent_id}`
- Ensure `amount` is a string (not number) to avoid floating-point precision issues
- `idempotency_key` must be unique per request — reusing one returns the original result

### Wallet shows 0 balance after deposit
- On-chain deposits require block confirmations. Base/Polygon: ~2-5 min, BNB: ~1-3 min, Solana: ~30s
- Check transaction status: `GET /v1/transactions/{tx_id}`

### `chain not supported` error
- Supported chains: `base`, `polygon`, `bnb`, `solana`
- Chain names are lowercase

## Webhooks

### Not receiving webhook events
- Verify your URL is publicly accessible (not `localhost`)
- Check webhook registration: `GET /v1/webhooks`
- Events are retried 3 times with exponential backoff
- Verify HMAC-SHA256 signature validation isn't rejecting valid payloads

### Webhook signature mismatch
- The signature is computed over the raw request body — don't parse/re-serialize before verifying
- Use the webhook secret from registration, not your API key
- Python: `hmac.compare_digest()` for timing-safe comparison

## Telegram Bot

### Bot not responding to `/start`
- Ensure `BOT_TOKEN` in `.env` is correct
- Only one gateway can use a bot token at a time (409 conflict otherwise)
- Check bot is running: `systemctl status agentpay-bot`

### Stars payment not crediting
- Telegram Stars have a processing delay (~5-30s)
- Check `/v1/transactions?type=stars` for the transaction status

## Docker / Self-Hosting

### Database connection refused
- Ensure Postgres is running: `docker compose ps`
- Default port is 5432 — check `DATABASE_URL` in `.env`
- First run needs `docker compose up -d postgres` before the API

### Port 8080 already in use
- Change `API_PORT` in `.env` or stop the conflicting service
- If using Cloudflare Tunnel, the tunnel config must match the port

## SDK

### Python: `ImportError: No module named 'agentpay'`
- Install: `pip install agentpay` (or from source: `pip install ./sdk/agentpay`)
- Requires Python 3.9+

### TypeScript: types not resolving
- Ensure `agentpay` is installed: `npm install agentpay`
- If using from source, run `npm run build` in `sdk/ts/` first

### Retry/timeout issues
- Default timeout: 30s. Override: `AgentPay(api_key, timeout=60)`
- SDK retries 429/5xx up to 3 times with exponential backoff
- For long-running operations, increase timeout

## x402 Protocol

### `402 Payment Required` not handled
- Your HTTP client must support the x402 flow: inspect `X-Payment` response header
- Use the SDK's built-in x402 client: `from agentpay.x402 import X402Client`

## Still stuck?

- API docs: `https://leofundmybot.dev/docs`
- GitHub Issues: `https://github.com/lfp22092002-ops/agentpay/issues`
- Telegram: `@FundmyAIbot`
