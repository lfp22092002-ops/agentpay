# Troubleshooting

Common issues and solutions when running AgentPay.

## Bot Won't Start

**Symptom**: `agentpay-bot` exits immediately or loops.

1. **Check bot token**: Ensure `BOT_TOKEN` in `.env` is valid. Get a new one from [@BotFather](https://t.me/BotFather).
2. **Token conflict**: Only one process can use a bot token at a time. If another instance is running (different server, local dev), stop it first — Telegram returns `409 Conflict`.
3. **Check logs**: `docker compose logs agentpay-bot --tail=50`

## API Returns 500

**Symptom**: All API calls return Internal Server Error.

1. **Database not ready**: The API needs PostgreSQL running. Check: `docker compose ps` — is `db` healthy?
2. **Missing migrations**: Run `docker compose exec api python -c "from database import init_db; import asyncio; asyncio.run(init_db())"` to initialize tables.
3. **Check logs**: `docker compose logs agentpay-api --tail=50`

## Webhook Delivery Failures

**Symptom**: Webhooks not arriving at your endpoint.

1. **URL reachable?**: AgentPay must be able to reach your webhook URL. `localhost` won't work — use a tunnel (ngrok, Cloudflare Tunnel) for development.
2. **HTTPS required**: Webhook URLs must use HTTPS.
3. **Signature verification failing?**: Ensure you're using the correct webhook secret and comparing HMAC-SHA256 hex digests. See `examples/webhook_receiver.py`.
4. **Timeout**: Endpoints must respond within 10 seconds or the delivery is marked failed.

## Stars Payments Not Working

**Symptom**: `/fund` command doesn't show payment option.

1. **Bot must have payments enabled**: In BotFather, use `/mybots` → your bot → Payments → enable a provider.
2. **Stars require Telegram 10.0+**: Ensure the user's app is updated.
3. **Test mode**: Stars payments don't work in test mode — use a real Telegram account.

## USDC Deposits Not Detected

**Symptom**: Sent USDC to agent wallet but balance doesn't update.

1. **Correct chain?**: Each agent has separate addresses per chain (Base, Polygon, BNB, Solana). Sending to the wrong chain means funds go to a different address.
2. **Confirmation time**: Base/Polygon: ~2-5 min. BNB: ~3 min. Solana: ~30 sec.
3. **Minimum amount**: Very small amounts (< $0.01) may be ignored due to gas economics.

## SDK Connection Errors

**Symptom**: `ConnectionError` or `TimeoutError` from Python/TS SDK.

```python
# Python — increase timeout
from agentpay import AgentPay
client = AgentPay(api_key="...", base_url="https://leofundmybot.dev", timeout=30)
```

```typescript
// TypeScript — increase timeout
import { AgentPay } from 'agentpay';
const client = new AgentPay({ apiKey: '...', baseUrl: 'https://leofundmybot.dev', timeout: 30000 });
```

1. **Check base URL**: Default is `https://leofundmybot.dev`. If self-hosting, set `base_url` / `baseUrl`.
2. **API key valid?**: Use `/myagents` in the Telegram bot to verify your key is active.
3. **Rate limited?**: SDK retries automatically on 429. If persistent, you're exceeding limits.

## Docker Issues

### Container keeps restarting
```bash
docker compose logs <service> --tail=100
```
Usually a missing env var or database connection issue.

### Port already in use
```bash
# Find what's using port 8080
lsof -i :8080
# Or change the port in docker-compose.yml
```

### Out of disk space
```bash
# Clean up Docker artifacts
docker system prune -a --volumes
```

## MCP Server Issues

**Symptom**: MCP tools not showing up in your AI agent.

1. **stdio mode**: Ensure `AGENTPAY_API_KEY` is set in the environment where the MCP server runs.
2. **HTTP mode**: The `x-api-key` header must be included in every request to `https://leofundmybot.dev/mcp`.
3. **Tool list empty?**: Check that the API key has sufficient permissions (agent-level keys can only access their own wallet).

## Still Stuck?

- **GitHub Issues**: [github.com/lfp22092002-ops/agentpay/issues](https://github.com/lfp22092002-ops/agentpay/issues)
- **Telegram**: [@FundmyAIbot](https://t.me/FundmyAIbot) — `/help` for bot commands
- **API Docs**: `https://leofundmybot.dev/docs` (interactive Swagger UI)
