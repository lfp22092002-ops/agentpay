# AgentPay Quick Deploy — Linux PC

## Prerequisites
- Docker + Docker Compose installed
- Bot token from @BotFather
- Cloudflare Tunnel configured (`leofundmybot.dev` → `localhost:8080`)

## Steps

### 1. Create `.env`
```bash
cd /home/leo/.openclaw/workspace/projects/agentpay
cp .env.example .env
```

Edit `.env`:
```
BOT_TOKEN=<from BotFather>
API_SECRET=<run: openssl rand -hex 32>
ENVIRONMENT=production
```

### 2. Change default Postgres password
Edit `docker-compose.yml` — replace `agentpay2026` with something random.
Update `DATABASE_URL` in `.env` to match.

### 3. Start everything
```bash
docker compose up -d
```

Launches: PostgreSQL 16, Redis 7, API (port 8080), Bot.

### 4. Verify
```bash
docker compose ps                  # all services healthy?
curl http://localhost:8080/v1/health   # should return 200
```

### 5. Cloudflare Tunnel
```bash
cloudflared tunnel run agentpay
```
Or set up as systemd service for persistence.

### 6. Test
- Open https://leofundmybot.dev/docs — API docs should load
- Message @FundmyAIbot on Telegram — `/start` should respond

## Maintenance
- Logs: `docker compose logs -f api`
- DB backup: `docker compose exec db pg_dump -U agentpay agentpay > backup.sql`
- Update: `git pull && docker compose up -d --build`
