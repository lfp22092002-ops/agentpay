# AgentPay — Production Deployment Guide

One-command deploy to any VPS (tested on Ubuntu 24.04).

## Prerequisites
- Ubuntu 24.04 server, 2GB+ RAM
- Domain pointing to server (e.g., `leofundmybot.dev`)
- SSL cert via Certbot

## One-Line Deploy
```bash
curl -fsSL https://raw.githubusercontent.com/lfp22092002-ops/agentpay/main/deploy/deploy.sh | bash -s -- \
  --domain leofundmybot.dev \
  --bot-token YOUR_BOT_TOKEN \
  --api-secret YOUR_API_SECRET \
  --telegram-admin-id 5360481016
```

## Manual Deploy

### 1. Clone & Setup
```bash
git clone https://github.com/lfp22092002-ops/agentpay.git
cd agentpay
pip install -e .
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env: BOT_TOKEN, API_SECRET, DOMAIN, etc.
```

### 3. Database
```bash
docker-compose up -d db redis
alembic upgrade head
```

### 4. Run Services
```bash
# API server
python -m api.main --host 0.0.0.0 --port 8080 &

# Bot
python -m bot.main &

# Cloudflare Tunnel (optional, for HTTPS)
cloudflared tunnel --url http://localhost:8080
```

### 5. Systemd (persistent)
```bash
cp deploy/agentpay.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now agentpay-api
```

## Environment Variables

| Variable | Required | Example |
|----------|----------|---------|
| BOT_TOKEN | ✅ | `7123456789:AAH...` |
| API_SECRET | ✅ | `super-secret-key-32chars!` |
| DATABASE_URL | ✅ | `postgresql+asyncpg://user:pass@localhost/db` |
| REDIS_URL | ❌ | `redis://localhost:6379/0` (in-memory cache) |
| DOMAIN | ❌ | `leofundmybot.dev` |
| ENVIRONMENT | ❌ | `production` |

## Post-Deploy Checklist
- [ ] `/v1/health` returns `{"status": "ok"}`
- [ ] Bot responds to `/start` in Telegram
- [ ] SSL cert valid (HTTPS redirects)
- [ ] Firewall: ports 80, 443 open
- [ ] Cloudflare tunnel running (if using CF)

## Troubleshooting
```bash
# Check logs
journalctl -u agentpay-api --tail -50

# Test endpoint
curl https://leofundmybot.dev/v1/health

# Bot check
curl https://api.telegram.org/botTOKEN/getMe
```
