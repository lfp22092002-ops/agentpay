# Self-Hosting AgentPay

Deploy your own AgentPay instance — full control, no third-party dependency.

## Requirements

- Docker & Docker Compose (recommended) **or** Python 3.11+ & PostgreSQL 14+
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- A domain with HTTPS (Cloudflare Tunnel, Caddy, nginx, etc.)

---

## Option 1: Docker Compose (Recommended)

```bash
git clone https://github.com/lfp22092002-ops/agentpay.git
cd agentpay

# Configure environment
cp .env.example .env
```

Edit `.env`:
```ini
BOT_TOKEN=your_telegram_bot_token
API_SECRET=generate-a-64-char-random-string
ENVIRONMENT=production
```

Generate a secure API secret:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Start everything:
```bash
docker compose up -d
```

This starts:
- **PostgreSQL 16** — persistent data
- **Redis 7** — rate limiting & caching
- **AgentPay API** — port 8080
- **AgentPay Bot** — Telegram bot polling

Verify:
```bash
curl http://localhost:8080/v1/health
# {"status":"ok","service":"agentpay","version":"0.1.0"}
```

---

## Option 2: Manual Setup

### 1. Database

```bash
sudo apt install postgresql
sudo -u postgres createuser agentpay -P
sudo -u postgres createdb agentpay -O agentpay
```

### 2. Python Environment

```bash
git clone https://github.com/lfp22092002-ops/agentpay.git
cd agentpay

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration

```bash
cp .env.example .env
# Edit with your values:
# BOT_TOKEN, API_SECRET, DATABASE_URL
```

### 4. Database Migrations

```bash
alembic upgrade head
```

### 5. Run

In separate terminals (or use systemd/supervisor):

```bash
# API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080

# Bot
python -m bot.main
```

---

## HTTPS & Domain

AgentPay requires HTTPS for webhooks, Telegram Mini App, and the `.well-known/ai-plugin.json` manifest.

### Cloudflare Tunnel (easiest)

```bash
cloudflared tunnel create agentpay
cloudflared tunnel route dns agentpay your-domain.dev
cloudflared tunnel run --url http://localhost:8080 agentpay
```

### Caddy (auto-TLS)

```
your-domain.dev {
    reverse_proxy localhost:8080
}
```

### nginx + Let's Encrypt

```nginx
server {
    listen 443 ssl;
    server_name your-domain.dev;

    ssl_certificate /etc/letsencrypt/live/your-domain.dev/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.dev/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## systemd Services (production)

Create `/etc/systemd/system/agentpay-api.service`:

```ini
[Unit]
Description=AgentPay API
After=network.target postgresql.service

[Service]
Type=simple
User=agentpay
WorkingDirectory=/opt/agentpay
EnvironmentFile=/opt/agentpay/.env
ExecStart=/opt/agentpay/venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/agentpay-bot.service`:

```ini
[Unit]
Description=AgentPay Telegram Bot
After=network.target postgresql.service agentpay-api.service

[Service]
Type=simple
User=agentpay
WorkingDirectory=/opt/agentpay
EnvironmentFile=/opt/agentpay/.env
ExecStart=/opt/agentpay/venv/bin/python -m bot.main
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable agentpay-api agentpay-bot
sudo systemctl start agentpay-api agentpay-bot
```

---

## Wallet Encryption

Agent wallet private keys are encrypted at rest using Fernet with a PBKDF2-derived key from your `API_SECRET`. **If you lose your API_SECRET, wallet keys are unrecoverable.** Back it up securely.

---

## Monitoring

Health endpoint:
```bash
curl https://your-domain.dev/v1/health
```

Logs (systemd):
```bash
journalctl -u agentpay-api -f
journalctl -u agentpay-bot -f
```

Logs (Docker):
```bash
docker compose logs -f api
docker compose logs -f bot
```

---

## Updating

```bash
cd agentpay
git pull origin main

# Docker
docker compose build
docker compose up -d

# Manual
source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
sudo systemctl restart agentpay-api agentpay-bot
```
