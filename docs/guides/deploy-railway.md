# Deploy AgentPay to Railway

[Railway](https://railway.app) is the easiest way to get AgentPay running in the cloud — no server setup, free tier available.

## Prerequisites

- [Railway account](https://railway.app) (free)
- [GitHub account](https://github.com) with a fork of [AgentPay](https://github.com/lfp22092002-ops/agentpay)
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))

## Step 1 — Fork the Repo

Fork `lfp22092002-ops/agentpay` to your own GitHub account. Railway will deploy from your fork.

## Step 2 — Create a New Railway Project

1. Go to [railway.app](https://railway.app) → **New Project**
2. Select **Deploy from GitHub repo**
3. Choose your forked `agentpay` repo
4. Railway auto-detects the `Dockerfile` — click **Deploy**

## Step 3 — Add PostgreSQL

1. In your Railway project, click **+ New** → **Database** → **PostgreSQL**
2. Railway provisions a Postgres instance and injects `DATABASE_URL` automatically

## Step 4 — Set Environment Variables

In your Railway service → **Variables**, add:

```env
BOT_TOKEN=your_telegram_bot_token
API_SECRET=your_random_secret_32_chars_min
ADMIN_TELEGRAM_ID=your_telegram_user_id
ENVIRONMENT=production
BASE_URL=https://your-railway-domain.up.railway.app
```

To generate a secure `API_SECRET`:
```bash
openssl rand -hex 32
```

> `DATABASE_URL` is injected automatically by Railway — do not set it manually.

## Step 5 — Run Migrations

In Railway → your service → **Shell** (or via Railway CLI):

```bash
alembic upgrade head
```

## Step 6 — Set a Custom Domain (Optional)

1. In Railway → your service → **Settings** → **Domains**
2. Add your custom domain (e.g. `agentpay.yourdomain.com`)
3. Point your DNS CNAME to the Railway-provided hostname
4. Railway handles TLS automatically

## Step 7 — Verify

Once deployed, check:

```bash
curl https://your-railway-domain.up.railway.app/v1/health
# Expected: {"status": "ok", "version": "0.1.0"}
```

Open your Telegram bot and send `/start` — it should respond.

## Scaling & Costs

| Plan | Price | RAM | What it covers |
|------|-------|-----|----------------|
| Hobby | $5/mo | 512MB | Perfect for personal use / testing |
| Pro | $20/mo | 8GB | Production with real users |

The Hobby plan is sufficient for early beta and low traffic.

## Troubleshooting

**Bot not responding**
- Check `BOT_TOKEN` is correct
- Ensure Railway service is running (green status)
- Check logs: Railway dashboard → **Logs**

**Database errors on startup**
- Run `alembic upgrade head` via Railway Shell
- Ensure `DATABASE_URL` is present in variables

**`/v1/health` returns 500**
- Most likely a missing env var — check all required variables are set

---

*For self-hosted Linux deployment, see [docs/guides/](./)*
*For Docker Compose local dev, see [DEPLOYMENT.md](../../DEPLOYMENT.md)*
