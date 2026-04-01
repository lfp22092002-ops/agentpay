# AgentPay Launch Checklist

## Pre-Launch (G-dependent)

- [ ] Generate production `API_SECRET` (`openssl rand -hex 32`)
- [ ] Set real `BOT_TOKEN` in `.env`
- [ ] Cloudflare Tunnel: route `leofundmybot.dev` → `localhost:8080`
- [ ] DNS: verify `leofundmybot.dev` resolves correctly
- [ ] HTTPS: confirm tunnel provides TLS termination
- [ ] Lithic: apply for production API key (virtual cards)
- [ ] PyPI: publish `agentpay` SDK (`pip install agentpay`)

## Security (before public access)

- [ ] Change default Postgres password in `docker-compose.yml`
- [ ] Rotate any dev API keys/secrets from testing
- [ ] Verify rate limiting is active (`/v1/` endpoints)
- [ ] Confirm CORS origins are locked to `leofundmybot.dev`
- [ ] Test webhook HMAC verification with invalid signatures (should 401)
- [ ] Verify wallet encryption keys are backed up securely

## Infrastructure

- [ ] Docker Compose running on mini PC (see `docs/guides/agentpay-docker-deploy.md`)
- [ ] Postgres backups scheduled (daily `pg_dump`)
- [ ] Health endpoint monitored (`/v1/health`)
- [ ] Log rotation configured (Docker default or logrotate)

## Marketing / Launch

- [ ] Product Hunt listing prepared
- [ ] X (@autonomousaibot) announcement thread ready
- [ ] README has live URL and working badges
- [ ] Landing page at `leofundmybot.dev/` loads correctly
- [ ] API docs at `/docs` and `/redoc` accessible

## Post-Launch

- [ ] Monitor error rates first 24h
- [ ] Check Telegram bot responds to `/start`
- [ ] Verify first real transaction flow (deposit → spend → webhook)
- [ ] Collect user feedback, iterate

---
*Created: 2026-03-29*
