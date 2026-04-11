# AgentPay v0.1.0 — Release Checklist

Pre-launch checklist. Each item has an owner: **G** (needs human action) or **Bot** (automated/done).

## Infrastructure

- [ ] **G** — Get Lithic production API key (currently sandbox)
- [ ] **G** — Verify `leofundmybot.dev` DNS + Cloudflare tunnel active
- [ ] **Bot** ✅ — Docker Compose + healthchecks configured
- [ ] **Bot** ✅ — HTTPS via Cloudflare tunnel
- [ ] **G** — Start services: `docker compose up -d` on Mini PC

## Code & CI

- [ ] **Bot** ✅ — CI pipeline (`.github/workflows/ci.yml`)
- [ ] **Bot** ✅ — Publish workflow (`.github/workflows/publish.yml`)
- [ ] **Bot** ✅ — 614+ tests passing
- [ ] **G** — Tag `v0.1.0` and push: `git tag v0.1.0 && git push --tags`
- [ ] **G** — Set PyPI + npm tokens as GitHub secrets for publish workflow

## SDKs

- [ ] **Bot** ✅ — Python SDK (`sdk/agentpay/`) — sync + async, retries, typed
- [ ] **Bot** ✅ — TypeScript SDK (`sdk/ts/`) v0.1.1 — retries, webhook verify, typed
- [ ] **G** — Publish Python SDK: `cd sdk && pip install build twine && python -m build && twine upload dist/*`
- [ ] **G** — Publish TS SDK: `cd sdk/ts && npm publish`

## Telegram Bot

- [ ] **G** — Confirm @FundmyAIbot token in `.env` is production token
- [ ] **G** — Set bot commands via BotFather (19 commands)
- [ ] **Bot** ✅ — Bot code ready (`bot/`)

## Marketing

- [ ] **Bot** ✅ — Product Hunt draft (`docs/product-hunt-launch.md`)
- [ ] **Bot** ✅ — Landing page (`landing/`)
- [ ] **Bot** ✅ — llms.txt + llms-install.md for AI discoverability
- [ ] **G** — Submit to Product Hunt
- [ ] **G** — Post on Hacker News (Show HN)
- [ ] **G** — Tweet launch from @autonomousaibot

## Post-Launch

- [ ] Monitor error logs first 48h
- [ ] Respond to GitHub issues within 24h
- [ ] Track first 10 users, gather feedback
- [ ] Iterate on SDK based on real usage

---

*Created: 2026-04-11 | Status: Waiting on G for infrastructure tokens*
