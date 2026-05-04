# Changelog

All notable changes to AgentPay will be documented in this file.

## [Unreleased]

### Added
- **Quick-start setup script**: `scripts/setup.sh` — auto `.env`, secrets, docker compose
- **Docker Compose dev mode**: `docker-compose.dev.yml` for hot-reload development
- **Examples**: FastAPI billing middleware, Express.js billing middleware (TS), MCP server with payment-gated tools
- **Makefile targets**: `test-all`, `ts-test`, `sdk-publish`
- **Healthcheck**: Added to API service in `docker-compose.yml`
- **Streamable HTTP MCP transport**: `mcp/server_http.py` — alternative to stdio for web-native integrations
- **Wallet route tests**: 19 tests covering spend, refund, transfer, chains, card, webhook, approvals, x402, wallet GET/all, send-usdc/native
- **Health endpoint tests**: Route-level test coverage for `/v1/health` and `/v1/health/detailed`
- **MCP server tests**: 19 handler tests + tool registry + error paths
- **Middleware tests**: Security headers, rate limiting, exception handler
- **Python sync SDK tests**: 21 unit tests covering all client methods, errors, retries
- **TypeScript SDK tests**: 24 unit tests covering all endpoints, error handling, retries
- **PR template**: `.github/PULL_REQUEST_TEMPLATE.md` for consistent contributions
- **llms-full.txt**: Comprehensive API docs for LLM discoverability (7.3KB)
- **Competitive landscape docs**: Razorpay, ReFiBuy, Rye, Affirm+Stripe, Stars deadline
- **Payee whitelists**: Added to landing page + llms.txt + getting-started guide
- **`GET /v1/health/detailed`**: New endpoint returning DB connectivity, latency, and uptime
- **`GET /v1/admin/agents`**: Paginated agent listing endpoint for ops/admin dashboards
- **Good First Issues table**: Added to `CONTRIBUTING.md` to attract open-source contributors
- **Railway deployment guide**: `docs/guides/deploy-railway.md` — one-click cloud setup
- **VPS deployment guide**: `docs/guides/deploy-vps.md` — DigitalOcean/Hetzner self-hosting
- **Product Hunt launch copy**: `docs/product-hunt-copy.md` — ready-to-submit listing
- **Admin agents coverage tests**: 4 tests for `GET /v1/admin/agents` (auth, pagination)
- **Payee rules coverage tests**: max_amount_usd list path + deny-only allowance logic
- **Auth rejection tests**: 401 coverage for balance, transactions, rotate-key, export endpoints

### Changed
- **Dockerfile**: Multi-stage build (smaller image, no gcc in runtime)
- **OpenAPI spec**: Added payee-rules endpoints (GET/POST/DELETE) — total 39 endpoints
- **CODEOWNERS**: Core payment + security paths require review
- **CONTRIBUTING.md**: Added `pip-audit` security scanning instructions + good first issues
- **README**: Added CI/license/Python badges + guides & tutorials section
- **llms.txt**: Added links to Railway and VPS deployment guides
- **Test count**: 626 tests (up from 621), 71% API coverage

### Security
- **requirements-lock.txt**: Pinned deps, fixed 15 CVEs (aiohttp, cryptography, pygments)

### Fixed
- Ruff lint errors: unused imports, E712 comparisons
- Alembic env import fix
- Analytics test documented as Postgres-only (SQLite `cast(Date)` incompatibility)

### CI
- Added `pip-audit` security scanning job to CI pipeline

### Docs
- MCP client usage example (`examples/mcp_client.py`)

## [0.1.1] — 2026-03-12

### Added
- **TypeScript SDK**: Full-featured client with 5 error classes (`sdk/ts/`)
- **TypeScript SDK Tests**: 29 tests covering all methods + error handling (vitest)
- **MCP Streamable HTTP**: `/mcp` endpoint with session management (`Mcp-Session-Id`)
- **MCP Session Lifecycle**: Create, validate, expire (TTL), delete sessions per spec
- **MCP Directory Submissions**: mcp.so (#782), Cline MCP Marketplace (#867) — submitted
- **MCP Discovery Artifacts**: `smithery.yaml`, LobeHub plugin config, `.well-known/mcp/server-card.json`
- **GitHub Issue Templates**: Bug report + feature request templates
- **CODE_OF_CONDUCT.md**: Contributor Covenant 2.1
- **11 MCP session tests** (`test_mcp_sessions.py`)
- **19 miniapp auth coverage tests** (`test_miniapp_auth_coverage.py`)

### Fixed
- MCP `protocolVersion` updated from `2024-11-05` to `2025-11-25` (latest spec)
- README + dev.to article: aligned SDK code examples with actual method signatures
- Python SDK README: `get_wallets()` → `get_wallet(chain=)`, `set_webhook()` → `register_webhook()`

### Stats
- **621 tests** (592 Python + 29 TypeScript), 2 skipped

## [0.1.0] — 2026-03-09

### Added
- **Agent Management**: Create, list, update, delete agents with per-agent API keys
- **Balance & Spending**: Deposit via Telegram Stars, spend via API, refund transactions
- **Multi-Chain USDC Wallets**: Base, Polygon, BNB Chain, Solana — auto-provisioned per agent
- **Agent Identity (KYA)**: Trust scores (0-100), public directory, verified badges, category filtering
- **Approval Workflows**: Auto-approve below threshold, Telegram notification above
- **Webhooks**: HMAC-SHA256 signed events for transactions, approvals, balance changes
- **Security**: SHA-256 hashed API keys (Stripe-style), Fernet+PBKDF2 encrypted wallet keys, CORS lockdown, rate limiting, security headers
- **x402 Protocol**: HTTP-native micropayment support (probe + pay endpoints)
- **Agent-to-Agent Transfers**: Move funds between agents instantly
- **CSV Export**: Download transaction history for accounting
- **Telegram Bot**: @FundmyAIbot — 19 commands for agent management
- **Mini App Dashboard**: 5-tab Telegram Mini App with JWT auth
- **Python SDK**: Sync + async clients with Pydantic v2 models (`sdk/agentpay/`)
- **MCP Tools**: 8 tool definitions for agent framework integration
- **Landing Page**: SEO-optimized at leofundmybot.dev with docs site
- **CI/CD**: GitHub Actions with Python 3.11/3.12/3.13 matrix, 504 tests, 88% coverage
- **Admin Panel**: Revenue tracking, withdrawal endpoints
- **Idempotent Operations**: Prevent duplicate transactions via idempotency keys
- **Key Rotation**: Rotate API keys without downtime

### Infrastructure
- Self-hosted on Ubuntu 24.04 (Ryzen 7 6800H, 32GB RAM)
- Cloudflare Tunnel for HTTPS
- PostgreSQL + Alembic migrations
- systemd services: agentpay-api, agentpay-bot, agentpay-tunnel
