# Changelog

All notable changes to AgentPay will be documented in this file.

## [Unreleased]

### Fixed
- Replaced all `datetime.utcnow()` calls with `datetime.now(timezone.utc)` — eliminates Python 3.12+ deprecation warnings across entire codebase
- Fixed `test_auth_dev_mode` — patch BOT_TOKEN properly for dev mode testing
- Fixed `test_validation.py` — updated 20 broken imports from monolith `api.main` to modular `api.models`, `api.middleware`, `api.routes.identity`
- Fixed `x402_protocol.py` — restored `pay_to` variable removed during lint cleanup (F821)

### Added
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — Python 3.11/3.12/3.13 tests, ruff lint, TypeScript SDK build
- **Bot handler test suite** (`test_bot.py`) — 40 tests covering all 19 commands and callback handlers
- **Route integration tests** (`test_routes.py`) — 21 tests for API endpoints (spend, refund, transfer, chains, webhook, card, x402, balance)
- **SDK test suite** (`test_sdk.py`) — 26 tests for sync/async Python clients
- **Approval workflow tests** (`test_approvals.py`) — 12 tests
- **Encryption tests** (`test_encryption.py`) — 10 tests
- **Webhook tests** (`test_webhooks.py`) — 14 tests
- **Refund & transfer tests** (`test_refund_transfer.py`) — 7 tests
- **TypeScript SDK** (`sdk/ts/`) — full client mirroring Python SDK, zero runtime deps, ESM + CJS
- **OpenAPI spec export** (`openapi.json`) — 37 endpoints, 39KB
- **CONTRIBUTING.md** — dev setup, test instructions, PR workflow
- CI badge in README.md
- **Examples directory** (`examples/`) — basic_usage.py/ts, openai_agent.py, webhook_receiver.py
- **Admin endpoint tests** (`test_admin.py`) — 8 tests
- **Identity/directory tests** (`test_identity.py`) — 15 tests
- **Mini App endpoint tests** (`test_miniapp.py`) — 32 tests
- Total test count: **283 tests** (282 passing, 1 skipped)

### Changed
- Lint cleanup: 161 auto-fixes + manual fixes (unused imports, variables, whitespace, `is_()` comparisons)
- conftest.py: BOT_TOKEN set to valid aiogram format for test isolation

## [0.1.0] — 2026-02-22

### Added
- Initial release of AgentPay
- Telegram bot (`@FundmyAIbot`) with 19 commands
- REST API with 37+ endpoints (FastAPI)
- Fund agents via Telegram Stars
- Multi-chain USDC wallets (Base, Polygon, BNB Chain, Solana)
- Virtual Visa cards via Lithic
- Agent Identity System (KYA — Know Your Agent) with trust scoring
- Webhook system with HMAC-SHA256 signing
- Approval workflows for spends exceeding thresholds
- x402 protocol support (client + server + probe)
- Idempotency support for all write operations
- Rate limiting (per-IP and per-agent)
- Python SDK with sync + async clients
- MCP tool definitions for agent integration
- Mini App dashboard (5-tab, JWT auth)
- Landing page + documentation site
- CSV transaction export
- Agent-to-agent transfers and refunds
- Platform revenue tracking (2% fee, currently 0% for beta)
