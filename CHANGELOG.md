# Changelog

All notable changes to AgentPay will be documented in this file.

## [Unreleased]

### Fixed
- Replaced all `datetime.utcnow()` calls with `datetime.now(timezone.utc)` — eliminates Python 3.12+ deprecation warnings across entire codebase (models, API routes, bot, webhooks, tests)

### Added
- Approval workflow test suite (`test_approvals.py`) — 12 tests covering creation, resolution, timeout, and querying
- Encryption test suite (`test_encryption.py`) — 10 tests covering salt management, encrypt/decrypt roundtrips, cross-secret isolation
- Webhook test suite (`test_webhooks.py`) — 14 tests covering signing, event building, delivery, filtering, and Telegram notifications
- Refund & transfer test suite (`test_refund_transfer.py`) — 6 tests covering success paths and edge cases
- SDK test suite (`test_sdk.py`) — 26 tests covering sync/async clients, all endpoints, and error handling
- Total test count: **150 passing** (up from 65)

## [0.1.0] — 2026-02-22

### Added
- Initial release of AgentPay
- Telegram bot (`@FundmyAIbot`) with 19 commands
- REST API with 37+ endpoints (FastAPI)
- Fund agents via Telegram Stars
- Multi-chain USDC wallets (Base, Polygon, BNB Chain, Solana)
- Agent Identity System (KYA — Know Your Agent) with trust scoring
- Webhook system with HMAC-SHA256 signing
- Approval workflows for spends exceeding thresholds
- Idempotency support for all write operations
- Rate limiting (per-IP and per-agent)
- Security: SHA-256 hashed API keys, Fernet+PBKDF2 wallet encryption, CORS lockdown
- Python SDK (sync + async clients)
- MCP tool definitions for agent integration
- x402 protocol support (client + server + probe)
- Telegram Mini App (5-tab dashboard)
- Landing page with SEO optimization
- API docs site
- CSV transaction export
- Cloudflare Tunnel deployment
