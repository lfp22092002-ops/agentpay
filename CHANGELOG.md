# Changelog

All notable changes to AgentPay will be documented in this file.

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
