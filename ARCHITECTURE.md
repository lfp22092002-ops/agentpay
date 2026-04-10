# Architecture

High-level overview of AgentPay's codebase for contributors and integrators.

## System Diagram

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Telegram    │────▶│   Bot        │────▶│                  │
│  Users       │     │  (aiogram)   │     │                  │
└─────────────┘     └──────────────┘     │                  │
                                          │   PostgreSQL     │
┌─────────────┐     ┌──────────────┐     │   (wallets,      │
│  AI Agents   │────▶│   REST API   │────▶│    txns,         │
│  (SDKs/MCP) │     │  (FastAPI)   │     │    agents)       │
└─────────────┘     └──────────────┘     │                  │
                                          │                  │
┌─────────────┐     ┌──────────────┐     │                  │
│  Mini App    │────▶│   /app/      │────▶│                  │
│  (Web UI)    │     │  (static)    │     └──────────────────┘
└─────────────┘     └──────────────┘
                           │
                    ┌──────┴──────┐
                    │  Providers   │
                    ├─────────────┤
                    │ Coinbase    │  EVM wallets (Base, Polygon, BNB)
                    │ Solana      │  SOL/SPL wallets
                    │ Lithic      │  Virtual Visa cards
                    │ Telegram ★  │  Stars payments
                    │ x402        │  HTTP 402 protocol
                    └─────────────┘
```

## Directory Structure

| Directory      | Purpose                                             |
|---------------|-----------------------------------------------------|
| `api/`        | FastAPI application — routes, middleware, models     |
| `api/routes/` | Endpoint groups: agents, wallets, admin, health, etc |
| `bot/`        | Telegram bot (aiogram) — 19 commands                |
| `core/`       | Business logic: encryption, approvals, webhooks      |
| `models/`     | SQLAlchemy models + Pydantic schemas                |
| `providers/`  | Payment backends (Coinbase, Solana, Lithic, Stars, x402) |
| `mcp/`        | MCP server (stdio + Streamable HTTP)                |
| `integrations/` | Framework adapters (LangChain, CrewAI, ADK, etc) |
| `sdk/`        | Python SDK (`pip install agentpay`)                 |
| `sdk-ts/`     | TypeScript SDK (`npm install agentpay`)             |
| `miniapp/`    | Telegram Mini App (5-tab web dashboard)             |
| `landing/`    | Landing page served at `/`                          |
| `examples/`   | 24 integration examples across frameworks           |
| `tests/`      | 614+ pytest tests                                   |
| `alembic/`    | Database migrations                                 |
| `deploy/`     | Deployment configs                                  |
| `scripts/`    | Setup, utility scripts                              |

## Key Design Decisions

### Multi-chain wallets
Each agent gets isolated wallets per chain (Base, Polygon, BNB, Solana). Wallet private keys are encrypted at rest using Fernet + PBKDF2. See `core/encryption.py`.

### Provider abstraction
Payment providers implement a common interface in `providers/`. Adding a new chain means adding one provider file — no changes to API routes.

### API authentication
Agents authenticate via SHA-256 hashed API keys. Keys are generated per-agent and stored hashed in the database. See `api/dependencies.py`.

### Webhook delivery
Transaction events are delivered via webhooks with HMAC-SHA256 signatures. Idempotency keys prevent duplicate processing. See `core/webhooks.py`.

### MCP integration
The MCP server (`mcp/server.py`) exposes AgentPay operations as tools, allowing LLM agents to make payments via the Model Context Protocol without custom HTTP code.

### Revenue model
2% transaction fee on spends. The fee is calculated and recorded per transaction in the database.

## API Overview

39 endpoints across these groups:

- **Agents** — CRUD, API key management
- **Wallets** — deposit, spend, refund, transfer, balance
- **Admin** — revenue, stats
- **Payee Rules** — whitelist management
- **Identity** — ERC-8004 on-chain identity
- **Health** — liveness/readiness checks
- **MCP** — Streamable HTTP transport
- **Mini App** — JWT-authenticated dashboard endpoints

Full OpenAPI docs available at `/docs` (Swagger) and `/redoc`.

## Running Locally

```bash
# Quick start
./scripts/setup.sh          # generates .env, secrets
docker compose up -d         # API + Bot + Postgres

# Development (hot-reload)
docker compose -f docker-compose.dev.yml up

# Tests
TESTING=1 pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and [DEPLOYMENT.md](DEPLOYMENT.md) for production setup.
