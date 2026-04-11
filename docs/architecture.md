# AgentPay Architecture

## System Overview

```mermaid
graph TB
    subgraph "User Layer"
        TG[Telegram Bot<br/>@FundmyAIbot]
        MINI[Mini App<br/>Dashboard]
    end

    subgraph "Agent Layer"
        SDK_PY[Python SDK]
        SDK_TS[TypeScript SDK]
        MCP[MCP Server]
        X402[x402 Protocol]
    end

    subgraph "Core API"
        API[FastAPI Server<br/>Port 8080]
        AUTH[Auth Middleware<br/>SHA-256 API Keys]
        RATE[Rate Limiter]
        WH[Webhook Dispatcher<br/>HMAC-SHA256]
    end

    subgraph "Business Logic"
        WALLET[Wallet Engine<br/>Spend / Refund / Transfer]
        APPROVE[Approval Workflows]
        BUDGET[Budget & Limits]
        IDEMP[Idempotency Layer]
    end

    subgraph "Payment Rails"
        STARS[Telegram Stars]
        BASE[Base USDC]
        POLY[Polygon USDC]
        BNB[BNB Chain]
        SOL[Solana]
        CARD[Virtual Visa<br/>via Lithic]
    end

    subgraph "Data"
        PG[(PostgreSQL)]
    end

    TG --> API
    MINI --> API
    SDK_PY --> API
    SDK_TS --> API
    MCP --> API
    X402 --> API

    API --> AUTH --> RATE
    RATE --> WALLET
    WALLET --> APPROVE
    WALLET --> BUDGET
    WALLET --> IDEMP
    WALLET --> PG
    WALLET --> WH

    WALLET --> STARS
    WALLET --> BASE
    WALLET --> POLY
    WALLET --> BNB
    WALLET --> SOL
    WALLET --> CARD
```

## Request Flow

```mermaid
sequenceDiagram
    participant Agent
    participant SDK
    participant API
    participant Auth
    participant Wallet
    participant DB
    participant Webhook

    Agent->>SDK: spend(amount, description)
    SDK->>API: POST /v1/spend
    API->>Auth: Verify API key (SHA-256)
    Auth->>API: ✓ Authenticated
    API->>Wallet: Process spend
    Wallet->>DB: Check balance + budget
    DB-->>Wallet: OK
    Wallet->>DB: Debit + record tx
    Wallet-->>API: Transaction result
    API-->>SDK: 200 OK + tx receipt
    Wallet--)Webhook: POST callback (HMAC signed)
```

## Component Responsibilities

| Component | Role |
|-----------|------|
| **Telegram Bot** | User-facing: create agents, fund wallets, set rules, view history |
| **Mini App** | 5-tab dashboard: overview, agents, transactions, wallets, settings |
| **FastAPI API** | 24+ REST endpoints, OpenAPI docs at `/docs` |
| **Python SDK** | Sync + async client, retries, typed responses |
| **TypeScript SDK** | Full client, webhook verification, retry with backoff |
| **MCP Server** | Tool definitions for AI agent frameworks (stdio + HTTP) |
| **x402 Protocol** | HTTP 402-based pay-per-request for web services |
| **Wallet Engine** | Core ledger: spend, refund, transfer, multi-chain |
| **Approval Workflows** | Human-in-the-loop for high-value transactions |
| **PostgreSQL** | Persistent storage: agents, wallets, transactions, keys |

## Deployment

```
Cloudflare Tunnel (leofundmybot.dev)
  └─► API (port 8080)
       ├─► /v1/*    → REST API
       ├─► /app/*   → Mini App (static)
       ├─► /        → Landing page
       └─► /docs    → OpenAPI docs
```

## Security Model

- **API Keys**: SHA-256 hashed at rest, never stored plaintext
- **Wallet Encryption**: Fernet + PBKDF2 for private keys
- **Webhooks**: HMAC-SHA256 signed, timing-safe verification
- **Rate Limiting**: Per-key and global limits
- **CORS**: Locked to allowed origins
- **Idempotency**: Duplicate request protection via idempotency keys

See [Security Architecture](security-architecture.md) for full details.
