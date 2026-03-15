# AgentPay Security Model

## Overview

AgentPay is a payment infrastructure layer for autonomous AI agents. This document describes the security model, threat landscape, and mitigations implemented in the AgentPay platform.

## Threat Model

### Actors
1. **Agent Owner** — creates and funds agents, sets spending rules
2. **Agent** — autonomous software that holds API keys and makes transactions
3. **External Service** — receives payments from agents (via API or x402)
4. **Attacker** — attempts to steal funds, impersonate agents, or manipulate transactions

### Attack Surface
- API endpoints (REST, MCP, x402)
- API key storage and transmission
- Wallet private keys
- Webhook delivery
- Telegram bot interactions
- Agent-to-agent transfers

## Authentication & Authorization

### API Key Security
- **Stripe-style key format**: `ap_` prefix + 40-character random string
- **One-way hashing**: API keys are SHA-256 hashed before database storage
- **Show-once policy**: Full key displayed only at creation time, never again
- **Prefix storage**: Only `ap_XXXX...` prefix stored for identification
- **Key rotation**: Agents can rotate keys via `/v1/agent/rotate-key` — old key invalidated immediately

### Rate Limiting
- Per-endpoint rate limits via SlowAPI (token bucket algorithm)
- Identity endpoints: 10-30 requests/minute
- Transaction endpoints: 60 requests/minute
- Directory (public): 60 requests/minute
- MCP: session-scoped limits

## Wallet Security

### Private Key Protection
- **Encryption**: Fernet symmetric encryption (AES-128-CBC)
- **Key derivation**: PBKDF2-HMAC-SHA256 with 100,000 iterations
- **Per-deployment salt**: Unique salt per AgentPay installation
- **At-rest encryption**: Private keys never stored in plaintext
- **No key export**: API does not expose private keys to agents or owners

### Multi-Chain Architecture
- Separate wallet per chain (Base, Polygon, BNB Chain, Solana)
- Auto-provisioned on agent creation
- Chain-specific address derivation

## Transaction Security

### Budget Governance
- **Per-transaction limit** (`tx_limit_usd`): Maximum single spend
- **Daily limit** (`daily_limit_usd`): Rolling 24-hour spending cap
- **Balance checks**: Real-time balance verification before every transaction
- **Idempotency keys**: Prevent duplicate transactions from retries

### Approval Workflows
- Transactions exceeding auto-approve threshold require human approval
- Approval request sent via Telegram push notification
- Time-limited approval windows
- Explicit approve/deny actions (no auto-approve on timeout)

### Transaction Audit Trail
- Every transaction logged with: timestamp, amount, description, status, idempotency key
- CSV export for accounting and compliance
- No transaction deletion (append-only log)

## Webhook Security

### HMAC-SHA256 Signatures
- Every webhook delivery signed with per-agent webhook secret
- Signature in `X-AgentPay-Signature` header
- Payload: `timestamp.body` to prevent replay attacks
- Recipients MUST verify signatures before processing

### Delivery
- HTTPS-only webhook endpoints
- Retry with exponential backoff on failure
- Delivery status tracking

## x402 Protocol Security

### Payment Flow
1. Agent sends HTTP request to x402-gated resource
2. Server responds with `402 Payment Required` + payment requirements
3. Agent evaluates cost against budget rules
4. If within limits, agent signs and sends payment
5. Server verifies payment on-chain, delivers resource

### Safeguards
- Budget rules apply to x402 payments (same limits as API spend)
- Agent cannot be tricked into overpaying — price checked against requirements
- On-chain settlement provides payment finality and auditability

## Agent Identity (KYA — Know Your Agent)

### Trust Score (0-100)
Composite score based on:
- **Account age** (0-15 pts): Longer-lived agents score higher
- **Transaction count** (0-25 pts): More completed transactions = more trust
- **Volume** (0-25 pts): Higher cumulative volume indicates established agent
- **Profile completeness** (0-15 pts): Name, description, homepage, logo
- **Verified badge** (0-20 pts): Manual verification by AgentPay team

### ERC-8004 Compatibility
- Agent registration files served in ERC-8004 Trustless Agents format
- Enables cross-platform agent discovery and trust verification
- Trust scores mapped to ERC-8004 Reputation Registry signals

## Network Security

### Transport
- HTTPS enforced (TLS 1.2+) via Cloudflare Tunnel
- CORS lockdown: only allowed origins can call API
- Security headers: `X-Content-Type-Options`, `X-Frame-Options`, `Strict-Transport-Security`
- No sensitive data in URL parameters

### Infrastructure
- PostgreSQL with encrypted connections
- Systemd service isolation
- Non-root process execution
- Minimal attack surface (no unnecessary ports exposed)

## Known Limitations & Mitigations

| Risk | Status | Mitigation |
|---|---|---|
| Prompt injection causing agent to overspend | Mitigated | Budget rules enforced server-side, not in agent logic |
| API key leakage in agent logs | Partially mitigated | Key rotation available; recommend env vars over hardcoding |
| Wallet key compromise via server breach | Mitigated | Fernet encryption + PBKDF2 derivation |
| Replay attacks on webhooks | Mitigated | Timestamp in HMAC payload |
| DDoS on API | Mitigated | Rate limiting + Cloudflare protection |
| Agent impersonation | Mitigated | SHA-256 hashed keys, no key reuse |
| Unauthorized fund transfer | Mitigated | Agent can only transfer to agents owned by same user |

## Responsible Disclosure

Security issues should be reported via:
- Email: security@leofundmybot.dev
- GitHub: See `SECURITY.md` in the repository

We commit to acknowledging reports within 48 hours and providing a fix timeline within 7 days.

---

*Last updated: March 15, 2026*
*AgentPay v0.1.1*
