# AgentPay Security Architecture

> How AgentPay protects AI agent wallets, transactions, and credentials.

## Overview

AgentPay provides payment infrastructure for autonomous AI agents. Unlike human users, agents operate 24/7 without supervision, making security architecture fundamentally different from traditional fintech. This document describes our layered security model.

## Threat Model

### Agent-Specific Threats
1. **Prompt Injection** — Adversary manipulates agent context to trigger unauthorized payments
2. **Credential Exfiltration** — Agent leaks API key through tool output or logs
3. **Budget Exhaustion** — Compromised agent drains wallet via rapid small transactions
4. **Replay Attacks** — Intercepted transaction requests replayed to double-spend
5. **Webhook Spoofing** — Forged webhook events trick consuming systems

### Infrastructure Threats
6. **API Key Compromise** — Stolen keys used for unauthorized access
7. **Wallet Key Extraction** — Private keys accessed from database
8. **Man-in-the-Middle** — Traffic interception between agent and API

## Security Layers

### Layer 1: Authentication — API Keys (Stripe-style)

**Design:**
- Keys prefixed `ap_` + 40 random characters
- Only shown once at creation — never stored in plaintext
- Stored as SHA-256 hashes in database
- Key prefix stored separately for identification

**Why this matters for agents:**
Unlike human users who can use OAuth flows, agents need static credentials. Our hash-only storage means even a full database breach doesn't expose usable keys. Agents rotate keys via API without downtime.

```
Agent sends: X-API-Key: ap_abc123...
Server computes: SHA-256(ap_abc123...) → matches stored hash
```

### Layer 2: Spending Controls — Defense in Depth

**Three independent limits:**

| Control | Purpose | Default |
|---|---|---|
| `tx_limit_usd` | Max per transaction | $25 |
| `daily_limit_usd` | Max per 24h period | $50 |
| `balance` | Hard floor — can't spend more than funded | $0 |

**Approval Workflows:**
- Transactions exceeding configurable threshold require human approval via Telegram
- Approval requests include: amount, description, agent name, remaining budget
- Timeout: unapproved transactions auto-reject after 5 minutes
- All approvals logged with timestamp and approver ID

**Why three limits:** A compromised agent can't drain funds with many small transactions (daily limit), can't make one large transaction (tx limit), and can't spend money that doesn't exist (balance). Each limit operates independently.

### Layer 3: Wallet Encryption — At Rest

**Multi-chain wallet keys encrypted using:**
- Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256)
- Key derived via PBKDF2 with 100,000 iterations
- Per-deployment encryption key (not hardcoded)
- Keys decrypted only at transaction signing time, never held in memory

**Chain coverage:** Base, Polygon, BNB Chain, Solana — each with independent encrypted keypair.

### Layer 4: Transport Security

- **HTTPS only** — enforced via Cloudflare Tunnel (named tunnel, permanent)
- **CORS lockdown** — origin whitelist, no wildcard
- **Security headers** — HSTS, X-Content-Type-Options, X-Frame-Options, CSP
- **Rate limiting** — per-endpoint limits (e.g., 10/min for writes, 60/min for reads)

### Layer 5: Webhook Integrity — HMAC-SHA256

**Outbound webhooks signed with:**
```
X-AgentPay-Signature: sha256=HMAC(webhook_secret, request_body)
```

**Verification flow:**
1. Agent registers webhook URL + receives secret
2. AgentPay signs every event payload with HMAC-SHA256
3. Consumer verifies signature before processing
4. Replay protection via unique event IDs + timestamp

### Layer 6: Idempotency — Replay Prevention

- Every transaction accepts optional `idempotency_key`
- Duplicate requests with same key return original result (no double-spend)
- Keys expire after 24 hours
- Critical for agents that may retry on network errors

### Layer 7: Agent Identity (KYA — Know Your Agent)

**Trust Score (0-100) based on:**
- Account age (0-15 pts)
- Transaction history (0-25 pts)
- Transaction volume (0-25 pts)
- Profile completeness (0-15 pts)
- Verified status (0-20 pts)

**ERC-8004 Compatibility:**
- Registration endpoint serves agent identity in ERC-8004 Trustless Agents format
- `.well-known/erc-8004.json` discovery file
- Maps trust scores to on-chain Identity + Reputation registries
- Enables cross-platform agent discovery and trust verification

## x402 Protocol Security

AgentPay implements x402 HTTP-native micropayments with additional safeguards:

- **Probe before pay** — Agent checks price/terms before committing
- **Per-request authorization** — Each x402 payment is individually authorized against spending limits
- **Stablecoin only** — USDC payments (no volatile assets)
- **On-chain verification** — Transaction receipts verifiable on Base/Polygon/BNB/Solana

## What We Don't Do

Transparency about limitations:

- **No PCI DSS** — We don't handle card numbers (crypto + Stars only)
- **No KYC** — Agents can't provide ID; we use KYA (behavioral trust) instead
- **No insurance** — Self-hosted deployments are self-secured
- **No multisig** — Single-key wallets (simplicity tradeoff; multisig planned for Phase 3)
- **No HSM** — Software key management (cost tradeoff for indie devs)

## Incident Response

1. **Key rotation** — Single API call, zero downtime
2. **Agent deactivation** — Instant freeze via API or Telegram bot
3. **Transaction reversal** — Refund endpoint for completed transactions
4. **Audit trail** — Full transaction history with CSV export
5. **Webhook alerts** — Real-time notifications for suspicious activity

## Recommendations for Agent Developers

1. **Set conservative limits** — Start with $5/tx, $25/day; increase based on trust
2. **Enable approval workflows** — Human-in-the-loop for high-value transactions
3. **Rotate keys regularly** — Monthly rotation recommended
4. **Monitor webhooks** — Set up alerts for unusual spending patterns
5. **Use idempotency keys** — Prevent accidental double-spends on retries
6. **Isolate agent keys** — One API key per agent, never shared across agents

## Architecture Diagram

```
┌─────────────┐     HTTPS      ┌──────────────────┐
│  AI Agent   │ ──────────────▶│   AgentPay API   │
│ (Claude,    │  X-API-Key     │                  │
│  GPT, etc.) │  + HMAC        │  ┌────────────┐  │
└─────────────┘                │  │ Rate Limit │  │
                               │  │ Auth Check │  │
                               │  │ Spend Ctrl │  │
                               │  └─────┬──────┘  │
                               │        │         │
                               │  ┌─────▼──────┐  │
                               │  │ Encrypted  │  │
                               │  │ Wallet Keys│  │
                               │  └─────┬──────┘  │
                               │        │         │
                               └────────┼─────────┘
                                        │
                          ┌─────────────┼─────────────┐
                          │             │             │
                     ┌────▼───┐   ┌────▼───┐   ┌────▼───┐
                     │  Base  │   │Polygon │   │  BNB   │
                     │  USDC  │   │  USDC  │   │  USDC  │
                     └────────┘   └────────┘   └────────┘
```

---

*Last updated: March 15, 2026*
*Version: 1.0*
*Contact: hello@leofundmybot.dev*
