# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability in AgentPay, **please do not open a public issue.**

Email: **hello@leofundmybot.dev**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We aim to respond within 48 hours and will credit reporters in the changelog (unless you prefer anonymity).

## Security Architecture

### API Key Security
- API keys are hashed with **SHA-256** before storage (Stripe-style)
- Raw keys are shown **once** at creation and never stored
- Key rotation available via `/rotatekey` — old key invalidated immediately
- Keys prefixed with `ap_` for easy identification in logs

### Wallet Encryption
- Private keys encrypted with **Fernet** (AES-128-CBC + HMAC-SHA256)
- Key derivation via **PBKDF2** with per-wallet salts
- Encrypted keys stored in `data/wallets/` — never in database
- Decryption only at transaction signing time

### Transport & Access
- All traffic over **HTTPS** via Cloudflare Tunnel
- **CORS** locked to `leofundmybot.dev` and Telegram origins
- **Rate limiting** on all endpoints (slowapi)
- **Security headers**: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Strict-Transport-Security

### Spending Controls
- Per-agent **daily limits** and **per-transaction limits**
- **Auto-approve threshold** — spending above threshold requires Telegram approval
- **Idempotency keys** prevent duplicate charges
- All transactions logged with full audit trail

### Telegram Authentication
- Mini App auth uses Telegram's `initData` HMAC validation
- JWT tokens for session management (HS256, 24h expiry)
- Admin operations restricted to owner Telegram ID

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Current |

## Scope

The following are **in scope** for security reports:
- API authentication bypass
- Wallet key exposure
- Unauthorized fund transfers
- SQL injection / XSS in API or Mini App
- Rate limit bypass
- Webhook signature forgery

**Out of scope**:
- Telegram Bot API vulnerabilities (report to Telegram)
- Cloudflare infrastructure issues
- Social engineering attacks
- Self-hosted instances with modified code
