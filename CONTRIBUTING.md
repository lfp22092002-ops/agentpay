# Contributing to AgentPay

Thanks for your interest in contributing! AgentPay is an open-source payment layer for AI agents, and we welcome improvements of all kinds.

## Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL 14+ (production) or SQLite (tests)
- Node.js 18+ (TypeScript SDK only)

### Setup

```bash
git clone https://github.com/lfp22092002-ops/agentpay.git
cd agentpay

# Create virtualenv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy env template
cp .env.example .env
# Edit .env with your Telegram bot token and DB URL
```

### Running Tests

Tests use SQLite in-memory — no external services needed:

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=api --cov=bot --cov=core --cov=models --cov=providers --cov-report=term-missing

# Single file
pytest tests/test_api.py -v
```

### Running Locally

```bash
# Start the API server
python -m api.main

# Start the Telegram bot (separate terminal)
python -m bot.main
```

## Project Structure

```
agentpay/
├── api/                 # FastAPI REST API
│   ├── main.py          # App factory, middleware
│   ├── models.py        # Pydantic request/response models
│   ├── middleware.py     # Rate limiting, auth
│   ├── dependencies.py  # Shared FastAPI dependencies
│   └── routes/          # Endpoint modules
│       ├── agents.py    # Balance, transactions, key rotation
│       ├── wallets.py   # Spend, refund, transfer, chains, webhooks
│       ├── identity.py  # Agent identity & trust scores
│       ├── miniapp.py   # Telegram Mini App endpoints
│       ├── admin.py     # Revenue & withdrawal (admin only)
│       └── health.py    # Health check
├── bot/                 # Telegram bot (aiogram)
│   └── main.py          # All bot commands and handlers
├── core/                # Business logic
│   ├── wallet.py        # User/agent CRUD, spend, refund, transfer
│   ├── approvals.py     # Human-in-the-loop approval workflow
│   └── webhooks.py      # Webhook delivery with retries
├── models/              # Database layer
│   ├── database.py      # SQLAlchemy async engine
│   └── schema.py        # ORM models (User, Agent, Transaction, etc.)
├── providers/           # External integrations
│   ├── coinbase_wallet.py  # Coinbase wallet provider
│   ├── lithic_card.py      # Virtual card (Lithic API)
│   ├── local_wallet.py     # Self-hosted EVM wallets
│   ├── solana_wallet.py    # Solana wallet support
│   ├── telegram_stars.py   # Stars pricing
│   └── x402_protocol.py    # x402 payment protocol
├── sdk/                 # Client SDKs
│   ├── agentpay/        # Python SDK (pip install agentpay)
│   └── ts/              # TypeScript SDK (npm)
├── mcp/                 # MCP tool definitions
├── tests/               # 250+ tests
└── alembic/             # Database migrations
```

## Code Style

- **Linter**: `ruff` with rules E, F, W (ignore E501 for line length, E402 for test files)
- **Formatter**: Standard Python formatting, no black/yapf enforced
- **Type hints**: Use them where practical, especially in public APIs
- **Docstrings**: Required for public functions and classes

```bash
# Check lint
ruff check . --select E,F,W --ignore E501

# Auto-fix
ruff check . --select E,F,W --ignore E501 --fix

# Security audit (also runs in CI)
pip install pip-audit
pip-audit --strict
```

## Writing Tests

Tests live in `tests/` and use pytest + pytest-asyncio. Key patterns:

- **conftest.py**: Sets up in-memory SQLite, creates test fixtures (users, agents, wallets)
- **Mocking**: Use `unittest.mock.patch` for external services (Telegram, Lithic, blockchain RPCs)
- **Async**: All DB-touching tests are `@pytest.mark.asyncio`

```python
class TestMyFeature:
    @pytest.mark.asyncio
    async def test_something(self, db: AsyncSession, test_agent):
        # db and test_agent come from conftest fixtures
        result = await my_function(db, test_agent)
        assert result is not None
```

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass: `pytest tests/ -v`
4. Ensure lint passes: `ruff check . --select E,F,W --ignore E501`
5. Keep commits atomic and descriptive
6. Open a PR with a clear description of what and why

## Areas to Contribute

- **More test coverage**: `providers/` and `api/routes/miniapp.py` need more tests
- **Documentation**: API guides, integration examples, deployment docs
- **SDK improvements**: Error handling, retry logic, more examples
- **New payment providers**: Stripe, LemonSqueezy, other crypto chains
- **Security**: Audit, hardening, pen testing

## License

MIT — see [LICENSE](LICENSE).
