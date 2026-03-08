# Contributing to AgentPay

Thanks for your interest in contributing to AgentPay! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/lfp22092002-ops/agentpay.git
cd agentpay
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests use SQLite in-memory — no external services required.

## Code Style

- Python 3.12+
- Use `datetime.now(timezone.utc)` instead of `datetime.utcnow()`
- Type hints everywhere
- Async-first (FastAPI + SQLAlchemy async)

## Pull Requests

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a PR with a clear description

## Reporting Issues

Open an issue on GitHub with:
- What you expected
- What happened instead
- Steps to reproduce

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
