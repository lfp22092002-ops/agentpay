.PHONY: help install dev test lint coverage run-api run-bot docker clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

dev:  ## Install dev + production dependencies
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov httpx ruff aiosqlite

test:  ## Run all tests
	python -m pytest tests/ -v --tb=short

test-quick:  ## Run tests (quiet, fast)
	python -m pytest tests/ -q --tb=line

lint:  ## Run ruff linter
	ruff check .

lint-fix:  ## Auto-fix lint issues
	ruff check . --fix

coverage:  ## Run tests with coverage report
	python -m pytest tests/ --cov=. --cov-report=term-missing --cov-report=html

run-api:  ## Start the API server
	python -m api.main

run-bot:  ## Start the Telegram bot
	python -m bot.main

docker:  ## Build and run with Docker Compose
	docker compose up --build -d

docker-down:  ## Stop Docker Compose
	docker compose down

sdk-build:  ## Build Python SDK
	cd sdk && python -m build

ts-build:  ## Build TypeScript SDK
	cd sdk/ts && npm install && npx tsup

clean:  ## Remove build artifacts
	rm -rf __pycache__ .pytest_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
