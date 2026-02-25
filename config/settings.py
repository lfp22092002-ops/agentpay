import os
from dotenv import load_dotenv

load_dotenv()

# Bot
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://agentpay:agentpay2026@localhost:5432/agentpay")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# API
API_SECRET = os.getenv("API_SECRET", "agentpay-secret-change-me")
if API_SECRET == "agentpay-secret-change-me" and ENVIRONMENT == "production":
    import warnings
    warnings.warn("⚠️ CRITICAL: Using default API_SECRET in production! Set API_SECRET in .env", stacklevel=2)
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))

# Stars rate: 1 Star ≈ $0.013 for developers
STARS_TO_USD_RATE = 0.013

# Platform fee
PLATFORM_FEE_PERCENT = 2.0  # 2% on transactions

# Spending limits
DEFAULT_DAILY_LIMIT_USD = 50.0
DEFAULT_TRANSACTION_LIMIT_USD = 25.0
MAX_DAILY_LIMIT_USD = 10000.0
