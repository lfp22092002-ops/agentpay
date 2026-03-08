"""
Shared fixtures for AgentPay tests.

Uses SQLite in-memory for fast isolated tests — no PostgreSQL/Redis needed.
"""
import asyncio
import os
import sys
from decimal import Decimal
from datetime import datetime
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Override env before any project imports
os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"
os.environ["BOT_TOKEN"] = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz0123456789a"
os.environ["API_SECRET"] = "test-secret-key-for-tests"

# Build a test engine BEFORE importing models.database so we can swap it in
_test_engine = create_async_engine("sqlite+aiosqlite://", echo=False)
_test_session_factory = async_sessionmaker(_test_engine, class_=AsyncSession, expire_on_commit=False)

# Monkey-patch the module-level create_async_engine call
# We do this by pre-creating the module with our engine
import importlib
import types

# First, make sure config.settings is loadable (it reads env vars, which we set above)
import config.settings  # noqa
# Force token override (dotenv may have loaded a different value)
config.settings.BOT_TOKEN = os.environ["BOT_TOKEN"]

# Now create models.database module manually with our test engine
db_mod = types.ModuleType("models.database")
db_mod.__file__ = os.path.join(PROJECT_ROOT, "models", "database.py")
db_mod.engine = _test_engine
db_mod.async_session = _test_session_factory
db_mod.AsyncSession = AsyncSession


class Base(DeclarativeBase):
    pass


db_mod.Base = Base


async def _get_db():
    async with _test_session_factory() as session:
        yield session


async def _init_db():
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


db_mod.get_db = _get_db
db_mod.init_db = _init_db
db_mod.create_async_engine = create_async_engine
db_mod.async_sessionmaker = async_sessionmaker

# Register it in sys.modules BEFORE any other import tries to load it
sys.modules["models.database"] = db_mod

# NOW import the schema and wallet modules (they import from models.database)
from models.schema import (
    User, Agent, Wallet, Transaction,
    TransactionType, TransactionStatus, PaymentMethod,
    AgentIdentity, PlatformRevenue,
)
from core.wallet import hash_api_key, generate_api_key


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def engine():
    """Create a fresh in-memory SQLite engine per test."""
    eng = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def db(engine):
    """Provide a fresh async session per test."""
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def test_user(db: AsyncSession):
    """Create a test user."""
    user = User(
        telegram_id=12345678,
        username="testuser",
        first_name="Test",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_agent(db: AsyncSession, test_user: User):
    """Create a test agent with a known API key."""
    full_key = "ap_test_key_1234567890abcdef1234567890abcdef"
    key_hash = hash_api_key(full_key)
    agent = Agent(
        user_id=test_user.id,
        name="test-agent",
        api_key_hash=key_hash,
        api_key_prefix="ap_test_...",
        balance_usd=Decimal("100.0000"),
        daily_limit_usd=Decimal("50.0000"),
        tx_limit_usd=Decimal("25.0000"),
        auto_approve_usd=Decimal("10.0000"),
        is_active=True,
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    # Attach the raw key for test convenience
    agent._test_api_key = full_key
    return agent


@pytest_asyncio.fixture
async def test_agent_with_wallet(db: AsyncSession, test_agent: Agent):
    """Create a test agent with an associated wallet."""
    wallet = Wallet(
        agent_id=test_agent.id,
        wallet_type="internal",
    )
    db.add(wallet)
    await db.commit()
    return test_agent


@pytest_asyncio.fixture
async def second_user(db: AsyncSession):
    """Create a second test user."""
    user = User(
        telegram_id=99999999,
        username="seconduser",
        first_name="Second",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@pytest_asyncio.fixture
async def second_agent(db: AsyncSession, test_user: User):
    """Create a second agent belonging to the same user (for transfer tests)."""
    full_key = "ap_second_key_abcdef1234567890abcdef1234567890"
    key_hash = hash_api_key(full_key)
    agent = Agent(
        user_id=test_user.id,
        name="second-agent",
        api_key_hash=key_hash,
        api_key_prefix="ap_secon...",
        balance_usd=Decimal("50.0000"),
        daily_limit_usd=Decimal("50.0000"),
        tx_limit_usd=Decimal("25.0000"),
        is_active=True,
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    agent._test_api_key = full_key
    return agent


@pytest_asyncio.fixture
async def other_user_agent(db: AsyncSession, second_user: User):
    """Create an agent belonging to a different user."""
    full_key = "ap_other_key_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
    key_hash = hash_api_key(full_key)
    agent = Agent(
        user_id=second_user.id,
        name="other-agent",
        api_key_hash=key_hash,
        api_key_prefix="ap_other...",
        balance_usd=Decimal("200.0000"),
        daily_limit_usd=Decimal("100.0000"),
        tx_limit_usd=Decimal("50.0000"),
        is_active=True,
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    agent._test_api_key = full_key
    return agent
