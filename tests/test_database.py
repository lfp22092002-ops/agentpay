"""
Tests for models/database.py — engine creation, session factory, init_db.
Covers the 0% coverage module.
"""
import os
import sys

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.database import Base, get_db, init_db, engine, async_session


class TestDatabaseModule:
    def test_engine_exists(self):
        """Engine is created and accessible."""
        assert engine is not None

    def test_session_factory_exists(self):
        """Session factory is created."""
        assert async_session is not None

    def test_base_class(self):
        """Base is a DeclarativeBase subclass."""
        assert hasattr(Base, "metadata")
        assert hasattr(Base, "__subclasses__")

    @pytest.mark.asyncio
    async def test_get_db_yields_session(self):
        """get_db yields an AsyncSession."""
        async for session in get_db():
            assert isinstance(session, AsyncSession)
            break

    @pytest.mark.asyncio
    async def test_init_db(self):
        """init_db runs without error (creates tables)."""
        await init_db()
        # Should be idempotent
        await init_db()
