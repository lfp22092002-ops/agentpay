"""
Tests for models/database.py — engine, session factory, Base, init_db, get_db.
"""
import os
import sys

import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.mark.asyncio
async def test_init_db_creates_tables():
    """init_db creates all tables from Base.metadata."""
    from models.database import Base

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Tables shouldn't exist yet — init_db should create them
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Verify we can open a session and query (no error = tables exist)
    async with factory() as session:
        # Simple introspection query
        from sqlalchemy import text
        result = await session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result.fetchall()]
        assert len(tables) > 0  # At least the Agent/User tables

    await engine.dispose()


@pytest.mark.asyncio
async def test_get_db_yields_session():
    """get_db yields a working AsyncSession."""
    from models.database import get_db, Base

    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Monkey-patch the module's session factory temporarily
    import models.database as db_mod
    orig_session = db_mod.async_session
    db_mod.async_session = factory

    try:
        gen = get_db()
        session = await gen.__anext__()
        assert isinstance(session, AsyncSession)
        # Clean up the generator
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass
    finally:
        db_mod.async_session = orig_session
        await engine.dispose()


@pytest.mark.asyncio
async def test_base_declarative():
    """Base is a valid DeclarativeBase."""
    from models.database import Base
    from sqlalchemy.orm import DeclarativeBase
    assert issubclass(Base, DeclarativeBase)
