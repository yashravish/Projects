"""Async SQLAlchemy engine + session factory.

`get_session()` is the FastAPI dependency that yields a session per request.
The engine is created lazily so importing this module does not require a
running database (important for unit tests and Alembic).
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import get_settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        # In pytest, each test may run a fresh asyncio event loop. The default
        # QueuePool reuses asyncpg connections that are bound to a closed loop
        # and the next test then hits "loop is closed" / "different loop".
        if settings.env == "test":
            _engine = create_async_engine(
                settings.database_url,
                poolclass=NullPool,
                pool_pre_ping=True,
                future=True,
            )
        else:
            _engine = create_async_engine(
                settings.database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                future=True,
            )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            autoflush=False,
            class_=AsyncSession,
        )
    return _sessionmaker


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields an AsyncSession that is closed on exit."""
    sm = get_sessionmaker()
    async with sm() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def async_session_factory() -> AsyncSession:
    """Direct factory for use outside the FastAPI request lifecycle.

    The Celery worker is the canonical caller — it has no request-scoped
    dependency injection so it builds a session manually with this.
    """
    return get_sessionmaker()()
