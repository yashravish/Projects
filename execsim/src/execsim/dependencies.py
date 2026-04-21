"""FastAPI dependency injection."""

from functools import lru_cache
from typing import Generator

from sqlalchemy.orm import Session

from execsim.config import Settings
from execsim.db.engine import build_session_factory


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


@lru_cache(maxsize=1)
def _session_factory():
    """Return cached session factory. Internal use only."""
    return build_session_factory(get_settings())


def get_db() -> Generator[Session, None, None]:
    """Yield a database session, closing it after use.

    Used as a FastAPI dependency. The session is closed (not committed)
    after the request completes.
    """
    factory = _session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()
