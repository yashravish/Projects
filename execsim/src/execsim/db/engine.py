"""SQLAlchemy engine and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from execsim.config import Settings


def build_engine(settings: Settings):
    """Create a SQLAlchemy engine from settings.

    Args:
        settings: Application settings containing database_url.

    Returns:
        A SQLAlchemy Engine instance with connection pooling.
    """
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


def build_session_factory(settings: Settings) -> sessionmaker[Session]:
    """Create a session factory bound to the engine.

    Args:
        settings: Application settings containing database_url.

    Returns:
        A sessionmaker configured with expire_on_commit=False.
    """
    engine = build_engine(settings)
    return sessionmaker(bind=engine, expire_on_commit=False)
