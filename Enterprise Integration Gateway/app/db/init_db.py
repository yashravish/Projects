"""
Database initializer.

Called once at application startup. In production, prefer running
Alembic migrations (`alembic upgrade head`) rather than create_all().
This module provides a lightweight fallback for local/test environments.
"""
import logging

from sqlalchemy.orm import Session

from app.db.session import SessionLocal, engine
from app.db.base import Base

# Import all models so Base.metadata is populated before create_all()
import app.models.customer       # noqa: F401
import app.models.order          # noqa: F401
import app.models.shipment       # noqa: F401
import app.models.sync_job       # noqa: F401
import app.models.failed_record  # noqa: F401

logger = logging.getLogger(__name__)


def init_db() -> None:
    """Create all tables if they do not already exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized")
    except Exception as exc:
        logger.error("Failed to initialize database tables", exc_info=exc)
        # Do not re-raise — allows app to start even if DB is temporarily
        # unavailable (e.g. in test environments or during rolling deploys).
