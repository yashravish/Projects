"""
Shared pytest fixtures.

Uses an in-memory SQLite database so tests run without a real PostgreSQL instance.
Each test gets a fresh database — this provides clean isolation for sync_service
tests that call db.commit() internally.
"""
import os

# ── Set test environment BEFORE any app code is imported ──────────────────────
# This ensures Settings() picks up the right DATABASE_URL for the session engine.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SCHEDULER_ENABLED", "false")
os.environ.setdefault("LOG_FORMAT", "text")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("CRM_BASE_URL", "http://localhost:8001")
os.environ.setdefault("VENDOR_BASE_URL", "http://localhost:8001")
os.environ.setdefault("REDIS_ENABLED", "false")
os.environ.setdefault("KAFKA_ENABLED", "false")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# ── Import models BEFORE app so Base.metadata is populated ────────────────────
import app.models.customer       # noqa: F401
import app.models.failed_record  # noqa: F401
import app.models.order          # noqa: F401
import app.models.shipment       # noqa: F401
import app.models.sync_job       # noqa: F401

# ── Import app AFTER env vars and models ──────────────────────────────────────
from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app as fastapi_app  # aliased — `import app.xxx` binds 'app' as package


def _make_engine():
    """
    Create a fresh SQLite in-memory engine for one test function.

    Uses StaticPool so all connections share the same in-memory database
    (required for SQLite :memory: to work correctly across multiple
    SQLAlchemy connections within the same process).
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="function")
def engine():
    """Fresh in-memory SQLite engine per test — guarantees clean state."""
    _engine = _make_engine()
    yield _engine
    _engine.dispose()


@pytest.fixture(scope="function")
def db(engine):
    """Database session bound to the per-test SQLite engine."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def client(engine):
    """
    FastAPI TestClient with the test database injected.

    Creates its own session factory from the test engine so API routes
    use the same in-memory SQLite DB as test seed data.
    """
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    fastapi_app.dependency_overrides[get_db] = override_get_db
    with TestClient(fastapi_app, raise_server_exceptions=False) as c:
        yield c
    fastapi_app.dependency_overrides.clear()
