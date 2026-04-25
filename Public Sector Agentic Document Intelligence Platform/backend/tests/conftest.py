"""Test fixtures.

Integration tests require a Postgres with pgvector. The compose file exposes
postgres on localhost:5432; CI provides a service container. Tests that need
the database take the `db_session` fixture; pure-unit tests don't.

Each test creates its own throw-away schema namespace by truncating tenant
tables before the test (cheaper than tearing down the whole schema).
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Force test config defaults BEFORE importing app modules.
# Use assignment (not setdefault) for ENV so docker-compose ``.env`` cannot
# leave us in ``development`` while tests expect ``test`` Settings behaviour.
os.environ["ENV"] = "test"
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("SEED_ON_BOOT", "false")
os.environ.setdefault("JWT_SECRET", "test-secret-please-do-not-use-in-production-32b")
os.environ.setdefault(
    "DATABASE_URL",
    os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+asyncpg://psdi:psdi_dev_password@localhost:5432/psdi",
    ),
)
os.environ.setdefault(
    "DATABASE_URL_SYNC",
    os.environ.get(
        "TEST_DATABASE_URL_SYNC",
        "postgresql+psycopg://psdi:psdi_dev_password@localhost:5432/psdi",
    ),
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:5173")

import tempfile

_TEST_UPLOADS = tempfile.mkdtemp(prefix="psdi-test-uploads-")
os.environ.setdefault("LOCAL_UPLOAD_DIR", _TEST_UPLOADS)

_TEST_MODELS = tempfile.mkdtemp(prefix="psdi-test-models-")
os.environ.setdefault("MODELS_DIR", _TEST_MODELS)

# If anything imported `get_settings` before conftest (unlikely), clear it.
try:
    from app.config import get_settings as _get_settings

    _get_settings.cache_clear()
except ImportError:
    pass


def _postgres_available() -> bool:
    """Best-effort check whether the configured Postgres is reachable."""
    import socket

    url = os.environ["DATABASE_URL"]
    # naive parse: postgresql+asyncpg://user:pw@host:port/db
    try:
        host_part = url.split("@", 1)[1].split("/", 1)[0]
        host, port = host_part.split(":")
        with socket.create_connection((host, int(port)), timeout=1.0):
            return True
    except OSError:
        return False
    except Exception:  # pragma: no cover
        return False


requires_postgres = pytest.mark.skipif(
    not _postgres_available(),
    reason="postgres not reachable; set TEST_DATABASE_URL or run via docker compose",
)


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    from app.db.session import get_sessionmaker

    sm = get_sessionmaker()
    async with sm() as session:
        # Truncate tenant tables for test isolation.
        await session.execute(
            text(
                "TRUNCATE TABLE retention_runs, retention_policies, "
                "registered_models, training_jobs, "
                "audit_logs, system_cards, evaluation_runs, "
                "query_runs, chunks, documents, users, organizations "
                "RESTART IDENTITY CASCADE"
            )
        )
        await session.commit()
        yield session


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def register_payload() -> dict[str, Any]:
    return {
        "email": "alice@example.gov",
        "password": "AnalystPass!2026",
        "organization_name": "Test Agency",
    }
