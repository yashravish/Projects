"""Shared test fixtures and configuration."""

import os
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app.db.database import set_db_path, init_db
from app.main import app

# Use an in-memory or temp database for tests
TEST_DB_PATH = ":memory:"


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db(tmp_path):
    """Create a fresh test database for each test."""
    db_path = str(tmp_path / "test.db")
    set_db_path(db_path)
    await init_db()
    yield
    # Database is cleaned up when tmp_path is removed


@pytest_asyncio.fixture
async def client():
    """Provide an async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
