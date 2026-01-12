"""Pytest configuration and fixtures."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.middleware import limiter
from app.main import app


@pytest.fixture(autouse=True)
def reset_rate_limiter() -> None:
    """Reset rate limiter before each test to prevent cross-test interference."""
    limiter.reset()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

