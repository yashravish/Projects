"""
Unit tests for the sliding-window rate limiter.

Uses fakeredis for in-memory Redis simulation — no real Redis required.
"""
import os
import time

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SCHEDULER_ENABLED", "false")
os.environ.setdefault("REDIS_ENABLED", "true")
os.environ.setdefault("KAFKA_ENABLED", "false")

import pytest
import fakeredis
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from app.core.rate_limiter import RateLimiter
from app.core import redis_client as redis_module


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """Inject a fakeredis instance into the redis_client module."""
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_module, "_redis_client", fake)
    yield fake
    fake.flushall()


def _make_request(path="/api/v1/sync/crm", client_ip="127.0.0.1"):
    """Create a mock FastAPI Request object."""
    request = MagicMock()
    request.url.path = path
    request.client.host = client_ip
    return request


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self):
        limiter = RateLimiter(requests_per_minute=5)
        request = _make_request()

        # Should allow 5 requests without raising
        for _ in range(5):
            await limiter(request)

    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self):
        limiter = RateLimiter(requests_per_minute=3)
        request = _make_request()

        # Allow first 3
        for _ in range(3):
            await limiter(request)

        # 4th should be rejected
        with pytest.raises(HTTPException) as exc_info:
            await limiter(request)

        assert exc_info.value.status_code == 429
        assert "rate_limit_exceeded" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_retry_after_header(self):
        limiter = RateLimiter(requests_per_minute=1)
        request = _make_request()

        await limiter(request)

        with pytest.raises(HTTPException) as exc_info:
            await limiter(request)

        assert "Retry-After" in exc_info.value.headers
        retry_after = int(exc_info.value.headers["Retry-After"])
        assert 0 < retry_after <= 61

    @pytest.mark.asyncio
    async def test_different_ips_independent(self):
        limiter = RateLimiter(requests_per_minute=1)

        req1 = _make_request(client_ip="10.0.0.1")
        req2 = _make_request(client_ip="10.0.0.2")

        await limiter(req1)
        await limiter(req2)  # Should not raise — different IP

    @pytest.mark.asyncio
    async def test_different_paths_independent(self):
        limiter = RateLimiter(requests_per_minute=1)

        req1 = _make_request(path="/api/v1/sync/crm")
        req2 = _make_request(path="/api/v1/sync/vendor")

        await limiter(req1)
        await limiter(req2)  # Should not raise — different path

    @pytest.mark.asyncio
    async def test_bypassed_when_redis_unavailable(self, monkeypatch):
        monkeypatch.setattr(redis_module, "_redis_client", None)
        limiter = RateLimiter(requests_per_minute=1)
        request = _make_request()

        # Should allow unlimited requests when Redis is down
        for _ in range(10):
            await limiter(request)
