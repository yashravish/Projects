"""
Unit tests for the Redis-backed response caching utilities.

Uses fakeredis for in-memory Redis simulation — no real Redis required.
"""
import json
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SCHEDULER_ENABLED", "false")
os.environ.setdefault("REDIS_ENABLED", "true")
os.environ.setdefault("KAFKA_ENABLED", "false")

import pytest
import fakeredis

from app.core.cache import _build_cache_key, cached, invalidate_cache
from app.core import redis_client as redis_module


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """Inject a fakeredis instance into the redis_client module."""
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_module, "_redis_client", fake)
    yield fake
    fake.flushall()


# ── Cache key generation ──────────────────────────────────────────────────────


class TestBuildCacheKey:
    def test_deterministic_key(self):
        key1 = _build_cache_key("customers", "/api/v1/customers", {"skip": "0", "limit": "50"})
        key2 = _build_cache_key("customers", "/api/v1/customers", {"skip": "0", "limit": "50"})
        assert key1 == key2

    def test_different_params_different_keys(self):
        key1 = _build_cache_key("customers", "/api/v1/customers", {"skip": "0"})
        key2 = _build_cache_key("customers", "/api/v1/customers", {"skip": "10"})
        assert key1 != key2

    def test_param_order_irrelevant(self):
        key1 = _build_cache_key("orders", "/api/v1/orders", {"a": "1", "b": "2"})
        key2 = _build_cache_key("orders", "/api/v1/orders", {"b": "2", "a": "1"})
        assert key1 == key2

    def test_key_includes_prefix_and_path(self):
        key = _build_cache_key("customers", "/api/v1/customers", {})
        assert key.startswith("cache:customers:/api/v1/customers:")


# ── @cached decorator ────────────────────────────────────────────────────────


class TestCachedDecorator:
    def test_caches_result(self, fake_redis):
        call_count = 0

        @cached(prefix="test", ttl=60)
        def my_handler(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"data": "hello"}

        result1 = my_handler(skip=0, limit=10)
        result2 = my_handler(skip=0, limit=10)

        assert result1 == {"data": "hello"}
        assert result2 == {"data": "hello"}
        assert call_count == 1  # second call served from cache

    def test_different_params_not_cached(self, fake_redis):
        call_count = 0

        @cached(prefix="test")
        def my_handler(**kwargs):
            nonlocal call_count
            call_count += 1
            return {"page": kwargs.get("skip", 0)}

        my_handler(skip=0)
        my_handler(skip=10)

        assert call_count == 2

    def test_falls_back_when_redis_unavailable(self, monkeypatch):
        monkeypatch.setattr(redis_module, "_redis_client", None)

        @cached(prefix="test")
        def my_handler(**kwargs):
            return {"fallback": True}

        result = my_handler()
        assert result == {"fallback": True}


# ── Cache invalidation ───────────────────────────────────────────────────────


class TestInvalidateCache:
    def test_invalidates_matching_keys(self, fake_redis):
        fake_redis.setex("cache:customers:/a:abc", 60, '{"x":1}')
        fake_redis.setex("cache:customers:/b:def", 60, '{"x":2}')
        fake_redis.setex("cache:orders:/c:ghi", 60, '{"x":3}')

        deleted = invalidate_cache("customers")
        assert deleted == 2

        assert fake_redis.get("cache:customers:/a:abc") is None
        assert fake_redis.get("cache:orders:/c:ghi") is not None

    def test_invalidates_multiple_prefixes(self, fake_redis):
        fake_redis.setex("cache:customers:key1", 60, '{"x":1}')
        fake_redis.setex("cache:orders:key2", 60, '{"x":2}')
        fake_redis.setex("cache:shipments:key3", 60, '{"x":3}')

        deleted = invalidate_cache("customers", "orders")
        assert deleted >= 2

    def test_returns_zero_when_redis_unavailable(self, monkeypatch):
        monkeypatch.setattr(redis_module, "_redis_client", None)
        assert invalidate_cache("customers") == 0
