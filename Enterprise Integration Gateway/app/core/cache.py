"""
Response caching utilities backed by Redis.

Provides a ``@cached`` decorator for FastAPI route handlers and a helper
to invalidate cache entries by prefix after write operations (e.g. sync).
"""
import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable, Optional

from app.core.config import settings
from app.core.redis_client import get_redis

logger = logging.getLogger(__name__)

# ── Key helpers ────────────────────────────────────────────────────────────────


def _build_cache_key(prefix: str, path: str, query_params: dict[str, Any]) -> str:
    """
    Generate a deterministic cache key from the request path and query params.

    Format: ``cache:{prefix}:{path}:{md5_of_sorted_params}``
    """
    sorted_params = json.dumps(dict(sorted(query_params.items())), default=str)
    param_hash = hashlib.md5(sorted_params.encode()).hexdigest()[:12]
    return f"cache:{prefix}:{path}:{param_hash}"


# ── Decorator ──────────────────────────────────────────────────────────────────


def cached(prefix: str, ttl: Optional[int] = None) -> Callable:
    """
    Decorator that caches the JSON-serializable return value of a
    FastAPI route handler in Redis with a configurable TTL.

    Usage::

        @router.get("/items")
        @cached(prefix="items", ttl=120)
        def list_items(skip: int = 0, limit: int = 50, db=Depends(get_db)):
            ...

    If Redis is unavailable the handler executes normally (no caching).
    """
    _ttl = ttl or settings.CACHE_TTL_SECONDS

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            r = get_redis()
            if r is None:
                return func(*args, **kwargs)

            # Extract request path and query params for key generation
            request = kwargs.get("request")
            if request is not None:
                path = request.url.path
                query_params = dict(request.query_params)
            else:
                path = func.__name__
                query_params = {
                    k: v for k, v in kwargs.items()
                    if k not in ("db", "request", "redis")
                }

            cache_key = _build_cache_key(prefix, path, query_params)

            # ── Cache hit ──────────────────────────────────────────────────
            try:
                cached_value = r.get(cache_key)
                if cached_value is not None:
                    logger.debug("cache_hit", extra={"key": cache_key})
                    return json.loads(cached_value)
            except Exception:
                pass  # graceful degradation

            # ── Cache miss — execute handler ───────────────────────────────
            result = func(*args, **kwargs)

            try:
                r.setex(cache_key, _ttl, json.dumps(result, default=str))
                logger.debug("cache_set", extra={"key": cache_key, "ttl": _ttl})
            except Exception:
                pass  # graceful degradation

            return result

        return wrapper

    return decorator


# ── Invalidation ───────────────────────────────────────────────────────────────


def invalidate_cache(*prefixes: str) -> int:
    """
    Delete all cache keys matching the given prefixes.

    Uses ``SCAN`` to avoid blocking Redis on large keyspaces.
    Returns the total number of keys deleted.
    """
    r = get_redis()
    if r is None:
        return 0

    deleted = 0
    for prefix in prefixes:
        pattern = f"cache:{prefix}:*"
        try:
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    r.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            logger.info("cache_invalidated", extra={"prefix": prefix, "deleted": deleted})
        except Exception as exc:
            logger.warning("cache_invalidation_failed", extra={"prefix": prefix, "error": str(exc)})

    return deleted
