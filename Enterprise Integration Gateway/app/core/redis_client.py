"""
Redis connection factory and FastAPI dependency.

Provides a singleton Redis connection pool that is shared across the application.
Designed for graceful degradation — if Redis is unavailable, the application
continues to operate without caching or rate limiting.
"""
import logging
from typing import Optional

import redis

from app.core.config import settings

logger = logging.getLogger(__name__)

_redis_pool: Optional[redis.ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None


def init_redis() -> Optional[redis.Redis]:
    """
    Initialize the Redis connection pool and return a client instance.

    Called once during application startup. Returns None if Redis is
    disabled or if the connection fails (graceful degradation).
    """
    global _redis_pool, _redis_client

    if not settings.REDIS_ENABLED:
        logger.info("redis_disabled", extra={"reason": "REDIS_ENABLED=false"})
        return None

    try:
        _redis_pool = redis.ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=20,
            decode_responses=True,
        )
        _redis_client = redis.Redis(connection_pool=_redis_pool)
        _redis_client.ping()
        logger.info("redis_connected", extra={"url": settings.REDIS_URL})
        return _redis_client
    except (redis.ConnectionError, redis.TimeoutError) as exc:
        logger.warning(
            "redis_connection_failed",
            extra={"url": settings.REDIS_URL, "error": str(exc)},
        )
        _redis_client = None
        return None


def get_redis() -> Optional[redis.Redis]:
    """
    FastAPI dependency that provides the Redis client.

    Returns None if Redis is not initialized or unavailable.
    """
    return _redis_client


def get_redis_status() -> dict:
    """Return Redis connectivity status for health checks."""
    if not settings.REDIS_ENABLED:
        return {"enabled": False, "status": "disabled"}
    if _redis_client is None:
        return {"enabled": True, "status": "disconnected"}
    try:
        _redis_client.ping()
        info = _redis_client.info(section="memory")
        return {
            "enabled": True,
            "status": "ok",
            "used_memory_human": info.get("used_memory_human", "unknown"),
        }
    except Exception as exc:
        return {"enabled": True, "status": "error", "error": str(exc)}


def shutdown_redis() -> None:
    """Close the Redis connection pool on application shutdown."""
    global _redis_client, _redis_pool
    if _redis_client is not None:
        try:
            _redis_client.close()
            logger.info("redis_disconnected")
        except Exception:
            pass
        _redis_client = None
    if _redis_pool is not None:
        _redis_pool.disconnect()
        _redis_pool = None
