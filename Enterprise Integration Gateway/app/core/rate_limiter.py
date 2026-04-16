"""
Sliding-window rate limiter backed by Redis sorted sets.

Limits the number of requests a client can make to a specific endpoint
within a rolling time window. Returns HTTP 429 with a ``Retry-After``
header when the limit is exceeded.
"""
import logging
import time

from fastapi import HTTPException, Request, status

from app.core.config import settings
from app.core.redis_client import get_redis

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    FastAPI dependency that enforces a per-endpoint sliding-window rate limit.

    Usage::

        rate_limiter = RateLimiter(requests_per_minute=30)

        @router.post("/sync/crm")
        def trigger_sync(_=Depends(rate_limiter)):
            ...

    Algorithm:
      1. Each request adds ``(timestamp, unique_id)`` to a Redis sorted set
      2. Entries older than the window are pruned
      3. If the remaining count exceeds the limit → reject with 429

    If Redis is unavailable, the limiter is bypassed (open policy).
    """

    def __init__(self, requests_per_minute: int | None = None) -> None:
        self.rpm = requests_per_minute or settings.RATE_LIMIT_RPM
        self.window_seconds = 60

    def _key(self, path: str, client_ip: str) -> str:
        return f"ratelimit:{path}:{client_ip}"

    async def __call__(self, request: Request) -> None:
        r = get_redis()
        if r is None:
            return  # open policy when Redis is down

        client_ip = request.client.host if request.client else "unknown"
        key = self._key(request.url.path, client_ip)
        now = time.time()
        window_start = now - self.window_seconds

        try:
            pipe = r.pipeline()
            # Remove entries outside the current window
            pipe.zremrangebyscore(key, 0, window_start)
            # Count remaining entries in the window
            pipe.zcard(key)
            # Add the current request
            pipe.zadd(key, {f"{now}": now})
            # Set expiry so keys don't persist forever
            pipe.expire(key, self.window_seconds + 10)
            results = pipe.execute()

            current_count = results[1]

            if current_count >= self.rpm:
                # Calculate retry-after based on the oldest entry in the window
                oldest = r.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(self.window_seconds - (now - oldest[0][1])) + 1
                else:
                    retry_after = self.window_seconds

                logger.warning(
                    "rate_limit_exceeded",
                    extra={
                        "path": request.url.path,
                        "client_ip": client_ip,
                        "count": current_count,
                        "limit": self.rpm,
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "rate_limit_exceeded",
                        "message": f"Rate limit of {self.rpm} requests per minute exceeded.",
                        "retry_after_seconds": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

        except HTTPException:
            raise
        except Exception as exc:
            # Graceful degradation — allow the request through
            logger.warning(
                "rate_limiter_error",
                extra={"error": str(exc), "path": request.url.path},
            )


# ── Pre-configured instances ──────────────────────────────────────────────────

sync_rate_limiter = RateLimiter()
