"""
Simple synchronous HTTP retry helper with exponential back-off.

Usage:
    response = retry_request(client.get, "https://api.example.com/data", max_retries=3)
"""
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def retry_request(
    fn: Callable[..., httpx.Response],
    *args: Any,
    max_retries: int | None = None,
    backoff_factor: float | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """
    Call *fn* with *args / **kwargs*, retrying on transient errors.

    Retries on:
      - httpx.TimeoutException
      - httpx.ConnectError
      - HTTP 429, 5xx responses

    Raises the last exception if all retries are exhausted.
    """
    _max = max_retries if max_retries is not None else settings.HTTP_MAX_RETRIES
    _factor = backoff_factor if backoff_factor is not None else settings.HTTP_RETRY_BACKOFF_FACTOR

    last_exc: Exception | None = None

    for attempt in range(1, _max + 2):  # +2: first attempt + N retries
        try:
            response = fn(*args, **kwargs)
            if response.status_code not in _RETRYABLE_STATUS_CODES:
                return response

            last_exc = httpx.HTTPStatusError(
                f"HTTP {response.status_code}",
                request=response.request,
                response=response,
            )
            logger.warning(
                "retryable_http_error",
                extra={"attempt": attempt, "status_code": response.status_code},
            )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            logger.warning(
                "retryable_connection_error",
                extra={"attempt": attempt, "error": str(exc)},
            )

        if attempt <= _max:
            wait = _factor * (2 ** (attempt - 1))
            logger.info("retry_backoff", extra={"wait_seconds": wait, "next_attempt": attempt + 1})
            time.sleep(wait)

    raise last_exc  # type: ignore[misc]
