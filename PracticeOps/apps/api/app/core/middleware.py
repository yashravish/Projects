"""Middleware for request processing.

Provides:
- Request ID middleware for request correlation
- Rate limiting middleware for auth endpoints
"""

import uuid
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.errors import ErrorCode
from app.core.logging import get_logger, request_id_var

logger = get_logger(__name__)


# =============================================================================
# Request ID Middleware
# =============================================================================


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and attach request IDs to each request."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        """Generate request ID and attach to request/response."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Store in context variable for logging
        request_id_var.set(request_id)

        # Attach to request state for access in route handlers
        request.state.request_id = request_id

        # Log request start
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            # Log request completion
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
            )

            return response
        except Exception as e:
            # Log request error
            logger.error(
                "request_error",
                method=request.method,
                path=request.url.path,
                error=str(e),
                exc_info=True,
            )
            raise
        finally:
            # Clear context variable
            request_id_var.set(None)


# =============================================================================
# Rate Limiting
# =============================================================================


def get_client_ip(request: Request) -> str:
    """Extract client IP for rate limiting.

    Checks X-Forwarded-For header for proxy scenarios,
    falls back to direct client IP.
    """
    # Check for proxy headers first
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        # Get the first IP in the chain (original client)
        return x_forwarded_for.split(",")[0].strip()

    # Fall back to direct client IP
    return get_remote_address(request)


# Create rate limiter instance with in-memory storage
limiter = Limiter(key_func=get_client_ip, default_limits=[])


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handle rate limit exceeded errors with standard error format."""
    logger.warning(
        "rate_limit_exceeded",
        client_ip=get_client_ip(request),
        path=request.url.path,
        limit=str(exc.detail),
    )

    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": ErrorCode.RATE_LIMITED.value,
                "message": "Too many requests. Please try again later.",
                "field": None,
            }
        },
        headers={"Retry-After": "60"},
    )


def setup_rate_limiting(app: FastAPI) -> None:
    """Configure rate limiting for the application."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


# Rate limit decorator for auth endpoints
# 10 requests per minute per IP
AUTH_RATE_LIMIT = "10/minute"

