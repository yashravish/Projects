"""Structured logging configuration via structlog.

All logs are emitted as JSON in non-development environments. In development
we use a console renderer for readability. A `request_id` contextvar is bound
on each HTTP request by the middleware in `app.main`.
"""
from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any, cast

import structlog
from structlog.contextvars import merge_contextvars
from structlog.stdlib import BoundLogger

from app.config import get_settings

_REDACTED_HEADERS = {"authorization", "cookie", "set-cookie", "x-api-key"}


def _redact_headers(
    _: Any, __: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Redact sensitive header values if a `headers` key is present."""
    headers = event_dict.get("headers")
    if isinstance(headers, dict):
        event_dict["headers"] = {
            k: ("<redacted>" if k.lower() in _REDACTED_HEADERS else v)
            for k, v in headers.items()
        }
    return event_dict


def _redact_request_body(
    _: Any, __: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Drop request bodies for auth routes; truncate large bodies elsewhere."""
    route = event_dict.get("route", "")
    if isinstance(route, str) and route.startswith("/api/v1/auth"):
        event_dict.pop("body", None)
    body = event_dict.get("body")
    if isinstance(body, str) and len(body) > 200:
        event_dict["body"] = body[:200] + "...<truncated>"
    return event_dict


def configure_logging() -> None:
    """Configure structlog and the standard logging module to share processors."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[structlog.types.Processor] = [
        merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        _redact_headers,
        _redact_request_body,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.env == "development":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(colors=False)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging through our processors as well.
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy libraries.
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "asyncio"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def get_logger(name: str | None = None) -> BoundLogger:
    logger = structlog.get_logger(name) if name else structlog.get_logger()
    return cast(BoundLogger, logger)
