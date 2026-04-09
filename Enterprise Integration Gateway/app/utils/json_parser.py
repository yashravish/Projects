"""
Utilities for safely parsing and validating JSON payloads from external sources.
"""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def safe_parse_json(raw: str | bytes, context: str = "") -> dict | list | None:
    """
    Parse a JSON string, returning None and logging on failure instead of raising.

    Args:
        raw: Raw JSON string or bytes.
        context: Human-readable label for log messages (e.g. "CRM customers response").

    Returns:
        Parsed Python object or None if parsing fails.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(
            "json_parse_error",
            extra={"context": context, "error": str(exc), "snippet": str(raw)[:200]},
        )
        return None


def extract_list(payload: dict | list, key: str) -> list[dict[str, Any]]:
    """
    Extract a list from a JSON payload.

    Handles both:
      - {"customers": [...]}     → pass key="customers"
      - [...]                    → pass key="" to return as-is

    Returns an empty list on failure.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        value = payload.get(key, [])
        if isinstance(value, list):
            return value
        logger.warning(
            "unexpected_json_structure",
            extra={"key": key, "found_type": type(value).__name__},
        )
    return []


def get_nested(data: dict, *keys: str, default: Any = None) -> Any:
    """
    Safely retrieve a nested value from a dict.

    Example:
        get_nested(record, "billingAddress", "street")
    """
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is None:
            return default
    return current
