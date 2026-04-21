"""Shared helper for creating validation alerts."""

import uuid
from datetime import datetime, timezone
from typing import Any

from execsim.db.models import CheckType, Severity, ValidationAlert


def create_alert(
    run_id: uuid.UUID,
    check_type: CheckType,
    severity: Severity,
    message: str,
    detail: dict[str, Any],
) -> ValidationAlert:
    """Create a ValidationAlert ORM object.

    Args:
        run_id: UUID of the simulation run.
        check_type: Type of validation check.
        severity: Alert severity level.
        message: Human-readable description.
        detail: Structured detail for debugging.

    Returns:
        ValidationAlert (not yet added to session).
    """
    return ValidationAlert(
        id=uuid.uuid4(),
        run_id=run_id,
        check_type=check_type,
        severity=severity,
        message=message,
        detail=detail,
        created_at=datetime.now(timezone.utc),
    )
