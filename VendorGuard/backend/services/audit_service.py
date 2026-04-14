"""Audit logging service for tracking user actions."""

import structlog
from sqlalchemy.orm import Session

from backend.models import AuditLog

logger = structlog.get_logger()


def log_action(
    db: Session,
    user_id: int | None,
    action: str,
    entity_type: str = "",
    entity_id: int | None = None,
    details: str = "",
    ip_address: str = "",
):
    entry = AuditLog(
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        details=details,
        ip_address=ip_address,
    )
    db.add(entry)
    db.commit()
    logger.info(
        "audit_event",
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        user_id=user_id,
    )
