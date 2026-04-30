"""Audit log service. Records every important business action."""
from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.audit_log import AuditAction, AuditLog
from app.repositories.audit_log_repository import AuditLogRepository


logger = logging.getLogger(__name__)


class AuditLogService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = AuditLogRepository(db)

    def log_action(
        self,
        *,
        user_id: Optional[int],
        action: AuditAction,
        entity_type: str,
        entity_id: Optional[int] = None,
        details: Optional[str] = None,
    ) -> AuditLog:
        log = AuditLog(
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details,
        )
        self.repo.add(log)
        logger.info(
            "audit user_id=%s action=%s entity_type=%s entity_id=%s",
            user_id,
            action.value,
            entity_type,
            entity_id,
        )
        return log

    def list_logs(
        self,
        action: Optional[AuditAction] = None,
        limit: int = 100,
    ) -> List[AuditLog]:
        return self.repo.list_logs(action=action, limit=limit)
