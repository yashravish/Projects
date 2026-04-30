"""Data access object for audit logs."""
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.audit_log import AuditAction, AuditLog


class AuditLogRepository:
    """Encapsulates queries against the audit_logs table."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def add(self, log: AuditLog) -> AuditLog:
        self.db.add(log)
        self.db.flush()
        return log

    def list_logs(
        self,
        action: Optional[AuditAction] = None,
        limit: int = 100,
    ) -> List[AuditLog]:
        stmt = (
            select(AuditLog)
            .options(selectinload(AuditLog.user))
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        if action is not None:
            stmt = stmt.where(AuditLog.action == action)
        return list(self.db.scalars(stmt).all())
