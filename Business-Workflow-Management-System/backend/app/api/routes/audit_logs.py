"""Audit log routes."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.audit_log import AuditAction
from app.models.user import User
from app.schemas.audit_log import AuditLogRead
from app.services.audit_log_service import AuditLogService


router = APIRouter(prefix="/audit-logs", tags=["audit-logs"])


@router.get("", response_model=List[AuditLogRead])
def list_audit_logs(
    action: Optional[AuditAction] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> List[AuditLogRead]:
    service = AuditLogService(db)
    logs = service.list_logs(action=action, limit=limit)
    return [AuditLogRead.model_validate(log) for log in logs]
