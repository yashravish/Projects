"""Audit log schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.models.audit_log import AuditAction
from app.schemas.user import UserSummary


class AuditLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: Optional[int]
    user: Optional[UserSummary]
    action: AuditAction
    entity_type: str
    entity_id: Optional[int]
    details: Optional[str]
    created_at: datetime
