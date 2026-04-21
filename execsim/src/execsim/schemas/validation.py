"""Schemas for validation alerts."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from execsim.schemas.common import CheckTypeEnum, SeverityEnum


class ValidationAlertSchema(BaseModel):
    """A single validation alert."""
    id: UUID
    run_id: UUID
    check_type: CheckTypeEnum
    severity: SeverityEnum
    message: str
    detail: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertListParams(BaseModel):
    """Query parameters for listing alerts."""
    run_id: Optional[UUID] = None
    check_type: Optional[CheckTypeEnum] = None
    severity: Optional[SeverityEnum] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
