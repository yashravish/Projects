"""Workflow event schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.models.task import TaskStatus
from app.schemas.user import UserSummary


class WorkflowEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    task_id: int
    from_status: Optional[TaskStatus]
    to_status: TaskStatus
    changed_by_user_id: int
    changed_by: UserSummary
    note: Optional[str]
    created_at: datetime
