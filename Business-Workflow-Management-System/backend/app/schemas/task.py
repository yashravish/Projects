"""Task schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.task import TaskPriority, TaskStatus
from app.schemas.user import UserSummary


class TaskBase(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=5000)
    priority: TaskPriority = TaskPriority.MEDIUM


class TaskCreate(TaskBase):
    assigned_to_user_id: int


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=5000)
    priority: Optional[TaskPriority] = None
    assigned_to_user_id: Optional[int] = None


class TaskRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str
    status: TaskStatus
    priority: TaskPriority
    assigned_to_user_id: int
    created_by_user_id: int
    assignee: UserSummary
    creator: UserSummary
    created_at: datetime
    updated_at: datetime


class TaskActionRequest(BaseModel):
    """Optional comment for submit/approve/reject/complete actions."""

    comment: Optional[str] = Field(default=None, max_length=500)
