"""Workflow event ORM model. Tracks every status change for a task."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin
from app.models.task import TaskStatus

if TYPE_CHECKING:
    from app.models.task import Task
    from app.models.user import User


class WorkflowEvent(Base, IdMixin):
    __tablename__ = "workflow_events"

    task_id: Mapped[int] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_status: Mapped[Optional[TaskStatus]] = mapped_column(
        SAEnum(TaskStatus, name="task_status", create_type=False),
        nullable=True,
    )
    to_status: Mapped[TaskStatus] = mapped_column(
        SAEnum(TaskStatus, name="task_status", create_type=False),
        nullable=False,
    )
    changed_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    note: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    task: Mapped["Task"] = relationship("Task", back_populates="workflow_events")
    changed_by: Mapped["User"] = relationship("User", back_populates="workflow_events")
