"""Task ORM model."""
from __future__ import annotations

import enum
from typing import List, TYPE_CHECKING

from sqlalchemy import Enum as SAEnum, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.workflow_event import WorkflowEvent
    from app.models.approval import Approval


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


class TaskPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Task(Base, IdMixin, TimestampMixin):
    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_status_priority", "status", "priority"),
    )

    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    status: Mapped[TaskStatus] = mapped_column(
        SAEnum(TaskStatus, name="task_status", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TaskStatus.PENDING,
        index=True,
    )
    priority: Mapped[TaskPriority] = mapped_column(
        SAEnum(TaskPriority, name="task_priority", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TaskPriority.MEDIUM,
        index=True,
    )

    assigned_to_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    created_by_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    assignee: Mapped["User"] = relationship(
        "User",
        foreign_keys=[assigned_to_user_id],
        back_populates="assigned_tasks",
    )
    creator: Mapped["User"] = relationship(
        "User",
        foreign_keys=[created_by_user_id],
        back_populates="created_tasks",
    )
    workflow_events: Mapped[List["WorkflowEvent"]] = relationship(
        "WorkflowEvent",
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="WorkflowEvent.created_at",
    )
    approvals: Mapped[List["Approval"]] = relationship(
        "Approval",
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="Approval.created_at",
    )

    def __repr__(self) -> str:
        return f"<Task id={self.id} status={self.status.value} title={self.title!r}>"
