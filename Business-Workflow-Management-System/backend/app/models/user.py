"""User ORM model."""
from __future__ import annotations

import enum
from typing import List, TYPE_CHECKING

from sqlalchemy import Enum as SAEnum, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.task import Task
    from app.models.workflow_event import WorkflowEvent
    from app.models.approval import Approval
    from app.models.audit_log import AuditLog


class UserRole(str, enum.Enum):
    """Roles available within the workflow system."""

    ANALYST = "analyst"
    MANAGER = "manager"


class User(Base, IdMixin, TimestampMixin):
    __tablename__ = "users"

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        SAEnum(UserRole, name="user_role", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        index=True,
    )

    created_tasks: Mapped[List["Task"]] = relationship(
        "Task",
        foreign_keys="Task.created_by_user_id",
        back_populates="creator",
    )
    assigned_tasks: Mapped[List["Task"]] = relationship(
        "Task",
        foreign_keys="Task.assigned_to_user_id",
        back_populates="assignee",
    )
    workflow_events: Mapped[List["WorkflowEvent"]] = relationship(
        "WorkflowEvent",
        back_populates="changed_by",
    )
    approvals: Mapped[List["Approval"]] = relationship(
        "Approval",
        back_populates="approved_by",
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="user",
    )

    @property
    def is_manager(self) -> bool:
        return self.role == UserRole.MANAGER

    @property
    def is_analyst(self) -> bool:
        return self.role == UserRole.ANALYST

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} role={self.role.value}>"
