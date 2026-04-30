"""Aggregate imports for all ORM models so Alembic can discover them."""
from app.models.user import User, UserRole
from app.models.task import Task, TaskPriority, TaskStatus
from app.models.workflow_event import WorkflowEvent
from app.models.approval import Approval, ApprovalDecision
from app.models.audit_log import AuditAction, AuditLog

__all__ = [
    "User",
    "UserRole",
    "Task",
    "TaskPriority",
    "TaskStatus",
    "WorkflowEvent",
    "Approval",
    "ApprovalDecision",
    "AuditAction",
    "AuditLog",
]
