"""Workflow state-machine service.

Centralizes valid status transitions, transition validation, and workflow
event creation for tasks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from fastapi import HTTPException, status as http_status
from sqlalchemy.orm import Session

from app.models.task import Task, TaskStatus
from app.models.user import User
from app.models.workflow_event import WorkflowEvent
from app.repositories.task_repository import TaskRepository


# Allowed task status transitions. The single source of truth.
VALID_TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.IN_PROGRESS, TaskStatus.SUBMITTED},
    TaskStatus.IN_PROGRESS: {TaskStatus.SUBMITTED},
    TaskStatus.SUBMITTED: {TaskStatus.APPROVED, TaskStatus.REJECTED},
    TaskStatus.APPROVED: {TaskStatus.COMPLETED},
    TaskStatus.REJECTED: {TaskStatus.IN_PROGRESS, TaskStatus.SUBMITTED},
    TaskStatus.COMPLETED: set(),
}


class WorkflowService:
    """Validates status transitions and records workflow events."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = TaskRepository(db)

    @staticmethod
    def is_valid_transition(from_status: TaskStatus, to_status: TaskStatus) -> bool:
        """Return True if the given transition is allowed."""
        return to_status in VALID_TRANSITIONS.get(from_status, set())

    def assert_valid_transition(self, from_status: TaskStatus, to_status: TaskStatus) -> None:
        """Raise an HTTP 400 if the transition is not allowed."""
        if not self.is_valid_transition(from_status, to_status):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid status transition: {from_status.value} -> {to_status.value}"
                ),
            )

    def transition(
        self,
        task: Task,
        to_status: TaskStatus,
        changed_by: User,
        note: Optional[str] = None,
    ) -> WorkflowEvent:
        """Validate the transition, mutate the task, and emit a workflow event."""
        from_status = task.status
        self.assert_valid_transition(from_status, to_status)
        task.status = to_status
        event = WorkflowEvent(
            task_id=task.id,
            from_status=from_status,
            to_status=to_status,
            changed_by_user_id=changed_by.id,
            note=note,
        )
        self.db.add(event)
        self.db.flush()
        return event

    def record_initial_event(
        self,
        task: Task,
        changed_by: User,
        note: Optional[str] = None,
    ) -> WorkflowEvent:
        """Record an event for task creation (no previous status)."""
        event = WorkflowEvent(
            task_id=task.id,
            from_status=None,
            to_status=task.status,
            changed_by_user_id=changed_by.id,
            note=note or "Task created",
        )
        self.db.add(event)
        self.db.flush()
        return event

    def get_task_history(self, task_id: int) -> List[WorkflowEvent]:
        return self.repo.list_workflow_events(task_id)
