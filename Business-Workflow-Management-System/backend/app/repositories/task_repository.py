"""Data access object for tasks and related workflow entities."""
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.task import Task, TaskPriority, TaskStatus
from app.models.workflow_event import WorkflowEvent


class TaskRepository:
    """Encapsulates queries against the tasks table."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, task_id: int) -> Optional[Task]:
        stmt = (
            select(Task)
            .options(selectinload(Task.assignee), selectinload(Task.creator))
            .where(Task.id == task_id)
        )
        return self.db.scalars(stmt).first()

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None,
        assigned_to_user_id: Optional[int] = None,
    ) -> List[Task]:
        stmt = (
            select(Task)
            .options(selectinload(Task.assignee), selectinload(Task.creator))
            .order_by(Task.created_at.desc())
        )
        if status is not None:
            stmt = stmt.where(Task.status == status)
        if priority is not None:
            stmt = stmt.where(Task.priority == priority)
        if assigned_to_user_id is not None:
            stmt = stmt.where(Task.assigned_to_user_id == assigned_to_user_id)
        return list(self.db.scalars(stmt).all())

    def list_workflow_events(self, task_id: int) -> List[WorkflowEvent]:
        stmt = (
            select(WorkflowEvent)
            .options(selectinload(WorkflowEvent.changed_by))
            .where(WorkflowEvent.task_id == task_id)
            .order_by(WorkflowEvent.created_at.asc())
        )
        return list(self.db.scalars(stmt).all())

    def add(self, task: Task) -> Task:
        self.db.add(task)
        self.db.flush()
        return task

    def delete(self, task: Task) -> None:
        self.db.delete(task)
        self.db.flush()
