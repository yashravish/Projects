"""Task service - implements business logic for the task CRUD + workflow API."""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import HTTPException, status as http_status
from sqlalchemy.orm import Session

from app.models.audit_log import AuditAction
from app.models.task import Task, TaskPriority, TaskStatus
from app.models.user import User
from app.repositories.task_repository import TaskRepository
from app.repositories.user_repository import UserRepository
from app.schemas.task import TaskCreate, TaskUpdate
from app.services.approval_service import ApprovalService
from app.services.audit_log_service import AuditLogService
from app.services.workflow_service import WorkflowService


logger = logging.getLogger(__name__)


class TaskService:
    """Encapsulates CRUD and lifecycle business rules for tasks."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.repo = TaskRepository(db)
        self.users = UserRepository(db)
        self.workflow = WorkflowService(db)
        self.approvals = ApprovalService(db)
        self.audit = AuditLogService(db)

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None,
        assigned_to_user_id: Optional[int] = None,
    ) -> List[Task]:
        return self.repo.list_tasks(
            status=status,
            priority=priority,
            assigned_to_user_id=assigned_to_user_id,
        )

    def get_task_or_404(self, task_id: int) -> Task:
        task = self.repo.get(task_id)
        if task is None:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found.",
            )
        return task

    def _ensure_creator_or_manager(self, task: Task, user: User, *, action: str) -> None:
        """Analysts may only act on tasks they created; managers may act on any."""
        if user.is_analyst and task.created_by_user_id != user.id:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail=f"Analysts can only {action} tasks they created.",
            )

    def create_task(self, data: TaskCreate, current_user: User) -> Task:
        assignee = self.users.get(data.assigned_to_user_id)
        if assignee is None:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=f"Assignee user {data.assigned_to_user_id} not found.",
            )
        task = Task(
            title=data.title.strip(),
            description=data.description or "",
            priority=data.priority,
            status=TaskStatus.PENDING,
            assigned_to_user_id=assignee.id,
            created_by_user_id=current_user.id,
        )
        self.repo.add(task)
        self.workflow.record_initial_event(task=task, changed_by=current_user)
        self.audit.log_action(
            user_id=current_user.id,
            action=AuditAction.CREATE_TASK,
            entity_type="task",
            entity_id=task.id,
            details=f"Created task '{task.title}' assigned to user {assignee.id}",
        )
        self.db.commit()
        self.db.refresh(task)
        return task

    def update_task(self, task_id: int, data: TaskUpdate, current_user: User) -> Task:
        task = self.get_task_or_404(task_id)
        if task.status == TaskStatus.COMPLETED:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Completed tasks cannot be edited.",
            )
        self._ensure_creator_or_manager(task, current_user, action="edit")

        changed_fields: List[str] = []
        if data.title is not None and data.title.strip() != task.title:
            task.title = data.title.strip()
            changed_fields.append("title")
        if data.description is not None and data.description != task.description:
            task.description = data.description
            changed_fields.append("description")
        if data.priority is not None and data.priority != task.priority:
            task.priority = data.priority
            changed_fields.append("priority")
        if (
            data.assigned_to_user_id is not None
            and data.assigned_to_user_id != task.assigned_to_user_id
        ):
            assignee = self.users.get(data.assigned_to_user_id)
            if assignee is None:
                raise HTTPException(
                    status_code=http_status.HTTP_400_BAD_REQUEST,
                    detail=f"Assignee user {data.assigned_to_user_id} not found.",
                )
            task.assigned_to_user_id = assignee.id
            changed_fields.append("assigned_to_user_id")

        self.audit.log_action(
            user_id=current_user.id,
            action=AuditAction.UPDATE_TASK,
            entity_type="task",
            entity_id=task.id,
            details=(
                f"Updated fields: {', '.join(changed_fields)}"
                if changed_fields
                else "No-op update"
            ),
        )
        self.db.commit()
        self.db.refresh(task)
        return task

    def delete_task(self, task_id: int, current_user: User) -> None:
        task = self.get_task_or_404(task_id)
        if task.status in (TaskStatus.APPROVED, TaskStatus.COMPLETED):
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Approved or completed tasks cannot be deleted.",
            )
        self._ensure_creator_or_manager(task, current_user, action="delete")
        task_id_value = task.id
        title = task.title
        self.repo.delete(task)
        self.audit.log_action(
            user_id=current_user.id,
            action=AuditAction.DELETE_TASK,
            entity_type="task",
            entity_id=task_id_value,
            details=f"Deleted task '{title}'",
        )
        self.db.commit()

    def submit_task(self, task_id: int, current_user: User, comment: Optional[str] = None) -> Task:
        task = self.get_task_or_404(task_id)
        self._ensure_creator_or_manager(task, current_user, action="submit")
        self.workflow.transition(
            task=task,
            to_status=TaskStatus.SUBMITTED,
            changed_by=current_user,
            note=comment or "Submitted for review",
        )
        self.audit.log_action(
            user_id=current_user.id,
            action=AuditAction.SUBMIT_TASK,
            entity_type="task",
            entity_id=task.id,
            details=comment or "Submitted task",
        )
        self.db.commit()
        self.db.refresh(task)
        return task

    def approve_task(self, task_id: int, manager: User, comment: Optional[str] = None) -> Task:
        task = self.get_task_or_404(task_id)
        self.approvals.approve_task(task=task, manager=manager, comment=comment)
        self.audit.log_action(
            user_id=manager.id,
            action=AuditAction.APPROVE_TASK,
            entity_type="task",
            entity_id=task.id,
            details=comment or "Approved task",
        )
        self.db.commit()
        self.db.refresh(task)
        return task

    def reject_task(self, task_id: int, manager: User, comment: Optional[str] = None) -> Task:
        task = self.get_task_or_404(task_id)
        self.approvals.reject_task(task=task, manager=manager, comment=comment)
        self.audit.log_action(
            user_id=manager.id,
            action=AuditAction.REJECT_TASK,
            entity_type="task",
            entity_id=task.id,
            details=comment or "Rejected task",
        )
        self.db.commit()
        self.db.refresh(task)
        return task

    def complete_task(self, task_id: int, manager: User, comment: Optional[str] = None) -> Task:
        task = self.get_task_or_404(task_id)
        if not manager.is_manager:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Only managers can complete tasks.",
            )
        self.workflow.transition(
            task=task,
            to_status=TaskStatus.COMPLETED,
            changed_by=manager,
            note=comment or "Completed by manager",
        )
        self.audit.log_action(
            user_id=manager.id,
            action=AuditAction.COMPLETE_TASK,
            entity_type="task",
            entity_id=task.id,
            details=comment or "Completed task",
        )
        self.db.commit()
        self.db.refresh(task)
        return task
