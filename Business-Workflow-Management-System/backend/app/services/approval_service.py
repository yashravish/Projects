"""Approval service - records approve/reject decisions on submitted tasks."""
from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, status as http_status
from sqlalchemy.orm import Session

from app.models.approval import Approval, ApprovalDecision
from app.models.task import Task, TaskStatus
from app.models.user import User
from app.services.workflow_service import WorkflowService


class ApprovalService:
    """Encapsulates the manager approval / rejection workflow."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.workflow = WorkflowService(db)

    def _ensure_manager(self, user: User) -> None:
        if not user.is_manager:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Only managers can approve or reject tasks.",
            )

    def _ensure_submitted(self, task: Task) -> None:
        if task.status != TaskStatus.SUBMITTED:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Only submitted tasks can be approved or rejected.",
            )

    def approve_task(self, task: Task, manager: User, comment: Optional[str] = None) -> Approval:
        return self._record_decision(
            task=task,
            manager=manager,
            comment=comment,
            decision=ApprovalDecision.APPROVED,
            target_status=TaskStatus.APPROVED,
            default_note="Approved by manager",
        )

    def reject_task(self, task: Task, manager: User, comment: Optional[str] = None) -> Approval:
        return self._record_decision(
            task=task,
            manager=manager,
            comment=comment,
            decision=ApprovalDecision.REJECTED,
            target_status=TaskStatus.REJECTED,
            default_note="Rejected by manager",
        )

    def _record_decision(
        self,
        *,
        task: Task,
        manager: User,
        comment: Optional[str],
        decision: ApprovalDecision,
        target_status: TaskStatus,
        default_note: str,
    ) -> Approval:
        self._ensure_manager(manager)
        self._ensure_submitted(task)
        self.workflow.transition(
            task=task,
            to_status=target_status,
            changed_by=manager,
            note=comment or default_note,
        )
        approval = Approval(
            task_id=task.id,
            approved_by_user_id=manager.id,
            decision=decision,
            comment=comment,
        )
        self.db.add(approval)
        self.db.flush()
        return approval
