"""Reporting service - SQL aggregation queries for the dashboard and reports."""
from __future__ import annotations

import csv
import io
from typing import Dict, List

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.approval import Approval, ApprovalDecision
from app.models.task import Task, TaskPriority, TaskStatus
from app.models.user import User
from app.schemas.report import (
    ApprovalSummaryEntry,
    ApprovalSummaryReport,
    PriorityCount,
    StatusCount,
    TasksByPriorityReport,
    TasksByStatusReport,
    UserWorkloadEntry,
    UserWorkloadReport,
)


class ReportService:
    """Aggregation queries grouped by status, priority, user, and decision."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def tasks_by_status(self) -> TasksByStatusReport:
        stmt = select(Task.status, func.count(Task.id)).group_by(Task.status)
        rows = self.db.execute(stmt).all()
        counts: Dict[TaskStatus, int] = {s: 0 for s in TaskStatus}
        for status, count in rows:
            counts[status] = int(count)
        items = [StatusCount(status=key, count=value) for key, value in counts.items()]
        return TasksByStatusReport(items=items, total=sum(counts.values()))

    def tasks_by_priority(self) -> TasksByPriorityReport:
        stmt = select(Task.priority, func.count(Task.id)).group_by(Task.priority)
        rows = self.db.execute(stmt).all()
        counts: Dict[TaskPriority, int] = {p: 0 for p in TaskPriority}
        for priority, count in rows:
            counts[priority] = int(count)
        items = [PriorityCount(priority=key, count=value) for key, value in counts.items()]
        return TasksByPriorityReport(items=items, total=sum(counts.values()))

    def user_workload(self) -> UserWorkloadReport:
        users_stmt = select(User).order_by(User.name)
        users = list(self.db.scalars(users_stmt).all())

        breakdown_stmt = (
            select(Task.assigned_to_user_id, Task.status, func.count(Task.id))
            .group_by(Task.assigned_to_user_id, Task.status)
        )
        rows = self.db.execute(breakdown_stmt).all()

        per_user: Dict[int, Dict[TaskStatus, int]] = {
            user.id: {s: 0 for s in TaskStatus} for user in users
        }
        for user_id, status, count in rows:
            if user_id in per_user:
                per_user[user_id][status] = int(count)

        items: List[UserWorkloadEntry] = []
        for user in users:
            counts = per_user[user.id]
            items.append(
                UserWorkloadEntry(
                    user_id=user.id,
                    name=user.name,
                    email=user.email,
                    role=user.role,
                    total=sum(counts.values()),
                    pending=counts[TaskStatus.PENDING],
                    in_progress=counts[TaskStatus.IN_PROGRESS],
                    submitted=counts[TaskStatus.SUBMITTED],
                    approved=counts[TaskStatus.APPROVED],
                    rejected=counts[TaskStatus.REJECTED],
                    completed=counts[TaskStatus.COMPLETED],
                )
            )
        return UserWorkloadReport(items=items)

    def approval_summary(self) -> ApprovalSummaryReport:
        stmt = (
            select(Approval.decision, func.count(Approval.id))
            .group_by(Approval.decision)
        )
        rows = self.db.execute(stmt).all()
        counts: Dict[ApprovalDecision, int] = {d: 0 for d in ApprovalDecision}
        for decision, count in rows:
            counts[decision] = int(count)
        items = [
            ApprovalSummaryEntry(decision=key, count=value) for key, value in counts.items()
        ]
        return ApprovalSummaryReport(items=items, total=sum(counts.values()))

    def export_tasks_csv(self) -> str:
        """Build a CSV string containing all tasks for download."""
        stmt = (
            select(Task, User.name.label("assignee_name"), User.email.label("assignee_email"))
            .join(User, User.id == Task.assigned_to_user_id)
            .order_by(Task.id)
        )
        rows = self.db.execute(stmt).all()

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "id",
                "title",
                "description",
                "status",
                "priority",
                "assignee_name",
                "assignee_email",
                "created_by_user_id",
                "created_at",
                "updated_at",
            ]
        )
        for task, assignee_name, assignee_email in rows:
            writer.writerow(
                [
                    task.id,
                    task.title,
                    task.description.replace("\n", " "),
                    task.status.value,
                    task.priority.value,
                    assignee_name,
                    assignee_email,
                    task.created_by_user_id,
                    task.created_at.isoformat() if task.created_at else "",
                    task.updated_at.isoformat() if task.updated_at else "",
                ]
            )
        return buffer.getvalue()
