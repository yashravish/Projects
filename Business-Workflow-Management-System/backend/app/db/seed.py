"""Seed script for development.

Inserts demo users, tasks, workflow events, approvals, and audit logs so the
app feels like a real business system on first launch.

Idempotent: skips if seed users already exist.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.db.session import SessionLocal
from app.models.approval import Approval, ApprovalDecision
from app.models.audit_log import AuditAction, AuditLog
from app.models.task import Task, TaskPriority, TaskStatus
from app.models.user import User, UserRole
from app.models.workflow_event import WorkflowEvent


logger = logging.getLogger("seed")


SEED_USERS = [
    {
        "name": "Megan Manager",
        "email": "manager@example.com",
        "password": "password123",
        "role": UserRole.MANAGER,
    },
    {
        "name": "Marcus Director",
        "email": "marcus.manager@example.com",
        "password": "password123",
        "role": UserRole.MANAGER,
    },
    {
        "name": "Anna Analyst",
        "email": "analyst@example.com",
        "password": "password123",
        "role": UserRole.ANALYST,
    },
    {
        "name": "Brian Carter",
        "email": "brian.analyst@example.com",
        "password": "password123",
        "role": UserRole.ANALYST,
    },
    {
        "name": "Carla Reyes",
        "email": "carla.analyst@example.com",
        "password": "password123",
        "role": UserRole.ANALYST,
    },
]


SEED_TASKS = [
    {
        "title": "Q3 vendor contract renewal review",
        "description": "Review renewal terms for vendor #482, flag pricing changes.",
        "priority": TaskPriority.HIGH,
        "status": TaskStatus.PENDING,
        "assignee_email": "analyst@example.com",
        "creator_email": "analyst@example.com",
    },
    {
        "title": "Reconcile June expense report variances",
        "description": "Compare GL postings to expense reports and document gaps.",
        "priority": TaskPriority.MEDIUM,
        "status": TaskStatus.IN_PROGRESS,
        "assignee_email": "analyst@example.com",
        "creator_email": "analyst@example.com",
    },
    {
        "title": "Customer onboarding SLA audit",
        "description": "Audit last 30 onboarding tickets for SLA breaches.",
        "priority": TaskPriority.MEDIUM,
        "status": TaskStatus.SUBMITTED,
        "assignee_email": "brian.analyst@example.com",
        "creator_email": "brian.analyst@example.com",
    },
    {
        "title": "Update procurement workflow documentation",
        "description": "Refresh runbook with the new approval thresholds.",
        "priority": TaskPriority.LOW,
        "status": TaskStatus.SUBMITTED,
        "assignee_email": "carla.analyst@example.com",
        "creator_email": "carla.analyst@example.com",
    },
    {
        "title": "Investigate duplicate invoice posting INV-2034",
        "description": "Trace duplicate posting and submit corrective entry.",
        "priority": TaskPriority.HIGH,
        "status": TaskStatus.APPROVED,
        "assignee_email": "analyst@example.com",
        "creator_email": "analyst@example.com",
    },
    {
        "title": "Annual access review - Finance group",
        "description": "Confirm role mappings for the finance Active Directory group.",
        "priority": TaskPriority.MEDIUM,
        "status": TaskStatus.COMPLETED,
        "assignee_email": "brian.analyst@example.com",
        "creator_email": "brian.analyst@example.com",
    },
    {
        "title": "Dashboard data refresh failure RCA",
        "description": "Root-cause the Tuesday morning ETL failure and document remediation.",
        "priority": TaskPriority.HIGH,
        "status": TaskStatus.REJECTED,
        "assignee_email": "carla.analyst@example.com",
        "creator_email": "carla.analyst@example.com",
    },
    {
        "title": "Quarterly headcount reconciliation",
        "description": "Reconcile HRIS headcount against the GL allocation file.",
        "priority": TaskPriority.MEDIUM,
        "status": TaskStatus.PENDING,
        "assignee_email": "carla.analyst@example.com",
        "creator_email": "carla.analyst@example.com",
    },
    {
        "title": "Refactor month-end close checklist",
        "description": "Modernize the close checklist template based on Q2 retro feedback.",
        "priority": TaskPriority.LOW,
        "status": TaskStatus.IN_PROGRESS,
        "assignee_email": "brian.analyst@example.com",
        "creator_email": "brian.analyst@example.com",
    },
    {
        "title": "Approve travel reimbursement batch #214",
        "description": "Validate batch totals and route to AP for payment.",
        "priority": TaskPriority.MEDIUM,
        "status": TaskStatus.SUBMITTED,
        "assignee_email": "analyst@example.com",
        "creator_email": "analyst@example.com",
    },
    {
        "title": "Vendor risk assessment - data processor X",
        "description": "Complete the risk assessment template and attach SOC 2.",
        "priority": TaskPriority.HIGH,
        "status": TaskStatus.APPROVED,
        "assignee_email": "carla.analyst@example.com",
        "creator_email": "carla.analyst@example.com",
    },
    {
        "title": "Backfill missing audit log entries for May 14",
        "description": "Coordinate with platform team to replay missing audit events.",
        "priority": TaskPriority.LOW,
        "status": TaskStatus.COMPLETED,
        "assignee_email": "analyst@example.com",
        "creator_email": "analyst@example.com",
    },
]


def _build_user(data: dict) -> User:
    return User(
        name=data["name"],
        email=data["email"],
        hashed_password=hash_password(data["password"]),
        role=data["role"],
    )


def _users_by_email(users: List[User]) -> dict[str, User]:
    return {user.email: user for user in users}


def _drive_status(
    db: Session,
    task: Task,
    target_status: TaskStatus,
    creator: User,
    manager: User,
) -> None:
    """Walk the workflow state machine until the task reaches target_status."""
    history: List[tuple[TaskStatus | None, TaskStatus, User, str]] = [
        (None, TaskStatus.PENDING, creator, "Task created"),
    ]

    if target_status == TaskStatus.PENDING:
        pass
    elif target_status == TaskStatus.IN_PROGRESS:
        history.append((TaskStatus.PENDING, TaskStatus.IN_PROGRESS, creator, "Started work"))
    elif target_status == TaskStatus.SUBMITTED:
        history.append((TaskStatus.PENDING, TaskStatus.IN_PROGRESS, creator, "Started work"))
        history.append(
            (TaskStatus.IN_PROGRESS, TaskStatus.SUBMITTED, creator, "Submitted for review")
        )
    elif target_status == TaskStatus.APPROVED:
        history.append((TaskStatus.PENDING, TaskStatus.IN_PROGRESS, creator, "Started work"))
        history.append(
            (TaskStatus.IN_PROGRESS, TaskStatus.SUBMITTED, creator, "Submitted for review")
        )
        history.append(
            (TaskStatus.SUBMITTED, TaskStatus.APPROVED, manager, "Approved by manager")
        )
    elif target_status == TaskStatus.REJECTED:
        history.append((TaskStatus.PENDING, TaskStatus.IN_PROGRESS, creator, "Started work"))
        history.append(
            (TaskStatus.IN_PROGRESS, TaskStatus.SUBMITTED, creator, "Submitted for review")
        )
        history.append(
            (
                TaskStatus.SUBMITTED,
                TaskStatus.REJECTED,
                manager,
                "Needs more detail before approval",
            )
        )
    elif target_status == TaskStatus.COMPLETED:
        history.append((TaskStatus.PENDING, TaskStatus.IN_PROGRESS, creator, "Started work"))
        history.append(
            (TaskStatus.IN_PROGRESS, TaskStatus.SUBMITTED, creator, "Submitted for review")
        )
        history.append(
            (TaskStatus.SUBMITTED, TaskStatus.APPROVED, manager, "Approved by manager")
        )
        history.append(
            (TaskStatus.APPROVED, TaskStatus.COMPLETED, manager, "Completed")
        )

    for from_status, to_status, actor, note in history:
        event = WorkflowEvent(
            task_id=task.id,
            from_status=from_status,
            to_status=to_status,
            changed_by_user_id=actor.id,
            note=note,
        )
        db.add(event)

    task.status = target_status

    if target_status == TaskStatus.APPROVED:
        db.add(
            Approval(
                task_id=task.id,
                approved_by_user_id=manager.id,
                decision=ApprovalDecision.APPROVED,
                comment="Looks good - proceed.",
            )
        )
    elif target_status == TaskStatus.REJECTED:
        db.add(
            Approval(
                task_id=task.id,
                approved_by_user_id=manager.id,
                decision=ApprovalDecision.REJECTED,
                comment="Please add supporting evidence.",
            )
        )
    elif target_status == TaskStatus.COMPLETED:
        db.add(
            Approval(
                task_id=task.id,
                approved_by_user_id=manager.id,
                decision=ApprovalDecision.APPROVED,
                comment="Approved and later completed.",
            )
        )


def _add_audit_logs(db: Session, manager: User, analyst: User, tasks: List[Task]) -> None:
    db.add(
        AuditLog(
            user_id=manager.id,
            action=AuditAction.LOGIN,
            entity_type="user",
            entity_id=manager.id,
            details=f"User {manager.email} logged in",
        )
    )
    db.add(
        AuditLog(
            user_id=analyst.id,
            action=AuditAction.LOGIN,
            entity_type="user",
            entity_id=analyst.id,
            details=f"User {analyst.email} logged in",
        )
    )
    for task in tasks:
        db.add(
            AuditLog(
                user_id=task.created_by_user_id,
                action=AuditAction.CREATE_TASK,
                entity_type="task",
                entity_id=task.id,
                details=f"Created task '{task.title}'",
            )
        )
        if task.status in (
            TaskStatus.SUBMITTED,
            TaskStatus.APPROVED,
            TaskStatus.REJECTED,
            TaskStatus.COMPLETED,
        ):
            db.add(
                AuditLog(
                    user_id=task.created_by_user_id,
                    action=AuditAction.SUBMIT_TASK,
                    entity_type="task",
                    entity_id=task.id,
                    details="Submitted task",
                )
            )
        if task.status == TaskStatus.APPROVED:
            db.add(
                AuditLog(
                    user_id=manager.id,
                    action=AuditAction.APPROVE_TASK,
                    entity_type="task",
                    entity_id=task.id,
                    details="Approved task",
                )
            )
        elif task.status == TaskStatus.REJECTED:
            db.add(
                AuditLog(
                    user_id=manager.id,
                    action=AuditAction.REJECT_TASK,
                    entity_type="task",
                    entity_id=task.id,
                    details="Rejected task",
                )
            )
        elif task.status == TaskStatus.COMPLETED:
            db.add(
                AuditLog(
                    user_id=manager.id,
                    action=AuditAction.APPROVE_TASK,
                    entity_type="task",
                    entity_id=task.id,
                    details="Approved task",
                )
            )
            db.add(
                AuditLog(
                    user_id=manager.id,
                    action=AuditAction.COMPLETE_TASK,
                    entity_type="task",
                    entity_id=task.id,
                    details="Completed task",
                )
            )


def seed(db: Session) -> None:
    existing = db.scalars(
        select(User).where(User.email == "manager@example.com")
    ).first()
    if existing is not None:
        logger.info("Seed data already present, skipping.")
        return

    logger.info("Seeding users...")
    users = [_build_user(u) for u in SEED_USERS]
    db.add_all(users)
    db.flush()
    by_email = _users_by_email(users)

    primary_manager = by_email["manager@example.com"]

    logger.info("Seeding tasks, workflow events, approvals, and audit logs...")
    created_tasks: List[Task] = []
    for spec in SEED_TASKS:
        assignee = by_email[spec["assignee_email"]]
        creator = by_email[spec["creator_email"]]
        task = Task(
            title=spec["title"],
            description=spec["description"],
            priority=spec["priority"],
            status=TaskStatus.PENDING,
            assigned_to_user_id=assignee.id,
            created_by_user_id=creator.id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(task)
        db.flush()
        _drive_status(
            db=db,
            task=task,
            target_status=spec["status"],
            creator=creator,
            manager=primary_manager,
        )
        created_tasks.append(task)

    _add_audit_logs(
        db=db,
        manager=primary_manager,
        analyst=by_email["analyst@example.com"],
        tasks=created_tasks,
    )

    db.commit()
    logger.info("Seed complete: %s users, %s tasks.", len(users), len(created_tasks))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    db = SessionLocal()
    try:
        seed(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
