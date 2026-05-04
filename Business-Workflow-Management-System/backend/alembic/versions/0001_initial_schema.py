"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-30 00:00:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


user_role_enum = sa.Enum("analyst", "manager", name="user_role")
task_status_enum = sa.Enum(
    "pending",
    "in_progress",
    "submitted",
    "approved",
    "rejected",
    "completed",
    name="task_status",
)
task_priority_enum = sa.Enum("low", "medium", "high", name="task_priority")
approval_decision_enum = sa.Enum("approved", "rejected", name="approval_decision")
audit_action_enum = sa.Enum(
    "login",
    "create_task",
    "update_task",
    "delete_task",
    "submit_task",
    "approve_task",
    "reject_task",
    "complete_task",
    name="audit_action",
)


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column(
            "role",
            user_role_enum,
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_role", "users", ["role"])

    op.create_table(
        "tasks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "status",
            task_status_enum,
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "priority",
            task_priority_enum,
            nullable=False,
            server_default="medium",
        ),
        sa.Column(
            "assigned_to_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "created_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_tasks_status", "tasks", ["status"])
    op.create_index("ix_tasks_priority", "tasks", ["priority"])
    op.create_index("ix_tasks_assigned_to_user_id", "tasks", ["assigned_to_user_id"])
    op.create_index("ix_tasks_created_by_user_id", "tasks", ["created_by_user_id"])
    op.create_index("ix_tasks_status_priority", "tasks", ["status", "priority"])

    op.create_table(
        "workflow_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "task_id",
            sa.Integer(),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "from_status",
            task_status_enum,
            nullable=True,
        ),
        sa.Column(
            "to_status",
            task_status_enum,
            nullable=False,
        ),
        sa.Column(
            "changed_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("note", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_workflow_events_task_id", "workflow_events", ["task_id"])
    op.create_index(
        "ix_workflow_events_changed_by_user_id",
        "workflow_events",
        ["changed_by_user_id"],
    )

    op.create_table(
        "approvals",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "task_id",
            sa.Integer(),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "approved_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "decision",
            approval_decision_enum,
            nullable=False,
        ),
        sa.Column("comment", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_approvals_task_id", "approvals", ["task_id"])
    op.create_index(
        "ix_approvals_approved_by_user_id",
        "approvals",
        ["approved_by_user_id"],
    )
    op.create_index("ix_approvals_decision", "approvals", ["decision"])

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "action",
            audit_action_enum,
            nullable=False,
        ),
        sa.Column("entity_type", sa.String(length=50), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=True),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_entity_type", "audit_logs", ["entity_type"])
    op.create_index("ix_audit_logs_entity_id", "audit_logs", ["entity_id"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_entity_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_entity_type", table_name="audit_logs")
    op.drop_index("ix_audit_logs_action", table_name="audit_logs")
    op.drop_index("ix_audit_logs_user_id", table_name="audit_logs")
    op.drop_table("audit_logs")

    op.drop_index("ix_approvals_decision", table_name="approvals")
    op.drop_index("ix_approvals_approved_by_user_id", table_name="approvals")
    op.drop_index("ix_approvals_task_id", table_name="approvals")
    op.drop_table("approvals")

    op.drop_index(
        "ix_workflow_events_changed_by_user_id",
        table_name="workflow_events",
    )
    op.drop_index("ix_workflow_events_task_id", table_name="workflow_events")
    op.drop_table("workflow_events")

    op.drop_index("ix_tasks_status_priority", table_name="tasks")
    op.drop_index("ix_tasks_created_by_user_id", table_name="tasks")
    op.drop_index("ix_tasks_assigned_to_user_id", table_name="tasks")
    op.drop_index("ix_tasks_priority", table_name="tasks")
    op.drop_index("ix_tasks_status", table_name="tasks")
    op.drop_table("tasks")

    op.drop_index("ix_users_role", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")

    bind = op.get_bind()
    audit_action_enum.drop(bind, checkfirst=True)
    approval_decision_enum.drop(bind, checkfirst=True)
    task_priority_enum.drop(bind, checkfirst=True)
    task_status_enum.drop(bind, checkfirst=True)
    user_role_enum.drop(bind, checkfirst=True)
