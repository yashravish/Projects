"""Extend notification_preferences with no_log_days and weekly_digest_enabled.

Revision ID: 007
Revises: 006
Create Date: 2026-01-02

These fields are required for Milestone 11 notification jobs:
- no_log_days: Number of days without practice before sending a reminder
- weekly_digest_enabled: Whether to include user in weekly leader digest
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "007_notification_prefs"
down_revision = "006_add_ticket_workflow_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add no_log_days column with default value 3
    op.add_column(
        "notification_preferences",
        sa.Column(
            "no_log_days",
            sa.Integer(),
            server_default="3",
            nullable=False,
        ),
    )

    # Add weekly_digest_enabled column with default value true
    op.add_column(
        "notification_preferences",
        sa.Column(
            "weekly_digest_enabled",
            sa.Boolean(),
            server_default="true",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("notification_preferences", "weekly_digest_enabled")
    op.drop_column("notification_preferences", "no_log_days")

