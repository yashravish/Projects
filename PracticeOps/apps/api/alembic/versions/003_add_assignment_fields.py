"""Add missing fields to assignments table.

Revision ID: 003_add_assignment_fields
Revises: 002_add_invite_email
Create Date: 2026-01-01

Bug fix: Milestone 1 schema was missing required fields for Milestone 5:
- priority: Required for sorting (priority DESC)
- created_by: Required for RBAC (creator can edit)
- song_ref: Optional reference to a song
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers
revision = "003_add_assignment_fields"
down_revision = "002_add_invite_email"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add priority, created_by, and song_ref columns to assignments table."""
    # Add priority column with default value
    op.add_column(
        "assignments",
        sa.Column(
            "priority",
            sa.Enum("LOW", "MEDIUM", "BLOCKING", name="priority_enum", create_type=False),
            nullable=False,
            server_default="LOW",
        ),
    )

    # Add created_by column (nullable initially to allow existing rows)
    op.add_column(
        "assignments",
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )

    # Add song_ref column (optional)
    op.add_column(
        "assignments",
        sa.Column("song_ref", sa.String(100), nullable=True),
    )

    # Create index for priority-based queries
    op.create_index(
        "ix_assignments_cycle_priority",
        "assignments",
        ["cycle_id", "priority"],
    )


def downgrade() -> None:
    """Remove priority, created_by, and song_ref columns from assignments table."""
    op.drop_index("ix_assignments_cycle_priority", table_name="assignments")
    op.drop_column("assignments", "song_ref")
    op.drop_column("assignments", "created_by")
    op.drop_column("assignments", "priority")

