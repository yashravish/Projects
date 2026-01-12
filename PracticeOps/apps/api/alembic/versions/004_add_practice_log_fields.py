"""Add missing fields to practice_logs table.

Revision ID: 004_add_practice_log_fields
Revises: 003_add_assignment_fields
Create Date: 2026-01-01

Bug fix: Milestone 1 schema was missing required fields for Milestone 6:
- rating_1_5: Optional 1-5 practice quality rating
- blocked_flag: Whether user was blocked during practice
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers
revision = "004_add_practice_log_fields"
down_revision = "003_add_assignment_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add rating_1_5 and blocked_flag columns to practice_logs table."""
    # Add rating_1_5 column (1-5 scale, nullable)
    op.add_column(
        "practice_logs",
        sa.Column("rating_1_5", sa.Integer(), nullable=True),
    )

    # Add blocked_flag column with default False
    op.add_column(
        "practice_logs",
        sa.Column(
            "blocked_flag",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    """Remove rating_1_5 and blocked_flag columns from practice_logs table."""
    op.drop_column("practice_logs", "blocked_flag")
    op.drop_column("practice_logs", "rating_1_5")

