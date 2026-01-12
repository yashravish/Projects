"""Add claimable field and make owner_id nullable on tickets.

Revision ID: 005_add_ticket_claimable
Revises: 004_add_practice_log_fields
Create Date: 2026-01-01

Bug fix: Milestone 1 schema was missing required fields for Milestone 7:
- claimable: Whether ticket can be claimed by team members
- owner_id: Must be nullable for claimable tickets (no owner until claimed)
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers
revision = "005_add_ticket_claimable"
down_revision = "004_add_practice_log_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add claimable column and make owner_id nullable."""
    # Add claimable column with default False
    op.add_column(
        "tickets",
        sa.Column(
            "claimable",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    # Make owner_id nullable for claimable tickets
    op.alter_column(
        "tickets",
        "owner_id",
        existing_type=sa.UUID(),
        nullable=True,
    )


def downgrade() -> None:
    """Remove claimable column and make owner_id NOT NULL."""
    # Make owner_id NOT NULL again
    # Note: This will fail if there are NULL owner_id values
    op.alter_column(
        "tickets",
        "owner_id",
        existing_type=sa.UUID(),
        nullable=False,
    )

    op.drop_column("tickets", "claimable")

