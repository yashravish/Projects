"""Add workflow fields to tickets table.

Revision ID: 006_add_ticket_workflow_fields
Revises: 005_add_ticket_claimable
Create Date: 2026-01-01

Milestone 1 gap fix: The original schema was missing fields required
for status transitions and verification workflow:
- resolved_note: Note provided when resolving a ticket
- verified_by: FK to users.id, the leader who verified the ticket
- verified_note: Optional note provided during verification
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers
revision = "006_add_ticket_workflow_fields"
down_revision = "005_add_ticket_claimable"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add workflow fields to tickets table."""
    # Add resolved_note column
    op.add_column(
        "tickets",
        sa.Column("resolved_note", sa.Text(), nullable=True),
    )

    # Add verified_by column with FK to users
    op.add_column(
        "tickets",
        sa.Column("verified_by", sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        op.f("fk_tickets_verified_by_users"),
        "tickets",
        "users",
        ["verified_by"],
        ["id"],
        ondelete="SET NULL",
    )

    # Add verified_note column
    op.add_column(
        "tickets",
        sa.Column("verified_note", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Remove workflow fields from tickets table."""
    op.drop_constraint(
        op.f("fk_tickets_verified_by_users"), "tickets", type_="foreignkey"
    )
    op.drop_column("tickets", "verified_note")
    op.drop_column("tickets", "verified_by")
    op.drop_column("tickets", "resolved_note")

