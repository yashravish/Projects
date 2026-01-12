"""Add email field to invites table.

Revision ID: 002_add_invite_email
Revises: 001_initial_schema
Create Date: 2026-01-01

Required for invite acceptance flow where:
- Invite can optionally be tied to a specific email
- If email exists in invite and account exists, reject with ACCOUNT_EXISTS_LOGIN_REQUIRED
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers
revision = "002_add_invite_email"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add email column to invites table."""
    op.add_column(
        "invites",
        sa.Column("email", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    """Remove email column from invites table."""
    op.drop_column("invites", "email")

