"""audit chain + retention policies

Stage 6 — extends the existing `audit_logs` table with tamper-evident chain
columns (`prev_hash`, `entry_hash`), an `outcome` enum, and a `request_id`
field; adds two new tables for governance:

  * `retention_policies` — one row per (organization, resource_type) → TTL
                            in days. ttl_days=0 means retain forever.
  * `retention_runs`     — log of executed retention sweeps with per-resource
                            purged counts.

The unique `(organization_id, entry_hash)` index makes the per-tenant hash
chain auditable in O(n) without scanning rows from sibling tenants.

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-25 19:30:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── audit_logs: chain columns + outcome + request_id ──────────────────
    op.add_column(
        "audit_logs",
        sa.Column(
            "outcome",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'success'"),
        ),
    )
    op.add_column(
        "audit_logs",
        sa.Column("request_id", sa.Text(), nullable=True),
    )
    op.add_column(
        "audit_logs",
        sa.Column("prev_hash", sa.Text(), nullable=True),
    )
    # `entry_hash` cannot be NOT NULL while existing rows are unbacfilled.
    # In practice the table is empty at this point in every deployment we
    # ship — the prior stage tests TRUNCATE it on every run and no service
    # writes to it. We backfill any straggler rows with a deterministic
    # hash of (id, action, resource_type, created_at) so the constraint
    # holds even when running against an old database.
    op.add_column(
        "audit_logs",
        sa.Column("entry_hash", sa.Text(), nullable=True),
    )
    op.execute(
        """
        UPDATE audit_logs
        SET entry_hash = encode(
            digest(
                COALESCE(id::text, '') || ':' ||
                COALESCE(action, '') || ':' ||
                COALESCE(resource_type, '') || ':' ||
                COALESCE(created_at::text, ''),
                'sha256'
            ),
            'hex'
        )
        WHERE entry_hash IS NULL
        """
    )
    op.alter_column("audit_logs", "entry_hash", nullable=False)

    op.create_check_constraint(
        "audit_logs_outcome_check",
        "audit_logs",
        "outcome IN ('success','denied','error')",
    )
    op.create_index(
        "ix_audit_logs_org_resource",
        "audit_logs",
        ["organization_id", "resource_type", "resource_id"],
    )
    op.create_unique_constraint(
        "uq_audit_logs_org_entry_hash",
        "audit_logs",
        ["organization_id", "entry_hash"],
    )

    # ── retention_policies ────────────────────────────────────────────────
    op.create_table(
        "retention_policies",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("resource_type", sa.Text(), nullable=False),
        sa.Column(
            "ttl_days",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "organization_id",
            "resource_type",
            name="uq_retention_policies_org_resource",
        ),
        sa.CheckConstraint("ttl_days >= 0", name="retention_policies_ttl_nonneg"),
    )
    op.create_index(
        "ix_retention_policies_organization_id",
        "retention_policies",
        ["organization_id"],
    )

    # ── retention_runs ────────────────────────────────────────────────────
    op.create_table(
        "retention_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "triggered_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'running'"),
        ),
        sa.Column(
            "purged_counts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running','success','failed')",
            name="retention_runs_status_check",
        ),
    )
    op.create_index(
        "ix_retention_runs_organization_id",
        "retention_runs",
        ["organization_id"],
    )
    op.create_index(
        "ix_retention_runs_started_at",
        "retention_runs",
        ["started_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_retention_runs_started_at", table_name="retention_runs"
    )
    op.drop_index(
        "ix_retention_runs_organization_id", table_name="retention_runs"
    )
    op.drop_table("retention_runs")

    op.drop_index(
        "ix_retention_policies_organization_id", table_name="retention_policies"
    )
    op.drop_table("retention_policies")

    op.drop_constraint(
        "uq_audit_logs_org_entry_hash", "audit_logs", type_="unique"
    )
    op.drop_index("ix_audit_logs_org_resource", table_name="audit_logs")
    op.drop_constraint(
        "audit_logs_outcome_check", "audit_logs", type_="check"
    )
    op.drop_column("audit_logs", "entry_hash")
    op.drop_column("audit_logs", "prev_hash")
    op.drop_column("audit_logs", "request_id")
    op.drop_column("audit_logs", "outcome")
