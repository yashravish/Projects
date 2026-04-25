"""training_jobs + registered_models

Adds the two tables that back the Stage 5 model registry:

  * `training_jobs`     — every train run (local or SageMaker) writes one row.
  * `registered_models` — once a job succeeds, its artifact is registered
                          here. `stage` is the lifecycle column
                          (staging / production / archived). The unique
                          (org, name, version) constraint prevents duplicate
                          registrations under the same logical name.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-25 13:30:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _timestamps() -> list[sa.Column]:  # type: ignore[type-arg]
    return [
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
    ]


def upgrade() -> None:
    op.create_table(
        "training_jobs",
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
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("backend", sa.Text(), nullable=False),
        sa.Column("framework", sa.Text(), nullable=False),
        sa.Column("framework_version", sa.Text(), nullable=True),
        sa.Column(
            "status", sa.Text(), nullable=False, server_default=sa.text("'pending'")
        ),
        sa.Column("artifact_uri", sa.Text(), nullable=True),
        sa.Column("external_job_id", sa.Text(), nullable=True),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "manifest",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("log_excerpt", sa.Text(), nullable=True),
        sa.Column(
            "duration_s",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("mlflow_run_id", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "status IN ('pending','running','success','failed')",
            name="training_jobs_status_check",
        ),
    )
    op.create_index(
        "ix_training_jobs_organization_id", "training_jobs", ["organization_id"]
    )
    op.create_index("ix_training_jobs_created_at", "training_jobs", ["created_at"])

    op.create_table(
        "registered_models",
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
            "training_job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("training_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "promoted_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("framework", sa.Text(), nullable=False),
        sa.Column("framework_version", sa.Text(), nullable=True),
        sa.Column("backend", sa.Text(), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("local_dir", sa.Text(), nullable=True),
        sa.Column(
            "stage",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'staging'"),
        ),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "manifest",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.CheckConstraint(
            "stage IN ('staging','production','archived')",
            name="registered_models_stage_check",
        ),
        sa.UniqueConstraint(
            "organization_id",
            "name",
            "version",
            name="uq_registered_models_org_name_version",
        ),
    )
    op.create_index(
        "ix_registered_models_organization_id",
        "registered_models",
        ["organization_id"],
    )
    op.create_index(
        "ix_registered_models_name_stage",
        "registered_models",
        ["name", "stage"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_registered_models_name_stage", table_name="registered_models"
    )
    op.drop_index(
        "ix_registered_models_organization_id", table_name="registered_models"
    )
    op.drop_table("registered_models")
    op.drop_index("ix_training_jobs_created_at", table_name="training_jobs")
    op.drop_index(
        "ix_training_jobs_organization_id", table_name="training_jobs"
    )
    op.drop_table("training_jobs")
