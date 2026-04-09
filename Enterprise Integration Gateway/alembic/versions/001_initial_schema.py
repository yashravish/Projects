"""Initial schema: customers, orders, shipments, sync_jobs, failed_records

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── customers ──────────────────────────────────────────────────────────────
    op.create_table(
        "customers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("external_id", sa.String(100), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("phone", sa.String(50), nullable=True),
        sa.Column("company", sa.String(255), nullable=True),
        sa.Column("address_line1", sa.String(255), nullable=True),
        sa.Column("address_line2", sa.String(255), nullable=True),
        sa.Column("city", sa.String(100), nullable=True),
        sa.Column("state", sa.String(100), nullable=True),
        sa.Column("country", sa.String(100), nullable=True),
        sa.Column("postal_code", sa.String(20), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),
        sa.Column("raw_data", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
    )
    op.create_index("ix_customers_external_id", "customers", ["external_id"])
    op.create_index("ix_customers_email", "customers", ["email"])
    op.create_index("ix_customers_source_status", "customers", ["source", "status"])

    # ── orders ──────────────────────────────────────────────────────────────────
    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("external_id", sa.String(100), nullable=False),
        sa.Column("customer_id", sa.Integer(), sa.ForeignKey("customers.id"), nullable=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("order_number", sa.String(100), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("total_amount", sa.Numeric(12, 2), nullable=True),
        sa.Column("currency", sa.String(10), nullable=False, server_default="USD"),
        sa.Column("order_date", sa.DateTime(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("raw_data", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
    )
    op.create_index("ix_orders_external_id", "orders", ["external_id"])
    op.create_index("ix_orders_order_number", "orders", ["order_number"])
    op.create_index("ix_orders_customer_id", "orders", ["customer_id"])
    op.create_index("ix_orders_source_status", "orders", ["source", "status"])

    # ── shipments ──────────────────────────────────────────────────────────────
    op.create_table(
        "shipments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("external_id", sa.String(100), nullable=False),
        sa.Column("order_id", sa.Integer(), sa.ForeignKey("orders.id"), nullable=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("tracking_number", sa.String(100), nullable=True),
        sa.Column("carrier", sa.String(100), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("estimated_delivery", sa.DateTime(), nullable=True),
        sa.Column("actual_delivery", sa.DateTime(), nullable=True),
        sa.Column("weight_kg", sa.Numeric(8, 3), nullable=True),
        sa.Column("raw_data", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
    )
    op.create_index("ix_shipments_external_id", "shipments", ["external_id"])
    op.create_index("ix_shipments_tracking_number", "shipments", ["tracking_number"])
    op.create_index("ix_shipments_order_id", "shipments", ["order_id"])
    op.create_index("ix_shipments_source_status", "shipments", ["source", "status"])

    # ── sync_jobs ──────────────────────────────────────────────────────────────
    op.create_table(
        "sync_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("correlation_id", sa.String(36), nullable=False),
        sa.Column("job_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("triggered_by", sa.String(50), nullable=False, server_default="scheduler"),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("records_processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_inserted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_updated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("job_metadata", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("correlation_id"),
    )
    op.create_index("ix_sync_jobs_correlation_id", "sync_jobs", ["correlation_id"])
    op.create_index("ix_sync_jobs_job_type_status", "sync_jobs", ["job_type", "status"])
    op.create_index("ix_sync_jobs_created_at", "sync_jobs", ["created_at"])

    # ── failed_records ─────────────────────────────────────────────────────────
    op.create_table(
        "failed_records",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sync_job_id", sa.Integer(), sa.ForeignKey("sync_jobs.id"), nullable=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("record_type", sa.String(50), nullable=False),
        sa.Column("external_id", sa.String(100), nullable=True),
        sa.Column("raw_data", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending_retry"),
        sa.Column("last_retried_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_failed_records_external_id", "failed_records", ["external_id"])
    op.create_index("ix_failed_records_source_status", "failed_records", ["source", "status"])
    op.create_index("ix_failed_records_sync_job_id", "failed_records", ["sync_job_id"])


def downgrade() -> None:
    op.drop_table("failed_records")
    op.drop_table("sync_jobs")
    op.drop_table("shipments")
    op.drop_table("orders")
    op.drop_table("customers")
