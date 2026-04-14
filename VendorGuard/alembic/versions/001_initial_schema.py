"""Initial schema

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("username", sa.String(100), unique=True, nullable=False, index=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), server_default=""),
        sa.Column("role", sa.String(50), server_default="analyst"),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "vendors",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, index=True),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("website", sa.String(500), server_default=""),
        sa.Column("business_owner", sa.String(255), server_default=""),
        sa.Column("vendor_contact", sa.String(255), server_default=""),
        sa.Column("hosting_model", sa.String(100), server_default=""),
        sa.Column("deployment_scope", sa.String(100), server_default=""),
        sa.Column("internet_exposed", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("handles_sensitive_data", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("data_types_json", sa.Text(), server_default="[]"),
        sa.Column("compliance_attestations_json", sa.Text(), server_default="[]"),
        sa.Column("status", sa.String(50), server_default="active"),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "vendor_integrations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("vendor_id", sa.Integer(), sa.ForeignKey("vendors.id"), nullable=False),
        sa.Column("system_name", sa.String(255), nullable=False),
        sa.Column("integration_type", sa.String(100), server_default=""),
        sa.Column("data_flow_direction", sa.String(50), server_default="bidirectional"),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "control_domains",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("code", sa.String(10), unique=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("nist_mapping", sa.String(255), server_default=""),
        sa.Column("iso_mapping", sa.String(255), server_default=""),
    )

    op.create_table(
        "assessments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("vendor_id", sa.Integer(), sa.ForeignKey("vendors.id"), nullable=False),
        sa.Column("assessment_type", sa.String(100), server_default="initial"),
        sa.Column("phase", sa.String(50), server_default="pre_implementation"),
        sa.Column("assessor_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("overall_inherent_risk", sa.String(50), nullable=True),
        sa.Column("inherent_risk_score", sa.Float(), nullable=True),
        sa.Column("overall_residual_risk", sa.String(50), nullable=True),
        sa.Column("residual_risk_score", sa.Float(), nullable=True),
        sa.Column("status", sa.String(50), server_default="draft"),
        sa.Column("executive_summary", sa.Text(), nullable=True),
        sa.Column("ai_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "assessment_answers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("assessment_id", sa.Integer(), sa.ForeignKey("assessments.id"), nullable=False),
        sa.Column("question_key", sa.String(255), nullable=False),
        sa.Column("section", sa.String(100), server_default=""),
        sa.Column("question_text", sa.Text(), server_default=""),
        sa.Column("answer", sa.Text(), server_default=""),
        sa.Column("notes", sa.Text(), server_default=""),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "findings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("assessment_id", sa.Integer(), sa.ForeignKey("assessments.id"), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("severity", sa.String(50), server_default="Low"),
        sa.Column("likelihood", sa.String(50), server_default="Moderate"),
        sa.Column("impact", sa.String(50), server_default="Moderate"),
        sa.Column("control_domain_id", sa.Integer(), sa.ForeignKey("control_domains.id"), nullable=True),
        sa.Column("recommendation", sa.Text(), server_default=""),
        sa.Column("owner", sa.String(255), server_default=""),
        sa.Column("due_date", sa.Date(), nullable=True),
        sa.Column("remediation_status", sa.String(50), server_default="open"),
        sa.Column("source_rule", sa.String(255), server_default=""),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "remediation_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("finding_id", sa.Integer(), sa.ForeignKey("findings.id"), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("assigned_to", sa.String(255), server_default=""),
        sa.Column("priority", sa.String(50), server_default="Medium"),
        sa.Column("status", sa.String(50), server_default="open"),
        sa.Column("due_date", sa.Date(), nullable=True),
        sa.Column("completion_date", sa.Date(), nullable=True),
        sa.Column("notes", sa.Text(), server_default=""),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "assessment_templates",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("category", sa.String(100), server_default="general"),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("questions_json", sa.Text(), server_default="[]"),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("action", sa.String(255), nullable=False),
        sa.Column("entity_type", sa.String(100), server_default=""),
        sa.Column("entity_id", sa.Integer(), nullable=True),
        sa.Column("details", sa.Text(), server_default=""),
        sa.Column("ip_address", sa.String(50), server_default=""),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "generated_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("assessment_id", sa.Integer(), sa.ForeignKey("assessments.id"), nullable=False),
        sa.Column("report_type", sa.String(100), server_default="full"),
        sa.Column("file_path", sa.String(500), nullable=True),
        sa.Column("generated_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("generated_reports")
    op.drop_table("audit_logs")
    op.drop_table("assessment_templates")
    op.drop_table("remediation_items")
    op.drop_table("findings")
    op.drop_table("assessment_answers")
    op.drop_table("assessments")
    op.drop_table("control_domains")
    op.drop_table("vendor_integrations")
    op.drop_table("vendors")
    op.drop_table("users")
