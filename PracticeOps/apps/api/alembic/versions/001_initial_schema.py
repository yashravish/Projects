"""Initial database schema.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-01-01

Creates all tables, enums, constraints, and indexes for PracticeOps.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ==========================================================================
    # ENUMS - Create PostgreSQL enum types first
    # ==========================================================================

    role_enum = postgresql.ENUM(
        "MEMBER", "SECTION_LEADER", "ADMIN",
        name="role_enum",
        create_type=True,
    )
    role_enum.create(op.get_bind(), checkfirst=True)

    assignment_type_enum = postgresql.ENUM(
        "SONG_WORK", "TECHNIQUE", "MEMORIZATION", "LISTENING",
        name="assignment_type_enum",
        create_type=True,
    )
    assignment_type_enum.create(op.get_bind(), checkfirst=True)

    assignment_scope_enum = postgresql.ENUM(
        "TEAM", "SECTION",
        name="assignment_scope_enum",
        create_type=True,
    )
    assignment_scope_enum.create(op.get_bind(), checkfirst=True)

    priority_enum = postgresql.ENUM(
        "LOW", "MEDIUM", "BLOCKING",
        name="priority_enum",
        create_type=True,
    )
    priority_enum.create(op.get_bind(), checkfirst=True)

    ticket_category_enum = postgresql.ENUM(
        "PITCH", "RHYTHM", "MEMORY", "BLEND", "TECHNIQUE", "OTHER",
        name="ticket_category_enum",
        create_type=True,
    )
    ticket_category_enum.create(op.get_bind(), checkfirst=True)

    ticket_visibility_enum = postgresql.ENUM(
        "PRIVATE", "SECTION", "TEAM",
        name="ticket_visibility_enum",
        create_type=True,
    )
    ticket_visibility_enum.create(op.get_bind(), checkfirst=True)

    ticket_status_enum = postgresql.ENUM(
        "OPEN", "IN_PROGRESS", "BLOCKED", "RESOLVED", "VERIFIED",
        name="ticket_status_enum",
        create_type=True,
    )
    ticket_status_enum.create(op.get_bind(), checkfirst=True)

    ticket_activity_type_enum = postgresql.ENUM(
        "CREATED", "COMMENT", "STATUS_CHANGE", "VERIFIED", "CLAIMED", "REASSIGNED",
        name="ticket_activity_type_enum",
        create_type=True,
    )
    ticket_activity_type_enum.create(op.get_bind(), checkfirst=True)

    # ==========================================================================
    # TABLES - Create in dependency order
    # ==========================================================================

    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # --- teams ---
    op.create_table(
        "teams",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # --- team_memberships ---
    op.create_table(
        "team_memberships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "team_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("teams.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "role",
            postgresql.ENUM(name="role_enum", create_type=False),
            nullable=False,
        ),
        sa.Column("section", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("team_id", "user_id", name="uq_team_memberships_team_user"),
    )
    op.create_index(
        "ix_team_memberships_team_role",
        "team_memberships",
        ["team_id", "role"],
    )
    op.create_index(
        "ix_team_memberships_team_section",
        "team_memberships",
        ["team_id", "section"],
    )

    # --- rehearsal_cycles ---
    op.create_table(
        "rehearsal_cycles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "team_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("teams.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("team_id", "date", name="uq_rehearsal_cycles_team_date"),
    )

    # --- assignments ---
    op.create_table(
        "assignments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "cycle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("rehearsal_cycles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "type",
            postgresql.ENUM(name="assignment_type_enum", create_type=False),
            nullable=False,
        ),
        sa.Column(
            "scope",
            postgresql.ENUM(name="assignment_scope_enum", create_type=False),
            nullable=False,
        ),
        sa.Column("section", sa.String(50), nullable=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_assignments_cycle_scope",
        "assignments",
        ["cycle_id", "scope"],
    )
    op.create_index(
        "ix_assignments_cycle_section",
        "assignments",
        ["cycle_id", "section"],
    )

    # --- practice_logs ---
    op.create_table(
        "practice_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "team_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("teams.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "cycle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("rehearsal_cycles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("duration_minutes", sa.Integer, nullable=False),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_practice_logs_user_cycle",
        "practice_logs",
        ["user_id", "cycle_id"],
    )
    op.create_index(
        "ix_practice_logs_team_cycle",
        "practice_logs",
        ["team_id", "cycle_id"],
    )
    op.create_index(
        "ix_practice_logs_cycle_occurred",
        "practice_logs",
        ["cycle_id", sa.text("occurred_at DESC")],
    )

    # --- practice_log_assignments ---
    op.create_table(
        "practice_log_assignments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "practice_log_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("practice_logs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "assignment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("assignments.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )

    # --- invites ---
    op.create_table(
        "invites",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "team_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("teams.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token", sa.String(64), unique=True, nullable=False),
        sa.Column(
            "role",
            postgresql.ENUM(name="role_enum", create_type=False),
            nullable=False,
        ),
        sa.Column("section", sa.String(50), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # --- tickets ---
    op.create_table(
        "tickets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "team_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("teams.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "cycle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("rehearsal_cycles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "claimed_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "category",
            postgresql.ENUM(name="ticket_category_enum", create_type=False),
            nullable=False,
        ),
        sa.Column(
            "priority",
            postgresql.ENUM(name="priority_enum", create_type=False),
            nullable=False,
            server_default="LOW",
        ),
        sa.Column(
            "status",
            postgresql.ENUM(name="ticket_status_enum", create_type=False),
            nullable=False,
            server_default="OPEN",
        ),
        sa.Column(
            "visibility",
            postgresql.ENUM(name="ticket_visibility_enum", create_type=False),
            nullable=False,
            server_default="PRIVATE",
        ),
        sa.Column("section", sa.String(50), nullable=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("song_ref", sa.String(100), nullable=True),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_tickets_cycle_status",
        "tickets",
        ["cycle_id", "status"],
    )
    op.create_index(
        "ix_tickets_cycle_priority",
        "tickets",
        ["cycle_id", "priority"],
    )
    op.create_index(
        "ix_tickets_team_cycle",
        "tickets",
        ["team_id", "cycle_id"],
    )
    op.create_index(
        "ix_tickets_visibility_section",
        "tickets",
        ["visibility", "section"],
    )

    # --- ticket_activity ---
    op.create_table(
        "ticket_activity",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "ticket_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tickets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "type",
            postgresql.ENUM(name="ticket_activity_type_enum", create_type=False),
            nullable=False,
        ),
        sa.Column("content", sa.Text, nullable=True),
        sa.Column(
            "old_status",
            postgresql.ENUM(name="ticket_status_enum", create_type=False),
            nullable=True,
        ),
        sa.Column(
            "new_status",
            postgresql.ENUM(name="ticket_status_enum", create_type=False),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # --- notification_preferences ---
    op.create_table(
        "notification_preferences",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "team_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("teams.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email_enabled", sa.Boolean, server_default="true", nullable=False),
        sa.Column("deadline_reminder_hours", sa.Integer, server_default="24", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("team_id", "user_id", name="uq_notification_preferences_team_user"),
    )


def downgrade() -> None:
    # ==========================================================================
    # TABLES - Drop in reverse dependency order
    # ==========================================================================

    op.drop_table("notification_preferences")
    op.drop_table("ticket_activity")

    # tickets indexes
    op.drop_index("ix_tickets_visibility_section", table_name="tickets")
    op.drop_index("ix_tickets_team_cycle", table_name="tickets")
    op.drop_index("ix_tickets_cycle_priority", table_name="tickets")
    op.drop_index("ix_tickets_cycle_status", table_name="tickets")
    op.drop_table("tickets")

    op.drop_table("invites")
    op.drop_table("practice_log_assignments")

    # practice_logs indexes
    op.drop_index("ix_practice_logs_cycle_occurred", table_name="practice_logs")
    op.drop_index("ix_practice_logs_team_cycle", table_name="practice_logs")
    op.drop_index("ix_practice_logs_user_cycle", table_name="practice_logs")
    op.drop_table("practice_logs")

    # assignments indexes
    op.drop_index("ix_assignments_cycle_section", table_name="assignments")
    op.drop_index("ix_assignments_cycle_scope", table_name="assignments")
    op.drop_table("assignments")

    op.drop_table("rehearsal_cycles")

    # team_memberships indexes
    op.drop_index("ix_team_memberships_team_section", table_name="team_memberships")
    op.drop_index("ix_team_memberships_team_role", table_name="team_memberships")
    op.drop_table("team_memberships")

    op.drop_table("teams")
    op.drop_table("users")

    # ==========================================================================
    # ENUMS - Drop in any order after tables
    # ==========================================================================

    op.execute("DROP TYPE IF EXISTS ticket_activity_type_enum")
    op.execute("DROP TYPE IF EXISTS ticket_status_enum")
    op.execute("DROP TYPE IF EXISTS ticket_visibility_enum")
    op.execute("DROP TYPE IF EXISTS ticket_category_enum")
    op.execute("DROP TYPE IF EXISTS priority_enum")
    op.execute("DROP TYPE IF EXISTS assignment_scope_enum")
    op.execute("DROP TYPE IF EXISTS assignment_type_enum")
    op.execute("DROP TYPE IF EXISTS role_enum")

