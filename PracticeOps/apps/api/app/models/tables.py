"""SQLAlchemy 2.0 table definitions.

All models use:
- Mapped[] type hints for full type inference
- mapped_column() for column definitions
- UUID primary keys
- TIMESTAMPTZ for all timestamps
- Explicit foreign keys and constraints
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.enums import (
    AssignmentScope,
    AssignmentType,
    Priority,
    Role,
    TicketActivityType,
    TicketCategory,
    TicketStatus,
    TicketVisibility,
)


class User(Base):
    """User account."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    memberships: Mapped[list["TeamMembership"]] = relationship(back_populates="user")
    practice_logs: Mapped[list["PracticeLog"]] = relationship(back_populates="user")


class Team(Base):
    """Musical team (a cappella group, choir, ensemble)."""

    __tablename__ = "teams"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    memberships: Mapped[list["TeamMembership"]] = relationship(back_populates="team")
    cycles: Mapped[list["RehearsalCycle"]] = relationship(back_populates="team")
    invites: Mapped[list["Invite"]] = relationship(back_populates="team")


class TeamMembership(Base):
    """Association between users and teams with role and section."""

    __tablename__ = "team_memberships"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[Role] = mapped_column(
        Enum(Role, name="role_enum", create_type=False), nullable=False
    )
    section: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("team_id", "user_id", name="uq_team_memberships_team_user"),
        Index("ix_team_memberships_team_role", "team_id", "role"),
        Index("ix_team_memberships_team_section", "team_id", "section"),
    )

    # Relationships
    team: Mapped["Team"] = relationship(back_populates="memberships")
    user: Mapped["User"] = relationship(back_populates="memberships")


class RehearsalCycle(Base):
    """A rehearsal cycle (typically a week leading to a rehearsal date)."""

    __tablename__ = "rehearsal_cycles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("team_id", "date", name="uq_rehearsal_cycles_team_date"),
    )

    # Relationships
    team: Mapped["Team"] = relationship(back_populates="cycles")
    assignments: Mapped[list["Assignment"]] = relationship(back_populates="cycle")
    tickets: Mapped[list["Ticket"]] = relationship(back_populates="cycle")
    practice_logs: Mapped[list["PracticeLog"]] = relationship(back_populates="cycle")


class Assignment(Base):
    """Practice assignment for a rehearsal cycle."""

    __tablename__ = "assignments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    cycle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rehearsal_cycles.id", ondelete="CASCADE"), nullable=False
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    type: Mapped[AssignmentType] = mapped_column(
        Enum(AssignmentType, name="assignment_type_enum", create_type=False), nullable=False
    )
    scope: Mapped[AssignmentScope] = mapped_column(
        Enum(AssignmentScope, name="assignment_scope_enum", create_type=False), nullable=False
    )
    priority: Mapped[Priority] = mapped_column(
        Enum(Priority, name="priority_enum", create_type=False),
        nullable=False,
        server_default="LOW",
    )
    section: Mapped[str | None] = mapped_column(String(50), nullable=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    song_ref: Mapped[str | None] = mapped_column(String(100), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Indexes
    __table_args__ = (
        Index("ix_assignments_cycle_scope", "cycle_id", "scope"),
        Index("ix_assignments_cycle_section", "cycle_id", "section"),
        Index("ix_assignments_cycle_priority", "cycle_id", "priority"),
    )

    # Relationships
    cycle: Mapped["RehearsalCycle"] = relationship(back_populates="assignments")
    creator: Mapped[Optional["User"]] = relationship(foreign_keys=[created_by])
    practice_log_assignments: Mapped[list["PracticeLogAssignment"]] = relationship(
        back_populates="assignment"
    )


class PracticeLog(Base):
    """Record of a practice session."""

    __tablename__ = "practice_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    cycle_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rehearsal_cycles.id", ondelete="SET NULL"), nullable=True
    )
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    rating_1_5: Mapped[int | None] = mapped_column(Integer, nullable=True)
    blocked_flag: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Indexes
    __table_args__ = (
        Index("ix_practice_logs_user_cycle", "user_id", "cycle_id"),
        Index("ix_practice_logs_team_cycle", "team_id", "cycle_id"),
        Index("ix_practice_logs_cycle_occurred", "cycle_id", "occurred_at"),
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="practice_logs")
    team: Mapped["Team"] = relationship()
    cycle: Mapped[Optional["RehearsalCycle"]] = relationship(back_populates="practice_logs")
    assignments: Mapped[list["PracticeLogAssignment"]] = relationship(
        back_populates="practice_log", cascade="all, delete-orphan"
    )


class PracticeLogAssignment(Base):
    """Association between practice logs and assignments worked on."""

    __tablename__ = "practice_log_assignments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    practice_log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("practice_logs.id", ondelete="CASCADE"), nullable=False
    )
    assignment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assignments.id", ondelete="CASCADE"), nullable=False
    )

    # Relationships
    practice_log: Mapped["PracticeLog"] = relationship(back_populates="assignments")
    assignment: Mapped["Assignment"] = relationship(back_populates="practice_log_assignments")


class Invite(Base):
    """Team invitation token.

    Security: The `token` column stores a SHA-256 hash of the actual token.
    The raw token is returned only once when the invite is created and never persisted.
    """

    __tablename__ = "invites"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[Role] = mapped_column(
        Enum(Role, name="role_enum", create_type=False), nullable=False
    )
    section: Mapped[str | None] = mapped_column(String(50), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    team: Mapped["Team"] = relationship(back_populates="invites")
    creator: Mapped["User"] = relationship()


class Ticket(Base):
    """Practice issue ticket."""

    __tablename__ = "tickets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    cycle_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rehearsal_cycles.id", ondelete="SET NULL"), nullable=True
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    created_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    claimed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    claimable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    category: Mapped[TicketCategory] = mapped_column(
        Enum(TicketCategory, name="ticket_category_enum", create_type=False), nullable=False
    )
    priority: Mapped[Priority] = mapped_column(
        Enum(Priority, name="priority_enum", create_type=False),
        nullable=False,
        server_default="LOW",
    )
    status: Mapped[TicketStatus] = mapped_column(
        Enum(TicketStatus, name="ticket_status_enum", create_type=False),
        nullable=False,
        server_default="OPEN",
    )
    visibility: Mapped[TicketVisibility] = mapped_column(
        Enum(TicketVisibility, name="ticket_visibility_enum", create_type=False),
        nullable=False,
        server_default="PRIVATE",
    )
    section: Mapped[str | None] = mapped_column(String(50), nullable=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    song_ref: Mapped[str | None] = mapped_column(String(100), nullable=True)
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verified_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    verified_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Indexes
    __table_args__ = (
        Index("ix_tickets_cycle_status", "cycle_id", "status"),
        Index("ix_tickets_cycle_priority", "cycle_id", "priority"),
        Index("ix_tickets_team_cycle", "team_id", "cycle_id"),
        Index("ix_tickets_visibility_section", "visibility", "section"),
    )

    # Relationships
    team: Mapped["Team"] = relationship()
    cycle: Mapped[Optional["RehearsalCycle"]] = relationship(back_populates="tickets")
    owner: Mapped[Optional["User"]] = relationship(foreign_keys=[owner_id])
    creator: Mapped["User"] = relationship(foreign_keys=[created_by])
    claimer: Mapped[Optional["User"]] = relationship(foreign_keys=[claimed_by])
    verifier: Mapped[Optional["User"]] = relationship(foreign_keys=[verified_by])
    activities: Mapped[list["TicketActivity"]] = relationship(
        back_populates="ticket", cascade="all, delete-orphan"
    )


class TicketActivity(Base):
    """Activity log entry for a ticket."""

    __tablename__ = "ticket_activity"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[TicketActivityType] = mapped_column(
        Enum(TicketActivityType, name="ticket_activity_type_enum", create_type=False),
        nullable=False,
    )
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    old_status: Mapped[TicketStatus | None] = mapped_column(
        Enum(TicketStatus, name="ticket_status_enum", create_type=False), nullable=True
    )
    new_status: Mapped[TicketStatus | None] = mapped_column(
        Enum(TicketStatus, name="ticket_status_enum", create_type=False), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    ticket: Mapped["Ticket"] = relationship(back_populates="activities")
    user: Mapped["User"] = relationship()


class NotificationPreference(Base):
    """User notification preferences per team."""

    __tablename__ = "notification_preferences"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    email_enabled: Mapped[bool] = mapped_column(Boolean, server_default="true", nullable=False)
    deadline_reminder_hours: Mapped[int] = mapped_column(
        Integer, server_default="24", nullable=False
    )
    no_log_days: Mapped[int] = mapped_column(
        Integer, server_default="3", nullable=False
    )
    weekly_digest_enabled: Mapped[bool] = mapped_column(
        Boolean, server_default="true", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("team_id", "user_id", name="uq_notification_preferences_team_user"),
    )

    # Relationships
    user: Mapped["User"] = relationship()
    team: Mapped["Team"] = relationship()

