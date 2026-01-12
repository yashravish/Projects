"""Tests for database constraints.

These tests verify that PostgreSQL constraints work correctly:
- Unique membership constraint (team_id, user_id)
- Unique rehearsal cycle constraint (team_id, date)
- Unique notification preferences constraint (team_id, user_id)

Requires a running PostgreSQL database.
"""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings


@pytest.fixture(scope="function")
async def db_session():
    """Create database session with transaction rollback."""
    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
    )
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session() as session:
        yield session
        await session.rollback()
    await engine.dispose()


class TestTeamMembershipConstraints:
    """Tests for team_memberships unique constraint."""

    @pytest.mark.asyncio
    async def test_duplicate_membership_rejected(self, db_session: AsyncSession) -> None:
        """Inserting duplicate (team_id, user_id) should raise IntegrityError."""
        # Create user
        user_id = uuid.uuid4()
        await db_session.execute(
            text("""
                INSERT INTO users (id, email, password_hash, display_name)
                VALUES (:id, :email, :password_hash, :display_name)
            """),
            {
                "id": user_id,
                "email": f"test-{user_id}@example.com",
                "password_hash": "hash",
                "display_name": "Test User",
            },
        )

        # Create team
        team_id = uuid.uuid4()
        await db_session.execute(
            text("""
                INSERT INTO teams (id, name)
                VALUES (:id, :name)
            """),
            {"id": team_id, "name": "Test Team"},
        )

        # First membership - should succeed
        await db_session.execute(
            text("""
                INSERT INTO team_memberships (id, team_id, user_id, role)
                VALUES (:id, :team_id, :user_id, :role)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": team_id,
                "user_id": user_id,
                "role": "MEMBER",
            },
        )

        # Second membership with same team_id + user_id - should fail
        with pytest.raises(IntegrityError) as exc_info:
            await db_session.execute(
                text("""
                    INSERT INTO team_memberships (id, team_id, user_id, role)
                    VALUES (:id, :team_id, :user_id, :role)
                """),
                {
                    "id": uuid.uuid4(),
                    "team_id": team_id,
                    "user_id": user_id,
                    "role": "ADMIN",
                },
            )
            await db_session.flush()

        assert "uq_team_memberships_team_user" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_same_user_different_teams_allowed(self, db_session: AsyncSession) -> None:
        """Same user can be member of multiple teams."""
        # Create user
        user_id = uuid.uuid4()
        await db_session.execute(
            text("""
                INSERT INTO users (id, email, password_hash, display_name)
                VALUES (:id, :email, :password_hash, :display_name)
            """),
            {
                "id": user_id,
                "email": f"multi-team-{user_id}@example.com",
                "password_hash": "hash",
                "display_name": "Multi Team User",
            },
        )

        # Create two teams
        team1_id = uuid.uuid4()
        team2_id = uuid.uuid4()
        await db_session.execute(
            text("INSERT INTO teams (id, name) VALUES (:id, :name)"),
            {"id": team1_id, "name": "Team 1"},
        )
        await db_session.execute(
            text("INSERT INTO teams (id, name) VALUES (:id, :name)"),
            {"id": team2_id, "name": "Team 2"},
        )

        # Memberships in both teams - should succeed
        await db_session.execute(
            text("""
                INSERT INTO team_memberships (id, team_id, user_id, role)
                VALUES (:id, :team_id, :user_id, :role)
            """),
            {"id": uuid.uuid4(), "team_id": team1_id, "user_id": user_id, "role": "MEMBER"},
        )
        await db_session.execute(
            text("""
                INSERT INTO team_memberships (id, team_id, user_id, role)
                VALUES (:id, :team_id, :user_id, :role)
            """),
            {"id": uuid.uuid4(), "team_id": team2_id, "user_id": user_id, "role": "MEMBER"},
        )

        # Verify both exist
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM team_memberships WHERE user_id = :user_id"),
            {"user_id": user_id},
        )
        count = result.scalar()
        assert count == 2


class TestRehearsalCycleConstraints:
    """Tests for rehearsal_cycles unique constraint."""

    @pytest.mark.asyncio
    async def test_duplicate_cycle_date_rejected(self, db_session: AsyncSession) -> None:
        """Inserting duplicate (team_id, date) should raise IntegrityError."""
        # Create team
        team_id = uuid.uuid4()
        await db_session.execute(
            text("INSERT INTO teams (id, name) VALUES (:id, :name)"),
            {"id": team_id, "name": "Cycle Test Team"},
        )

        cycle_date = datetime(2026, 3, 15, 19, 0, 0, tzinfo=UTC)

        # First cycle - should succeed
        await db_session.execute(
            text("""
                INSERT INTO rehearsal_cycles (id, team_id, name, date)
                VALUES (:id, :team_id, :name, :date)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": team_id,
                "name": "Week 1",
                "date": cycle_date,
            },
        )

        # Second cycle with same team_id + date - should fail
        with pytest.raises(IntegrityError) as exc_info:
            await db_session.execute(
                text("""
                    INSERT INTO rehearsal_cycles (id, team_id, name, date)
                    VALUES (:id, :team_id, :name, :date)
                """),
                {
                    "id": uuid.uuid4(),
                    "team_id": team_id,
                    "name": "Week 1 Duplicate",
                    "date": cycle_date,
                },
            )
            await db_session.flush()

        assert "uq_rehearsal_cycles_team_date" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_same_date_different_teams_allowed(self, db_session: AsyncSession) -> None:
        """Different teams can have cycles on the same date."""
        # Create two teams
        team1_id = uuid.uuid4()
        team2_id = uuid.uuid4()
        await db_session.execute(
            text("INSERT INTO teams (id, name) VALUES (:id, :name)"),
            {"id": team1_id, "name": "Team A"},
        )
        await db_session.execute(
            text("INSERT INTO teams (id, name) VALUES (:id, :name)"),
            {"id": team2_id, "name": "Team B"},
        )

        cycle_date = datetime(2026, 4, 1, 19, 0, 0, tzinfo=UTC)

        # Both teams can have a cycle on the same date
        await db_session.execute(
            text("""
                INSERT INTO rehearsal_cycles (id, team_id, name, date)
                VALUES (:id, :team_id, :name, :date)
            """),
            {"id": uuid.uuid4(), "team_id": team1_id, "name": "Team A Week", "date": cycle_date},
        )
        await db_session.execute(
            text("""
                INSERT INTO rehearsal_cycles (id, team_id, name, date)
                VALUES (:id, :team_id, :name, :date)
            """),
            {"id": uuid.uuid4(), "team_id": team2_id, "name": "Team B Week", "date": cycle_date},
        )

        # Verify both exist
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM rehearsal_cycles WHERE date = :date"),
            {"date": cycle_date},
        )
        count = result.scalar()
        assert count == 2


class TestNotificationPreferencesConstraints:
    """Tests for notification_preferences unique constraint."""

    @pytest.mark.asyncio
    async def test_duplicate_preferences_rejected(self, db_session: AsyncSession) -> None:
        """Inserting duplicate (team_id, user_id) should raise IntegrityError."""
        # Create user
        user_id = uuid.uuid4()
        await db_session.execute(
            text("""
                INSERT INTO users (id, email, password_hash, display_name)
                VALUES (:id, :email, :password_hash, :display_name)
            """),
            {
                "id": user_id,
                "email": f"notif-{user_id}@example.com",
                "password_hash": "hash",
                "display_name": "Notif User",
            },
        )

        # Create team
        team_id = uuid.uuid4()
        await db_session.execute(
            text("INSERT INTO teams (id, name) VALUES (:id, :name)"),
            {"id": team_id, "name": "Notif Team"},
        )

        # First preference - should succeed
        await db_session.execute(
            text("""
                INSERT INTO notification_preferences (id, user_id, team_id)
                VALUES (:id, :user_id, :team_id)
            """),
            {"id": uuid.uuid4(), "user_id": user_id, "team_id": team_id},
        )

        # Second preference with same user_id + team_id - should fail
        with pytest.raises(IntegrityError) as exc_info:
            await db_session.execute(
                text("""
                    INSERT INTO notification_preferences (id, user_id, team_id)
                    VALUES (:id, :user_id, :team_id)
                """),
                {"id": uuid.uuid4(), "user_id": user_id, "team_id": team_id},
            )
            await db_session.flush()

        assert "uq_notification_preferences_team_user" in str(exc_info.value)


class TestForeignKeyConstraints:
    """Tests for foreign key constraints."""

    @pytest.mark.asyncio
    async def test_membership_requires_valid_team(self, db_session: AsyncSession) -> None:
        """Membership with non-existent team should fail."""
        user_id = uuid.uuid4()
        await db_session.execute(
            text("""
                INSERT INTO users (id, email, password_hash, display_name)
                VALUES (:id, :email, :password_hash, :display_name)
            """),
            {
                "id": user_id,
                "email": f"fk-test-{user_id}@example.com",
                "password_hash": "hash",
                "display_name": "FK Test User",
            },
        )

        with pytest.raises(IntegrityError):
            await db_session.execute(
                text("""
                    INSERT INTO team_memberships (id, team_id, user_id, role)
                    VALUES (:id, :team_id, :user_id, :role)
                """),
                {
                    "id": uuid.uuid4(),
                    "team_id": uuid.uuid4(),  # Non-existent team
                    "user_id": user_id,
                    "role": "MEMBER",
                },
            )
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_cascade_delete_team_removes_memberships(
        self, db_session: AsyncSession
    ) -> None:
        """Deleting a team should cascade delete its memberships."""
        # Create user
        user_id = uuid.uuid4()
        await db_session.execute(
            text("""
                INSERT INTO users (id, email, password_hash, display_name)
                VALUES (:id, :email, :password_hash, :display_name)
            """),
            {
                "id": user_id,
                "email": f"cascade-{user_id}@example.com",
                "password_hash": "hash",
                "display_name": "Cascade User",
            },
        )

        # Create team
        team_id = uuid.uuid4()
        await db_session.execute(
            text("INSERT INTO teams (id, name) VALUES (:id, :name)"),
            {"id": team_id, "name": "Cascade Team"},
        )

        # Create membership
        await db_session.execute(
            text("""
                INSERT INTO team_memberships (id, team_id, user_id, role)
                VALUES (:id, :team_id, :user_id, :role)
            """),
            {"id": uuid.uuid4(), "team_id": team_id, "user_id": user_id, "role": "MEMBER"},
        )

        # Delete team
        await db_session.execute(
            text("DELETE FROM teams WHERE id = :id"),
            {"id": team_id},
        )

        # Verify membership is gone
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM team_memberships WHERE team_id = :team_id"),
            {"team_id": team_id},
        )
        count = result.scalar()
        assert count == 0


class TestEnumConstraints:
    """Tests for enum type constraints."""

    @pytest.mark.asyncio
    async def test_invalid_role_rejected(self, db_session: AsyncSession) -> None:
        """Invalid role value should be rejected."""
        user_id = uuid.uuid4()
        await db_session.execute(
            text("""
                INSERT INTO users (id, email, password_hash, display_name)
                VALUES (:id, :email, :password_hash, :display_name)
            """),
            {
                "id": user_id,
                "email": f"enum-{user_id}@example.com",
                "password_hash": "hash",
                "display_name": "Enum User",
            },
        )

        team_id = uuid.uuid4()
        await db_session.execute(
            text("INSERT INTO teams (id, name) VALUES (:id, :name)"),
            {"id": team_id, "name": "Enum Team"},
        )

        with pytest.raises((IntegrityError, Exception)):
            await db_session.execute(
                text("""
                    INSERT INTO team_memberships (id, team_id, user_id, role)
                    VALUES (:id, :team_id, :user_id, 'INVALID_ROLE')
                """),
                {"id": uuid.uuid4(), "team_id": team_id, "user_id": user_id},
            )
            await db_session.flush()

    @pytest.mark.asyncio
    async def test_valid_ticket_status_accepted(self, db_session: AsyncSession) -> None:
        """Valid ticket status values should be accepted."""
        # Create user
        user_id = uuid.uuid4()
        await db_session.execute(
            text("""
                INSERT INTO users (id, email, password_hash, display_name)
                VALUES (:id, :email, :password_hash, :display_name)
            """),
            {
                "id": user_id,
                "email": f"status-{user_id}@example.com",
                "password_hash": "hash",
                "display_name": "Status User",
            },
        )

        # Create team
        team_id = uuid.uuid4()
        await db_session.execute(
            text("INSERT INTO teams (id, name) VALUES (:id, :name)"),
            {"id": team_id, "name": "Status Team"},
        )

        # Create ticket with each valid status
        for status in ["OPEN", "IN_PROGRESS", "BLOCKED", "RESOLVED", "VERIFIED"]:
            ticket_id = uuid.uuid4()
            await db_session.execute(
                text("""
                    INSERT INTO tickets (
                        id, team_id, owner_id, created_by, category,
                        priority, status, visibility, title
                    )
                    VALUES (
                        :id, :team_id, :owner_id, :created_by, :category,
                        :priority, :status, :visibility, :title
                    )
                """),
                {
                    "id": ticket_id,
                    "team_id": team_id,
                    "owner_id": user_id,
                    "created_by": user_id,
                    "category": "PITCH",
                    "priority": "LOW",
                    "status": status,
                    "visibility": "PRIVATE",
                    "title": f"Test ticket with status {status}",
                },
            )

        # Verify all 5 tickets exist
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM tickets WHERE team_id = :team_id"),
            {"team_id": team_id},
        )
        count = result.scalar()
        assert count == 5

