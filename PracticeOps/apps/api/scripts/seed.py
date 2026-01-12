"""Seed script for initial database data.

Creates:
- One admin user
- One team
- One team membership (admin)
- One upcoming rehearsal cycle
- Two assignments

Run with: python -m scripts.seed
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_maker, engine
from app.models import (
    Assignment,
    AssignmentScope,
    AssignmentType,
    RehearsalCycle,
    Role,
    Team,
    TeamMembership,
    User,
)


async def seed_database() -> None:
    """Seed the database with initial data."""
    async with async_session_maker() as session:
        # Check if data already exists
        result = await session.execute(select(User).limit(1))
        if result.scalar_one_or_none() is not None:
            print("Database already seeded. Skipping.")
            return

        await create_seed_data(session)
        await session.commit()
        print("Database seeded successfully.")


async def create_seed_data(session: AsyncSession) -> None:
    """Create all seed data within a transaction."""

    # --- Admin User ---
    # Password: "admin123" hashed with bcrypt
    # Note: In production, use passlib to generate this
    admin_user = User(
        id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        email="admin@example.com",
        # This is bcrypt hash of "admin123" - for dev only
        password_hash="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.V1IbK.1P7Jm1Wy",
        display_name="Admin User",
    )
    session.add(admin_user)

    # --- Team ---
    team = Team(
        id=uuid.UUID("00000000-0000-0000-0000-000000000010"),
        name="Demo A Cappella Group",
    )
    session.add(team)

    # --- Team Membership ---
    membership = TeamMembership(
        id=uuid.UUID("00000000-0000-0000-0000-000000000100"),
        team_id=team.id,
        user_id=admin_user.id,
        role=Role.ADMIN,
        section="Tenor",
    )
    session.add(membership)

    # --- Rehearsal Cycle ---
    # Upcoming cycle: next Monday
    now = datetime.now(UTC)
    days_until_monday = (7 - now.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7  # Next Monday, not today
    next_rehearsal = now + timedelta(days=days_until_monday)
    next_rehearsal = next_rehearsal.replace(hour=19, minute=0, second=0, microsecond=0)

    cycle = RehearsalCycle(
        id=uuid.UUID("00000000-0000-0000-0000-000000001000"),
        team_id=team.id,
        name=f"Week of {next_rehearsal.strftime('%B %d')}",
        date=next_rehearsal,
    )
    session.add(cycle)

    # --- Assignments ---
    assignment1 = Assignment(
        id=uuid.UUID("00000000-0000-0000-0000-000000010000"),
        cycle_id=cycle.id,
        type=AssignmentType.SONG_WORK,
        scope=AssignmentScope.TEAM,
        title="Learn measures 1-32 of 'Bohemian Rhapsody'",
        description="Focus on pitch accuracy in the harmony sections.",
        due_at=next_rehearsal - timedelta(hours=2),
    )
    session.add(assignment1)

    assignment2 = Assignment(
        id=uuid.UUID("00000000-0000-0000-0000-000000100000"),
        cycle_id=cycle.id,
        type=AssignmentType.MEMORIZATION,
        scope=AssignmentScope.SECTION,
        section="Tenor",
        title="Memorize tenor part for 'Africa'",
        description="Full memorization required - no sheet music at next rehearsal.",
        due_at=next_rehearsal - timedelta(hours=2),
    )
    session.add(assignment2)


async def main() -> None:
    """Main entry point."""
    try:
        await seed_database()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

