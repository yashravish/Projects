#!/usr/bin/env python3
"""
Seed Demo Data Script

Creates a complete demo dataset for PracticeOps:
- Demo team with cycles, assignments, tickets
- Demo admin user
- Sample members in different sections
- Practice logs and activities

Usage:
    python -m scripts.seed_demo

Or via Docker:
    docker compose exec api python -m scripts.seed_demo
"""

import asyncio
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.core.security import hash_password
from app.models import (
    Assignment,
    AssignmentScope,
    AssignmentType,
    PracticeLog,
    Priority,
    RehearsalCycle,
    Role,
    Team,
    TeamMembership,
    Ticket,
    TicketActivity,
    TicketActivityType,
    TicketCategory,
    TicketStatus,
    TicketVisibility,
    User,
)

# =============================================================================
# Configuration
# =============================================================================

DEMO_TEAM_NAME = "Harmonia Choir"
DEMO_ADMIN_EMAIL = "demo@practiceops.app"
DEMO_ADMIN_PASSWORD = "demo1234"
DEMO_ADMIN_NAME = "Alex Director"

SECTIONS = ["Soprano", "Alto", "Tenor", "Bass"]

DEMO_MEMBERS = [
    # Section leaders
    {"email": "sarah@practiceops.app", "name": "Sarah Mitchell", "role": "SECTION_LEADER", "section": "Soprano"},
    {"email": "marcus@practiceops.app", "name": "Marcus Chen", "role": "SECTION_LEADER", "section": "Alto"},
    {"email": "elena@practiceops.app", "name": "Elena Rodriguez", "role": "SECTION_LEADER", "section": "Tenor"},
    {"email": "james@practiceops.app", "name": "James Thompson", "role": "SECTION_LEADER", "section": "Bass"},
    # Regular members
    {"email": "olivia@practiceops.app", "name": "Olivia Parker", "role": "MEMBER", "section": "Soprano"},
    {"email": "noah@practiceops.app", "name": "Noah Kim", "role": "MEMBER", "section": "Soprano"},
    {"email": "emma@practiceops.app", "name": "Emma Wilson", "role": "MEMBER", "section": "Alto"},
    {"email": "liam@practiceops.app", "name": "Liam Davis", "role": "MEMBER", "section": "Alto"},
    {"email": "ava@practiceops.app", "name": "Ava Martinez", "role": "MEMBER", "section": "Tenor"},
    {"email": "mason@practiceops.app", "name": "Mason Brown", "role": "MEMBER", "section": "Tenor"},
    {"email": "sophia@practiceops.app", "name": "Sophia Lee", "role": "MEMBER", "section": "Bass"},
    {"email": "ethan@practiceops.app", "name": "Ethan Garcia", "role": "MEMBER", "section": "Bass"},
]

SONG_REFS = [
    "Ave Maria",
    "Hallelujah",
    "Bohemian Rhapsody",
    "Bridge Over Troubled Water",
    "Amazing Grace",
]

ASSIGNMENT_TITLES = [
    "Learn mm. 1-32",
    "Memorize verse 1",
    "Work on dynamics in chorus",
    "Practice alto entrance at m. 45",
    "Polish final section",
    "Review text pronunciation",
    "Blend practice with recording",
]

TICKET_TITLES = [
    "Pitch drift in measure 24",
    "Rhythm unclear at bridge",
    "Need to memorize verse 2",
    "Breath support issues in long phrases",
    "Blend problem with tenors",
    "Tempo inconsistency in finale",
    "Diction muddy in fast passages",
]


# =============================================================================
# Seed Functions
# =============================================================================


async def clear_demo_data(db: AsyncSession) -> None:
    """Remove existing demo data if present."""
    # Find demo team
    result = await db.execute(select(Team).where(Team.name == DEMO_TEAM_NAME))
    team = result.scalar_one_or_none()
    
    if team:
        print(f"🗑️  Clearing existing demo data for team: {team.name}")
        
        # Delete in order to respect foreign keys
        # Get all cycles for this team
        cycles = await db.execute(select(RehearsalCycle).where(RehearsalCycle.team_id == team.id))
        cycle_ids = [c.id for c in cycles.scalars().all()]
        
        # Delete ticket activities for this team's tickets
        await db.execute(
            TicketActivity.__table__.delete().where(
                TicketActivity.ticket_id.in_(
                    select(Ticket.id).where(Ticket.team_id == team.id)
                )
            )
        )
        
        # Delete tickets
        await db.execute(Ticket.__table__.delete().where(Ticket.team_id == team.id))
        
        # Delete practice logs
        await db.execute(PracticeLog.__table__.delete().where(PracticeLog.team_id == team.id))
        
        # Delete assignments
        if cycle_ids:
            await db.execute(Assignment.__table__.delete().where(Assignment.cycle_id.in_(cycle_ids)))
        
        # Delete cycles
        await db.execute(RehearsalCycle.__table__.delete().where(RehearsalCycle.team_id == team.id))
        
        # Delete memberships
        await db.execute(TeamMembership.__table__.delete().where(TeamMembership.team_id == team.id))
        
        # Delete team
        await db.delete(team)
        
        await db.commit()
        print("✅ Existing demo data cleared")


async def create_demo_team(db: AsyncSession) -> Team:
    """Create the demo team."""
    team = Team(name=DEMO_TEAM_NAME)
    db.add(team)
    await db.commit()
    await db.refresh(team)
    print(f"✅ Created team: {team.name}")
    return team


async def create_demo_users(db: AsyncSession, team: Team) -> dict[str, User]:
    """Create demo admin and members."""
    users: dict[str, User] = {}
    
    # Create admin user
    admin = User(
        email=DEMO_ADMIN_EMAIL,
        display_name=DEMO_ADMIN_NAME,
        password_hash=hash_password(DEMO_ADMIN_PASSWORD),
    )
    db.add(admin)
    await db.flush()
    
    # Create admin membership
    admin_membership = TeamMembership(
        team_id=team.id,
        user_id=admin.id,
        role=Role.ADMIN,
        section=None,
    )
    db.add(admin_membership)
    users["admin"] = admin
    print(f"✅ Created admin: {admin.email}")
    
    # Create demo members
    for member_data in DEMO_MEMBERS:
        user = User(
            email=member_data["email"],
            display_name=member_data["name"],
            password_hash=hash_password("demo1234"),  # Same password for all demo users
        )
        db.add(user)
        await db.flush()
        
        membership = TeamMembership(
            team_id=team.id,
            user_id=user.id,
            role=Role[member_data["role"]],
            section=member_data["section"],
        )
        db.add(membership)
        users[member_data["email"]] = user
    
    await db.commit()
    print(f"✅ Created {len(DEMO_MEMBERS)} demo members")
    
    return users


async def create_demo_cycles(db: AsyncSession, team: Team) -> list[RehearsalCycle]:
    """Create rehearsal cycles - past, current, and future."""
    cycles = []
    now = datetime.now(UTC)
    
    # Past cycle (2 weeks ago)
    past_cycle = RehearsalCycle(
        team_id=team.id,
        name="Week 1 - Kickoff",
        date=now - timedelta(days=14),
    )
    db.add(past_cycle)
    cycles.append(past_cycle)
    
    # Current cycle (this week - 3 days from now)
    current_cycle = RehearsalCycle(
        team_id=team.id,
        name="Week 2 - Deep Work",
        date=now + timedelta(days=3),
    )
    db.add(current_cycle)
    cycles.append(current_cycle)
    
    # Future cycle (next week)
    future_cycle = RehearsalCycle(
        team_id=team.id,
        name="Week 3 - Polish",
        date=now + timedelta(days=10),
    )
    db.add(future_cycle)
    cycles.append(future_cycle)
    
    await db.commit()
    for cycle in cycles:
        await db.refresh(cycle)
    
    print(f"✅ Created {len(cycles)} rehearsal cycles")
    return cycles


async def create_demo_assignments(
    db: AsyncSession,
    cycles: list[RehearsalCycle],
    admin: User
) -> list[Assignment]:
    """Create sample assignments."""
    assignments = []
    
    for cycle in cycles:
        # Create 3-5 assignments per cycle
        num_assignments = random.randint(3, 5)
        
        for i in range(num_assignments):
            # Mix of TEAM and SECTION scope
            is_team_scope = random.random() > 0.4
            section = None if is_team_scope else random.choice(SECTIONS)
            
            assignment = Assignment(
                cycle_id=cycle.id,
                created_by=admin.id,
                type=random.choice(list(AssignmentType)),
                scope=AssignmentScope.TEAM if is_team_scope else AssignmentScope.SECTION,
                priority=random.choice(list(Priority)),
                section=section,
                title=random.choice(ASSIGNMENT_TITLES),
                song_ref=random.choice(SONG_REFS) if random.random() > 0.3 else None,
                description="Focus on accuracy and blend. Use recordings for reference.",
                due_at=cycle.date,
            )
            db.add(assignment)
            assignments.append(assignment)
    
    await db.commit()
    print(f"✅ Created {len(assignments)} assignments")
    return assignments


async def create_demo_tickets(
    db: AsyncSession,
    team: Team,
    cycles: list[RehearsalCycle],
    users: dict[str, User]
) -> list[Ticket]:
    """Create sample tickets in various states."""
    tickets = []
    user_list = list(users.values())
    current_cycle = cycles[1]  # The "current" cycle
    
    # Create tickets in different states
    ticket_configs = [
        # Open tickets
        {"status": TicketStatus.OPEN, "visibility": TicketVisibility.TEAM, "priority": Priority.BLOCKING},
        {"status": TicketStatus.OPEN, "visibility": TicketVisibility.SECTION, "priority": Priority.MEDIUM},
        {"status": TicketStatus.OPEN, "visibility": TicketVisibility.PRIVATE, "priority": Priority.LOW},
        # In progress
        {"status": TicketStatus.IN_PROGRESS, "visibility": TicketVisibility.TEAM, "priority": Priority.MEDIUM},
        {"status": TicketStatus.IN_PROGRESS, "visibility": TicketVisibility.SECTION, "priority": Priority.BLOCKING},
        # Blocked
        {"status": TicketStatus.BLOCKED, "visibility": TicketVisibility.TEAM, "priority": Priority.BLOCKING},
        # Resolved (awaiting verification)
        {"status": TicketStatus.RESOLVED, "visibility": TicketVisibility.TEAM, "priority": Priority.MEDIUM},
        {"status": TicketStatus.RESOLVED, "visibility": TicketVisibility.SECTION, "priority": Priority.LOW},
        # Verified (complete)
        {"status": TicketStatus.VERIFIED, "visibility": TicketVisibility.TEAM, "priority": Priority.MEDIUM},
    ]
    
    for i, config in enumerate(ticket_configs):
        owner = random.choice(user_list)
        section = random.choice(SECTIONS) if config["visibility"] != TicketVisibility.TEAM else random.choice(SECTIONS)
        
        ticket = Ticket(
            team_id=team.id,
            cycle_id=current_cycle.id,
            owner_id=owner.id if config["status"] != TicketStatus.OPEN else None,
            created_by=owner.id,
            category=random.choice(list(TicketCategory)),
            priority=config["priority"],
            status=config["status"],
            visibility=config["visibility"],
            section=section,
            title=TICKET_TITLES[i % len(TICKET_TITLES)],
            description="This needs attention before the next rehearsal.",
            song_ref=random.choice(SONG_REFS),
            claimable=config["status"] == TicketStatus.OPEN,
            due_at=current_cycle.date - timedelta(days=random.randint(0, 2)),
        )
        
        if config["status"] == TicketStatus.RESOLVED:
            ticket.resolved_at = datetime.now(UTC) - timedelta(hours=random.randint(1, 48))
            ticket.resolved_note = "Fixed after focused practice session"
        
        if config["status"] == TicketStatus.VERIFIED:
            ticket.resolved_at = datetime.now(UTC) - timedelta(hours=random.randint(48, 96))
            ticket.resolved_note = "Fixed after focused practice session"
            ticket.verified_at = datetime.now(UTC) - timedelta(hours=random.randint(1, 24))
            ticket.verified_note = "Confirmed resolved in section rehearsal"
            ticket.verified_by = users["admin"].id
        
        db.add(ticket)
        tickets.append(ticket)
    
    await db.commit()
    print(f"✅ Created {len(tickets)} tickets")
    return tickets


async def create_demo_practice_logs(
    db: AsyncSession,
    team: Team,
    cycles: list[RehearsalCycle],
    users: dict[str, User]
) -> None:
    """Create sample practice logs for demo users."""
    user_list = [u for u in users.values()]
    current_cycle = cycles[1]
    
    logs_created = 0
    
    for user in user_list:
        # Each user has 2-7 practice sessions in the past week
        num_sessions = random.randint(2, 7)
        
        for _ in range(num_sessions):
            log = PracticeLog(
                user_id=user.id,
                team_id=team.id,
                cycle_id=current_cycle.id,
                duration_minutes=random.choice([15, 20, 30, 45, 60, 90]),
                rating_1_5=random.randint(2, 5),
                blocked_flag=random.random() < 0.1,  # 10% chance of blocked
                notes=random.choice([
                    "Good session, made progress on difficult passage",
                    "Focused on breath control",
                    "Worked with piano recording",
                    "Memorization practice",
                    None,
                ]),
                occurred_at=datetime.now(UTC) - timedelta(
                    days=random.randint(0, 6),
                    hours=random.randint(0, 23),
                ),
            )
            db.add(log)
            logs_created += 1
    
    await db.commit()
    print(f"✅ Created {logs_created} practice logs")


async def main() -> None:
    """Run the seed script."""
    print("\n🎵 PracticeOps Demo Seed Script")
    print("=" * 40)
    
    # Create database connection
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    async with async_session() as db:
        try:
            # Clear existing demo data
            await clear_demo_data(db)
            
            # Create fresh demo data
            team = await create_demo_team(db)
            users = await create_demo_users(db, team)
            cycles = await create_demo_cycles(db, team)
            await create_demo_assignments(db, cycles, users["admin"])
            await create_demo_tickets(db, team, cycles, users)
            await create_demo_practice_logs(db, team, cycles, users)
            
            print("\n" + "=" * 40)
            print("🎉 Demo data seeded successfully!")
            print("=" * 40)
            print(f"\n📧 Demo Admin Login:")
            print(f"   Email:    {DEMO_ADMIN_EMAIL}")
            print(f"   Password: {DEMO_ADMIN_PASSWORD}")
            print(f"\n📧 Demo Member Login (any):")
            print(f"   Email:    {DEMO_MEMBERS[0]['email']}")
            print(f"   Password: demo1234")
            print("\n")
            
        except Exception as e:
            print(f"\n❌ Error seeding data: {e}")
            await db.rollback()
            raise
        finally:
            await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

