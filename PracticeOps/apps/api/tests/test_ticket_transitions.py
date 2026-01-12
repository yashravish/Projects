"""Tests for Milestone 8: Ticket Status Transitions + Verification.

Tests cover:
- Happy path: OPEN -> IN_PROGRESS -> RESOLVED -> VERIFIED
- Transition rules enforcement
- RBAC for verification
- Activity logging
- Edge cases (invalid transitions, missing content)
"""

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.database import get_db
from app.main import app
from app.models.enums import (
    Priority,
    Role,
    TicketActivityType,
    TicketCategory,
    TicketStatus,
    TicketVisibility,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create database session with cleanup after each test."""
    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
    )
    try:
        session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with session_maker() as session:
            # Clean up before test
            await session.execute(text("DELETE FROM ticket_activity"))
            await session.execute(text("DELETE FROM tickets"))
            await session.execute(text("DELETE FROM practice_log_assignments"))
            await session.execute(text("DELETE FROM practice_logs"))
            await session.execute(text("DELETE FROM assignments"))
            await session.execute(text("DELETE FROM rehearsal_cycles"))
            await session.execute(text("DELETE FROM invites"))
            await session.execute(text("DELETE FROM team_memberships"))
            await session.execute(text("DELETE FROM teams"))
            await session.execute(text("DELETE FROM users"))
            await session.commit()

            yield session

            # Clean up after test
            await session.execute(text("DELETE FROM ticket_activity"))
            await session.execute(text("DELETE FROM tickets"))
            await session.execute(text("DELETE FROM practice_log_assignments"))
            await session.execute(text("DELETE FROM practice_logs"))
            await session.execute(text("DELETE FROM assignments"))
            await session.execute(text("DELETE FROM rehearsal_cycles"))
            await session.execute(text("DELETE FROM invites"))
            await session.execute(text("DELETE FROM team_memberships"))
            await session.execute(text("DELETE FROM teams"))
            await session.execute(text("DELETE FROM users"))
            await session.commit()
    finally:
        await engine.dispose()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for testing with database override."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def register_user(
    client: AsyncClient,
    email: str,
    name: str = "Test User",
    password: str = "password123",
) -> dict:
    """Helper to register a user and return auth tokens."""
    response = await client.post(
        "/auth/register",
        json={"email": email, "name": name, "password": password},
    )
    assert response.status_code == 201
    return response.json()


def auth_header(access_token: str) -> dict[str, str]:
    """Create authorization header from access token."""
    return {"Authorization": f"Bearer {access_token}"}


async def create_team(client: AsyncClient, auth_data: dict, name: str = "Test Team") -> str:
    """Helper to create a team and return team_id."""
    response = await client.post(
        "/teams",
        json={"name": name},
        headers=auth_header(auth_data["access_token"]),
    )
    assert response.status_code == 201
    return response.json()["team"]["id"]


async def create_cycle(
    client: AsyncClient, team_id: str, auth_data: dict, date: datetime, label: str = "Test Cycle"
) -> str:
    """Helper to create a cycle and return cycle_id."""
    response = await client.post(
        f"/teams/{team_id}/cycles",
        json={"date": date.strftime("%Y-%m-%d"), "label": label},
        headers=auth_header(auth_data["access_token"]),
    )
    assert response.status_code == 201
    return response.json()["cycle"]["id"]


async def add_member(
    db_session: AsyncSession,
    team_id: str,
    user_id: str,
    role: Role = Role.MEMBER,
    section: str | None = None,
) -> None:
    """Helper to add a membership directly to the database."""
    await db_session.execute(
        text(
            """
            INSERT INTO team_memberships (id, team_id, user_id, role, section)
            VALUES (:id, :team_id, :user_id, :role, :section)
            """
        ),
        {
            "id": uuid.uuid4(),
            "team_id": uuid.UUID(team_id),
            "user_id": uuid.UUID(user_id),
            "role": role.value,
            "section": section,
        },
    )
    await db_session.commit()


async def create_ticket(
    client: AsyncClient,
    cycle_id: str,
    auth_data: dict,
    title: str = "Test Ticket",
    visibility: TicketVisibility = TicketVisibility.TEAM,
    section: str | None = None,
) -> dict:
    """Helper to create a ticket and return its data."""
    payload = {
        "title": title,
        "category": TicketCategory.PITCH.value,
        "priority": Priority.LOW.value,
        "visibility": visibility.value,
    }
    if section:
        payload["section"] = section

    response = await client.post(
        f"/cycles/{cycle_id}/tickets",
        json=payload,
        headers=auth_header(auth_data["access_token"]),
    )
    assert response.status_code == 201
    return response.json()["ticket"]


class TestTransitionTicket:
    """Tests for POST /tickets/{id}/transition."""

    async def test_open_to_in_progress(self, client: AsyncClient) -> None:
        """Owner can transition from OPEN to IN_PROGRESS."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)
        assert ticket["status"] == TicketStatus.OPEN.value

        # Transition to IN_PROGRESS
        response = await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value},
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 200
        result = response.json()["ticket"]
        assert result["status"] == TicketStatus.IN_PROGRESS.value

    async def test_in_progress_to_blocked(self, client: AsyncClient) -> None:
        """Owner can transition from IN_PROGRESS to BLOCKED."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)

        # Transition to IN_PROGRESS first
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value},
            headers=auth_header(admin_data["access_token"]),
        )

        # Transition to BLOCKED
        response = await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.BLOCKED.value},
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 200
        result = response.json()["ticket"]
        assert result["status"] == TicketStatus.BLOCKED.value

    async def test_in_progress_to_resolved_with_note(self, client: AsyncClient) -> None:
        """Owner can transition from IN_PROGRESS to RESOLVED with required note."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)

        # Transition to IN_PROGRESS
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value},
            headers=auth_header(admin_data["access_token"]),
        )

        # Transition to RESOLVED with note
        response = await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={
                "to_status": TicketStatus.RESOLVED.value,
                "content": "Fixed the pitch issue by practicing scales",
            },
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 200
        result = response.json()["ticket"]
        assert result["status"] == TicketStatus.RESOLVED.value
        assert result["resolved_note"] == "Fixed the pitch issue by practicing scales"
        assert result["resolved_at"] is not None

    async def test_resolved_without_note_fails(self, client: AsyncClient) -> None:
        """Transitioning to RESOLVED without content returns VALIDATION_ERROR."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)

        # Transition to IN_PROGRESS
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value},
            headers=auth_header(admin_data["access_token"]),
        )

        # Try to transition to RESOLVED without note
        response = await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.RESOLVED.value},  # No content
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    async def test_open_to_resolved_fails(self, client: AsyncClient) -> None:
        """Cannot transition directly from OPEN to RESOLVED (must go through IN_PROGRESS)."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)

        # Try to transition directly to RESOLVED
        response = await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={
                "to_status": TicketStatus.RESOLVED.value,
                "content": "Trying to skip IN_PROGRESS",
            },
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    async def test_open_to_verified_fails(self, client: AsyncClient) -> None:
        """Cannot transition to VERIFIED via /transition endpoint."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)

        # Try to transition directly to VERIFIED
        response = await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.VERIFIED.value},
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    async def test_verified_is_terminal(self, client: AsyncClient) -> None:
        """Cannot transition out of VERIFIED state."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)

        # Go through the full workflow
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.RESOLVED.value, "content": "Done"},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/tickets/{ticket['id']}/verify",
            json={"content": "Verified by admin"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Try to transition out of VERIFIED
        response = await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.OPEN.value},
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    async def test_non_owner_cannot_transition(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """Non-owner member cannot transition a ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)

        # Add another member
        member_data = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member_data["user"]["id"], Role.MEMBER)

        # Member tries to transition admin's ticket
        response = await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value},
            headers=auth_header(member_data["access_token"]),
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_transition_creates_activity(self, client: AsyncClient) -> None:
        """Each transition creates a ticket_activity record."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)

        # Transition to IN_PROGRESS
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value, "content": "Starting work"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Check activities
        activity_response = await client.get(
            f"/tickets/{ticket['id']}/activity",
            headers=auth_header(admin_data["access_token"]),
        )
        assert activity_response.status_code == 200
        activities = activity_response.json()["items"]

        # Should have CREATED and STATUS_CHANGE activities
        activity_types = [a["type"] for a in activities]
        assert TicketActivityType.CREATED.value in activity_types
        assert TicketActivityType.STATUS_CHANGE.value in activity_types

        # Find the STATUS_CHANGE activity
        status_change = next(a for a in activities if a["type"] == TicketActivityType.STATUS_CHANGE.value)
        assert status_change["old_status"] == TicketStatus.OPEN.value
        assert status_change["new_status"] == TicketStatus.IN_PROGRESS.value
        assert status_change["content"] == "Starting work"


class TestVerifyTicket:
    """Tests for POST /tickets/{id}/verify."""

    async def test_admin_can_verify(self, client: AsyncClient) -> None:
        """Admin can verify a ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)

        # Go through workflow to RESOLVED
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.RESOLVED.value, "content": "Done"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Admin verifies
        response = await client.post(
            f"/tickets/{ticket['id']}/verify",
            json={"content": "Looks good!"},
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 200
        result = response.json()["ticket"]
        assert result["status"] == TicketStatus.VERIFIED.value
        assert result["verified_by"] == admin_data["user"]["id"]
        assert result["verified_at"] is not None
        assert result["verified_note"] == "Looks good!"

    async def test_section_leader_can_verify_section_ticket(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Section leader can verify a ticket in their section."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        # Create a section ticket
        ticket = await create_ticket(
            client,
            cycle_id,
            admin_data,
            visibility=TicketVisibility.SECTION,
            section="Soprano",
        )

        # Add section leader
        leader_data = await register_user(client, "leader@example.com")
        await add_member(db_session, team_id, leader_data["user"]["id"], Role.SECTION_LEADER, "Soprano")

        # Go through workflow
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.RESOLVED.value, "content": "Done"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Section leader verifies
        response = await client.post(
            f"/tickets/{ticket['id']}/verify",
            json={"content": "Section leader approval"},
            headers=auth_header(leader_data["access_token"]),
        )
        assert response.status_code == 200
        result = response.json()["ticket"]
        assert result["status"] == TicketStatus.VERIFIED.value
        assert result["verified_by"] == leader_data["user"]["id"]

    async def test_member_cannot_verify(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """Regular member cannot verify a ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)

        # Add member
        member_data = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member_data["user"]["id"], Role.MEMBER)

        # Go through workflow
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.RESOLVED.value, "content": "Done"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Member tries to verify
        response = await client.post(
            f"/tickets/{ticket['id']}/verify",
            json={"content": "Member trying to verify"},
            headers=auth_header(member_data["access_token"]),
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_section_leader_cannot_verify_other_section(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Section leader cannot verify ticket in another section."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        # Create a Soprano section ticket
        ticket = await create_ticket(
            client,
            cycle_id,
            admin_data,
            visibility=TicketVisibility.SECTION,
            section="Soprano",
        )

        # Add Alto section leader
        leader_data = await register_user(client, "leader@example.com")
        await add_member(db_session, team_id, leader_data["user"]["id"], Role.SECTION_LEADER, "Alto")

        # Go through workflow
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.RESOLVED.value, "content": "Done"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Alto section leader tries to verify Soprano ticket
        response = await client.post(
            f"/tickets/{ticket['id']}/verify",
            json={"content": "Wrong section leader"},
            headers=auth_header(leader_data["access_token"]),
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_cannot_verify_already_verified(self, client: AsyncClient) -> None:
        """Cannot verify an already verified ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)

        # Go through full workflow
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.RESOLVED.value, "content": "Done"},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/tickets/{ticket['id']}/verify",
            json={"content": "First verification"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Try to verify again
        response = await client.post(
            f"/tickets/{ticket['id']}/verify",
            json={"content": "Second verification"},
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    async def test_verify_creates_activity(self, client: AsyncClient) -> None:
        """Verification creates a VERIFIED activity record."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        ticket = await create_ticket(client, cycle_id, admin_data)

        # Go through workflow
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/tickets/{ticket['id']}/transition",
            json={"to_status": TicketStatus.RESOLVED.value, "content": "Done"},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/tickets/{ticket['id']}/verify",
            json={"content": "Verified!"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Check activities
        activity_response = await client.get(
            f"/tickets/{ticket['id']}/activity",
            headers=auth_header(admin_data["access_token"]),
        )
        activities = activity_response.json()["items"]

        # Find VERIFIED activity
        verified_activity = next(
            (a for a in activities if a["type"] == TicketActivityType.VERIFIED.value),
            None,
        )
        assert verified_activity is not None
        assert verified_activity["old_status"] == TicketStatus.RESOLVED.value
        assert verified_activity["new_status"] == TicketStatus.VERIFIED.value
        assert verified_activity["content"] == "Verified!"


class TestFullWorkflow:
    """Test complete ticket workflow from creation to verification."""

    async def test_happy_path_full_workflow(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """Test OPEN -> IN_PROGRESS -> RESOLVED -> VERIFIED workflow."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        # Add a section leader
        leader_data = await register_user(client, "leader@example.com")
        await add_member(db_session, team_id, leader_data["user"]["id"], Role.SECTION_LEADER, "Soprano")

        # Create a member who owns the ticket
        member_data = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member_data["user"]["id"], Role.MEMBER, "Soprano")

        # Member creates ticket
        ticket_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Member's Issue",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.MEDIUM.value,
                "visibility": TicketVisibility.SECTION.value,
                "section": "Soprano",
            },
            headers=auth_header(member_data["access_token"]),
        )
        assert ticket_response.status_code == 201
        ticket = ticket_response.json()["ticket"]
        ticket_id = ticket["id"]

        # Step 1: OPEN -> IN_PROGRESS
        response1 = await client.post(
            f"/tickets/{ticket_id}/transition",
            json={"to_status": TicketStatus.IN_PROGRESS.value, "content": "Starting to work on it"},
            headers=auth_header(member_data["access_token"]),
        )
        assert response1.status_code == 200
        assert response1.json()["ticket"]["status"] == TicketStatus.IN_PROGRESS.value

        # Step 2: IN_PROGRESS -> RESOLVED
        response2 = await client.post(
            f"/tickets/{ticket_id}/transition",
            json={
                "to_status": TicketStatus.RESOLVED.value,
                "content": "I practiced the section and the pitch is now correct",
            },
            headers=auth_header(member_data["access_token"]),
        )
        assert response2.status_code == 200
        result2 = response2.json()["ticket"]
        assert result2["status"] == TicketStatus.RESOLVED.value
        assert result2["resolved_note"] == "I practiced the section and the pitch is now correct"
        assert result2["resolved_at"] is not None

        # Step 3: Section leader verifies
        response3 = await client.post(
            f"/tickets/{ticket_id}/verify",
            json={"content": "Confirmed, pitch is now correct. Good job!"},
            headers=auth_header(leader_data["access_token"]),
        )
        assert response3.status_code == 200
        result3 = response3.json()["ticket"]
        assert result3["status"] == TicketStatus.VERIFIED.value
        assert result3["verified_by"] == leader_data["user"]["id"]
        assert result3["verified_at"] is not None
        assert result3["verified_note"] == "Confirmed, pitch is now correct. Good job!"

        # Verify all activity records
        activity_response = await client.get(
            f"/tickets/{ticket_id}/activity",
            headers=auth_header(leader_data["access_token"]),
        )
        activities = activity_response.json()["items"]

        # Should have: CREATED, STATUS_CHANGE (to IN_PROGRESS), STATUS_CHANGE (to RESOLVED), VERIFIED
        assert len(activities) == 4

        activity_types = [a["type"] for a in activities]
        assert activity_types.count(TicketActivityType.CREATED.value) == 1
        assert activity_types.count(TicketActivityType.STATUS_CHANGE.value) == 2
        assert activity_types.count(TicketActivityType.VERIFIED.value) == 1

