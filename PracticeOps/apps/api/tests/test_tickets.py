"""Tests for Milestone 7: Tickets.

Tests cover:
- Ticket creation (all visibility levels)
- Ticket listing with visibility enforcement
- Ticket claim flow (happy path + edge cases)
- Ticket update RBAC
- Ticket activity viewing
- Pagination with deterministic sorting
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


class TestCreateTicket:
    """Tests for POST /cycles/{cycle_id}/tickets."""

    async def test_member_creates_private_ticket(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """A member can create a PRIVATE ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        member_data = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member_data["user"]["id"], Role.MEMBER, "Soprano")

        response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "My Private Issue",
                "description": "Details about my issue",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.PRIVATE.value,
            },
            headers=auth_header(member_data["access_token"]),
        )
        assert response.status_code == 201
        ticket = response.json()["ticket"]
        assert ticket["title"] == "My Private Issue"
        assert ticket["visibility"] == TicketVisibility.PRIVATE.value
        assert ticket["status"] == TicketStatus.OPEN.value
        assert ticket["owner_id"] == member_data["user"]["id"]
        assert ticket["created_by"] == member_data["user"]["id"]
        assert ticket["claimable"] is False

    async def test_member_creates_section_ticket(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """A member can create a SECTION ticket with section specified."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        member_data = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member_data["user"]["id"], Role.MEMBER, "Soprano")

        response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Section Issue",
                "category": TicketCategory.BLEND.value,
                "priority": Priority.MEDIUM.value,
                "visibility": TicketVisibility.SECTION.value,
                "section": "Soprano",
            },
            headers=auth_header(member_data["access_token"]),
        )
        assert response.status_code == 201
        ticket = response.json()["ticket"]
        assert ticket["visibility"] == TicketVisibility.SECTION.value
        assert ticket["section"] == "Soprano"

    async def test_member_creates_team_ticket(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """A member can create a TEAM ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        member_data = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member_data["user"]["id"], Role.MEMBER, "Soprano")

        response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Team Issue",
                "category": TicketCategory.OTHER.value,
                "priority": Priority.BLOCKING.value,
                "visibility": TicketVisibility.TEAM.value,
            },
            headers=auth_header(member_data["access_token"]),
        )
        assert response.status_code == 201
        ticket = response.json()["ticket"]
        assert ticket["visibility"] == TicketVisibility.TEAM.value

    async def test_section_visibility_requires_section(self, client: AsyncClient) -> None:
        """Creating a SECTION ticket without section returns validation error."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Missing Section",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.SECTION.value,
                # section is missing
            },
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 422

    async def test_leader_creates_claimable_ticket(self, client: AsyncClient) -> None:
        """A leader (ADMIN/SECTION_LEADER) can create a claimable ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Claimable Issue",
                "category": TicketCategory.TECHNIQUE.value,
                "priority": Priority.MEDIUM.value,
                "visibility": TicketVisibility.TEAM.value,
                "claimable": True,
            },
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 201
        ticket = response.json()["ticket"]
        assert ticket["claimable"] is True
        assert ticket["owner_id"] is None
        assert ticket["claimed_by"] is None

    async def test_member_cannot_create_claimable_ticket(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """A regular member cannot create a claimable ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        member_data = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member_data["user"]["id"], Role.MEMBER, "Soprano")

        response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Member Claimable Attempt",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.TEAM.value,
                "claimable": True,
            },
            headers=auth_header(member_data["access_token"]),
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"


class TestListTickets:
    """Tests for GET /cycles/{cycle_id}/tickets."""

    async def test_member_sees_team_tickets(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """A member can see TEAM visibility tickets."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        # Admin creates TEAM ticket
        await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Team Issue",
                "category": TicketCategory.OTHER.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.TEAM.value,
            },
            headers=auth_header(admin_data["access_token"]),
        )

        member_data = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member_data["user"]["id"], Role.MEMBER, "Alto")

        response = await client.get(
            f"/cycles/{cycle_id}/tickets",
            headers=auth_header(member_data["access_token"]),
        )
        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) == 1
        assert items[0]["title"] == "Team Issue"

    async def test_member_sees_own_section_tickets(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """A member can see SECTION tickets for their own section."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        # Admin creates SECTION ticket for Soprano
        await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Soprano Issue",
                "category": TicketCategory.BLEND.value,
                "priority": Priority.MEDIUM.value,
                "visibility": TicketVisibility.SECTION.value,
                "section": "Soprano",
            },
            headers=auth_header(admin_data["access_token"]),
        )

        # Admin creates SECTION ticket for Alto
        await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Alto Issue",
                "category": TicketCategory.BLEND.value,
                "priority": Priority.MEDIUM.value,
                "visibility": TicketVisibility.SECTION.value,
                "section": "Alto",
            },
            headers=auth_header(admin_data["access_token"]),
        )

        # Soprano member should only see Soprano ticket
        soprano_member = await register_user(client, "soprano@example.com")
        await add_member(db_session, team_id, soprano_member["user"]["id"], Role.MEMBER, "Soprano")

        response = await client.get(
            f"/cycles/{cycle_id}/tickets",
            headers=auth_header(soprano_member["access_token"]),
        )
        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) == 1
        assert items[0]["title"] == "Soprano Issue"

    async def test_member_sees_own_private_tickets(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """A member can see their own PRIVATE tickets but not others'."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        member1 = await register_user(client, "member1@example.com")
        await add_member(db_session, team_id, member1["user"]["id"], Role.MEMBER, "Soprano")

        member2 = await register_user(client, "member2@example.com")
        await add_member(db_session, team_id, member2["user"]["id"], Role.MEMBER, "Soprano")

        # Member1 creates private ticket
        await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Member1 Private",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.PRIVATE.value,
            },
            headers=auth_header(member1["access_token"]),
        )

        # Member2 creates private ticket
        await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Member2 Private",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.PRIVATE.value,
            },
            headers=auth_header(member2["access_token"]),
        )

        # Member1 only sees their own ticket
        response = await client.get(
            f"/cycles/{cycle_id}/tickets",
            headers=auth_header(member1["access_token"]),
        )
        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) == 1
        assert items[0]["title"] == "Member1 Private"

    async def test_admin_sees_all_tickets(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """An admin can see all tickets regardless of visibility."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        member = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member["user"]["id"], Role.MEMBER, "Soprano")

        # Member creates PRIVATE ticket
        await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Private Issue",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.PRIVATE.value,
            },
            headers=auth_header(member["access_token"]),
        )

        # Admin can see it
        response = await client.get(
            f"/cycles/{cycle_id}/tickets",
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) == 1

    async def test_sorting_priority_desc(self, client: AsyncClient) -> None:
        """Tickets are sorted by priority DESC (BLOCKING first)."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        # Create tickets with different priorities
        await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Low Priority",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.TEAM.value,
            },
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Blocking Priority",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.BLOCKING.value,
                "visibility": TicketVisibility.TEAM.value,
            },
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Medium Priority",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.MEDIUM.value,
                "visibility": TicketVisibility.TEAM.value,
            },
            headers=auth_header(admin_data["access_token"]),
        )

        response = await client.get(
            f"/cycles/{cycle_id}/tickets",
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) == 3
        assert items[0]["title"] == "Blocking Priority"
        assert items[1]["title"] == "Medium Priority"
        assert items[2]["title"] == "Low Priority"

    async def test_pagination_deterministic(self, client: AsyncClient) -> None:
        """Pagination is deterministic - no duplicates or missing items."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        # Create 5 tickets
        for i in range(5):
            await client.post(
                f"/cycles/{cycle_id}/tickets",
                json={
                    "title": f"Ticket {i}",
                    "category": TicketCategory.PITCH.value,
                    "priority": Priority.LOW.value,
                    "visibility": TicketVisibility.TEAM.value,
                },
                headers=auth_header(admin_data["access_token"]),
            )

        # Get first page
        response1 = await client.get(
            f"/cycles/{cycle_id}/tickets?limit=2",
            headers=auth_header(admin_data["access_token"]),
        )
        assert response1.status_code == 200
        data1 = response1.json()
        assert len(data1["items"]) == 2
        assert data1["next_cursor"] is not None

        # Get second page
        response2 = await client.get(
            f"/cycles/{cycle_id}/tickets?limit=2&cursor={data1['next_cursor']}",
            headers=auth_header(admin_data["access_token"]),
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert len(data2["items"]) == 2
        assert data2["next_cursor"] is not None

        # Get third page
        response3 = await client.get(
            f"/cycles/{cycle_id}/tickets?limit=2&cursor={data2['next_cursor']}",
            headers=auth_header(admin_data["access_token"]),
        )
        assert response3.status_code == 200
        data3 = response3.json()
        assert len(data3["items"]) == 1
        assert data3["next_cursor"] is None

        # Verify no duplicates
        all_items = data1["items"] + data2["items"] + data3["items"]
        all_ids = {item["id"] for item in all_items}
        assert len(all_ids) == 5


class TestClaimTicket:
    """Tests for POST /tickets/{id}/claim."""

    async def test_member_claims_claimable_ticket(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """A member can claim a claimable ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        # Admin creates claimable ticket
        create_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Claimable Issue",
                "category": TicketCategory.TECHNIQUE.value,
                "priority": Priority.MEDIUM.value,
                "visibility": TicketVisibility.TEAM.value,
                "claimable": True,
            },
            headers=auth_header(admin_data["access_token"]),
        )
        ticket_id = create_response.json()["ticket"]["id"]

        member = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member["user"]["id"], Role.MEMBER, "Soprano")

        # Member claims ticket
        claim_response = await client.post(
            f"/tickets/{ticket_id}/claim",
            headers=auth_header(member["access_token"]),
        )
        assert claim_response.status_code == 200
        ticket = claim_response.json()["ticket"]
        assert ticket["owner_id"] == member["user"]["id"]
        assert ticket["claimed_by"] == member["user"]["id"]

    async def test_claim_creates_activity(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """Claiming a ticket creates a CLAIMED activity record."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        # Admin creates claimable ticket
        create_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Claimable Issue",
                "category": TicketCategory.TECHNIQUE.value,
                "priority": Priority.MEDIUM.value,
                "visibility": TicketVisibility.TEAM.value,
                "claimable": True,
            },
            headers=auth_header(admin_data["access_token"]),
        )
        ticket_id = create_response.json()["ticket"]["id"]

        member = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member["user"]["id"], Role.MEMBER, "Soprano")

        # Member claims ticket
        await client.post(
            f"/tickets/{ticket_id}/claim",
            headers=auth_header(member["access_token"]),
        )

        # Check activity
        activity_response = await client.get(
            f"/tickets/{ticket_id}/activity",
            headers=auth_header(member["access_token"]),
        )
        assert activity_response.status_code == 200
        activities = activity_response.json()["items"]
        # Should have CREATED and CLAIMED activities
        activity_types = [a["type"] for a in activities]
        assert TicketActivityType.CREATED.value in activity_types
        assert TicketActivityType.CLAIMED.value in activity_types

    async def test_double_claim_returns_conflict(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """Attempting to claim an already-claimed ticket returns CONFLICT."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        # Admin creates claimable ticket
        create_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Claimable Issue",
                "category": TicketCategory.TECHNIQUE.value,
                "priority": Priority.MEDIUM.value,
                "visibility": TicketVisibility.TEAM.value,
                "claimable": True,
            },
            headers=auth_header(admin_data["access_token"]),
        )
        ticket_id = create_response.json()["ticket"]["id"]

        member1 = await register_user(client, "member1@example.com")
        await add_member(db_session, team_id, member1["user"]["id"], Role.MEMBER, "Soprano")

        member2 = await register_user(client, "member2@example.com")
        await add_member(db_session, team_id, member2["user"]["id"], Role.MEMBER, "Alto")

        # Member1 claims ticket
        await client.post(
            f"/tickets/{ticket_id}/claim",
            headers=auth_header(member1["access_token"]),
        )

        # Member2 tries to claim (should fail)
        claim_response = await client.post(
            f"/tickets/{ticket_id}/claim",
            headers=auth_header(member2["access_token"]),
        )
        assert claim_response.status_code == 409
        assert claim_response.json()["error"]["code"] == "CONFLICT"

    async def test_cannot_claim_non_claimable_ticket(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """Attempting to claim a non-claimable ticket returns FORBIDDEN."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        # Admin creates non-claimable ticket
        create_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Non-Claimable Issue",
                "category": TicketCategory.TECHNIQUE.value,
                "priority": Priority.MEDIUM.value,
                "visibility": TicketVisibility.TEAM.value,
                "claimable": False,
            },
            headers=auth_header(admin_data["access_token"]),
        )
        ticket_id = create_response.json()["ticket"]["id"]

        member = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member["user"]["id"], Role.MEMBER, "Soprano")

        # Member tries to claim (should fail)
        claim_response = await client.post(
            f"/tickets/{ticket_id}/claim",
            headers=auth_header(member["access_token"]),
        )
        assert claim_response.status_code == 403

    async def test_out_of_section_cannot_claim_section_ticket(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """A member from another section cannot claim a SECTION-visibility claimable ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        # Admin creates SECTION claimable ticket for Soprano
        create_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Soprano Claimable Issue",
                "category": TicketCategory.BLEND.value,
                "priority": Priority.MEDIUM.value,
                "visibility": TicketVisibility.SECTION.value,
                "section": "Soprano",
                "claimable": True,
            },
            headers=auth_header(admin_data["access_token"]),
        )
        ticket_id = create_response.json()["ticket"]["id"]

        # Alto member tries to claim (should fail - can't see the ticket)
        alto_member = await register_user(client, "alto@example.com")
        await add_member(db_session, team_id, alto_member["user"]["id"], Role.MEMBER, "Alto")

        claim_response = await client.post(
            f"/tickets/{ticket_id}/claim",
            headers=auth_header(alto_member["access_token"]),
        )
        assert claim_response.status_code == 403


class TestUpdateTicket:
    """Tests for PATCH /tickets/{id}."""

    async def test_owner_can_update_ticket(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """The ticket owner can update their ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        member = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member["user"]["id"], Role.MEMBER, "Soprano")

        # Member creates ticket
        create_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Original Title",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.PRIVATE.value,
            },
            headers=auth_header(member["access_token"]),
        )
        ticket_id = create_response.json()["ticket"]["id"]

        # Member updates ticket
        update_response = await client.patch(
            f"/tickets/{ticket_id}",
            json={"title": "Updated Title", "priority": Priority.MEDIUM.value},
            headers=auth_header(member["access_token"]),
        )
        assert update_response.status_code == 200
        ticket = update_response.json()["ticket"]
        assert ticket["title"] == "Updated Title"
        assert ticket["priority"] == Priority.MEDIUM.value

    async def test_admin_can_update_any_ticket(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """An admin can update any ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        member = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member["user"]["id"], Role.MEMBER, "Soprano")

        # Member creates ticket
        create_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Member Ticket",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.PRIVATE.value,
            },
            headers=auth_header(member["access_token"]),
        )
        ticket_id = create_response.json()["ticket"]["id"]

        # Admin updates ticket
        update_response = await client.patch(
            f"/tickets/{ticket_id}",
            json={"title": "Admin Updated"},
            headers=auth_header(admin_data["access_token"]),
        )
        assert update_response.status_code == 200
        assert update_response.json()["ticket"]["title"] == "Admin Updated"

    async def test_other_member_cannot_update_ticket(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """A member who is not the owner cannot update another member's ticket."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        member1 = await register_user(client, "member1@example.com")
        await add_member(db_session, team_id, member1["user"]["id"], Role.MEMBER, "Soprano")

        member2 = await register_user(client, "member2@example.com")
        await add_member(db_session, team_id, member2["user"]["id"], Role.MEMBER, "Soprano")

        # Member1 creates ticket
        create_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Member1 Ticket",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.TEAM.value,
            },
            headers=auth_header(member1["access_token"]),
        )
        ticket_id = create_response.json()["ticket"]["id"]

        # Member2 tries to update (should fail)
        update_response = await client.patch(
            f"/tickets/{ticket_id}",
            json={"title": "Unauthorized Update"},
            headers=auth_header(member2["access_token"]),
        )
        assert update_response.status_code == 403


class TestGetTicketActivity:
    """Tests for GET /tickets/{id}/activity."""

    async def test_owner_can_view_activity(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """The ticket owner can view activity."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        member = await register_user(client, "member@example.com")
        await add_member(db_session, team_id, member["user"]["id"], Role.MEMBER, "Soprano")

        # Member creates ticket
        create_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "My Ticket",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.PRIVATE.value,
            },
            headers=auth_header(member["access_token"]),
        )
        ticket_id = create_response.json()["ticket"]["id"]

        # Member views activity
        response = await client.get(
            f"/tickets/{ticket_id}/activity",
            headers=auth_header(member["access_token"]),
        )
        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) >= 1
        # Should have at least CREATED activity
        assert any(a["type"] == TicketActivityType.CREATED.value for a in items)

    async def test_non_viewer_cannot_see_activity(self, client: AsyncClient, db_session: AsyncSession) -> None:
        """A member who can't view the ticket also can't view activity."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, team_id, admin_data, datetime.now(UTC))

        member1 = await register_user(client, "member1@example.com")
        await add_member(db_session, team_id, member1["user"]["id"], Role.MEMBER, "Soprano")

        member2 = await register_user(client, "member2@example.com")
        await add_member(db_session, team_id, member2["user"]["id"], Role.MEMBER, "Alto")

        # Member1 creates PRIVATE ticket
        create_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Private Ticket",
                "category": TicketCategory.PITCH.value,
                "priority": Priority.LOW.value,
                "visibility": TicketVisibility.PRIVATE.value,
            },
            headers=auth_header(member1["access_token"]),
        )
        ticket_id = create_response.json()["ticket"]["id"]

        # Member2 tries to view activity (should fail)
        response = await client.get(
            f"/tickets/{ticket_id}/activity",
            headers=auth_header(member2["access_token"]),
        )
        assert response.status_code == 403

