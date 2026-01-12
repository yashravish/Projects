"""Tests for assignments.

Tests cover:
- Happy paths:
  - ADMIN creates TEAM assignment
  - ADMIN creates SECTION assignment
  - SECTION_LEADER creates SECTION assignment for own section
  - Member sees TEAM assignments
  - Member sees SECTION assignments for their section only
  - Creator can update assignment
  - ADMIN can update any assignment
  - ADMIN can delete assignment
- Unauthorized:
  - MEMBER cannot create assignments
  - SECTION_LEADER cannot create TEAM assignment
  - SECTION_LEADER cannot create SECTION assignment for different section
  - Non-creator, non-admin cannot update assignment
  - Non-admin cannot delete assignment
- Pagination:
  - Sorting: priority DESC, due_at ASC, created_at DESC, id
  - Cursor pagination is deterministic
- Edge cases:
  - Visibility filters correctly (TEAM always visible, SECTION only for user's section)
"""

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.database import get_db
from app.main import app

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
            # Clean up before test (order matters due to FK constraints)
            await session.execute(text("DELETE FROM assignments"))
            await session.execute(text("DELETE FROM rehearsal_cycles"))
            await session.execute(text("DELETE FROM invites"))
            await session.execute(text("DELETE FROM team_memberships"))
            await session.execute(text("DELETE FROM teams"))
            await session.execute(text("DELETE FROM users"))
            await session.commit()

            yield session

            # Clean up after test
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


async def create_cycle(client: AsyncClient, auth_data: dict, team_id: str, days_ahead: int = 1) -> str:
    """Helper to create a cycle and return cycle_id."""
    date_str = (datetime.now(UTC) + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    response = await client.post(
        f"/teams/{team_id}/cycles",
        json={"date": date_str, "label": f"Test Cycle +{days_ahead}d"},
        headers=auth_header(auth_data["access_token"]),
    )
    assert response.status_code == 201
    return response.json()["cycle"]["id"]


async def add_membership(
    db_session: AsyncSession,
    team_id: str,
    user_id: str,
    role: str,
    section: str | None = None,
) -> None:
    """Add a user to a team with specified role."""
    await db_session.execute(
        text("""
            INSERT INTO team_memberships (id, team_id, user_id, role, section)
            VALUES (:id, :team_id, :user_id, :role, :section)
        """),
        {
            "id": uuid.uuid4(),
            "team_id": uuid.UUID(team_id),
            "user_id": uuid.UUID(user_id),
            "role": role,
            "section": section,
        },
    )
    await db_session.commit()


class TestCreateAssignment:
    """Tests for POST /cycles/{cycle_id}/assignments."""

    async def test_admin_creates_team_assignment(self, client: AsyncClient) -> None:
        """ADMIN can create TEAM scope assignment."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        response = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Learn new song",
                "type": "SONG_WORK",
                "scope": "TEAM",
                "priority": "MEDIUM",
                "song_ref": "Amazing Grace",
                "notes": "Focus on the chorus",
            },
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 201
        data = response.json()
        assert data["assignment"]["title"] == "Learn new song"
        assert data["assignment"]["scope"] == "TEAM"
        assert data["assignment"]["priority"] == "MEDIUM"
        assert data["assignment"]["song_ref"] == "Amazing Grace"
        assert data["assignment"]["notes"] == "Focus on the chorus"
        assert data["assignment"]["created_by"] == admin_data["user"]["id"]
        assert data["assignment"]["section"] is None

    async def test_admin_creates_section_assignment(self, client: AsyncClient) -> None:
        """ADMIN can create SECTION scope assignment."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        response = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Alto part practice",
                "type": "TECHNIQUE",
                "scope": "SECTION",
                "section": "Alto",
                "priority": "BLOCKING",
            },
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 201
        data = response.json()
        assert data["assignment"]["scope"] == "SECTION"
        assert data["assignment"]["section"] == "Alto"
        assert data["assignment"]["priority"] == "BLOCKING"

    async def test_section_leader_creates_own_section_assignment(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """SECTION_LEADER can create SECTION assignment for their own section."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create section leader
        leader_data = await register_user(client, "leader@example.com")
        await add_membership(db_session, team_id, leader_data["user"]["id"], "SECTION_LEADER", "Soprano")

        response = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Soprano warm-up",
                "type": "TECHNIQUE",
                "scope": "SECTION",
                "section": "Soprano",
                "priority": "LOW",
            },
            headers=auth_header(leader_data["access_token"]),
        )

        assert response.status_code == 201
        data = response.json()
        assert data["assignment"]["scope"] == "SECTION"
        assert data["assignment"]["section"] == "Soprano"

    async def test_section_leader_cannot_create_team_assignment(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """SECTION_LEADER cannot create TEAM scope assignment."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create section leader
        leader_data = await register_user(client, "leader@example.com")
        await add_membership(db_session, team_id, leader_data["user"]["id"], "SECTION_LEADER", "Soprano")

        response = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Team-wide practice",
                "type": "SONG_WORK",
                "scope": "TEAM",
                "priority": "MEDIUM",
            },
            headers=auth_header(leader_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_section_leader_cannot_create_other_section_assignment(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """SECTION_LEADER cannot create assignment for another section."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create section leader for Soprano
        leader_data = await register_user(client, "leader@example.com")
        await add_membership(db_session, team_id, leader_data["user"]["id"], "SECTION_LEADER", "Soprano")

        # Try to create assignment for Alto
        response = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Alto practice",
                "type": "TECHNIQUE",
                "scope": "SECTION",
                "section": "Alto",
                "priority": "LOW",
            },
            headers=auth_header(leader_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_member_cannot_create_assignment(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """MEMBER cannot create assignments."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create member
        member_data = await register_user(client, "member@example.com")
        await add_membership(db_session, team_id, member_data["user"]["id"], "MEMBER", "Tenor")

        response = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Member try",
                "type": "SONG_WORK",
                "scope": "TEAM",
                "priority": "LOW",
            },
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_team_scope_with_section_rejected(self, client: AsyncClient) -> None:
        """TEAM scope assignment with section field is rejected by validation."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        response = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Bad assignment",
                "type": "SONG_WORK",
                "scope": "TEAM",
                "section": "Soprano",  # Invalid for TEAM scope
                "priority": "LOW",
            },
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 422  # Validation error

    async def test_section_scope_without_section_rejected(self, client: AsyncClient) -> None:
        """SECTION scope assignment without section field is rejected."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        response = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Bad assignment",
                "type": "SONG_WORK",
                "scope": "SECTION",
                # Missing section field
                "priority": "LOW",
            },
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 422  # Validation error


class TestListAssignments:
    """Tests for GET /cycles/{cycle_id}/assignments."""

    async def test_member_sees_team_assignments(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Member sees TEAM scope assignments."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Admin creates TEAM assignment
        await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Team practice",
                "type": "SONG_WORK",
                "scope": "TEAM",
                "priority": "MEDIUM",
            },
            headers=auth_header(admin_data["access_token"]),
        )

        # Create member
        member_data = await register_user(client, "member@example.com")
        await add_membership(db_session, team_id, member_data["user"]["id"], "MEMBER", "Tenor")

        # Member lists assignments
        response = await client.get(
            f"/cycles/{cycle_id}/assignments",
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["title"] == "Team practice"
        assert data["items"][0]["scope"] == "TEAM"

    async def test_member_sees_own_section_assignments(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Member sees SECTION assignments for their section only."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Admin creates assignments for different sections
        await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Tenor part",
                "type": "TECHNIQUE",
                "scope": "SECTION",
                "section": "Tenor",
                "priority": "LOW",
            },
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Alto part",
                "type": "TECHNIQUE",
                "scope": "SECTION",
                "section": "Alto",
                "priority": "LOW",
            },
            headers=auth_header(admin_data["access_token"]),
        )

        # Create Tenor member
        member_data = await register_user(client, "member@example.com")
        await add_membership(db_session, team_id, member_data["user"]["id"], "MEMBER", "Tenor")

        # Member lists assignments
        response = await client.get(
            f"/cycles/{cycle_id}/assignments",
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        # Should only see Tenor section assignment (not Alto)
        assert len(data["items"]) == 1
        assert data["items"][0]["title"] == "Tenor part"
        assert data["items"][0]["section"] == "Tenor"

    async def test_member_sees_team_and_own_section(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Member sees both TEAM and their section's SECTION assignments."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create various assignments
        await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={"title": "Team A", "type": "SONG_WORK", "scope": "TEAM", "priority": "MEDIUM"},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={"title": "Tenor X", "type": "TECHNIQUE", "scope": "SECTION", "section": "Tenor", "priority": "LOW"},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={"title": "Alto Y", "type": "TECHNIQUE", "scope": "SECTION", "section": "Alto", "priority": "LOW"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Create Tenor member
        member_data = await register_user(client, "member@example.com")
        await add_membership(db_session, team_id, member_data["user"]["id"], "MEMBER", "Tenor")

        response = await client.get(
            f"/cycles/{cycle_id}/assignments",
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        # Should see TEAM + Tenor (not Alto)
        assert len(data["items"]) == 2
        titles = {item["title"] for item in data["items"]}
        assert titles == {"Team A", "Tenor X"}

    async def test_sorting_priority_desc(
        self, client: AsyncClient
    ) -> None:
        """Assignments sorted by priority DESC (BLOCKING first)."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create assignments with different priorities
        for priority in ["LOW", "BLOCKING", "MEDIUM"]:
            await client.post(
                f"/cycles/{cycle_id}/assignments",
                json={"title": f"Priority {priority}", "type": "SONG_WORK", "scope": "TEAM", "priority": priority},
                headers=auth_header(admin_data["access_token"]),
            )

        response = await client.get(
            f"/cycles/{cycle_id}/assignments",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        priorities = [item["priority"] for item in data["items"]]
        # BLOCKING first, then MEDIUM, then LOW
        assert priorities == ["BLOCKING", "MEDIUM", "LOW"]

    async def test_pagination_deterministic(
        self, client: AsyncClient
    ) -> None:
        """Pagination is deterministic - no duplicates or missing items."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create 7 assignments
        for i in range(7):
            await client.post(
                f"/cycles/{cycle_id}/assignments",
                json={"title": f"Assignment {i}", "type": "SONG_WORK", "scope": "TEAM", "priority": "LOW"},
                headers=auth_header(admin_data["access_token"]),
            )

        # Fetch in pages of 3
        all_ids: set[str] = set()
        cursor = None

        while True:
            url = f"/cycles/{cycle_id}/assignments?limit=3"
            if cursor:
                url += f"&cursor={cursor}"

            response = await client.get(url, headers=auth_header(admin_data["access_token"]))
            assert response.status_code == 200
            data = response.json()

            page_ids = {item["id"] for item in data["items"]}
            # Check no duplicates
            assert page_ids.isdisjoint(all_ids), "Duplicate items in pagination"
            all_ids.update(page_ids)

            if data["next_cursor"] is None:
                break
            cursor = data["next_cursor"]

        # Should have all 7 items
        assert len(all_ids) == 7

    async def test_filter_by_scope(self, client: AsyncClient) -> None:
        """Can filter assignments by scope."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create TEAM and SECTION assignments
        await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={"title": "Team 1", "type": "SONG_WORK", "scope": "TEAM", "priority": "LOW"},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={"title": "Section 1", "type": "TECHNIQUE", "scope": "SECTION", "section": "Tenor", "priority": "LOW"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Filter by TEAM scope
        response = await client.get(
            f"/cycles/{cycle_id}/assignments?scope=TEAM",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["scope"] == "TEAM"

    async def test_non_member_forbidden(self, client: AsyncClient) -> None:
        """Non-member cannot list assignments."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        other_data = await register_user(client, "other@example.com")

        response = await client.get(
            f"/cycles/{cycle_id}/assignments",
            headers=auth_header(other_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"


class TestUpdateAssignment:
    """Tests for PATCH /assignments/{id}."""

    async def test_creator_can_update(self, client: AsyncClient) -> None:
        """Assignment creator can update it."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create assignment
        create_resp = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={"title": "Original", "type": "SONG_WORK", "scope": "TEAM", "priority": "LOW"},
            headers=auth_header(admin_data["access_token"]),
        )
        assignment_id = create_resp.json()["assignment"]["id"]

        # Update assignment
        response = await client.patch(
            f"/assignments/{assignment_id}",
            json={"title": "Updated", "priority": "BLOCKING"},
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["assignment"]["title"] == "Updated"
        assert data["assignment"]["priority"] == "BLOCKING"

    async def test_admin_can_update_any(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Admin can update any assignment, even those created by others."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create section leader who creates an assignment
        leader_data = await register_user(client, "leader@example.com")
        await add_membership(db_session, team_id, leader_data["user"]["id"], "SECTION_LEADER", "Soprano")

        create_resp = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Leader Created",
                "type": "TECHNIQUE",
                "scope": "SECTION",
                "section": "Soprano",
                "priority": "LOW",
            },
            headers=auth_header(leader_data["access_token"]),
        )
        assignment_id = create_resp.json()["assignment"]["id"]

        # Admin updates the assignment
        response = await client.patch(
            f"/assignments/{assignment_id}",
            json={"title": "Admin Updated"},
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        assert response.json()["assignment"]["title"] == "Admin Updated"

    async def test_non_creator_non_admin_forbidden(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Non-creator, non-admin cannot update assignment."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Admin creates assignment
        create_resp = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={"title": "Admin Created", "type": "SONG_WORK", "scope": "TEAM", "priority": "LOW"},
            headers=auth_header(admin_data["access_token"]),
        )
        assignment_id = create_resp.json()["assignment"]["id"]

        # Create section leader (non-admin, non-creator)
        leader_data = await register_user(client, "leader@example.com")
        await add_membership(db_session, team_id, leader_data["user"]["id"], "SECTION_LEADER", "Soprano")

        # Leader tries to update
        response = await client.patch(
            f"/assignments/{assignment_id}",
            json={"title": "Attempted Update"},
            headers=auth_header(leader_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_member_cannot_update(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Member cannot update any assignment."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        create_resp = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={"title": "Test", "type": "SONG_WORK", "scope": "TEAM", "priority": "LOW"},
            headers=auth_header(admin_data["access_token"]),
        )
        assignment_id = create_resp.json()["assignment"]["id"]

        # Create member
        member_data = await register_user(client, "member@example.com")
        await add_membership(db_session, team_id, member_data["user"]["id"], "MEMBER", "Tenor")

        response = await client.patch(
            f"/assignments/{assignment_id}",
            json={"title": "Member Update"},
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"


class TestDeleteAssignment:
    """Tests for DELETE /assignments/{id}."""

    async def test_admin_can_delete(self, client: AsyncClient) -> None:
        """Admin can delete assignments."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        create_resp = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={"title": "To Delete", "type": "SONG_WORK", "scope": "TEAM", "priority": "LOW"},
            headers=auth_header(admin_data["access_token"]),
        )
        assignment_id = create_resp.json()["assignment"]["id"]

        # Delete assignment
        response = await client.delete(
            f"/assignments/{assignment_id}",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 204

        # Verify deletion
        list_resp = await client.get(
            f"/cycles/{cycle_id}/assignments",
            headers=auth_header(admin_data["access_token"]),
        )
        assert len(list_resp.json()["items"]) == 0

    async def test_section_leader_cannot_delete(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Section leader cannot delete assignments (even their own)."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create section leader
        leader_data = await register_user(client, "leader@example.com")
        await add_membership(db_session, team_id, leader_data["user"]["id"], "SECTION_LEADER", "Soprano")

        # Leader creates assignment
        create_resp = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={
                "title": "Leader Assignment",
                "type": "TECHNIQUE",
                "scope": "SECTION",
                "section": "Soprano",
                "priority": "LOW",
            },
            headers=auth_header(leader_data["access_token"]),
        )
        assignment_id = create_resp.json()["assignment"]["id"]

        # Leader tries to delete
        response = await client.delete(
            f"/assignments/{assignment_id}",
            headers=auth_header(leader_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_member_cannot_delete(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Member cannot delete assignments."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        create_resp = await client.post(
            f"/cycles/{cycle_id}/assignments",
            json={"title": "Test", "type": "SONG_WORK", "scope": "TEAM", "priority": "LOW"},
            headers=auth_header(admin_data["access_token"]),
        )
        assignment_id = create_resp.json()["assignment"]["id"]

        # Create member
        member_data = await register_user(client, "member@example.com")
        await add_membership(db_session, team_id, member_data["user"]["id"], "MEMBER", "Tenor")

        response = await client.delete(
            f"/assignments/{assignment_id}",
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_delete_nonexistent_returns_404(self, client: AsyncClient) -> None:
        """Deleting nonexistent assignment returns 404."""
        admin_data = await register_user(client, "admin@example.com")
        await create_team(client, admin_data)

        fake_id = uuid.uuid4()
        response = await client.delete(
            f"/assignments/{fake_id}",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 404

