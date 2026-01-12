"""Tests for practice logs.

Tests cover:
- Happy paths:
  - Create practice log with assignments
  - Member can see their own logs (me=true)
  - Section leader can see section logs (me=false)
  - Admin can see all logs (me=false)
  - Update practice log (owner only)
- Unauthorized:
  - Another user cannot PATCH someone else's log
  - Member cannot see others' logs with me=false
  - Section leader cannot see other sections
- Edge cases:
  - Invalid assignment_id from different cycle → VALIDATION_ERROR
  - Assignment not visible to user (different section) → VALIDATION_ERROR
  - Blocked flag returns suggested_ticket
- Pagination:
  - Deterministic ordering (occurred_at DESC, id)
  - Cursor behavior complies with contract
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


async def create_assignment(
    client: AsyncClient,
    auth_data: dict,
    cycle_id: str,
    title: str,
    scope: str = "TEAM",
    section: str | None = None,
) -> str:
    """Helper to create an assignment and return assignment_id."""
    payload = {
        "title": title,
        "type": "SONG_WORK",
        "scope": scope,
        "priority": "LOW",
    }
    if section:
        payload["section"] = section

    response = await client.post(
        f"/cycles/{cycle_id}/assignments",
        json=payload,
        headers=auth_header(auth_data["access_token"]),
    )
    assert response.status_code == 201
    return response.json()["assignment"]["id"]


class TestCreatePracticeLog:
    """Tests for POST /cycles/{cycle_id}/practice-logs."""

    async def test_create_with_assignments(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Create a practice log with two assignments and verify join table rows."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create two assignments
        assignment1_id = await create_assignment(client, admin_data, cycle_id, "Assignment 1")
        assignment2_id = await create_assignment(client, admin_data, cycle_id, "Assignment 2")

        # Create practice log with both assignments
        response = await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={
                "duration_min": 30,
                "notes": "Good practice session",
                "rating_1_5": 4,
                "blocked_flag": False,
                "assignment_ids": [assignment1_id, assignment2_id],
            },
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 201
        data = response.json()
        assert "practice_log" in data
        assert data["practice_log"]["duration_minutes"] == 30
        assert data["practice_log"]["notes"] == "Good practice session"
        assert data["practice_log"]["rating_1_5"] == 4
        assert data["practice_log"]["blocked_flag"] is False
        assert len(data["practice_log"]["assignments"]) == 2
        assert data["suggested_ticket"] is None  # Not blocked

        # Verify join table rows in DB
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM practice_log_assignments WHERE practice_log_id = :log_id"),
            {"log_id": uuid.UUID(data["practice_log"]["id"])},
        )
        count = result.scalar()
        assert count == 2

    async def test_create_without_assignments(self, client: AsyncClient) -> None:
        """Create a practice log without any assignments."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        response = await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={
                "duration_min": 15,
                "assignment_ids": [],
            },
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 201
        data = response.json()
        assert data["practice_log"]["duration_minutes"] == 15
        assert len(data["practice_log"]["assignments"]) == 0

    async def test_blocked_flag_returns_suggested_ticket(self, client: AsyncClient) -> None:
        """When blocked_flag=true, suggested_ticket is included."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        response = await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={
                "duration_min": 20,
                "blocked_flag": True,
                "assignment_ids": [],
            },
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 201
        data = response.json()
        assert data["practice_log"]["blocked_flag"] is True
        assert data["suggested_ticket"] is not None
        assert "title_suggestion" in data["suggested_ticket"]
        assert "due_date" in data["suggested_ticket"]
        assert data["suggested_ticket"]["visibility_default"] == "PRIVATE"
        assert data["suggested_ticket"]["priority_default"] == "MEDIUM"

    async def test_invalid_assignment_from_different_cycle(
        self, client: AsyncClient
    ) -> None:
        """Using assignment from different cycle returns VALIDATION_ERROR."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create two cycles
        cycle1_id = await create_cycle(client, admin_data, team_id, days_ahead=1)
        cycle2_id = await create_cycle(client, admin_data, team_id, days_ahead=2)

        # Create assignment in cycle 2
        assignment_id = await create_assignment(client, admin_data, cycle2_id, "Wrong Cycle Assignment")

        # Try to use it in cycle 1
        response = await client.post(
            f"/cycles/{cycle1_id}/practice-logs",
            json={
                "duration_min": 30,
                "assignment_ids": [assignment_id],
            },
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    async def test_section_assignment_not_visible_to_other_section(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Member in different section cannot log assignment for another section."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create section-scoped assignment for Soprano
        assignment_id = await create_assignment(
            client, admin_data, cycle_id, "Soprano Only", scope="SECTION", section="Soprano"
        )

        # Create Alto member
        alto_data = await register_user(client, "alto@example.com")
        await add_membership(db_session, team_id, alto_data["user"]["id"], "MEMBER", "Alto")

        # Alto member tries to log Soprano assignment
        response = await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={
                "duration_min": 30,
                "assignment_ids": [assignment_id],
            },
            headers=auth_header(alto_data["access_token"]),
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    async def test_duration_validation(self, client: AsyncClient) -> None:
        """Duration must be between 1 and 600 minutes."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Test below minimum
        response = await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={"duration_min": 0, "assignment_ids": []},
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 422

        # Test above maximum
        response = await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={"duration_min": 601, "assignment_ids": []},
            headers=auth_header(admin_data["access_token"]),
        )
        assert response.status_code == 422


class TestListPracticeLogs:
    """Tests for GET /cycles/{cycle_id}/practice-logs."""

    async def test_member_sees_own_logs_only(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Member with me=true sees only their own logs."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create member and add them to team
        member_data = await register_user(client, "member@example.com")
        await add_membership(db_session, team_id, member_data["user"]["id"], "MEMBER", "Tenor")

        # Admin creates a log
        await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={"duration_min": 30, "assignment_ids": []},
            headers=auth_header(admin_data["access_token"]),
        )

        # Member creates a log
        await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={"duration_min": 20, "assignment_ids": []},
            headers=auth_header(member_data["access_token"]),
        )

        # Member lists with me=true
        response = await client.get(
            f"/cycles/{cycle_id}/practice-logs?me=true",
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["duration_minutes"] == 20  # Only member's log

    async def test_member_cannot_see_others_with_me_false(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Member with me=false is forbidden."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create member
        member_data = await register_user(client, "member@example.com")
        await add_membership(db_session, team_id, member_data["user"]["id"], "MEMBER", "Tenor")

        response = await client.get(
            f"/cycles/{cycle_id}/practice-logs?me=false",
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_section_leader_sees_section_logs(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Section leader with me=false sees logs from their section only."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create section leader for Tenor
        leader_data = await register_user(client, "leader@example.com")
        await add_membership(db_session, team_id, leader_data["user"]["id"], "SECTION_LEADER", "Tenor")

        # Create Tenor member
        tenor_data = await register_user(client, "tenor@example.com")
        await add_membership(db_session, team_id, tenor_data["user"]["id"], "MEMBER", "Tenor")

        # Create Alto member
        alto_data = await register_user(client, "alto@example.com")
        await add_membership(db_session, team_id, alto_data["user"]["id"], "MEMBER", "Alto")

        # Tenor member creates a log
        await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={"duration_min": 25, "assignment_ids": []},
            headers=auth_header(tenor_data["access_token"]),
        )

        # Alto member creates a log
        await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={"duration_min": 35, "assignment_ids": []},
            headers=auth_header(alto_data["access_token"]),
        )

        # Section leader lists with me=false
        response = await client.get(
            f"/cycles/{cycle_id}/practice-logs?me=false",
            headers=auth_header(leader_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        # Should only see Tenor member's log (+ potentially leader's own if they had one)
        assert len(data["items"]) == 1
        assert data["items"][0]["duration_minutes"] == 25  # Tenor's log

    async def test_section_leader_cannot_view_other_sections(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Section leader cannot explicitly request another section."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create section leader for Tenor
        leader_data = await register_user(client, "leader@example.com")
        await add_membership(db_session, team_id, leader_data["user"]["id"], "SECTION_LEADER", "Tenor")

        # Try to access Alto section
        response = await client.get(
            f"/cycles/{cycle_id}/practice-logs?me=false&section=Alto",
            headers=auth_header(leader_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_admin_sees_all_logs(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Admin with me=false sees all team logs."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create members in different sections
        tenor_data = await register_user(client, "tenor@example.com")
        await add_membership(db_session, team_id, tenor_data["user"]["id"], "MEMBER", "Tenor")

        alto_data = await register_user(client, "alto@example.com")
        await add_membership(db_session, team_id, alto_data["user"]["id"], "MEMBER", "Alto")

        # Both create logs
        await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={"duration_min": 25, "assignment_ids": []},
            headers=auth_header(tenor_data["access_token"]),
        )
        await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={"duration_min": 35, "assignment_ids": []},
            headers=auth_header(alto_data["access_token"]),
        )

        # Admin lists with me=false
        response = await client.get(
            f"/cycles/{cycle_id}/practice-logs?me=false",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        # Admin sees both logs
        assert len(data["items"]) == 2

    async def test_pagination_deterministic(self, client: AsyncClient) -> None:
        """Pagination is deterministic - no duplicates or missing items."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create 7 practice logs
        for i in range(7):
            await client.post(
                f"/cycles/{cycle_id}/practice-logs",
                json={"duration_min": 10 + i, "assignment_ids": []},
                headers=auth_header(admin_data["access_token"]),
            )

        # Fetch in pages of 3
        all_ids: set[str] = set()
        cursor = None

        while True:
            url = f"/cycles/{cycle_id}/practice-logs?me=true&limit=3"
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

    async def test_sorting_occurred_at_desc(self, client: AsyncClient) -> None:
        """Logs are sorted by occurred_at DESC."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create logs with different occurred_at times
        times = [
            datetime.now(UTC) - timedelta(hours=2),
            datetime.now(UTC) - timedelta(hours=1),
            datetime.now(UTC),
        ]
        for t in times:
            await client.post(
                f"/cycles/{cycle_id}/practice-logs",
                json={
                    "duration_min": 30,
                    "occurred_at": t.isoformat(),
                    "assignment_ids": [],
                },
                headers=auth_header(admin_data["access_token"]),
            )

        response = await client.get(
            f"/cycles/{cycle_id}/practice-logs?me=true",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        # Most recent first
        occurred_ats = [item["occurred_at"] for item in data["items"]]
        assert occurred_ats == sorted(occurred_ats, reverse=True)


class TestUpdatePracticeLog:
    """Tests for PATCH /practice-logs/{id}."""

    async def test_owner_can_update(self, client: AsyncClient) -> None:
        """Owner can update their practice log."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create practice log
        create_resp = await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={"duration_min": 30, "notes": "Original", "assignment_ids": []},
            headers=auth_header(admin_data["access_token"]),
        )
        log_id = create_resp.json()["practice_log"]["id"]

        # Update it
        response = await client.patch(
            f"/practice-logs/{log_id}",
            json={"duration_min": 45, "notes": "Updated"},
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["practice_log"]["duration_minutes"] == 45
        assert data["practice_log"]["notes"] == "Updated"

    async def test_other_user_cannot_update(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Another user cannot PATCH someone else's practice log."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Admin creates a log
        create_resp = await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={"duration_min": 30, "assignment_ids": []},
            headers=auth_header(admin_data["access_token"]),
        )
        log_id = create_resp.json()["practice_log"]["id"]

        # Create another member
        member_data = await register_user(client, "member@example.com")
        await add_membership(db_session, team_id, member_data["user"]["id"], "MEMBER", "Tenor")

        # Member tries to update admin's log
        response = await client.patch(
            f"/practice-logs/{log_id}",
            json={"notes": "Hacked!"},
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_update_assignments(
        self, client: AsyncClient
    ) -> None:
        """Can update assignment_ids with re-validation."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)
        cycle_id = await create_cycle(client, admin_data, team_id)

        # Create assignments
        assignment1_id = await create_assignment(client, admin_data, cycle_id, "Assignment 1")
        assignment2_id = await create_assignment(client, admin_data, cycle_id, "Assignment 2")

        # Create log with assignment1
        create_resp = await client.post(
            f"/cycles/{cycle_id}/practice-logs",
            json={"duration_min": 30, "assignment_ids": [assignment1_id]},
            headers=auth_header(admin_data["access_token"]),
        )
        log_id = create_resp.json()["practice_log"]["id"]
        assert len(create_resp.json()["practice_log"]["assignments"]) == 1

        # Update to assignment2
        response = await client.patch(
            f"/practice-logs/{log_id}",
            json={"assignment_ids": [assignment2_id]},
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["practice_log"]["assignments"]) == 1
        assert data["practice_log"]["assignments"][0]["id"] == assignment2_id

    async def test_update_nonexistent_returns_404(self, client: AsyncClient) -> None:
        """Updating nonexistent practice log returns 404."""
        admin_data = await register_user(client, "admin@example.com")
        await create_team(client, admin_data)

        fake_id = uuid.uuid4()
        response = await client.patch(
            f"/practice-logs/{fake_id}",
            json={"notes": "Test"},
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 404

