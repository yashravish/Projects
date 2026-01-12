"""Tests for rehearsal cycles.

Tests cover:
- Happy paths:
  - Create cycle successfully
  - List cycles with pagination (upcoming/past)
  - Active chooses nearest upcoming cycle (>= today) correctly
  - Active falls back to latest past cycle if no upcoming
- Unauthorized:
  - Non-member blocked from accessing team cycles endpoints
- Edge cases:
  - If no cycles exist → { "cycle": null }
  - Duplicate date → CONFLICT
  - MEMBER cannot create cycle (only ADMIN/SECTION_LEADER can)
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
            # Clean up before test
            await session.execute(text("DELETE FROM rehearsal_cycles"))
            await session.execute(text("DELETE FROM invites"))
            await session.execute(text("DELETE FROM team_memberships"))
            await session.execute(text("DELETE FROM teams"))
            await session.execute(text("DELETE FROM users"))
            await session.commit()

            yield session

            # Clean up after test
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


class TestCreateCycle:
    """Tests for POST /teams/{team_id}/cycles."""

    async def test_create_cycle_admin_success(self, client: AsyncClient) -> None:
        """ADMIN can create a rehearsal cycle."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create cycle
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
        response = await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": tomorrow, "label": "Weekly Rehearsal"},
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 201
        data = response.json()
        assert "cycle" in data
        assert data["cycle"]["name"] == "Weekly Rehearsal"
        assert data["cycle"]["team_id"] == team_id

    async def test_create_cycle_auto_label(self, client: AsyncClient) -> None:
        """Cycle gets auto-generated label if not provided."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create cycle without label
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
        response = await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": tomorrow},
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 201
        data = response.json()
        assert "Rehearsal" in data["cycle"]["name"]

    async def test_create_cycle_section_leader_success(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """SECTION_LEADER can create a rehearsal cycle."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create section leader user
        leader_data = await register_user(client, "leader@example.com")
        await db_session.execute(
            text("""
                INSERT INTO team_memberships (id, team_id, user_id, role, section)
                VALUES (:id, :team_id, :user_id, :role, :section)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "user_id": uuid.UUID(leader_data["user"]["id"]),
                "role": "SECTION_LEADER",
                "section": "Soprano",
            },
        )
        await db_session.commit()

        # Section leader creates cycle
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
        response = await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": tomorrow, "label": "Leader Created"},
            headers=auth_header(leader_data["access_token"]),
        )

        assert response.status_code == 201
        assert response.json()["cycle"]["name"] == "Leader Created"

    async def test_create_cycle_member_forbidden(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """MEMBER cannot create a rehearsal cycle."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create member user
        member_data = await register_user(client, "member@example.com")
        await db_session.execute(
            text("""
                INSERT INTO team_memberships (id, team_id, user_id, role)
                VALUES (:id, :team_id, :user_id, :role)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "user_id": uuid.UUID(member_data["user"]["id"]),
                "role": "MEMBER",
            },
        )
        await db_session.commit()

        # Member tries to create cycle
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
        response = await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": tomorrow, "label": "Member Try"},
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_create_cycle_duplicate_date_conflict(self, client: AsyncClient) -> None:
        """Creating cycle with duplicate date returns CONFLICT."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create first cycle
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
        response1 = await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": tomorrow, "label": "First"},
            headers=auth_header(admin_data["access_token"]),
        )
        assert response1.status_code == 201

        # Try to create second cycle on same date
        response2 = await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": tomorrow, "label": "Duplicate"},
            headers=auth_header(admin_data["access_token"]),
        )

        assert response2.status_code == 409
        assert response2.json()["error"]["code"] == "CONFLICT"
        assert response2.json()["error"]["field"] == "date"

    async def test_create_cycle_non_member_forbidden(self, client: AsyncClient) -> None:
        """Non-member cannot create cycle."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create another user (not a member)
        other_data = await register_user(client, "other@example.com")

        # Non-member tries to create cycle
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
        response = await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": tomorrow},
            headers=auth_header(other_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"


class TestListCycles:
    """Tests for GET /teams/{team_id}/cycles."""

    async def test_list_upcoming_cycles(self, client: AsyncClient) -> None:
        """List upcoming cycles sorted by date ASC."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create multiple upcoming cycles
        for i in range(3):
            date_str = (datetime.now(UTC) + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            await client.post(
                f"/teams/{team_id}/cycles",
                json={"date": date_str, "label": f"Cycle {i + 1}"},
                headers=auth_header(admin_data["access_token"]),
            )

        # List upcoming cycles
        response = await client.get(
            f"/teams/{team_id}/cycles?upcoming=true",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 3
        # Verify ASC order (earliest date first)
        dates = [item["date"] for item in data["items"]]
        assert dates == sorted(dates)

    async def test_list_past_cycles(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """List past cycles sorted by date DESC."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create past cycles directly in DB
        for i in range(3):
            past_date = datetime.now(UTC) - timedelta(days=i + 1)
            await db_session.execute(
                text("""
                    INSERT INTO rehearsal_cycles (id, team_id, name, date)
                    VALUES (:id, :team_id, :name, :date)
                """),
                {
                    "id": uuid.uuid4(),
                    "team_id": uuid.UUID(team_id),
                    "name": f"Past Cycle {i + 1}",
                    "date": past_date,
                },
            )
        await db_session.commit()

        # List past cycles
        response = await client.get(
            f"/teams/{team_id}/cycles?upcoming=false",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 3
        # Verify DESC order (most recent first)
        dates = [item["date"] for item in data["items"]]
        assert dates == sorted(dates, reverse=True)

    async def test_list_cycles_member_access(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Any team member can list cycles."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create a cycle
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
        await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": tomorrow},
            headers=auth_header(admin_data["access_token"]),
        )

        # Create member user
        member_data = await register_user(client, "member@example.com")
        await db_session.execute(
            text("""
                INSERT INTO team_memberships (id, team_id, user_id, role)
                VALUES (:id, :team_id, :user_id, :role)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "user_id": uuid.UUID(member_data["user"]["id"]),
                "role": "MEMBER",
            },
        )
        await db_session.commit()

        # Member lists cycles
        response = await client.get(
            f"/teams/{team_id}/cycles",
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 200
        assert len(response.json()["items"]) == 1

    async def test_list_cycles_non_member_forbidden(self, client: AsyncClient) -> None:
        """Non-member cannot list cycles."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        other_data = await register_user(client, "other@example.com")

        response = await client.get(
            f"/teams/{team_id}/cycles",
            headers=auth_header(other_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_list_cycles_pagination(self, client: AsyncClient) -> None:
        """Pagination works correctly for cycles."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create 5 upcoming cycles
        for i in range(5):
            date_str = (datetime.now(UTC) + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            await client.post(
                f"/teams/{team_id}/cycles",
                json={"date": date_str, "label": f"Cycle {i + 1}"},
                headers=auth_header(admin_data["access_token"]),
            )

        # Get first page with limit=2
        response1 = await client.get(
            f"/teams/{team_id}/cycles?limit=2",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response1.status_code == 200
        data1 = response1.json()
        assert len(data1["items"]) == 2
        assert data1["next_cursor"] is not None

        # Get second page
        cursor = data1["next_cursor"]
        response2 = await client.get(
            f"/teams/{team_id}/cycles?limit=2&cursor={cursor}",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert len(data2["items"]) == 2

        # Verify no duplicate items
        page1_ids = {item["id"] for item in data1["items"]}
        page2_ids = {item["id"] for item in data2["items"]}
        assert page1_ids.isdisjoint(page2_ids)


class TestActiveCycle:
    """Tests for GET /teams/{team_id}/cycles/active."""

    async def test_active_chooses_nearest_upcoming(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Active cycle selects nearest upcoming cycle (>= today)."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create a past cycle
        past_date = datetime.now(UTC) - timedelta(days=5)
        await db_session.execute(
            text("""
                INSERT INTO rehearsal_cycles (id, team_id, name, date)
                VALUES (:id, :team_id, :name, :date)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "name": "Past Cycle",
                "date": past_date,
            },
        )
        await db_session.commit()

        # Create two upcoming cycles (day+1 and day+7)
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
        next_week = (datetime.now(UTC) + timedelta(days=7)).strftime("%Y-%m-%d")

        await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": next_week, "label": "Next Week"},
            headers=auth_header(admin_data["access_token"]),
        )
        await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": tomorrow, "label": "Tomorrow"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Get active cycle
        response = await client.get(
            f"/teams/{team_id}/cycles/active",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["cycle"] is not None
        # Should be tomorrow (nearest upcoming)
        assert data["cycle"]["name"] == "Tomorrow"

    async def test_active_falls_back_to_latest_past(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """If no upcoming cycles, active returns latest past cycle."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create only past cycles
        for days_ago in [10, 5, 3]:
            past_date = datetime.now(UTC) - timedelta(days=days_ago)
            await db_session.execute(
                text("""
                    INSERT INTO rehearsal_cycles (id, team_id, name, date)
                    VALUES (:id, :team_id, :name, :date)
                """),
                {
                    "id": uuid.uuid4(),
                    "team_id": uuid.UUID(team_id),
                    "name": f"Past {days_ago} days",
                    "date": past_date,
                },
            )
        await db_session.commit()

        # Get active cycle
        response = await client.get(
            f"/teams/{team_id}/cycles/active",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["cycle"] is not None
        # Should be the most recent past (3 days ago)
        assert data["cycle"]["name"] == "Past 3 days"

    async def test_active_no_cycles_returns_null(self, client: AsyncClient) -> None:
        """If no cycles exist, active returns null."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Get active cycle (no cycles exist)
        response = await client.get(
            f"/teams/{team_id}/cycles/active",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["cycle"] is None

    async def test_active_member_access(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Any team member can access active cycle."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create a cycle
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
        await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": tomorrow, "label": "Test"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Create member user
        member_data = await register_user(client, "member@example.com")
        await db_session.execute(
            text("""
                INSERT INTO team_memberships (id, team_id, user_id, role)
                VALUES (:id, :team_id, :user_id, :role)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "user_id": uuid.UUID(member_data["user"]["id"]),
                "role": "MEMBER",
            },
        )
        await db_session.commit()

        # Member gets active cycle
        response = await client.get(
            f"/teams/{team_id}/cycles/active",
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 200
        assert response.json()["cycle"] is not None

    async def test_active_non_member_forbidden(self, client: AsyncClient) -> None:
        """Non-member cannot access active cycle."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        other_data = await register_user(client, "other@example.com")

        response = await client.get(
            f"/teams/{team_id}/cycles/active",
            headers=auth_header(other_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_active_today_included_in_upcoming(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """A cycle scheduled for today is considered upcoming (>= today)."""
        admin_data = await register_user(client, "admin@example.com")
        team_id = await create_team(client, admin_data)

        # Create a cycle for today at midnight UTC
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        await db_session.execute(
            text("""
                INSERT INTO rehearsal_cycles (id, team_id, name, date)
                VALUES (:id, :team_id, :name, :date)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "name": "Today Cycle",
                "date": today,
            },
        )
        await db_session.commit()

        # Get active cycle
        response = await client.get(
            f"/teams/{team_id}/cycles/active",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["cycle"] is not None
        assert data["cycle"]["name"] == "Today Cycle"

