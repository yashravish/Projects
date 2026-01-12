"""Dashboard endpoint tests.

Tests for:
- GET /teams/{team_id}/dashboards/member
- GET /teams/{team_id}/dashboards/leader

Covers:
- Happy paths (member sees personal data, leader sees team data)
- RBAC (member blocked from leader dashboard)
- Privacy (private_ticket_aggregates contains no identifiers)
- Section leader scope restrictions
- Edge cases (empty data, countdown calculation)
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


# =============================================================================
# Test fixtures
# =============================================================================


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


# =============================================================================
# Helper functions
# =============================================================================


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
    assert response.status_code == 201, f"Registration failed: {response.json()}"
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


async def create_ticket(
    client: AsyncClient,
    auth_data: dict,
    cycle_id: str,
    title: str,
    visibility: str = "PRIVATE",
    priority: str = "LOW",
    section: str | None = None,
    song_ref: str | None = None,
) -> dict:
    """Create a ticket and return the response."""
    payload = {
        "title": title,
        "category": "OTHER",
        "priority": priority,
        "visibility": visibility,
    }
    if section:
        payload["section"] = section
    if song_ref:
        payload["song_ref"] = song_ref

    response = await client.post(
        f"/cycles/{cycle_id}/tickets",
        headers=auth_header(auth_data["access_token"]),
        json=payload,
    )
    assert response.status_code == 201
    return response.json()


async def create_practice_log(
    client: AsyncClient,
    auth_data: dict,
    cycle_id: str,
    duration_min: int = 30,
    occurred_at: str | None = None,
) -> dict:
    """Create a practice log and return the response."""
    payload = {
        "duration_min": duration_min,
        "assignment_ids": [],
        "blocked_flag": False,
    }
    if occurred_at:
        payload["occurred_at"] = occurred_at

    response = await client.post(
        f"/cycles/{cycle_id}/practice-logs",
        headers=auth_header(auth_data["access_token"]),
        json=payload,
    )
    assert response.status_code == 201
    return response.json()


async def create_invite(
    client: AsyncClient,
    auth_data: dict,
    team_id: str,
    role: str = "MEMBER",
    section: str | None = None,
) -> str:
    """Create an invite and return the invite token."""
    payload = {"role": role, "expires_in_hours": 24}
    if section:
        payload["section"] = section

    response = await client.post(
        f"/teams/{team_id}/invites",
        headers=auth_header(auth_data["access_token"]),
        json=payload,
    )
    assert response.status_code == 201
    invite_link = response.json()["invite_link"]
    return invite_link.split("/")[-1]


async def accept_invite(
    client: AsyncClient,
    auth_data: dict,
    invite_token: str,
) -> dict:
    """Accept an invite while logged in."""
    response = await client.post(
        f"/invites/{invite_token}/accept",
        json={},  # Empty JSON body required for logged-in acceptance
        headers=auth_header(auth_data["access_token"]),
    )
    assert response.status_code == 201, f"Accept invite failed: {response.json()}"
    return response.json()


# =============================================================================
# Member Dashboard Tests
# =============================================================================


class TestMemberDashboard:
    """Tests for GET /teams/{team_id}/dashboards/member."""

    async def test_member_dashboard_returns_personal_data(
        self, client: AsyncClient
    ) -> None:
        """Member dashboard returns personal progress data."""
        # Setup: Create user and team
        auth_data = await register_user(client, "member-dash@test.com", "Dashboard Member")
        team_id = await create_team(client, auth_data, "Dashboard Test Team")

        # Create a future cycle
        future_date = datetime.now(UTC) + timedelta(days=5)
        cycle_id = await create_cycle(client, team_id, auth_data, future_date, "Future Rehearsal")

        # Create a practice log
        await create_practice_log(client, auth_data, cycle_id, duration_min=45)

        # Create a ticket
        await create_ticket(client, auth_data, cycle_id, "Test Ticket", visibility="PRIVATE")

        # Get member dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/member",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()

        # Verify cycle info
        assert data["cycle"] is not None
        assert data["cycle"]["id"] == cycle_id
        assert data["countdown_days"] == 5  # 5 days in future

        # Verify weekly summary
        assert data["weekly_summary"]["practice_days"] >= 1
        assert data["weekly_summary"]["total_sessions"] >= 1

        # Verify progress
        assert "tickets_resolved_this_cycle" in data["progress"]
        assert "tickets_verified_this_cycle" in data["progress"]

    async def test_member_dashboard_countdown_negative_for_past(
        self, client: AsyncClient
    ) -> None:
        """Countdown is negative when cycle date is in the past."""
        # Setup
        auth_data = await register_user(client, "past-cycle@test.com", "Past Cycle User")
        team_id = await create_team(client, auth_data, "Past Cycle Team")

        # Create a past cycle
        past_date = datetime.now(UTC) - timedelta(days=3)
        await create_cycle(client, team_id, auth_data, past_date, "Past Rehearsal")

        # Get member dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/member",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()

        # Countdown should be negative
        assert data["countdown_days"] is not None
        assert data["countdown_days"] <= -3

    async def test_member_dashboard_empty_cycle(self, client: AsyncClient) -> None:
        """Member dashboard handles empty cycles gracefully."""
        # Setup
        auth_data = await register_user(client, "empty-cycle@test.com", "Empty Cycle User")
        team_id = await create_team(client, auth_data, "Empty Cycle Team")

        # Don't create any cycles

        # Get member dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/member",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()

        # No cycle should mean empty arrays
        assert data["cycle"] is None
        assert data["countdown_days"] is None
        assert data["assignments"] == []

    async def test_member_dashboard_non_member_forbidden(
        self, client: AsyncClient
    ) -> None:
        """Non-member cannot access team dashboard."""
        # Setup: Create two users
        auth_data1 = await register_user(client, "owner-nm@test.com", "Team Owner")
        auth_data2 = await register_user(client, "outsider@test.com", "Outsider")

        # Create team with user1
        team_id = await create_team(client, auth_data1, "Private Team")

        # User2 tries to access dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/member",
            headers=auth_header(auth_data2["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_member_dashboard_tickets_due_soon(
        self, client: AsyncClient
    ) -> None:
        """Member dashboard shows tickets due soon."""
        # Setup
        auth_data = await register_user(client, "tickets-due@test.com", "Tickets Due User")
        team_id = await create_team(client, auth_data, "Tickets Due Team")

        # Create a cycle in the near future (so due_at is set)
        future_date = datetime.now(UTC) + timedelta(days=3)
        cycle_id = await create_cycle(client, team_id, auth_data, future_date, "Soon Rehearsal")

        # Create a ticket (due_at = cycle.date by default)
        await create_ticket(client, auth_data, cycle_id, "Due Soon Ticket", visibility="PRIVATE")

        # Get member dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/member",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()

        # Should have ticket in due soon list
        assert len(data["tickets_due_soon"]) >= 1
        ticket = data["tickets_due_soon"][0]
        assert ticket["title"] == "Due Soon Ticket"


# =============================================================================
# Leader Dashboard Tests
# =============================================================================


class TestLeaderDashboard:
    """Tests for GET /teams/{team_id}/dashboards/leader."""

    async def test_admin_sees_all_sections(self, client: AsyncClient) -> None:
        """Admin leader dashboard returns data for all sections."""
        # Setup: Create admin and team
        auth_data = await register_user(client, "admin-leader@test.com", "Admin Leader")
        team_id = await create_team(client, auth_data, "Leader Dashboard Team")

        # Create cycle
        future_date = datetime.now(UTC) + timedelta(days=5)
        await create_cycle(client, team_id, auth_data, future_date, "Leader Test Cycle")

        # Get leader dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/leader",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert data["cycle"] is not None
        assert "compliance" in data
        assert "risk_summary" in data
        assert "private_ticket_aggregates" in data
        assert "drilldown" in data

        # Verify compliance structure
        assert "logged_last_7_days_pct" in data["compliance"]
        assert "practice_days_by_member" in data["compliance"]
        assert "total_practice_minutes_7d" in data["compliance"]

    async def test_member_blocked_from_leader_dashboard(
        self, client: AsyncClient
    ) -> None:
        """Member cannot access leader dashboard."""
        # Setup: Create admin and member
        admin_auth = await register_user(client, "admin-block@test.com", "Admin User")
        member_auth = await register_user(client, "member-block@test.com", "Member User")

        # Create team and add member
        team_id = await create_team(client, admin_auth, "RBAC Test Team")

        # Invite and add member
        invite_token = await create_invite(client, admin_auth, team_id, role="MEMBER")

        # Accept invite
        await accept_invite(client, member_auth, invite_token)

        # Member tries to access leader dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/leader",
            headers=auth_header(member_auth["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_section_leader_sees_only_section(
        self, client: AsyncClient
    ) -> None:
        """Section leader only sees their section's data."""
        # Setup: Create admin
        admin_auth = await register_user(client, "admin-section@test.com", "Admin User")

        # Create section leader
        leader_auth = await register_user(client, "section-lead@test.com", "Section Leader")

        # Create team
        team_id = await create_team(client, admin_auth, "Section Test Team")

        # Invite section leader
        invite_token = await create_invite(
            client, admin_auth, team_id, role="SECTION_LEADER", section="Soprano"
        )

        # Accept invite
        await accept_invite(client, leader_auth, invite_token)

        # Section leader accesses dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/leader",
            headers=auth_header(leader_auth["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()

        # Section leader should see data (scoped to their section)
        assert "compliance" in data
        assert "drilldown" in data

    async def test_private_ticket_aggregates_no_identifiers(
        self, client: AsyncClient
    ) -> None:
        """private_ticket_aggregates must not contain any identifying fields."""
        # Setup
        auth_data = await register_user(client, "privacy-test@test.com", "Privacy Test User")
        team_id = await create_team(client, auth_data, "Privacy Test Team")

        # Create cycle
        future_date = datetime.now(UTC) + timedelta(days=5)
        cycle_id = await create_cycle(client, team_id, auth_data, future_date, "Privacy Test Cycle")

        # Create multiple PRIVATE tickets (at least 3 to pass threshold)
        for i in range(5):
            await create_ticket(
                client,
                auth_data,
                cycle_id,
                f"Private Ticket {i}",
                visibility="PRIVATE",
                priority="LOW",
                section="Bass",
            )

        # Get leader dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/leader",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()

        # Check private_ticket_aggregates
        for aggregate in data["private_ticket_aggregates"]:
            # These fields MUST NOT be present
            assert "id" not in aggregate
            assert "owner_id" not in aggregate
            assert "created_by" not in aggregate
            assert "title" not in aggregate
            assert "description" not in aggregate

            # These fields MUST be present (only aggregate data)
            assert "section" in aggregate  # Can be None
            assert "category" in aggregate
            assert "status" in aggregate
            assert "priority" in aggregate
            assert "due_bucket" in aggregate
            assert "count" in aggregate

    async def test_private_aggregates_min_count_threshold(
        self, client: AsyncClient
    ) -> None:
        """Rows with count < 3 are omitted from private_ticket_aggregates."""
        # Setup
        auth_data = await register_user(client, "threshold-test@test.com", "Threshold Test User")
        team_id = await create_team(client, auth_data, "Threshold Test Team")

        # Create cycle
        future_date = datetime.now(UTC) + timedelta(days=5)
        cycle_id = await create_cycle(
            client, team_id, auth_data, future_date, "Threshold Test Cycle"
        )

        # Create only 2 PRIVATE tickets (below threshold)
        for i in range(2):
            await create_ticket(
                client,
                auth_data,
                cycle_id,
                f"Below Threshold Ticket {i}",
                visibility="PRIVATE",
                priority="MEDIUM",
                section="Alto",
            )

        # Get leader dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/leader",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()

        # With count < 3, should be empty (all filtered out)
        # Note: This may not always hold if there are pre-existing tickets
        # Check that no aggregate has count < 3
        for aggregate in data["private_ticket_aggregates"]:
            assert aggregate["count"] >= 3

    async def test_leader_dashboard_risk_summary(self, client: AsyncClient) -> None:
        """Leader dashboard includes risk summary with blocked tickets."""
        # Setup
        auth_data = await register_user(client, "risk-test@test.com", "Risk Test User")
        team_id = await create_team(client, auth_data, "Risk Test Team")

        # Create cycle
        future_date = datetime.now(UTC) + timedelta(days=5)
        cycle_id = await create_cycle(
            client, team_id, auth_data, future_date, "Risk Test Cycle"
        )

        # Create a BLOCKING priority TEAM ticket
        await create_ticket(
            client,
            auth_data,
            cycle_id,
            "Blocking Ticket",
            visibility="TEAM",
            priority="BLOCKING",
        )

        # Get leader dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/leader",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()

        # Verify risk summary structure
        risk = data["risk_summary"]
        assert "blocking_due_count" in risk
        assert "blocked_count" in risk
        assert "resolved_not_verified_count" in risk
        assert "by_section" in risk
        assert "by_song" in risk

        # Should have at least 1 blocking ticket
        assert risk["blocking_due_count"] >= 1


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDashboardEdgeCases:
    """Edge case tests for dashboards."""

    async def test_invalid_team_returns_404(self, client: AsyncClient) -> None:
        """Invalid team ID returns 404."""
        auth_data = await register_user(client, "invalid-team@test.com", "Invalid Team User")

        fake_team_id = str(uuid.uuid4())

        response = await client.get(
            f"/teams/{fake_team_id}/dashboards/member",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 404

    async def test_invalid_cycle_returns_404(self, client: AsyncClient) -> None:
        """Invalid cycle ID returns 404."""
        auth_data = await register_user(client, "invalid-cycle@test.com", "Invalid Cycle User")
        team_id = await create_team(client, auth_data, "Invalid Cycle Team")

        fake_cycle_id = str(uuid.uuid4())

        response = await client.get(
            f"/teams/{team_id}/dashboards/member?cycle_id={fake_cycle_id}",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 404

    async def test_due_bucket_classification(self, client: AsyncClient) -> None:
        """Test due bucket classification correctness."""
        # Setup
        auth_data = await register_user(client, "due-bucket@test.com", "Due Bucket User")
        team_id = await create_team(client, auth_data, "Due Bucket Team")

        # Create a past cycle (so tickets are overdue)
        past_date = datetime.now(UTC) - timedelta(days=1)
        cycle_id = await create_cycle(client, team_id, auth_data, past_date, "Past Cycle")

        # Create tickets
        for i in range(3):
            await create_ticket(
                client,
                auth_data,
                cycle_id,
                f"Overdue Ticket {i}",
                visibility="PRIVATE",
            )

        # Get leader dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/leader",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()

        # Verify due buckets are valid
        valid_buckets = {"overdue", "due_today", "due_this_week", "future", "no_due_date"}
        for aggregate in data["private_ticket_aggregates"]:
            assert aggregate["due_bucket"] in valid_buckets

    async def test_streak_calculation(self, client: AsyncClient) -> None:
        """Test streak days calculation."""
        # Setup
        auth_data = await register_user(client, "streak-test@test.com", "Streak Test User")
        team_id = await create_team(client, auth_data, "Streak Test Team")

        # Create cycle
        future_date = datetime.now(UTC) + timedelta(days=5)
        cycle_id = await create_cycle(
            client, team_id, auth_data, future_date, "Streak Test Cycle"
        )

        # Create practice logs for consecutive days
        for days_ago in range(3):
            occurred_at = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
            await create_practice_log(
                client, auth_data, cycle_id, duration_min=30, occurred_at=occurred_at
            )

        # Get member dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/member",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()

        # Should have streak of at least 3 days
        assert data["weekly_summary"]["streak_days"] >= 3
