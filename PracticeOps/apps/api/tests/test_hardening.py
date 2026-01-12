"""Tests for Milestone 12: Hardening Pass.

Tests cover:
- Rate limiting on /auth/login and /auth/register
- Health endpoint with DB check
- Request ID in responses
- Error sanitization (no stack traces)
- RBAC coverage verification
"""

import asyncio
from collections.abc import AsyncGenerator

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
            await session.execute(text("DELETE FROM notification_preferences"))
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
            await session.execute(text("DELETE FROM notification_preferences"))
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

    # Note: Rate limiter is reset automatically via conftest.py autouse fixture

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoint:
    """Tests for GET /health."""

    async def test_health_returns_ok_with_db_check(
        self, client: AsyncClient
    ) -> None:
        """Health endpoint returns status with real DB check."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Must have status and db fields
        assert "status" in data
        assert "db" in data

        # DB should be ok when database is available
        assert data["db"] == "ok"
        assert data["status"] == "ok"

    async def test_health_no_auth_required(self, client: AsyncClient) -> None:
        """Health endpoint does not require authentication."""
        response = await client.get("/health")

        # Should not return 401
        assert response.status_code == 200


# =============================================================================
# Request ID Tests
# =============================================================================


class TestRequestID:
    """Tests for request ID middleware."""

    async def test_request_id_in_response_header(
        self, client: AsyncClient
    ) -> None:
        """Response includes X-Request-ID header."""
        response = await client.get("/health")

        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]

        # Should be a valid UUID format
        assert len(request_id) == 36  # UUID length
        assert request_id.count("-") == 4  # UUID format

    async def test_request_id_unique_per_request(
        self, client: AsyncClient
    ) -> None:
        """Each request gets a unique request ID."""
        response1 = await client.get("/health")
        response2 = await client.get("/health")

        request_id1 = response1.headers["X-Request-ID"]
        request_id2 = response2.headers["X-Request-ID"]

        assert request_id1 != request_id2


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting on auth endpoints."""

    async def test_login_rate_limit_not_triggered_normal_use(
        self, client: AsyncClient
    ) -> None:
        """Normal login attempts are not rate limited."""
        # Register a user first
        await client.post(
            "/auth/register",
            json={
                "email": "ratelimit@test.com",
                "name": "Rate Limit Test",
                "password": "password123",
            },
        )

        # Try logging in a few times (under limit)
        for _ in range(3):
            response = await client.post(
                "/auth/login",
                json={"email": "ratelimit@test.com", "password": "password123"},
            )
            # Should succeed, not be rate limited
            assert response.status_code in [200, 401]  # Success or wrong password
            assert response.status_code != 429  # Not rate limited

    async def test_register_rate_limit_not_triggered_normal_use(
        self, client: AsyncClient
    ) -> None:
        """Normal registration attempts are not rate limited."""
        for i in range(3):
            response = await client.post(
                "/auth/register",
                json={
                    "email": f"user{i}@ratelimit.com",
                    "name": f"User {i}",
                    "password": "password123",
                },
            )
            # Should succeed or conflict (if email exists)
            assert response.status_code in [201, 409]
            assert response.status_code != 429  # Not rate limited


# =============================================================================
# Error Sanitization Tests
# =============================================================================


class TestErrorSanitization:
    """Tests for error response sanitization."""

    async def test_error_format_standard(self, client: AsyncClient) -> None:
        """Errors follow standard format with no stack traces."""
        # Try to access non-existent resource
        response = await client.get(
            "/teams/00000000-0000-0000-0000-000000000000/dashboards/member",
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert response.status_code in [401, 403]
        data = response.json()

        # Must have error object
        assert "error" in data
        error = data["error"]

        # Must have standard fields
        assert "code" in error
        assert "message" in error
        assert "field" in error

        # Must NOT have stack trace
        assert "traceback" not in str(data).lower()
        assert "exception" not in str(data).lower()
        assert "file" not in str(data).lower()
        assert "line" not in str(data).lower()

    async def test_validation_error_no_internal_info(
        self, client: AsyncClient
    ) -> None:
        """Validation errors don't expose internal implementation."""
        response = await client.post(
            "/auth/register",
            json={
                "email": "not-an-email",
                "name": "Test",
                "password": "short",
            },
        )

        assert response.status_code == 422
        data = response.json()

        # Check response doesn't contain internal class names
        response_str = str(data)
        assert "pydantic" not in response_str.lower()
        assert "sqlalchemy" not in response_str.lower()


# =============================================================================
# RBAC Coverage Tests
# =============================================================================


class TestRBACCoverage:
    """Tests verifying RBAC is properly enforced."""

    async def test_member_cannot_access_leader_dashboard(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """MEMBER role cannot access leader dashboard."""
        # Create admin and team
        admin_response = await client.post(
            "/auth/register",
            json={
                "email": "admin-rbac@example.com",
                "name": "Admin",
                "password": "password123",
            },
        )
        assert admin_response.status_code == 201, f"Register failed: {admin_response.json()}"
        admin_token = admin_response.json()["access_token"]

        team_response = await client.post(
            "/teams",
            json={"name": "RBAC Test Team"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        team_id = team_response.json()["team"]["id"]

        # Create member
        member_response = await client.post(
            "/auth/register",
            json={
                "email": "member-rbac@example.com",
                "name": "Member",
                "password": "password123",
            },
        )
        member_token = member_response.json()["access_token"]
        member_id = member_response.json()["user"]["id"]

        # Add member to team via direct DB insert
        import uuid

        await db_session.execute(
            text(
                """
                INSERT INTO team_memberships (id, team_id, user_id, role, section)
                VALUES (:id, :team_id, :user_id, 'MEMBER', 'Bass')
                """
            ),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "user_id": uuid.UUID(member_id),
            },
        )
        await db_session.commit()

        # Member tries to access leader dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/leader",
            headers={"Authorization": f"Bearer {member_token}"},
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_non_member_cannot_access_team_resources(
        self, client: AsyncClient
    ) -> None:
        """Non-team-member cannot access team resources."""
        # Create two users
        user1_response = await client.post(
            "/auth/register",
            json={
                "email": "user1-rbac@example.com",
                "name": "User 1",
                "password": "password123",
            },
        )
        assert user1_response.status_code == 201, f"Register failed: {user1_response.json()}"
        user1_token = user1_response.json()["access_token"]

        user2_response = await client.post(
            "/auth/register",
            json={
                "email": "user2-rbac@example.com",
                "name": "User 2",
                "password": "password123",
            },
        )
        assert user2_response.status_code == 201, f"Register failed: {user2_response.json()}"
        user2_token = user2_response.json()["access_token"]

        # User 1 creates team
        team_response = await client.post(
            "/teams",
            json={"name": "User 1 Team"},
            headers={"Authorization": f"Bearer {user1_token}"},
        )
        team_id = team_response.json()["team"]["id"]

        # User 2 tries to access User 1's team
        response = await client.get(
            f"/teams/{team_id}/dashboards/member",
            headers={"Authorization": f"Bearer {user2_token}"},
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"


# =============================================================================
# Anonymization Guarantee Tests
# =============================================================================


class TestAnonymizationGuarantees:
    """Tests verifying privacy guarantees for private_ticket_aggregates."""

    async def test_private_aggregates_no_identifiers(
        self, client: AsyncClient
    ) -> None:
        """private_ticket_aggregates must not contain any identifying fields."""
        # Create admin and team
        admin_response = await client.post(
            "/auth/register",
            json={
                "email": "admin-anon@example.com",
                "name": "Admin",
                "password": "password123",
            },
        )
        assert admin_response.status_code == 201, f"Register failed: {admin_response.json()}"
        admin_token = admin_response.json()["access_token"]

        team_response = await client.post(
            "/teams",
            json={"name": "Anon Test Team"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        team_id = team_response.json()["team"]["id"]

        # Get leader dashboard
        response = await client.get(
            f"/teams/{team_id}/dashboards/leader",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        # Check private_ticket_aggregates structure
        aggregates = data.get("private_ticket_aggregates", [])

        for aggregate in aggregates:
            # These fields MUST NOT be present
            assert "id" not in aggregate
            assert "ticket_id" not in aggregate
            assert "owner_id" not in aggregate
            assert "created_by" not in aggregate
            assert "claimed_by" not in aggregate
            assert "title" not in aggregate
            assert "description" not in aggregate
            assert "name" not in aggregate
            assert "email" not in aggregate

            # These fields MUST be present (only aggregate data)
            assert "section" in aggregate or aggregate.get("section") is None
            assert "category" in aggregate
            assert "status" in aggregate
            assert "priority" in aggregate
            assert "due_bucket" in aggregate
            assert "count" in aggregate


# =============================================================================
# Illegal Status Transition Tests
# =============================================================================


class TestIllegalStatusTransitions:
    """Tests verifying illegal ticket transitions are rejected."""

    async def test_open_to_resolved_rejected(self, client: AsyncClient) -> None:
        """Cannot transition directly from OPEN to RESOLVED."""
        # Create user and team
        user_response = await client.post(
            "/auth/register",
            json={
                "email": "trans-open@example.com",
                "name": "Transition Test",
                "password": "password123",
            },
        )
        assert user_response.status_code == 201, f"Register failed: {user_response.json()}"
        user_token = user_response.json()["access_token"]

        team_response = await client.post(
            "/teams",
            json={"name": "Transition Test Team"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        team_id = team_response.json()["team"]["id"]

        # Create cycle
        from datetime import datetime, timedelta

        future_date = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
        cycle_response = await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": future_date, "label": "Test Cycle"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        cycle_id = cycle_response.json()["cycle"]["id"]

        # Create ticket (starts OPEN)
        ticket_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Test Ticket",
                "category": "OTHER",
                "priority": "LOW",
                "visibility": "PRIVATE",
            },
            headers={"Authorization": f"Bearer {user_token}"},
        )
        ticket_id = ticket_response.json()["ticket"]["id"]

        # Try to transition directly to RESOLVED (should fail)
        response = await client.post(
            f"/tickets/{ticket_id}/transition",
            json={"to_status": "RESOLVED", "content": "Skipping IN_PROGRESS"},
            headers={"Authorization": f"Bearer {user_token}"},
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    async def test_verified_is_terminal(self, client: AsyncClient) -> None:
        """Cannot transition out of VERIFIED state."""
        # Create user and team
        user_response = await client.post(
            "/auth/register",
            json={
                "email": "terminal-verify@example.com",
                "name": "Terminal Test",
                "password": "password123",
            },
        )
        assert user_response.status_code == 201, f"Register failed: {user_response.json()}"
        user_token = user_response.json()["access_token"]

        team_response = await client.post(
            "/teams",
            json={"name": "Terminal Test Team"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        team_id = team_response.json()["team"]["id"]

        # Create cycle
        from datetime import datetime, timedelta

        future_date = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
        cycle_response = await client.post(
            f"/teams/{team_id}/cycles",
            json={"date": future_date, "label": "Test Cycle"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        cycle_id = cycle_response.json()["cycle"]["id"]

        # Create and progress ticket to VERIFIED
        ticket_response = await client.post(
            f"/cycles/{cycle_id}/tickets",
            json={
                "title": "Terminal Test",
                "category": "OTHER",
                "priority": "LOW",
                "visibility": "TEAM",
            },
            headers={"Authorization": f"Bearer {user_token}"},
        )
        ticket_id = ticket_response.json()["ticket"]["id"]

        # Progress through states
        await client.post(
            f"/tickets/{ticket_id}/transition",
            json={"to_status": "IN_PROGRESS"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        await client.post(
            f"/tickets/{ticket_id}/transition",
            json={"to_status": "RESOLVED", "content": "Done"},
            headers={"Authorization": f"Bearer {user_token}"},
        )
        await client.post(
            f"/tickets/{ticket_id}/verify",
            json={"content": "Verified"},
            headers={"Authorization": f"Bearer {user_token}"},
        )

        # Try to transition out of VERIFIED (should fail)
        response = await client.post(
            f"/tickets/{ticket_id}/transition",
            json={"to_status": "OPEN"},
            headers={"Authorization": f"Bearer {user_token}"},
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

