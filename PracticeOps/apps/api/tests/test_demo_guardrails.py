"""Tests for demo user write protection.

Verifies that demo users (users with @practiceops.app email) cannot perform
write operations while still being able to authenticate and read data.
"""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.core.security import hash_password
from app.database import get_db
from app.main import app
from app.models import RehearsalCycle, Role, Team, TeamMembership, User

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
            await session.execute(text("DELETE FROM practice_logs"))
            await session.execute(text("DELETE FROM tickets"))
            await session.execute(text("DELETE FROM assignments"))
            await session.execute(text("DELETE FROM rehearsal_cycles"))
            await session.execute(text("DELETE FROM team_memberships"))
            await session.execute(text("DELETE FROM teams"))
            await session.execute(text("DELETE FROM users"))
            await session.commit()

            yield session

            # Clean up after test
            await session.execute(text("DELETE FROM practice_logs"))
            await session.execute(text("DELETE FROM tickets"))
            await session.execute(text("DELETE FROM assignments"))
            await session.execute(text("DELETE FROM rehearsal_cycles"))
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


@pytest.fixture
async def demo_user_setup(db_session: AsyncSession) -> dict:
    """Create demo user and team for testing."""
    # Create team
    team = Team(name="Demo Team")
    db_session.add(team)
    await db_session.flush()

    # Create demo user (email ends with @practiceops.app)
    demo_user = User(
        email="demo@practiceops.app",
        display_name="Demo User",
        password_hash=hash_password("demo1234"),
    )
    db_session.add(demo_user)
    await db_session.flush()

    # Create membership
    membership = TeamMembership(
        team_id=team.id,
        user_id=demo_user.id,
        role=Role.MEMBER,
        section="Soprano",
    )
    db_session.add(membership)

    # Create a cycle
    cycle = RehearsalCycle(
        team_id=team.id,
        name="Current Cycle",
        date=datetime.now(UTC) + timedelta(days=7),
    )
    db_session.add(cycle)
    await db_session.commit()

    return {
        "team_id": str(team.id),
        "user_id": str(demo_user.id),
        "cycle_id": str(cycle.id),
    }


@pytest.fixture
async def regular_user_setup(db_session: AsyncSession) -> dict:
    """Create regular (non-demo) user and team for testing."""
    # Create team
    team = Team(name="Regular Team")
    db_session.add(team)
    await db_session.flush()

    # Create regular user (email does NOT end with @practiceops.app)
    regular_user = User(
        email="test@example.com",
        display_name="Test User",
        password_hash=hash_password("test1234"),
    )
    db_session.add(regular_user)
    await db_session.flush()

    # Create membership
    membership = TeamMembership(
        team_id=team.id,
        user_id=regular_user.id,
        role=Role.MEMBER,
        section="Alto",
    )
    db_session.add(membership)

    # Create a cycle
    cycle = RehearsalCycle(
        team_id=team.id,
        name="Current Cycle",
        date=datetime.now(UTC) + timedelta(days=7),
    )
    db_session.add(cycle)
    await db_session.commit()

    return {
        "team_id": str(team.id),
        "user_id": str(regular_user.id),
        "cycle_id": str(cycle.id),
    }


class TestDemoLogin:
    """Tests for demo user authentication."""

    async def test_demo_user_can_login(
        self, client: AsyncClient, demo_user_setup: dict
    ) -> None:
        """Demo users can authenticate successfully."""
        response = await client.post(
            "/auth/login",
            json={
                "email": "demo@practiceops.app",
                "password": "demo1234",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["user"]["email"] == "demo@practiceops.app"


class TestDemoWriteProtection:
    """Tests for demo user write operation blocking."""

    async def test_demo_user_blocked_from_creating_practice_log(
        self, client: AsyncClient, demo_user_setup: dict
    ) -> None:
        """Demo users cannot create practice logs."""
        # Login first
        login_response = await client.post(
            "/auth/login",
            json={"email": "demo@practiceops.app", "password": "demo1234"},
        )
        token = login_response.json()["access_token"]

        # Attempt to create practice log
        response = await client.post(
            f"/cycles/{demo_user_setup['cycle_id']}/practice-logs",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "duration_min": 30,
                "rating_1_5": 4,
                "blocked_flag": False,
            },
        )
        assert response.status_code == 403
        data = response.json()
        assert data["error"]["code"] == "FORBIDDEN"
        assert data["error"]["message"] == "Demo accounts are read-only."

    async def test_demo_user_blocked_from_creating_ticket(
        self, client: AsyncClient, demo_user_setup: dict
    ) -> None:
        """Demo users cannot create tickets."""
        # Login first
        login_response = await client.post(
            "/auth/login",
            json={"email": "demo@practiceops.app", "password": "demo1234"},
        )
        token = login_response.json()["access_token"]

        # Attempt to create ticket
        response = await client.post(
            f"/cycles/{demo_user_setup['cycle_id']}/tickets",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "category": "PITCH",
                "priority": "MEDIUM",
                "visibility": "PRIVATE",
                "title": "Test ticket",
                "description": "Test description",
            },
        )
        assert response.status_code == 403
        data = response.json()
        assert data["error"]["code"] == "FORBIDDEN"
        assert data["error"]["message"] == "Demo accounts are read-only."

    async def test_demo_user_can_read_data(
        self, client: AsyncClient, demo_user_setup: dict
    ) -> None:
        """Demo users can read data successfully."""
        # Login first
        login_response = await client.post(
            "/auth/login",
            json={"email": "demo@practiceops.app", "password": "demo1234"},
        )
        token = login_response.json()["access_token"]

        # Can fetch practice logs
        response = await client.get(
            f"/cycles/{demo_user_setup['cycle_id']}/practice-logs?me=true",
            headers={"Authorization": f"Bearer {token}"},
        )
        # Should return 200 with empty list (no logs yet)
        assert response.status_code == 200
        assert "items" in response.json()


class TestRegularUserUnaffected:
    """Tests to ensure non-demo users are not affected."""

    async def test_non_demo_user_can_create_practice_log(
        self, client: AsyncClient, regular_user_setup: dict
    ) -> None:
        """Non-demo users can create practice logs normally."""
        # Login first
        login_response = await client.post(
            "/auth/login",
            json={"email": "test@example.com", "password": "test1234"},
        )
        token = login_response.json()["access_token"]

        # Create practice log
        response = await client.post(
            f"/cycles/{regular_user_setup['cycle_id']}/practice-logs",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "duration_min": 45,
                "rating_1_5": 5,
                "blocked_flag": False,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["practice_log"]["duration_minutes"] == 45
