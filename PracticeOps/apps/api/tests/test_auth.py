"""Tests for authentication endpoints.

Tests:
- Happy path: register → login → /me
- Unauthorized: /me without token
- Edge case: wrong password

These tests require a running PostgreSQL database.
"""

import uuid
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.database import get_db
from app.main import app

# Check database availability once at module load
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
            await session.execute(text("DELETE FROM team_memberships"))
            await session.execute(text("DELETE FROM teams"))
            await session.execute(text("DELETE FROM users"))
            await session.commit()

            yield session

            # Clean up after test
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


class TestRegister:
    """Tests for POST /auth/register."""

    async def test_register_success(self, client: AsyncClient) -> None:
        """Successful registration returns tokens and user."""
        response = await client.post(
            "/auth/register",
            json={
                "email": "test@example.com",
                "name": "Test User",
                "password": "securepassword123",
            },
        )

        assert response.status_code == 201
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data
        assert data["user"]["email"] == "test@example.com"
        assert data["user"]["name"] == "Test User"
        assert "id" in data["user"]

    async def test_register_duplicate_email(self, client: AsyncClient) -> None:
        """Registering with existing email returns CONFLICT error."""
        # First registration
        await client.post(
            "/auth/register",
            json={
                "email": "duplicate@example.com",
                "name": "First User",
                "password": "securepassword123",
            },
        )

        # Second registration with same email
        response = await client.post(
            "/auth/register",
            json={
                "email": "duplicate@example.com",
                "name": "Second User",
                "password": "differentpassword",
            },
        )

        assert response.status_code == 409
        data = response.json()
        assert data["error"]["code"] == "CONFLICT"
        assert data["error"]["field"] == "email"

    async def test_register_invalid_email(self, client: AsyncClient) -> None:
        """Invalid email format returns validation error."""
        response = await client.post(
            "/auth/register",
            json={
                "email": "not-an-email",
                "name": "Test User",
                "password": "securepassword123",
            },
        )

        assert response.status_code == 422

    async def test_register_short_password(self, client: AsyncClient) -> None:
        """Password too short returns validation error."""
        response = await client.post(
            "/auth/register",
            json={
                "email": "test@example.com",
                "name": "Test User",
                "password": "short",
            },
        )

        assert response.status_code == 422


class TestLogin:
    """Tests for POST /auth/login."""

    async def test_login_success(self, client: AsyncClient) -> None:
        """Successful login returns tokens and user."""
        # Register first
        await client.post(
            "/auth/register",
            json={
                "email": "login@example.com",
                "name": "Login User",
                "password": "securepassword123",
            },
        )

        # Login
        response = await client.post(
            "/auth/login",
            json={
                "email": "login@example.com",
                "password": "securepassword123",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert "user" in data
        assert data["user"]["email"] == "login@example.com"

    async def test_login_wrong_password(self, client: AsyncClient) -> None:
        """Wrong password returns UNAUTHORIZED error."""
        # Register first
        await client.post(
            "/auth/register",
            json={
                "email": "wrongpass@example.com",
                "name": "Test User",
                "password": "correctpassword123",
            },
        )

        # Login with wrong password
        response = await client.post(
            "/auth/login",
            json={
                "email": "wrongpass@example.com",
                "password": "wrongpassword123",
            },
        )

        assert response.status_code == 401
        data = response.json()
        assert data["error"]["code"] == "UNAUTHORIZED"

    async def test_login_nonexistent_user(self, client: AsyncClient) -> None:
        """Login with nonexistent email returns UNAUTHORIZED error."""
        response = await client.post(
            "/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "anypassword123",
            },
        )

        assert response.status_code == 401
        data = response.json()
        assert data["error"]["code"] == "UNAUTHORIZED"


class TestRefresh:
    """Tests for POST /auth/refresh."""

    async def test_refresh_success(self, client: AsyncClient) -> None:
        """Valid refresh token returns new access token."""
        # Register to get tokens
        register_response = await client.post(
            "/auth/register",
            json={
                "email": "refresh@example.com",
                "name": "Refresh User",
                "password": "securepassword123",
            },
        )
        refresh_token = register_response.json()["refresh_token"]

        # Refresh
        response = await client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

    async def test_refresh_invalid_token(self, client: AsyncClient) -> None:
        """Invalid refresh token returns UNAUTHORIZED error."""
        response = await client.post(
            "/auth/refresh",
            json={"refresh_token": "invalid.token.here"},
        )

        assert response.status_code == 401
        data = response.json()
        assert data["error"]["code"] == "UNAUTHORIZED"

    async def test_refresh_with_access_token(self, client: AsyncClient) -> None:
        """Using access token for refresh returns UNAUTHORIZED error."""
        # Register to get tokens
        register_response = await client.post(
            "/auth/register",
            json={
                "email": "accessref@example.com",
                "name": "Access Refresh User",
                "password": "securepassword123",
            },
        )
        access_token = register_response.json()["access_token"]

        # Try to use access token as refresh token
        response = await client.post(
            "/auth/refresh",
            json={"refresh_token": access_token},
        )

        assert response.status_code == 401


class TestMe:
    """Tests for GET /me."""

    async def test_me_success(self, client: AsyncClient) -> None:
        """GET /me with valid token returns user info."""
        # Register
        register_response = await client.post(
            "/auth/register",
            json={
                "email": "me@example.com",
                "name": "Me User",
                "password": "securepassword123",
            },
        )
        access_token = register_response.json()["access_token"]

        # Get /me
        response = await client.get(
            "/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "user" in data
        assert data["user"]["email"] == "me@example.com"
        assert data["user"]["name"] == "Me User"
        # No team membership yet
        assert data["primary_team"] is None

    async def test_me_without_token(self, client: AsyncClient) -> None:
        """GET /me without token returns UNAUTHORIZED error."""
        response = await client.get("/me")

        assert response.status_code == 401
        data = response.json()
        assert data["error"]["code"] == "UNAUTHORIZED"

    async def test_me_with_invalid_token(self, client: AsyncClient) -> None:
        """GET /me with invalid token returns UNAUTHORIZED error."""
        response = await client.get(
            "/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )

        assert response.status_code == 401
        data = response.json()
        assert data["error"]["code"] == "UNAUTHORIZED"

    async def test_me_with_malformed_header(self, client: AsyncClient) -> None:
        """GET /me with malformed auth header returns UNAUTHORIZED error."""
        response = await client.get(
            "/me",
            headers={"Authorization": "InvalidFormat"},
        )

        assert response.status_code == 401
        data = response.json()
        assert data["error"]["code"] == "UNAUTHORIZED"


class TestHappyPath:
    """End-to-end happy path tests."""

    async def test_register_login_me_flow(self, client: AsyncClient) -> None:
        """Full happy path: register → login → /me."""
        # 1. Register
        register_response = await client.post(
            "/auth/register",
            json={
                "email": "happypath@example.com",
                "name": "Happy Path User",
                "password": "securepassword123",
            },
        )
        assert register_response.status_code == 201
        register_data = register_response.json()
        user_id = register_data["user"]["id"]

        # 2. Login
        login_response = await client.post(
            "/auth/login",
            json={
                "email": "happypath@example.com",
                "password": "securepassword123",
            },
        )
        assert login_response.status_code == 200
        login_data = login_response.json()
        assert login_data["user"]["id"] == user_id

        # 3. Get /me
        me_response = await client.get(
            "/me",
            headers={"Authorization": f"Bearer {login_data['access_token']}"},
        )
        assert me_response.status_code == 200
        me_data = me_response.json()
        assert me_data["user"]["id"] == user_id
        assert me_data["user"]["email"] == "happypath@example.com"
        assert me_data["user"]["name"] == "Happy Path User"


class TestMeWithTeamMembership:
    """Tests for GET /me with team membership."""

    async def test_me_with_team_membership(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """GET /me returns primary team when user has membership."""
        # 1. Register user
        register_response = await client.post(
            "/auth/register",
            json={
                "email": "teammember@example.com",
                "name": "Team Member",
                "password": "securepassword123",
            },
        )
        access_token = register_response.json()["access_token"]
        user_id = register_response.json()["user"]["id"]

        # 2. Create team and membership directly in DB
        team_id = uuid.uuid4()
        await db_session.execute(
            text("INSERT INTO teams (id, name) VALUES (:id, :name)"),
            {"id": team_id, "name": "Test Team"},
        )
        await db_session.execute(
            text("""
                INSERT INTO team_memberships (id, team_id, user_id, role, section)
                VALUES (:id, :team_id, :user_id, :role, :section)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": team_id,
                "user_id": uuid.UUID(user_id),
                "role": "ADMIN",
                "section": "Soprano",
            },
        )
        await db_session.commit()

        # 3. Get /me - should include team membership
        me_response = await client.get(
            "/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert me_response.status_code == 200
        me_data = me_response.json()

        assert me_data["primary_team"] is not None
        assert me_data["primary_team"]["team_id"] == str(team_id)
        assert me_data["primary_team"]["role"] == "ADMIN"
        assert me_data["primary_team"]["section"] == "Soprano"
