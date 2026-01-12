"""Tests for teams, memberships, and invites.

Tests cover:
- Happy Paths:
  - Create invite → accept while logged out → account + membership created
  - Create invite → accept while logged in → membership created
- Authorization:
  - MEMBER cannot create invite → FORBIDDEN
- Edge Cases:
  - Expired token → FORBIDDEN (INVITE_EXPIRED)
  - Reused token → CONFLICT (INVITE_ALREADY_USED)
  - Email exists but not logged in → CONFLICT (ACCOUNT_EXISTS_LOGIN_REQUIRED)
- Pagination:
  - Members list pagination respects sorting and cursor contract
"""

import hashlib
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
            await session.execute(text("DELETE FROM invites"))
            await session.execute(text("DELETE FROM team_memberships"))
            await session.execute(text("DELETE FROM teams"))
            await session.execute(text("DELETE FROM users"))
            await session.commit()

            yield session

            # Clean up after test
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
    client: AsyncClient, email: str, name: str = "Test User", password: str = "password123"
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


def hash_token(token: str) -> str:
    """Hash a token using SHA-256 (same as in routes/teams.py)."""
    return hashlib.sha256(token.encode()).hexdigest()


class TestCreateTeam:
    """Tests for POST /teams."""

    async def test_create_team_success(self, client: AsyncClient) -> None:
        """Any authenticated user can create a team."""
        # Register a user
        auth_data = await register_user(client, "creator@example.com")

        # Create team
        response = await client.post(
            "/teams",
            json={"name": "Test Team"},
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 201
        data = response.json()
        assert "team" in data
        assert data["team"]["name"] == "Test Team"
        assert "id" in data["team"]

    async def test_create_team_creator_becomes_admin(self, client: AsyncClient) -> None:
        """Creator automatically becomes ADMIN of the new team."""
        auth_data = await register_user(client, "admin@example.com")

        # Create team
        create_response = await client.post(
            "/teams",
            json={"name": "Admin Test Team"},
            headers=auth_header(auth_data["access_token"]),
        )
        assert create_response.status_code == 201
        team_id = create_response.json()["team"]["id"]

        # List members to verify role
        members_response = await client.get(
            f"/teams/{team_id}/members",
            headers=auth_header(auth_data["access_token"]),
        )
        assert members_response.status_code == 200
        members = members_response.json()["items"]
        assert len(members) == 1
        assert members[0]["role"] == "ADMIN"
        assert members[0]["email"] == "admin@example.com"

    async def test_create_team_requires_auth(self, client: AsyncClient) -> None:
        """Creating team without auth fails."""
        response = await client.post(
            "/teams",
            json={"name": "No Auth Team"},
        )
        assert response.status_code == 401


class TestListMembers:
    """Tests for GET /teams/{team_id}/members."""

    async def test_list_members_admin(self, client: AsyncClient) -> None:
        """ADMIN can list team members."""
        auth_data = await register_user(client, "admin@example.com")

        # Create team (user becomes ADMIN)
        create_response = await client.post(
            "/teams",
            json={"name": "Test Team"},
            headers=auth_header(auth_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # List members
        response = await client.get(
            f"/teams/{team_id}/members",
            headers=auth_header(auth_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "next_cursor" in data

    async def test_list_members_member_forbidden(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """MEMBER cannot list team members."""
        # Create admin user and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Test Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create member user
        member_data = await register_user(client, "member@example.com")

        # Manually add member to team with MEMBER role
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

        # Member tries to list members
        response = await client.get(
            f"/teams/{team_id}/members",
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_list_members_pagination(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Members list respects pagination and deterministic ordering."""
        # Create admin
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Pagination Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Add more members to the team
        for i in range(5):
            user_data = await register_user(client, f"member{i}@example.com")
            await db_session.execute(
                text("""
                    INSERT INTO team_memberships (id, team_id, user_id, role, created_at)
                    VALUES (:id, :team_id, :user_id, :role, :created_at)
                """),
                {
                    "id": uuid.uuid4(),
                    "team_id": uuid.UUID(team_id),
                    "user_id": uuid.UUID(user_data["user"]["id"]),
                    "role": "MEMBER",
                    "created_at": datetime.now(UTC) + timedelta(seconds=i + 1),
                },
            )
        await db_session.commit()

        # Request with limit=2 to test pagination
        response = await client.get(
            f"/teams/{team_id}/members?limit=2",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["next_cursor"] is not None

        # Follow cursor
        cursor = data["next_cursor"]
        response2 = await client.get(
            f"/teams/{team_id}/members?limit=2&cursor={cursor}",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert len(data2["items"]) == 2

        # Verify no duplicate items between pages
        page1_ids = {item["id"] for item in data["items"]}
        page2_ids = {item["id"] for item in data2["items"]}
        assert page1_ids.isdisjoint(page2_ids)


class TestUpdateMember:
    """Tests for PATCH /teams/{team_id}/members/{user_id}."""

    async def test_update_member_admin_only(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """ADMIN can update member role and section."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Update Test Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create member user
        member_data = await register_user(client, "member@example.com")
        member_id = member_data["user"]["id"]

        # Add member to team
        await db_session.execute(
            text("""
                INSERT INTO team_memberships (id, team_id, user_id, role)
                VALUES (:id, :team_id, :user_id, :role)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "user_id": uuid.UUID(member_id),
                "role": "MEMBER",
            },
        )
        await db_session.commit()

        # Admin updates member
        response = await client.patch(
            f"/teams/{team_id}/members/{member_id}",
            json={"role": "SECTION_LEADER", "section": "Soprano"},
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["membership"]["role"] == "SECTION_LEADER"
        assert data["membership"]["section"] == "Soprano"

    async def test_update_member_member_forbidden(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """MEMBER cannot update other members."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Update Test Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create two member users
        member1_data = await register_user(client, "member1@example.com")
        member2_data = await register_user(client, "member2@example.com")

        # Add both members to team
        for member_data in [member1_data, member2_data]:
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

        # Member1 tries to update Member2
        response = await client.patch(
            f"/teams/{team_id}/members/{member2_data['user']['id']}",
            json={"role": "ADMIN"},
            headers=auth_header(member1_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"


class TestCreateInvite:
    """Tests for POST /teams/{team_id}/invites."""

    async def test_create_invite_admin(self, client: AsyncClient) -> None:
        """ADMIN can create team invites."""
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Invite Test Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create invite
        response = await client.post(
            f"/teams/{team_id}/invites",
            json={"role": "MEMBER", "expires_in_hours": 168},
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 201
        data = response.json()
        assert "invite_link" in data
        assert "/invites/" in data["invite_link"]

    async def test_create_invite_member_forbidden(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """MEMBER cannot create invites."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Invite Test Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create member user
        member_data = await register_user(client, "member@example.com")

        # Add member to team
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

        # Member tries to create invite
        response = await client.post(
            f"/teams/{team_id}/invites",
            json={"role": "MEMBER"},
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"


class TestInviteAcceptance:
    """Tests for invite acceptance flows."""

    async def test_accept_invite_logged_out_creates_account(
        self, client: AsyncClient
    ) -> None:
        """Create invite → accept while logged out → account + membership created."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Invite Accept Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create invite
        invite_response = await client.post(
            f"/teams/{team_id}/invites",
            json={"role": "MEMBER", "section": "Alto"},
            headers=auth_header(admin_data["access_token"]),
        )
        invite_link = invite_response.json()["invite_link"]
        token = invite_link.split("/invites/")[1]

        # Accept invite while logged out (create new account)
        accept_response = await client.post(
            f"/invites/{token}/accept",
            json={
                "name": "New User",
                "email": "newuser@example.com",
                "password": "newpassword123",
            },
        )

        assert accept_response.status_code == 201
        data = accept_response.json()
        assert data["membership"]["role"] == "MEMBER"
        assert data["membership"]["section"] == "Alto"
        assert data["access_token"] is not None
        assert data["refresh_token"] is not None

    async def test_accept_invite_logged_in_creates_membership(
        self, client: AsyncClient
    ) -> None:
        """Create invite → accept while logged in → membership created."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Logged In Accept Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create another user (not part of team)
        user_data = await register_user(client, "existinguser@example.com")

        # Create invite
        invite_response = await client.post(
            f"/teams/{team_id}/invites",
            json={"role": "MEMBER", "section": "Tenor"},
            headers=auth_header(admin_data["access_token"]),
        )
        invite_link = invite_response.json()["invite_link"]
        token = invite_link.split("/invites/")[1]

        # Accept invite while logged in
        accept_response = await client.post(
            f"/invites/{token}/accept",
            json={},
            headers=auth_header(user_data["access_token"]),
        )

        assert accept_response.status_code == 201
        data = accept_response.json()
        assert data["membership"]["role"] == "MEMBER"
        assert data["membership"]["section"] == "Tenor"
        # No tokens returned for logged in users
        assert data["access_token"] is None
        assert data["refresh_token"] is None

    async def test_accept_invite_expired_token(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Expired token → FORBIDDEN."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Expired Token Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create an expired invite directly in DB
        raw_token = "expired-test-token"
        token_hash = hash_token(raw_token)
        await db_session.execute(
            text("""
                INSERT INTO invites (id, team_id, token, role, expires_at, created_by)
                VALUES (:id, :team_id, :token, :role, :expires_at, :created_by)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "token": token_hash,
                "role": "MEMBER",
                "expires_at": datetime.now(UTC) - timedelta(hours=1),  # Expired
                "created_by": uuid.UUID(admin_data["user"]["id"]),
            },
        )
        await db_session.commit()

        # Try to accept expired invite
        response = await client.post(
            f"/invites/{raw_token}/accept",
            json={
                "name": "New User",
                "email": "newuser@example.com",
                "password": "password123",
            },
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"
        assert "expired" in response.json()["error"]["message"].lower()

    async def test_accept_invite_reused_token(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Reused token → CONFLICT."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Reused Token Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create an already-used invite directly in DB
        raw_token = "used-test-token"
        token_hash = hash_token(raw_token)
        await db_session.execute(
            text("""
                INSERT INTO invites (id, team_id, token, role, expires_at, used_at, created_by)
                VALUES (:id, :team_id, :token, :role, :expires_at, :used_at, :created_by)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "token": token_hash,
                "role": "MEMBER",
                "expires_at": datetime.now(UTC) + timedelta(hours=168),
                "used_at": datetime.now(UTC),  # Already used
                "created_by": uuid.UUID(admin_data["user"]["id"]),
            },
        )
        await db_session.commit()

        # Try to accept already-used invite
        response = await client.post(
            f"/invites/{raw_token}/accept",
            json={
                "name": "New User",
                "email": "newuser@example.com",
                "password": "password123",
            },
        )

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "CONFLICT"
        assert "already been used" in response.json()["error"]["message"].lower()

    async def test_accept_invite_account_exists_login_required(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Email exists but not logged in → CONFLICT (ACCOUNT_EXISTS_LOGIN_REQUIRED)."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Account Exists Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create a user that already exists
        await register_user(client, "existing@example.com")

        # Create invite with existing email
        raw_token = "account-exists-token"
        token_hash = hash_token(raw_token)
        await db_session.execute(
            text("""
                INSERT INTO invites (id, team_id, token, email, role, expires_at, created_by)
                VALUES (:id, :team_id, :token, :email, :role, :expires_at, :created_by)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "token": token_hash,
                "email": "existing@example.com",
                "role": "MEMBER",
                "expires_at": datetime.now(UTC) + timedelta(hours=168),
                "created_by": uuid.UUID(admin_data["user"]["id"]),
            },
        )
        await db_session.commit()

        # Try to accept invite without logging in
        response = await client.post(
            f"/invites/{raw_token}/accept",
            json={
                "name": "New User",
                "email": "existing@example.com",
                "password": "password123",
            },
        )

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "CONFLICT"
        assert "log in" in response.json()["error"]["message"].lower()


class TestInvitePreview:
    """Tests for GET /invites/{token}."""

    async def test_preview_invite_valid(
        self, client: AsyncClient
    ) -> None:
        """Preview shows invite details for valid token."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Preview Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create invite
        invite_response = await client.post(
            f"/teams/{team_id}/invites",
            json={"role": "MEMBER", "section": "Bass", "email": "invitee@example.com"},
            headers=auth_header(admin_data["access_token"]),
        )
        invite_link = invite_response.json()["invite_link"]
        token = invite_link.split("/invites/")[1]

        # Preview invite
        response = await client.get(f"/invites/{token}")

        assert response.status_code == 200
        data = response.json()
        assert data["team_name"] == "Preview Team"
        assert data["role"] == "MEMBER"
        assert data["section"] == "Bass"
        assert data["email"] == "invitee@example.com"
        assert data["expired"] is False

    async def test_preview_invite_invalid_token(self, client: AsyncClient) -> None:
        """Preview returns 404 for invalid token."""
        response = await client.get("/invites/invalid-token")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"

    async def test_preview_invite_shows_expired(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Preview shows expired=true for expired invites."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Expired Preview Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create an expired invite directly in DB
        raw_token = "expired-preview-token"
        token_hash = hash_token(raw_token)
        await db_session.execute(
            text("""
                INSERT INTO invites (id, team_id, token, role, expires_at, created_by)
                VALUES (:id, :team_id, :token, :role, :expires_at, :created_by)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "token": token_hash,
                "role": "MEMBER",
                "expires_at": datetime.now(UTC) - timedelta(hours=1),
                "created_by": uuid.UUID(admin_data["user"]["id"]),
            },
        )
        await db_session.commit()

        # Preview expired invite
        response = await client.get(f"/invites/{raw_token}")

        assert response.status_code == 200
        assert response.json()["expired"] is True


class TestUpdateTeam:
    """Tests for PATCH /teams/{team_id}."""

    async def test_update_team_admin(self, client: AsyncClient) -> None:
        """ADMIN can update team name."""
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Original Name"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Update team name
        response = await client.patch(
            f"/teams/{team_id}",
            json={"name": "Updated Name"},
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["team"]["name"] == "Updated Name"
        assert data["team"]["id"] == team_id

    async def test_update_team_member_forbidden(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """MEMBER cannot update team."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Original Name"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create member
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

        # Member tries to update team
        response = await client.patch(
            f"/teams/{team_id}",
            json={"name": "Hacked Name"},
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

    async def test_update_team_not_found(self, client: AsyncClient) -> None:
        """Update non-existent team returns 404."""
        admin_data = await register_user(client, "admin@example.com")
        fake_team_id = str(uuid.uuid4())

        response = await client.patch(
            f"/teams/{fake_team_id}",
            json={"name": "New Name"},
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 403  # Not found, but shown as forbidden for security


class TestRemoveMember:
    """Tests for DELETE /teams/{team_id}/members/{user_id}."""

    async def test_remove_member_admin(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """ADMIN can remove team members."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Remove Member Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create member
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

        # Admin removes member
        response = await client.delete(
            f"/teams/{team_id}/members/{member_data['user']['id']}",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Member removed successfully"

        # Verify member is removed
        members_response = await client.get(
            f"/teams/{team_id}/members",
            headers=auth_header(admin_data["access_token"]),
        )
        member_ids = [m["user_id"] for m in members_response.json()["items"]]
        assert member_data["user"]["id"] not in member_ids

    async def test_remove_last_admin_forbidden(self, client: AsyncClient) -> None:
        """Cannot remove the last admin from a team."""
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Last Admin Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Admin tries to remove themselves (last admin)
        response = await client.delete(
            f"/teams/{team_id}/members/{admin_data['user']['id']}",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 403
        assert "last admin" in response.json()["error"]["message"].lower()

    async def test_remove_member_not_found(
        self, client: AsyncClient
    ) -> None:
        """Remove non-existent member returns 404."""
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Remove Test Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        fake_user_id = str(uuid.uuid4())
        response = await client.delete(
            f"/teams/{team_id}/members/{fake_user_id}",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"


class TestListInvites:
    """Tests for GET /teams/{team_id}/invites."""

    async def test_list_invites_admin(self, client: AsyncClient) -> None:
        """ADMIN can list team invites."""
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "List Invites Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create some invites
        for i in range(3):
            await client.post(
                f"/teams/{team_id}/invites",
                json={"role": "MEMBER", "email": f"invitee{i}@example.com"},
                headers=auth_header(admin_data["access_token"]),
            )

        # List invites
        response = await client.get(
            f"/teams/{team_id}/invites",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 3
        assert "next_cursor" in data

    async def test_list_invites_excludes_used_by_default(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """By default, only unused invites are returned."""
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "List Invites Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create an unused invite
        await client.post(
            f"/teams/{team_id}/invites",
            json={"role": "MEMBER"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Create a used invite directly in DB
        raw_token = "used-invite-token"
        token_hash = hash_token(raw_token)
        await db_session.execute(
            text("""
                INSERT INTO invites (id, team_id, token, role, expires_at, used_at, created_by)
                VALUES (:id, :team_id, :token, :role, :expires_at, :used_at, :created_by)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "token": token_hash,
                "role": "MEMBER",
                "expires_at": datetime.now(UTC) + timedelta(hours=168),
                "used_at": datetime.now(UTC),
                "created_by": uuid.UUID(admin_data["user"]["id"]),
            },
        )
        await db_session.commit()

        # List invites (should only show unused)
        response = await client.get(
            f"/teams/{team_id}/invites",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["used_at"] is None

    async def test_list_invites_include_used(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """include_used=true returns both used and unused invites."""
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "List All Invites Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create an unused invite
        await client.post(
            f"/teams/{team_id}/invites",
            json={"role": "MEMBER"},
            headers=auth_header(admin_data["access_token"]),
        )

        # Create a used invite directly in DB
        raw_token = "used-invite-token"
        token_hash = hash_token(raw_token)
        await db_session.execute(
            text("""
                INSERT INTO invites (id, team_id, token, role, expires_at, used_at, created_by)
                VALUES (:id, :team_id, :token, :role, :expires_at, :used_at, :created_by)
            """),
            {
                "id": uuid.uuid4(),
                "team_id": uuid.UUID(team_id),
                "token": token_hash,
                "role": "MEMBER",
                "expires_at": datetime.now(UTC) + timedelta(hours=168),
                "used_at": datetime.now(UTC),
                "created_by": uuid.UUID(admin_data["user"]["id"]),
            },
        )
        await db_session.commit()

        # List all invites
        response = await client.get(
            f"/teams/{team_id}/invites?include_used=true",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2

    async def test_list_invites_member_forbidden(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """MEMBER cannot list invites."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "List Invites Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create member
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

        # Member tries to list invites
        response = await client.get(
            f"/teams/{team_id}/invites",
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"


class TestRevokeInvite:
    """Tests for DELETE /invites/{invite_id}."""

    async def test_revoke_invite_by_id(self, client: AsyncClient) -> None:
        """ADMIN can revoke invite by UUID."""
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Revoke Invite Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create invite
        invite_response = await client.post(
            f"/teams/{team_id}/invites",
            json={"role": "MEMBER"},
            headers=auth_header(admin_data["access_token"]),
        )
        invite_link = invite_response.json()["invite_link"]
        token = invite_link.split("/invites/")[1]

        # Get invite ID from list
        list_response = await client.get(
            f"/teams/{team_id}/invites",
            headers=auth_header(admin_data["access_token"]),
        )
        invite_id = list_response.json()["items"][0]["id"]

        # Revoke by ID
        response = await client.delete(
            f"/invites/{invite_id}",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Invite revoked successfully"

        # Verify invite is gone
        preview_response = await client.get(f"/invites/{token}")
        assert preview_response.status_code == 404

    async def test_revoke_invite_by_token(self, client: AsyncClient) -> None:
        """ADMIN can revoke invite by raw token."""
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Revoke Token Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create invite
        invite_response = await client.post(
            f"/teams/{team_id}/invites",
            json={"role": "MEMBER"},
            headers=auth_header(admin_data["access_token"]),
        )
        invite_link = invite_response.json()["invite_link"]
        token = invite_link.split("/invites/")[1]

        # Revoke by token
        response = await client.delete(
            f"/invites/{token}",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Invite revoked successfully"

    async def test_revoke_used_invite_conflict(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Cannot revoke an already-used invite."""
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Used Invite Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create a used invite directly in DB
        invite_id = uuid.uuid4()
        raw_token = "used-revoke-token"
        token_hash = hash_token(raw_token)
        await db_session.execute(
            text("""
                INSERT INTO invites (id, team_id, token, role, expires_at, used_at, created_by)
                VALUES (:id, :team_id, :token, :role, :expires_at, :used_at, :created_by)
            """),
            {
                "id": invite_id,
                "team_id": uuid.UUID(team_id),
                "token": token_hash,
                "role": "MEMBER",
                "expires_at": datetime.now(UTC) + timedelta(hours=168),
                "used_at": datetime.now(UTC),
                "created_by": uuid.UUID(admin_data["user"]["id"]),
            },
        )
        await db_session.commit()

        # Try to revoke used invite
        response = await client.delete(
            f"/invites/{invite_id}",
            headers=auth_header(admin_data["access_token"]),
        )

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "CONFLICT"

    async def test_revoke_invite_not_admin_forbidden(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Non-admin cannot revoke invites."""
        # Create admin and team
        admin_data = await register_user(client, "admin@example.com")
        create_response = await client.post(
            "/teams",
            json={"name": "Revoke Forbidden Team"},
            headers=auth_header(admin_data["access_token"]),
        )
        team_id = create_response.json()["team"]["id"]

        # Create invite
        invite_response = await client.post(
            f"/teams/{team_id}/invites",
            json={"role": "MEMBER"},
            headers=auth_header(admin_data["access_token"]),
        )
        invite_link = invite_response.json()["invite_link"]
        token = invite_link.split("/invites/")[1]

        # Create member
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

        # Member tries to revoke
        response = await client.delete(
            f"/invites/{token}",
            headers=auth_header(member_data["access_token"]),
        )

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "FORBIDDEN"

