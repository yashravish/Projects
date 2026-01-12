"""Compliance insights endpoint tests."""

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
from app.services import openai_summary

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


def auth_header(access_token: str) -> dict[str, str]:
    """Create authorization header from access token."""
    return {"Authorization": f"Bearer {access_token}"}


async def register_user(
    client: AsyncClient, email: str, name: str, password: str = "password123"
) -> dict:
    response = await client.post(
        "/auth/register",
        json={"email": email, "name": name, "password": password},
    )
    assert response.status_code == 201, response.json()
    return response.json()


async def create_team(client: AsyncClient, auth_data: dict) -> str:
    response = await client.post(
        "/teams",
        json={"name": "Insights Team"},
        headers=auth_header(auth_data["access_token"]),
    )
    assert response.status_code == 201
    return response.json()["team"]["id"]


async def create_cycle(client: AsyncClient, team_id: str, auth_data: dict) -> str:
    future_date = datetime.now(UTC) + timedelta(days=2)
    response = await client.post(
        f"/teams/{team_id}/cycles",
        json={"date": future_date.strftime("%Y-%m-%d"), "label": "Insights Cycle"},
        headers=auth_header(auth_data["access_token"]),
    )
    assert response.status_code == 201
    return response.json()["cycle"]["id"]


async def create_practice_log(
    client: AsyncClient, auth_data: dict, cycle_id: str, occurred_at: datetime
) -> None:
    response = await client.post(
        f"/cycles/{cycle_id}/practice-logs",
        headers=auth_header(auth_data["access_token"]),
        json={
            "duration_min": 30,
            "assignment_ids": [],
            "blocked_flag": False,
            "occurred_at": occurred_at.isoformat(),
        },
    )
    assert response.status_code == 201, response.json()


async def test_compliance_insights_returns_ai_summary(
    client: AsyncClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compliance insights returns aggregates and AI summary when mocked."""
    admin_auth = await register_user(client, "insights-admin@test.com", "Insights Admin")
    team_id = await create_team(client, admin_auth)
    cycle_id = await create_cycle(client, team_id, admin_auth)

    member1_email = "member1@test.com"
    member2_email = "member2@test.com"

    member1_auth = await register_user(client, member1_email, "Member One")
    member2_auth = await register_user(client, member2_email, "Member Two")

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
            "user_id": uuid.UUID(member1_auth["user"]["id"]),
            "role": "MEMBER",
            "section": "Soprano",
        },
    )
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
            "user_id": uuid.UUID(member2_auth["user"]["id"]),
            "role": "MEMBER",
            "section": "Tenor",
        },
    )
    await db_session.commit()

    now = datetime.now(UTC)
    await create_practice_log(client, member1_auth, cycle_id, now - timedelta(days=1))
    await create_practice_log(client, member1_auth, cycle_id, now - timedelta(days=2))
    await create_practice_log(client, member1_auth, cycle_id, now - timedelta(days=3))
    await create_practice_log(client, member2_auth, cycle_id, now - timedelta(days=1))

    monkeypatch.setattr(settings, "openai_api_key", "test-key")

    async def fake_request_openai_summary(prompt: str) -> str:
        assert "Soprano" in prompt
        return "AI summary for compliance."

    monkeypatch.setattr(openai_summary, "_request_openai_summary", fake_request_openai_summary)

    response = await client.get(
        f"/teams/{team_id}/dashboards/leader/compliance-insights",
        headers=auth_header(admin_auth["access_token"]),
    )

    assert response.status_code == 200, response.json()
    data = response.json()
    assert data["summary"] == "AI summary for compliance."
    assert data["summary_source"] == "openai"
    sections = {section["section"] for section in data["sections"]}
    assert {"Soprano", "Tenor"}.issubset(sections)


async def test_compliance_insights_forbidden_for_member(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Non-leader members cannot access compliance insights."""
    admin_auth = await register_user(client, "insights-admin2@test.com", "Insights Admin 2")
    team_id = await create_team(client, admin_auth)

    member_email = "member3@test.com"
    member_auth = await register_user(client, member_email, "Member Three")

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
            "user_id": uuid.UUID(member_auth["user"]["id"]),
            "role": "MEMBER",
            "section": "Alto",
        },
    )
    await db_session.commit()

    response = await client.get(
        f"/teams/{team_id}/dashboards/leader/compliance-insights",
        headers=auth_header(member_auth["access_token"]),
    )

    assert response.status_code == 403
