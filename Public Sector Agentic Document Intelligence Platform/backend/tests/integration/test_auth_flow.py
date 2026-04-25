"""Integration tests for register / login / refresh / me."""
from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from tests.conftest import requires_postgres

pytestmark = [requires_postgres, pytest.mark.asyncio]


async def test_register_creates_user_and_returns_tokens(
    client: AsyncClient, db_session, register_payload: dict[str, Any]  # type: ignore[no-untyped-def]
) -> None:
    resp = await client.post("/api/v1/auth/register", json=register_payload)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["user"]["email"] == register_payload["email"]
    assert body["organization"]["name"] == register_payload["organization_name"]
    assert body["access_token"]
    assert body["refresh_token"]


async def test_register_duplicate_email_returns_409(
    client: AsyncClient, db_session, register_payload  # type: ignore[no-untyped-def]
) -> None:
    first = await client.post("/api/v1/auth/register", json=register_payload)
    assert first.status_code == 201
    second = await client.post("/api/v1/auth/register", json=register_payload)
    assert second.status_code == 409


async def test_login_success_and_failure(
    client: AsyncClient, db_session, register_payload  # type: ignore[no-untyped-def]
) -> None:
    await client.post("/api/v1/auth/register", json=register_payload)

    ok = await client.post(
        "/api/v1/auth/login",
        json={
            "email": register_payload["email"],
            "password": register_payload["password"],
        },
    )
    assert ok.status_code == 200
    assert "access_token" in ok.json()

    bad = await client.post(
        "/api/v1/auth/login",
        json={"email": register_payload["email"], "password": "wrong-password-123"},
    )
    assert bad.status_code == 401


async def test_me_requires_bearer_and_returns_user(
    client: AsyncClient, db_session, register_payload  # type: ignore[no-untyped-def]
) -> None:
    reg = await client.post("/api/v1/auth/register", json=register_payload)
    access = reg.json()["access_token"]

    unauth = await client.get("/api/v1/auth/me")
    assert unauth.status_code == 401

    me = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {access}"}
    )
    assert me.status_code == 200
    assert me.json()["email"] == register_payload["email"]


async def test_refresh_returns_new_access_token(
    client: AsyncClient, db_session, register_payload  # type: ignore[no-untyped-def]
) -> None:
    reg = await client.post("/api/v1/auth/register", json=register_payload)
    refresh = reg.json()["refresh_token"]

    resp = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
    assert resp.status_code == 200
    assert resp.json()["access_token"]


async def test_health_returns_component_statuses(
    client: AsyncClient, db_session  # type: ignore[no-untyped-def]
) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    for component in ("db", "redis", "mlflow", "openai"):
        assert component in body
