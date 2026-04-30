"""Authentication tests."""
from __future__ import annotations

from sqlalchemy import select

from app.tests.conftest import auth_header


def test_login_success(client, seeded_users):
    response = client.post(
        "/auth/login",
        json={"email": "analyst@test.com", "password": "password123"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["user"]["email"] == "analyst@test.com"
    assert body["user"]["role"] == "analyst"


def test_login_wrong_password(client, seeded_users):
    response = client.post(
        "/auth/login",
        json={"email": "analyst@test.com", "password": "wrongpassword"},
    )
    assert response.status_code == 401


def test_login_unknown_email(client, seeded_users):
    response = client.post(
        "/auth/login",
        json={"email": "ghost@test.com", "password": "password123"},
    )
    assert response.status_code == 401


def test_me_requires_authentication(client):
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_me_returns_current_user(client, seeded_users, manager_token):
    response = client.get("/auth/me", headers=auth_header(manager_token))
    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "manager@test.com"
    assert body["role"] == "manager"


def test_protected_tasks_route_rejects_unauthenticated(client):
    response = client.get("/tasks")
    assert response.status_code == 401


def test_login_creates_audit_log(client, seeded_users, db_session):
    from app.models.audit_log import AuditAction, AuditLog

    client.post(
        "/auth/login",
        json={"email": "manager@test.com", "password": "password123"},
    )
    db_session.expire_all()
    logs = list(
        db_session.scalars(
            select(AuditLog).where(AuditLog.action == AuditAction.LOGIN)
        ).all()
    )
    assert any(log.user_id == seeded_users["manager"].id for log in logs)
