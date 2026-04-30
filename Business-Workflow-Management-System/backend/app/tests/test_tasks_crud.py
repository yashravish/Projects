"""Task CRUD tests."""
from __future__ import annotations

from sqlalchemy import select

from app.tests.conftest import auth_header


def _create_task(client, token, seeded_users, **overrides):
    payload = {
        "title": overrides.get("title", "Reconcile invoices"),
        "description": overrides.get("description", "Match invoices to GL"),
        "priority": overrides.get("priority", "medium"),
        "assigned_to_user_id": overrides.get(
            "assigned_to_user_id", seeded_users["analyst"].id
        ),
    }
    return client.post("/tasks", json=payload, headers=auth_header(token))


def test_create_task_returns_201(client, seeded_users, analyst_token):
    response = _create_task(client, analyst_token, seeded_users)
    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Reconcile invoices"
    assert body["status"] == "pending"
    assert body["priority"] == "medium"
    assert body["assignee"]["id"] == seeded_users["analyst"].id


def test_list_tasks(client, seeded_users, analyst_token):
    _create_task(client, analyst_token, seeded_users, title="A")
    _create_task(client, analyst_token, seeded_users, title="B")
    response = client.get("/tasks", headers=auth_header(analyst_token))
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    titles = {t["title"] for t in body}
    assert titles == {"A", "B"}


def test_get_task_404(client, analyst_token):
    response = client.get("/tasks/9999", headers=auth_header(analyst_token))
    assert response.status_code == 404


def test_update_task(client, seeded_users, analyst_token):
    create_resp = _create_task(client, analyst_token, seeded_users)
    task_id = create_resp.json()["id"]
    update = client.put(
        f"/tasks/{task_id}",
        json={"title": "Updated title", "priority": "high"},
        headers=auth_header(analyst_token),
    )
    assert update.status_code == 200
    body = update.json()
    assert body["title"] == "Updated title"
    assert body["priority"] == "high"


def test_analyst_cannot_update_others_task(client, seeded_users, analyst_token, manager_token):
    response = _create_task(
        client,
        manager_token,
        seeded_users,
        assigned_to_user_id=seeded_users["analyst"].id,
    )
    task_id = response.json()["id"]
    update = client.put(
        f"/tasks/{task_id}",
        json={"title": "hijacked"},
        headers=auth_header(analyst_token),
    )
    assert update.status_code == 403


def test_delete_pending_task(client, seeded_users, analyst_token):
    response = _create_task(client, analyst_token, seeded_users)
    task_id = response.json()["id"]
    delete = client.delete(f"/tasks/{task_id}", headers=auth_header(analyst_token))
    assert delete.status_code == 204
    follow_up = client.get(f"/tasks/{task_id}", headers=auth_header(analyst_token))
    assert follow_up.status_code == 404


def test_cannot_delete_approved_task(
    client, seeded_users, analyst_token, manager_token
):
    response = _create_task(client, analyst_token, seeded_users)
    task_id = response.json()["id"]
    client.post(f"/tasks/{task_id}/submit", headers=auth_header(analyst_token))
    client.post(f"/tasks/{task_id}/approve", headers=auth_header(manager_token))
    delete = client.delete(f"/tasks/{task_id}", headers=auth_header(analyst_token))
    assert delete.status_code == 400


def test_create_task_creates_audit_log(client, seeded_users, analyst_token, db_session):
    from app.models.audit_log import AuditAction, AuditLog

    _create_task(client, analyst_token, seeded_users)
    db_session.expire_all()
    logs = list(
        db_session.scalars(
            select(AuditLog).where(AuditLog.action == AuditAction.CREATE_TASK)
        ).all()
    )
    assert len(logs) == 1
