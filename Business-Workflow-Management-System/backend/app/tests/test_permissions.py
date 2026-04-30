"""Role-based permission tests."""
from __future__ import annotations

from sqlalchemy import select

from app.tests.conftest import auth_header


def _create_task(client, token, seeded_users):
    payload = {
        "title": "Investigate AP discrepancy",
        "description": "",
        "priority": "medium",
        "assigned_to_user_id": seeded_users["analyst"].id,
    }
    return client.post("/tasks", json=payload, headers=auth_header(token))


def test_analyst_cannot_approve_task(client, seeded_users, analyst_token):
    create = _create_task(client, analyst_token, seeded_users)
    task_id = create.json()["id"]
    client.post(f"/tasks/{task_id}/submit", headers=auth_header(analyst_token))
    approve = client.post(
        f"/tasks/{task_id}/approve", headers=auth_header(analyst_token)
    )
    assert approve.status_code == 403


def test_analyst_cannot_reject_task(client, seeded_users, analyst_token):
    create = _create_task(client, analyst_token, seeded_users)
    task_id = create.json()["id"]
    client.post(f"/tasks/{task_id}/submit", headers=auth_header(analyst_token))
    reject = client.post(
        f"/tasks/{task_id}/reject", headers=auth_header(analyst_token)
    )
    assert reject.status_code == 403


def test_analyst_cannot_complete_task(client, seeded_users, analyst_token, manager_token):
    create = _create_task(client, analyst_token, seeded_users)
    task_id = create.json()["id"]
    client.post(f"/tasks/{task_id}/submit", headers=auth_header(analyst_token))
    client.post(f"/tasks/{task_id}/approve", headers=auth_header(manager_token))
    complete = client.post(
        f"/tasks/{task_id}/complete", headers=auth_header(analyst_token)
    )
    assert complete.status_code == 403


def test_manager_can_approve(client, seeded_users, analyst_token, manager_token):
    create = _create_task(client, analyst_token, seeded_users)
    task_id = create.json()["id"]
    client.post(f"/tasks/{task_id}/submit", headers=auth_header(analyst_token))
    approve = client.post(
        f"/tasks/{task_id}/approve", headers=auth_header(manager_token)
    )
    assert approve.status_code == 200


def test_audit_log_records_approve_and_reject(
    client, seeded_users, analyst_token, manager_token, db_session
):
    from app.models.audit_log import AuditAction, AuditLog

    create = _create_task(client, analyst_token, seeded_users)
    task_id = create.json()["id"]
    client.post(f"/tasks/{task_id}/submit", headers=auth_header(analyst_token))
    client.post(f"/tasks/{task_id}/approve", headers=auth_header(manager_token))

    db_session.expire_all()
    actions = {log.action for log in db_session.scalars(select(AuditLog)).all()}
    assert AuditAction.SUBMIT_TASK in actions
    assert AuditAction.APPROVE_TASK in actions
