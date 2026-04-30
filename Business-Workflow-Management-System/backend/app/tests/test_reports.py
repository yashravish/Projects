"""Reporting endpoint tests."""
from __future__ import annotations

from app.tests.conftest import auth_header


def _create_task(client, token, seeded_users, **overrides):
    payload = {
        "title": overrides.get("title", "Sample task"),
        "description": "",
        "priority": overrides.get("priority", "medium"),
        "assigned_to_user_id": overrides.get(
            "assigned_to_user_id", seeded_users["analyst"].id
        ),
    }
    return client.post("/tasks", json=payload, headers=auth_header(token))


def test_tasks_by_status_report(client, seeded_users, analyst_token, manager_token):
    _create_task(client, analyst_token, seeded_users, title="t1")
    create = _create_task(client, analyst_token, seeded_users, title="t2")
    task_id = create.json()["id"]
    client.post(f"/tasks/{task_id}/submit", headers=auth_header(analyst_token))

    response = client.get(
        "/reports/tasks-by-status", headers=auth_header(analyst_token)
    )
    assert response.status_code == 200
    body = response.json()
    counts = {item["status"]: item["count"] for item in body["items"]}
    assert counts["pending"] == 1
    assert counts["submitted"] == 1
    assert body["total"] == 2


def test_tasks_by_priority_report(client, seeded_users, analyst_token):
    _create_task(client, analyst_token, seeded_users, priority="high", title="a")
    _create_task(client, analyst_token, seeded_users, priority="high", title="b")
    _create_task(client, analyst_token, seeded_users, priority="low", title="c")
    response = client.get(
        "/reports/tasks-by-priority", headers=auth_header(analyst_token)
    )
    assert response.status_code == 200
    counts = {item["priority"]: item["count"] for item in response.json()["items"]}
    assert counts["high"] == 2
    assert counts["low"] == 1
    assert counts["medium"] == 0


def test_user_workload_report(client, seeded_users, analyst_token):
    _create_task(client, analyst_token, seeded_users, title="x")
    _create_task(client, analyst_token, seeded_users, title="y")
    response = client.get(
        "/reports/user-workload", headers=auth_header(analyst_token)
    )
    assert response.status_code == 200
    items = response.json()["items"]
    analyst_entry = next(
        item for item in items if item["user_id"] == seeded_users["analyst"].id
    )
    assert analyst_entry["total"] == 2
    assert analyst_entry["pending"] == 2


def test_approval_summary_report(
    client, seeded_users, analyst_token, manager_token
):
    create = _create_task(client, analyst_token, seeded_users, title="approve me")
    task_id = create.json()["id"]
    client.post(f"/tasks/{task_id}/submit", headers=auth_header(analyst_token))
    client.post(f"/tasks/{task_id}/approve", headers=auth_header(manager_token))

    create2 = _create_task(client, analyst_token, seeded_users, title="reject me")
    task2_id = create2.json()["id"]
    client.post(f"/tasks/{task2_id}/submit", headers=auth_header(analyst_token))
    client.post(f"/tasks/{task2_id}/reject", headers=auth_header(manager_token))

    response = client.get(
        "/reports/approval-summary", headers=auth_header(analyst_token)
    )
    assert response.status_code == 200
    counts = {item["decision"]: item["count"] for item in response.json()["items"]}
    assert counts["approved"] == 1
    assert counts["rejected"] == 1
    assert response.json()["total"] == 2


def test_csv_export(client, seeded_users, analyst_token):
    _create_task(client, analyst_token, seeded_users, title="csv item")
    response = client.get(
        "/reports/export/tasks.csv", headers=auth_header(analyst_token)
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    text = response.text
    assert "id,title" in text.splitlines()[0]
    assert "csv item" in text


def test_audit_logs_endpoint(client, seeded_users, analyst_token):
    _create_task(client, analyst_token, seeded_users)
    response = client.get("/audit-logs", headers=auth_header(analyst_token))
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert any(log["action"] == "create_task" for log in response.json())
